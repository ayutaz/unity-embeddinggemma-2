import json
import hashlib

import pytest
import torch
from safetensors.torch import save_file

from embeddinggemma_tools import audio_layer_replay
from test_audio_encoder import audio_models, mask


def capture():
    value = getattr(audio_layer_replay, "capture_parts", None)
    assert callable(value), "Pinned-input native layer replay is missing"
    return value


def fixture():
    audio, projection = audio_models()
    features = torch.randn(1, 25, 8, generator=torch.Generator().manual_seed(17))
    frame_mask = mask()
    with torch.no_grad():
        hidden, output_mask = audio.subsample_conv_projection(features, frame_mask.bool())
        positions = audio.rel_pos_enc(hidden)
    from embeddinggemma_tools.export import AudioEncoderExport
    blocked = AudioEncoderExport(audio, projection, frame_mask).blocked_mask
    with torch.no_grad():
        expected = audio.layers[0](hidden.clone(), attention_mask=blocked, position_embeddings=positions)
    return audio.layers[0], hidden, blocked, positions, expected


def test_replay_keeps_pinned_input_and_matches_independent_native_final():
    run = capture()
    layer, hidden, blocked, positions, expected = fixture()
    original = hidden.clone()
    values = run(layer, hidden, blocked, positions, expected)
    assert list(values) == ["feed_forward1", "norm_pre_attn", "self_attn", "norm_post_attn", "lconv1d", "feed_forward2", "norm_out"]
    assert torch.equal(hidden, original), "Replay must never mutate its pinned input"
    assert torch.equal(values["norm_out"], expected)
    assert all(torch.isfinite(value).all() for value in values.values())


@pytest.mark.parametrize("fail", [False, True])
def test_replay_preserves_external_hooks_and_removes_own_hooks(fail, monkeypatch):
    run = capture()
    layer, hidden, blocked, positions, expected = fixture()
    modules = [getattr(layer, name) for name in ("feed_forward1", "norm_pre_attn", "self_attn", "norm_post_attn", "lconv1d", "feed_forward2", "norm_out")]
    external = layer.norm_out.register_forward_hook(lambda *args: None)
    before = [set(value._forward_hooks) for value in modules]
    if fail:
        monkeypatch.setattr(layer.self_attn, "forward", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("fixture failure")))
    try:
        if fail:
            with pytest.raises(RuntimeError, match="fixture failure"):
                run(layer, hidden, blocked, positions, expected)
        else:
            run(layer, hidden, blocked, positions, expected)
        assert [set(value._forward_hooks) for value in modules] == before
    finally:
        external.remove()


def test_changed_native_final_is_rejected_before_any_result_is_accepted():
    run = capture()
    layer, hidden, blocked, positions, expected = fixture()
    with pytest.raises(ValueError, match="pinned.*final"):
        run(layer, hidden, blocked, positions, expected + 0.1)


def test_attention_projection_capture_matches_exact_native_inputs_and_final():
    run=getattr(audio_layer_replay,'capture_attention_projections',None)
    assert callable(run),'Pinned attention projection capture is missing'
    layer,hidden,blocked,positions,expected=fixture()
    original=hidden.clone();parts=capture()(layer,hidden,blocked,positions,expected)
    values=run(layer,hidden,blocked,positions,expected)
    assert list(values)==['q_proj','k_proj','v_proj','relative_k_proj','post']
    with torch.no_grad():
        for name in ['q_proj','k_proj','v_proj']:
            assert torch.equal(values[name],getattr(layer.self_attn,name)(parts['norm_pre_attn']))
        assert torch.equal(values['relative_k_proj'],layer.self_attn.relative_k_proj(positions))
    assert torch.equal(values['post'],parts['self_attn'])
    assert torch.equal(hidden,original)


def test_attention_projection_capture_rejects_changed_final_and_removes_hooks():
    run=getattr(audio_layer_replay,'capture_attention_projections',None)
    assert callable(run),'Pinned attention projection capture is missing'
    layer,hidden,blocked,positions,expected=fixture()
    modules=[layer.get_submodule('self_attn.'+name) for name in ['q_proj','k_proj','v_proj','relative_k_proj','post']]
    before=[set(module._forward_hooks) for module in modules]
    with pytest.raises(ValueError,match='pinned.*final'):run(layer,hidden,blocked,positions,expected+0.1)
    assert [set(module._forward_hooks) for module in modules]==before


def test_attention_operations_capture_preserves_native_final_and_bmm_results():
    run=getattr(audio_layer_replay,'capture_attention_operations',None)
    assert callable(run),'Pinned attention operation capture is missing'
    layer,hidden,blocked,positions,expected=fixture();before=hidden.clone()
    values=run(layer,hidden,blocked,positions,expected)
    assert len(values)==13
    for name in ['matrix_ac','matrix_bd','weighted_values']:
        assert torch.equal(values[name+'.output'],torch.bmm(values[name+'.left'],values[name+'.right']))
    assert torch.equal(values['softmax.output'],torch.softmax(values['softmax.input'],dim=-1))
    assert torch.equal(values['tanh.output'],torch.tanh(values['tanh.input']))
    assert torch.equal(hidden,before)


def test_attention_operation_capture_restores_external_hooks_on_failure():
    run=getattr(audio_layer_replay,'capture_attention_operations',None)
    assert callable(run),'Pinned attention operation capture is missing'
    layer,hidden,blocked,positions,expected=fixture();attention=layer.self_attn
    external=attention.register_forward_hook(lambda *args:None)
    before=(set(attention._forward_hooks),set(attention._forward_pre_hooks))
    try:
        with pytest.raises(ValueError,match='pinned.*final'):run(layer,hidden,blocked,positions,expected+0.1)
        assert (set(attention._forward_hooks),set(attention._forward_pre_hooks))==before
    finally:external.remove()


def test_producer_records_attention_operation_inputs_and_outputs(tmp_path):
    snapshot,context=bundle(tmp_path)
    report=preparer()(snapshot,context,tmp_path/'out',reference_source='a'*40,diagnostic_source='c'*40,detail_attention_ops=True)
    assert report['detail_attention_ops'] and report['native_final_exact_match']
    assert len(report['parts'])==20 and 'self_attn.matrix_ac.left' in report['parts']
    assert not report['full_encoder_validated'] and not report['unity_validated']


def test_convolution_capture_preserves_native_parts_and_exposes_glu_and_depthwise_operands():
    run=getattr(audio_layer_replay,'capture_convolution_parts',None)
    assert callable(run),'Pinned convolution capture is missing'
    layer,hidden,blocked,positions,expected=fixture();original=hidden.clone()
    values=run(layer,hidden,blocked,positions,expected)
    names=['pre_layer_norm','linear_start','depthwise_conv1d','conv_norm','act_fn','linear_end']
    assert set(values)=={n+'.'+k for n in names for k in ['input','output']}|{'depthwise_conv1d.weight','glu.input','glu.output'}
    with torch.no_grad():
        assert torch.equal(values['glu.output'],torch.nn.functional.glu(values['glu.input'],dim=-1))
        assert torch.equal(values['depthwise_conv1d.output'],layer.lconv1d.depthwise_conv1d(values['depthwise_conv1d.input']))
        assert torch.equal(values['act_fn.output'],layer.lconv1d.act_fn(values['act_fn.input']))
    assert torch.equal(values['depthwise_conv1d.weight'],layer.lconv1d.depthwise_conv1d.weight)
    assert torch.equal(values['glu.input'],values['linear_start.output'])
    assert torch.equal(values['glu.output'].transpose(1,2),values['depthwise_conv1d.input'])
    assert torch.equal(hidden,original)
    assert all(v.dtype==torch.float32 and torch.isfinite(v).all() for v in values.values())


@pytest.mark.parametrize('fail',[False,True])
def test_convolution_capture_restores_external_hooks_even_when_final_gate_fails(fail):
    run=getattr(audio_layer_replay,'capture_convolution_parts',None)
    assert callable(run),'Pinned convolution capture is missing'
    layer,hidden,blocked,positions,expected=fixture()
    modules=[layer.get_submodule('lconv1d.'+n) for n in ['pre_layer_norm','linear_start','depthwise_conv1d','conv_norm','act_fn','linear_end']]
    external=modules[0].register_forward_hook(lambda *args:None);before=[set(m._forward_hooks) for m in modules]
    try:
        if fail:
            with pytest.raises(ValueError,match='pinned.*final'):run(layer,hidden,blocked,positions,expected+.1)
        else:run(layer,hidden,blocked,positions,expected)
        assert [set(m._forward_hooks) for m in modules]==before
    finally:external.remove()


def test_producer_records_optional_convolution_inputs_outputs_and_original_weights(tmp_path):
    snapshot,context=bundle(tmp_path)
    report=preparer()(snapshot,context,tmp_path/'out',reference_source='a'*40,diagnostic_source='c'*40,detail_convolution=True)
    assert report['detail_convolution'] and report['native_final_exact_match']
    assert len(report['parts'])==22 and 'lconv1d.depthwise_conv1d.weight' in report['parts']
    assert not report['full_encoder_validated'] and not report['unity_validated']


def test_cli_passes_convolution_capture_flag_to_audited_producer(tmp_path,monkeypatch):
    calls=[]
    monkeypatch.setattr(audio_layer_replay,'prepare',lambda *args,**kwargs:calls.append(kwargs) or {'native_final_exact_match':True})
    result=audio_layer_replay.main(['--snapshot',str(tmp_path),'--context',str(tmp_path),'--output',str(tmp_path/'out'),
        '--reference-source','a'*40,'--diagnostic-source','c'*40,'--report',str(tmp_path/'report.json'),'--detail-convolution'])
    assert result==0 and calls[0]['detail_convolution'] is True


def test_loader_allocates_only_selected_layer_and_loads_exact_pinned_weights(tmp_path):
    load = getattr(audio_layer_replay, "load_layer", None)
    assert callable(load), "Selected audio layer loader is missing"
    audio, _ = audio_models()
    config = dict(audio_config=audio.config.to_dict())
    (tmp_path / "config.json").write_text(json.dumps(config))
    save_file({"audio_tower.layers.0." + key: value.contiguous() for key, value in audio.layers[0].state_dict().items()}, tmp_path / "model.safetensors")
    layer, loaded_config = load(tmp_path, index=0)
    assert loaded_config.hidden_size == audio.config.hidden_size
    assert layer.training is False
    assert all(torch.equal(value, audio.layers[0].state_dict()[key]) for key, value in layer.state_dict().items())
    assert len(layer.state_dict()) == len(audio.layers[0].state_dict())


def bundle(tmp_path):
    audio, _ = audio_models()
    layer, hidden, blocked, positions, expected = fixture()
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").write_text(json.dumps(dict(audio_config=audio.config.to_dict())))
    save_file({"audio_tower.layers.0." + key: value.contiguous() for key, value in layer.state_dict().items()}, snapshot / "model.safetensors")
    context = tmp_path / "context"
    context.mkdir()
    hidden.numpy().astype("<f4").tofile(context / "input.bin")
    expected.numpy().astype("<f4").tofile(context / "final.bin")
    def identity(path):
        data = path.read_bytes()
        return dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    metadata = dict(source_commit="a" * 40, model_revision="b" * 40,
                    input_sha256={name: identity(snapshot / name)["sha256"] for name in ("config.json", "model.safetensors")},
                    feature_rtol=1e-4, feature_atol=1e-5)
    reference = dict(metadata=metadata, profiles={"frames-25-valid-25": dict(feature_shape=[1, 25, 8])},
                     cases=[dict(id="silence", profile="frames-25-valid-25", stages=[
                         dict(name="subsample", shape=list(hidden.shape), expected="input.bin"),
                         dict(name="layer_0", shape=list(expected.shape), expected="final.bin")])])
    audit = dict(success=True, stage="audio_runtime_stage_context_only", context_validated=True, metadata=metadata,
                 files={name: identity(context / name) for name in ("input.bin", "final.bin")})
    for name, value in (("audio-stage-reference.json", reference), ("audio-stage-validation.json", audit)):
        (context / name).write_text(json.dumps(value))
    return snapshot, context


def preparer():
    value = getattr(audio_layer_replay, "prepare", None)
    assert callable(value), "Pinned layer replay producer is missing"
    return value


def test_producer_writes_parts_only_after_fixed_context_matches(tmp_path):
    run = preparer()
    snapshot, context = bundle(tmp_path)
    report = run(snapshot, context, tmp_path / "out", reference_source="a" * 40, diagnostic_source="c" * 40)
    assert report["context_validated"] is True
    assert report["native_final_exact_match"] is True
    assert report["reference_source"] == "a" * 40 and report["diagnostic_source"] == "c" * 40
    assert len(report["parts"]) == 7
    assert report["full_encoder_validated"] is False and report["unity_validated"] is False


def test_producer_records_optional_attention_projections_without_changing_acceptance(tmp_path):
    snapshot,context=bundle(tmp_path)
    report=preparer()(snapshot,context,tmp_path/'out',reference_source='a'*40,diagnostic_source='c'*40,detail_attention_projections=True)
    assert report['detail_attention_projections'] is True
    assert len(report['parts'])==12
    assert all('self_attn.'+name in report['parts'] for name in ['q_proj','k_proj','v_proj','relative_k_proj','post'])
    assert report['native_final_exact_match'] and not report['full_encoder_validated'] and not report['unity_validated']


def test_cli_passes_attention_detail_option_to_audited_producer(tmp_path,monkeypatch):
    calls=[]
    def prepare(*args,**kwargs):
        calls.append(kwargs)
        return dict(context_validated=True,native_final_exact_match=True,full_encoder_validated=False,unity_validated=False)
    monkeypatch.setattr(audio_layer_replay,'prepare',prepare)
    result=audio_layer_replay.main(['--snapshot',str(tmp_path),'--context',str(tmp_path),'--output',str(tmp_path/'out'),
        '--reference-source','a'*40,'--diagnostic-source','c'*40,'--report',str(tmp_path/'report.json'),'--detail-attention-projections'])
    assert result==0 and calls[0]['detail_attention_projections'] is True


@pytest.mark.parametrize("target", ["input.bin", "final.bin"])
def test_context_binary_corruption_is_rejected_before_loading_weights(tmp_path, monkeypatch, target):
    run = preparer()
    snapshot, context = bundle(tmp_path)
    (context / target).write_bytes(b"corrupted")
    monkeypatch.setattr(audio_layer_replay, "load_layer", lambda *args, **kwargs: pytest.fail("Audit must precede weights"))
    with pytest.raises(ValueError, match="hash|size"):
        run(snapshot, context, tmp_path / "out", reference_source="a" * 40, diagnostic_source="c" * 40)


@pytest.mark.parametrize("mutation", ["source", "tolerance", "weights", "final"])
def test_invalid_context_never_produces_internal_reference(tmp_path, mutation):
    snapshot, context = bundle(tmp_path)
    if mutation == "weights":
        (snapshot / "model.safetensors").write_bytes(b"changed")
    elif mutation == "final":
        path = context / "final.bin"
        import numpy as np
        values = np.fromfile(path, dtype="<f4") + 0.1
        values.tofile(path)
        audit_path = context / "audio-stage-validation.json"
        audit = json.loads(audit_path.read_text())
        audit["files"]["final.bin"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        audit_path.write_text(json.dumps(audit))
    else:
        for name in ("audio-stage-reference.json", "audio-stage-validation.json"):
            path = context / name
            value = json.loads(path.read_text())
            value["metadata"]["source_commit" if mutation == "source" else "feature_atol"] = (
                "d" * 40 if mutation == "source" else 0.1)
            path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        preparer()(snapshot, context, tmp_path / "out", reference_source="a" * 40, diagnostic_source="c" * 40)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("fail", [False, True])
def test_cli_records_replay_failure_separately_from_encoder_acceptance(tmp_path, monkeypatch, fail):
    main = getattr(audio_layer_replay, "main", None)
    assert callable(main), "Replay CLI and failure ledger are missing"
    def run(*args, **kwargs):
        if fail:
            raise ValueError("native replay does not match pinned layer final")
        return dict(context_validated=True, native_final_exact_match=True,
                    full_encoder_validated=False, unity_validated=False)
    monkeypatch.setattr(audio_layer_replay, "prepare", run)
    report_path = tmp_path / "report.json"
    code = main(["--snapshot", str(tmp_path), "--context", str(tmp_path), "--output", str(tmp_path / "out"),
                 "--reference-source", "a" * 40, "--diagnostic-source", "c" * 40,
                 "--report", str(report_path)])
    report = json.loads(report_path.read_text())
    assert code == (1 if fail else 0)
    assert report["success"] is (not fail)
    assert report["full_encoder_validated"] is False and report["unity_validated"] is False
    if fail:
        assert "pinned layer final" in report["error"]
