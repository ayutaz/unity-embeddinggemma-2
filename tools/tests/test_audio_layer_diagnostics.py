"""Localize saved-graph divergence using independently hooked native layers."""

import inspect
import json

import pytest
import torch

from embeddinggemma_tools import export, prepare
from test_audio_encoder import audio_models, mask


PARTS = ("feed_forward1", "norm_pre_attn", "self_attn", "norm_post_attn", "lconv1d", "feed_forward2")


def require_detail_contract():
    assert "detail_layer" in inspect.signature(export.AudioEncoderStagesExport).parameters, "Selected layer export contract is missing"
    assert "detail_layer" in inspect.signature(prepare.capture_official_audio_stages).parameters, "Selected native layer capture contract is missing"


def native_trace(audio, projection, features, frame_mask, detail_layer):
    values, handles = [], []
    def register(module, tuple_output=False):
        def capture(module, args, output):
            values.append((output[0] if tuple_output else output).detach().clone())
        handles.append(module.register_forward_hook(capture))
    try:
        register(audio.subsample_conv_projection, True)
        for index, layer in enumerate(audio.layers):
            if index == detail_layer:
                for part in PARTS:
                    register(getattr(layer, part), part == "self_attn")
            register(layer)
        register(audio.output_proj)
        with torch.no_grad():
            result = audio(input_features=features, attention_mask=frame_mask.bool())
            values.append(projection(result.last_hidden_state)[result.attention_mask].detach().clone())
    finally:
        for handle in handles:
            handle.remove()
    return tuple(values)


@pytest.mark.parametrize("selected", [0, 1])
def test_selected_layer_saved_trace_matches_independent_native_submodules(tmp_path, selected):
    require_detail_contract()
    audio, projection = audio_models()
    frame_mask = mask(25, 17)
    features = torch.randn(1, 25, 8) * frame_mask[:, :, None]
    wrapper = export.AudioEncoderStagesExport(audio, projection, frame_mask, detail_layer=selected)
    names = ["subsample"]
    for index in range(2):
        if index == selected:
            names.extend(f"layer_{index}.{part}" for part in PARTS)
        names.append(f"layer_{index}")
    names.extend(["output_projection", "projected_features"])
    assert wrapper.stage_names == tuple(names)
    expected = native_trace(audio, projection, features, frame_mask, selected)
    for actual, native in zip(wrapper(features), expected, strict=True):
        torch.testing.assert_close(actual, native, rtol=1e-5, atol=1e-6)
    graph = tmp_path / "detail.pt2"
    export.export_audio_pt2(wrapper, features, graph)
    changed = features * 0.7
    saved = torch.export.load(graph).module()(changed)
    for actual, native in zip(saved, native_trace(audio, projection, changed, frame_mask, selected), strict=True):
        torch.testing.assert_close(actual, native, rtol=1e-5, atol=1e-6)
    # Additional diagnostic outputs must leave the failed numerical path intact.
    normal = tmp_path / "normal.pt2"
    export.export_audio_pt2(export.AudioEncoderExport(audio, projection, frame_mask), features, normal)
    assert torch.equal(saved[-1], torch.export.load(normal).module()(changed))


@pytest.mark.parametrize("fail", [False, True])
def test_detailed_native_capture_removes_its_nested_hooks_even_on_failure(monkeypatch, fail):
    require_detail_contract()
    audio, projection = audio_models()
    frame_mask = mask()
    features = torch.randn(1, 25, 8)
    modules = [audio.subsample_conv_projection, *audio.layers, audio.output_proj,
               *(getattr(audio.layers[0], part) for part in PARTS)]
    external = audio.layers[0].norm_post_attn.register_forward_hook(lambda *args: None)
    before = [set(module._forward_hooks) for module in modules]
    expected = native_trace(audio, projection, features, frame_mask, 0)
    try:
        if fail:
            def broken(*args, **kwargs):
                raise RuntimeError("nested native capture failed")
            monkeypatch.setattr(audio.layers[0].lconv1d, "forward", broken)
            with pytest.raises(RuntimeError, match="nested native capture failed"):
                prepare.capture_official_audio_stages(audio, projection, features, frame_mask, detail_layer=0)
        else:
            actual = prepare.capture_official_audio_stages(audio, projection, features, frame_mask, detail_layer=0)
            for value, native in zip(actual, expected, strict=True):
                assert torch.equal(value, native)
        assert [set(module._forward_hooks) for module in modules] == before
    finally:
        external.remove()


@pytest.mark.parametrize("selected", [-1, 2, True, "0"])
def test_detail_selector_rejects_invalid_layer_before_export_or_native_forward(selected):
    require_detail_contract()
    audio, projection = audio_models()
    frame_mask, features = mask(), torch.randn(1, 25, 8)
    with pytest.raises(ValueError, match="detail layer"):
        export.AudioEncoderStagesExport(audio, projection, frame_mask, detail_layer=selected)
    with pytest.raises(ValueError, match="detail layer"):
        prepare.capture_official_audio_stages(audio, projection, features, frame_mask, detail_layer=selected)


@pytest.mark.parametrize("same_final", [True, False])
def test_layer_diagnosis_reports_final_path_agreement_without_claiming_encoder_success(tmp_path, same_final):
    diagnose = getattr(prepare, "diagnose_audio_layer_stages", None)
    assert diagnose is not None, "failure-only layer diagnostics are missing"
    audio, projection = audio_models()
    features, frame_mask = torch.randn(1, 25, 8), mask()
    normal = tmp_path / "normal.pt2"
    export.export_audio_pt2(export.AudioEncoderExport(audio, projection, frame_mask), features, normal)
    original = torch.export.load(normal).module()(features)
    if not same_final:
        original = original + 0.1
    result = diagnose(audio, projection, features, frame_mask, tmp_path, original, detail_layer=0)
    assert result["detail_layer"] == 0
    assert result["stage"] == "audio_encoder_layer_diagnostics_only"
    assert result["trace_matches_failed_graph"] is same_final
    assert (result["trace_vs_failed_graph_maximum_absolute_error"] == 0) is same_final
    assert len(result["stages"]) == 11
    assert result["first_divergent_stage"] is None
    assert (tmp_path / "audio-layer-0-stages.pt2").exists()


@pytest.mark.parametrize("same_final,detail_fails", [(True, False), (True, True), (False, False)])
def test_failed_coarse_trace_persists_before_localizing_first_nonzero_layer(tmp_path, monkeypatch, same_final, detail_fails):
    audio, projection = audio_models()
    features, frame_mask = torch.randn(1, 25, 8), mask()
    path = tmp_path / "normal.pt2"
    export.export_audio_pt2(export.AudioEncoderExport(audio, projection, frame_mask), features, path)
    original = torch.export.load(path).module()(features)
    if not same_final:
        original = original + 0.1
    compare = prepare.compare_audio_stage_outputs
    def inject_stage_difference(names, expected, actual):
        result = compare(names, expected, actual)
        # Control only localization logic. This is not a real numeric receipt.
        result["first_divergent_stage"] = "layer_1"
        result["stages"][0]["maximum_absolute_error"] = 0
        result["stages"][1]["maximum_absolute_error"] = 1e-6
        result["stages"][2].update(success=False, maximum_absolute_error=0.001)
        return result
    calls = []
    def diagnose(*args, **kwargs):
        calls.append(kwargs["detail_layer"])
        coarse = json.loads((tmp_path / "audio-stage-diagnostics.json").read_text())
        assert coarse["first_divergent_stage"] == "layer_1"
        assert coarse["trace_matches_failed_graph"] is True
        if detail_fails:
            raise RuntimeError("interrupted detail trace")
        return {"detail_layer": 0, "first_divergent_stage": "layer_0.self_attn"}
    monkeypatch.setattr(prepare, "compare_audio_stage_outputs", inject_stage_difference)
    monkeypatch.setattr(prepare, "diagnose_audio_layer_stages", diagnose, raising=False)
    result = prepare.diagnose_audio_stages(audio, projection, features, frame_mask, tmp_path, original)
    assert result["first_divergent_stage"] == "layer_1"
    assert result["trace_matches_failed_graph"] is same_final
    assert calls == ([0] if same_final else [])
    persisted = json.loads((tmp_path / "audio-stage-diagnostics.json").read_text())
    assert persisted["first_divergent_stage"] == "layer_1"
    if same_final and detail_fails:
        assert "interrupted detail trace" in result["layer_diagnostics_error"]
        assert "interrupted detail trace" in persisted["layer_diagnostics_error"]
    elif same_final:
        assert result["layer_diagnostics"]["detail_layer"] == 0
    else:
        assert "layer_diagnostics" not in result
