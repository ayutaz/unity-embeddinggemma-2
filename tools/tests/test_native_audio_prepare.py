import json

import onnx
import pytest

from embeddinggemma_tools import prepare, __main__ as cli
from test_audio_encoder_prepare import fixture


def test_native_candidate_keeps_all_six_signals_and_standard_import_graphs(tmp_path,monkeypatch):
    snapshot,inputs=fixture(tmp_path,monkeypatch)
    result=prepare.prepare_audio_encoder(snapshot,inputs,tmp_path/"out",source_commit="b"*40,native_silu=True)
    assert result["success"] is True and result["case_count"]==6
    reference=json.loads((tmp_path/"out/audio-encoder-reference.json").read_text())
    for profile in reference["profiles"].values():
        assert profile["import_format"]=="onnx"
        assert profile["native_graph"].endswith('.pt2')
        model=onnx.load(tmp_path/"out"/profile["graph"])
        onnx.checker.check_model(model)
        assert all(node.domain in ('','ai.onnx') for node in model.graph.node)
        assert not any(node.op_type=='Swish' for node in model.graph.node)
        assert {'Neg', 'Exp', 'Add', 'Div'} <= {node.op_type for node in model.graph.node}
    assert result["unity_validated"] is False


def test_cli_selects_native_candidate_without_changing_the_reference_suite(tmp_path,monkeypatch,capsys):
    calls=[]
    def run(*args,**kwargs):
        calls.append(kwargs)
        return dict(success=True,stage="audio_encoder_projector_only",profile_count=2,case_count=6,minimum_cosine=1.0,unity_validated=False,end_to_end_validated=False)
    monkeypatch.setattr(cli,"prepare_audio_encoder",run)
    cli.main(["prepare-audio-encoder","--snapshot",str(tmp_path/"snapshot"),"--preprocessing",str(tmp_path/"pre"),"--output",str(tmp_path/"out"),"--native-silu"])
    assert calls[0]["native_silu"] is True
    assert json.loads(capsys.readouterr().out)["case_count"]==6


def test_candidate_validates_all_six_onnx_outputs_against_official_features(tmp_path, monkeypatch):
    snapshot, inputs = fixture(tmp_path, monkeypatch)
    result = prepare.prepare_audio_encoder(
        snapshot, inputs, tmp_path / "out", source_commit="b" * 40, native_silu=True)
    assert result["onnx_numerically_validated"] is True
    assert result["onnx_case_count"] == 6
    assert result["onnx_minimum_cosine"] >= 0.999999
    reference = json.loads((tmp_path / "out/audio-encoder-reference.json").read_text())
    assert len(reference["cases"]) == 6
    assert all(row["onnx_minimum_cosine"] >= 0.999999 for row in reference["cases"])


def test_every_onnx_output_is_saved_and_audited_for_immutable_pinned_comparison(tmp_path, monkeypatch):
    import numpy as np
    import hashlib
    snapshot, inputs = fixture(tmp_path, monkeypatch)
    destination = tmp_path / "out"
    result = prepare.prepare_audio_encoder(snapshot, inputs, destination, source_commit="b"*40, native_silu=True)
    reference = json.loads((destination/"audio-encoder-reference.json").read_text())
    assert len(reference['cases']) == 6
    for row in reference['cases']:
        path = destination / row['onnx_features']
        spec = result['files'][row['onnx_features']]
        raw = path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == spec['sha256'] and len(raw) == spec['bytes']
        actual = np.fromfile(path, dtype='<f4').reshape(np.asarray(row['features']).shape)
        assert np.isfinite(actual).all()
        np.testing.assert_allclose(actual, np.asarray(row['features']), rtol=1e-4, atol=1e-5)


def test_cli_forwards_explicit_onnx_layernorm_candidate(tmp_path,monkeypatch,capsys):
    calls=[]
    def run(*args,**kwargs):
        calls.append(kwargs)
        return dict(success=True,stage="audio_encoder_projector_only",profile_count=2,case_count=6,
                    minimum_cosine=1.0,unity_validated=False,end_to_end_validated=False)
    monkeypatch.setattr(cli,"prepare_audio_encoder",run)
    cli.main(["prepare-audio-encoder","--snapshot",str(tmp_path/"snapshot"),"--preprocessing",str(tmp_path/"pre"),
              "--output",str(tmp_path/"out"),"--native-silu","--onnx-layernorm","torch_welford8"])
    assert calls[0]['onnx_layernorm']=='torch_welford8' and calls[0]['native_silu']


def test_cli_forwards_explicit_compensated_projection_candidate(tmp_path,monkeypatch):
    calls=[]
    def run(*args,**kwargs):
        calls.append(kwargs)
        return dict(success=True,stage="audio_encoder_projector_only",profile_count=2,case_count=6,
                    minimum_cosine=1.0,unity_validated=False,end_to_end_validated=False)
    monkeypatch.setattr(cli,"prepare_audio_encoder",run)
    cli.main(["prepare-audio-encoder","--snapshot",str(tmp_path/"snapshot"),"--preprocessing",str(tmp_path/"pre"),
              "--output",str(tmp_path/"out"),"--native-silu","--onnx-linear","compensated64"])
    assert calls[0]['onnx_linear']=='compensated64'


def test_structurally_valid_but_wrong_onnx_cannot_produce_success(tmp_path, monkeypatch):
    snapshot, inputs = fixture(tmp_path, monkeypatch)
    original = prepare.export_audio_native_onnx

    def corrupt(wrapper, features, path):
        original(wrapper, features, path)
        model = onnx.load(path)
        output = model.graph.output[0].name
        for node in model.graph.node:
            for index, name in enumerate(node.output):
                if name == output:
                    node.output[index] = "before_corruption"
        model.graph.node.extend([
            onnx.helper.make_node("Neg", ["before_corruption"], [output])])
        onnx.checker.check_model(model)
        onnx.save_model(model, path)

    monkeypatch.setattr(prepare, "export_audio_native_onnx", corrupt)
    destination = tmp_path / "out"
    with pytest.raises(ValueError, match="audio ONNX export mismatch"):
        prepare.prepare_audio_encoder(
            snapshot, inputs, destination, source_commit="b" * 40, native_silu=True)
    assert not (destination / "audio-encoder-validation.json").exists()
    failure = json.loads((destination / "audio-encoder-failure.json").read_text())
    assert failure["success"] is False
    assert failure["stage"] == "audio_encoder_onnx"
    assert failure["source_commit"] == "b" * 40
    assert failure["mismatched_elements"] > 0
