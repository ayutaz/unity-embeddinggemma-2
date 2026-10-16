import json

import onnx
from onnx.reference import ReferenceEvaluator
import pytest
import torch

from embeddinggemma_tools import export, prepare
from test_audio_encoder import audio_models, mask
from test_audio_encoder_prepare import fixture


def diagnostic():
    value = getattr(prepare, "diagnose_audio_onnx_stages", None)
    assert callable(value), "Failure-only ONNX stage diagnostic is missing"
    return value


@pytest.mark.parametrize("detail_layer", [None, 0])
def test_onnx_trace_matches_the_original_graph_before_localizing(tmp_path, detail_layer):
    run = diagnostic()
    audio, projection = audio_models()
    frame_mask = mask(25, 17)
    features = torch.randn(1, 25, 8, generator=torch.Generator().manual_seed(9)) * frame_mask[:, :, None]
    original = tmp_path / "original.onnx"
    export.export_audio_native_onnx(export.AudioEncoderExport(audio, projection, frame_mask), features, original)
    failed = torch.from_numpy(ReferenceEvaluator(str(original)).run(None, {"input": features.numpy()})[0])
    report = run(audio, projection, features, frame_mask, tmp_path, failed, detail_layer=detail_layer)
    assert report["trace_matches_failed_graph"] is True
    assert report["localization_validated"] is True
    assert report["encoders_validated"] is False and report["end_to_end_validated"] is False
    assert all(row["success"] for row in report["stages"])
    names = [row["name"] for row in report["stages"]]
    assert names[0] == "subsample" and names[-1] == "projected_features"
    if detail_layer == 0:
        assert "layer_0.feed_forward1" in names and "layer_0.self_attn" in names


def test_trace_with_a_different_final_path_cannot_claim_localization(tmp_path):
    run = diagnostic()
    audio, projection = audio_models()
    frame_mask = mask(25, 17)
    features = torch.zeros(1, 25, 8)
    with torch.no_grad():
        unrelated = export.AudioEncoderExport(audio, projection, frame_mask)(features) + 0.1
    report = run(audio, projection, features, frame_mask, tmp_path, unrelated)
    assert report["trace_matches_failed_graph"] is False
    assert report["localization_validated"] is False
    assert report["trace_vs_failed_graph_maximum_absolute_error"] > 0.01
    assert report["encoders_validated"] is False


def test_changed_native_reference_cannot_validate_localization(tmp_path, monkeypatch):
    run = diagnostic()
    audio, projection = audio_models()
    frame_mask = mask(25, 17)
    features = torch.zeros(1, 25, 8)
    original = tmp_path / "original.onnx"
    export.export_audio_native_onnx(export.AudioEncoderExport(audio, projection, frame_mask), features, original)
    failed = torch.from_numpy(ReferenceEvaluator(str(original)).run(None, {"input": features.numpy()})[0])
    capture = prepare.capture_official_audio_stages
    def changed(*args, **kwargs):
        stages = capture(*args, **kwargs)
        return (*stages[:-1], stages[-1] + 0.1)
    monkeypatch.setattr(prepare, "capture_official_audio_stages", changed)
    report = run(audio, projection, features, frame_mask, tmp_path, failed)
    assert report["trace_matches_failed_graph"] is True
    assert report["localization_validated"] is False, "A different native reference must invalidate the diagnostic context"
    assert report["native_reference_matches_original"] is False


@pytest.mark.parametrize("diagnostic_fails", [False, True])
def test_original_onnx_failure_survives_stage_diagnostic_success_or_failure(tmp_path, monkeypatch, diagnostic_fails):
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
        model.graph.node.extend([onnx.helper.make_node("Neg", ["before_corruption"], [output])])
        onnx.save_model(model, path)

    calls = []
    def probe(*args, **kwargs):
        calls.append(args)
        if diagnostic_fails:
            raise RuntimeError("secondary diagnostic failure")
        return dict(localization_validated=False, trace_matches_failed_graph=False)

    monkeypatch.setattr(prepare, "export_audio_native_onnx", corrupt)
    monkeypatch.setattr(prepare, "diagnose_audio_onnx_stages", probe, raising=False)
    destination = tmp_path / "out"
    with pytest.raises(ValueError, match="audio ONNX export mismatch"):
        prepare.prepare_audio_encoder(snapshot, inputs, destination, source_commit="b" * 40, native_silu=True)
    assert len(calls) == 1, "The ONNX failure must reach its own diagnostic path"
    report = json.loads((destination / "audio-encoder-failure.json").read_text())
    assert report["success"] is False and report["stage"] == "audio_encoder_onnx"
    assert report["mismatched_elements"] > 0
    assert not (destination / "audio-encoder-validation.json").exists()
    if diagnostic_fails:
        assert "secondary diagnostic failure" in report["onnx_stage_diagnostics_error"]
    else:
        assert report["onnx_stage_diagnostics"]["localization_validated"] is False
