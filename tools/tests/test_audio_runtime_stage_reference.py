import json
import hashlib

import pytest
import torch

from embeddinggemma_tools import prepare
from test_audio_encoder_prepare import fixture


def function():
    value = getattr(prepare, "prepare_audio_runtime_stages", None)
    assert callable(value), "Pinned native and original-ONNX stage reference producer is missing"
    return value


def source(tmp_path, monkeypatch):
    snapshot, inputs = fixture(tmp_path, monkeypatch)
    encoder = tmp_path / "encoder"
    prepare.prepare_audio_encoder(snapshot, inputs, encoder, source_commit="b" * 40, native_silu=True)
    return snapshot, encoder


def test_stage_reference_keeps_all_original_cases_and_independent_native_final(tmp_path, monkeypatch):
    run = function()
    snapshot, encoder = source(tmp_path, monkeypatch)
    # Native completion must remain usable for diagnostics when the numeric
    # ONNX receipt is failed; that status must survive context-only success.
    audit_path = encoder / "audio-encoder-validation.json"
    audit = json.loads(audit_path.read_text())
    audit.update(success=False, onnx_numerically_validated=False, onnx_failed_case_count=6)
    audit_path.write_text(json.dumps(audit))
    output = tmp_path / "stages"
    result = run(snapshot, encoder, output, source_commit="b" * 40)
    assert result["stage"] == "audio_runtime_stage_context_only"
    assert result["context_validated"] is True and result["case_count"] == 6
    assert result["full_encoder_validated"] is False and result["unity_validated"] is False
    assert result["original_onnx_numerically_validated"] is False
    reference = json.loads((output / "audio-stage-reference.json").read_text())
    assert len(reference["cases"]) == len({row["id"] for row in reference["cases"]}) == 6
    for row in reference["cases"]:
        assert row["native_final_matches_original"] is True
        assert row["trace_numpy_matches_original"] is True
        assert len(row["stages"]) == 5
        assert row["stages"][-1]["name"] == "projected_features"
        for stage in row["stages"]:
            spec = result["files"][stage["expected"]]
            raw = (output / stage["expected"]).read_bytes()
            assert hashlib.sha256(raw).hexdigest() == spec["sha256"]
    assert len(reference["profiles"]) == 2


@pytest.mark.parametrize("problem", ["source", "hash", "native_incomplete"])
def test_stage_reference_refuses_untrusted_inputs_before_loading_weights(tmp_path, monkeypatch, problem):
    run = function()
    snapshot, encoder = source(tmp_path, monkeypatch)
    if problem == "source":
        source_commit = "c" * 40
    elif problem == "hash":
        (encoder / "audio-encoder-reference.json").write_text('{}')
    else:
        report = json.loads((encoder / "audio-encoder-validation.json").read_text())
        report["native_reference_complete"] = False
        (encoder / "audio-encoder-validation.json").write_text(json.dumps(report))
    monkeypatch.setattr(prepare, "load_audio_models", lambda *args: pytest.fail("must audit before weights"))
    with pytest.raises(ValueError, match="stage reference"):
        run(snapshot, encoder, tmp_path / "stages", source_commit=source_commit if problem == "source" else "b" * 40)


def test_changed_native_stage_final_cannot_validate_original_context(tmp_path, monkeypatch):
    run = function()
    snapshot, encoder = source(tmp_path, monkeypatch)
    original = prepare.capture_official_audio_stages
    def changed(*args, **kwargs):
        values = original(*args, **kwargs)
        return (*values[:-1], values[-1] + 0.1)
    monkeypatch.setattr(prepare, "capture_official_audio_stages", changed)
    with pytest.raises(ValueError, match="native stage final.*original"):
        run(snapshot, encoder, tmp_path / "stages", source_commit="b" * 40)
    assert not (tmp_path / "stages/audio-stage-validation.json").exists()


def test_changed_trace_final_cannot_validate_original_numpy_context(tmp_path, monkeypatch):
    run = function()
    snapshot, encoder = source(tmp_path, monkeypatch)
    import onnx
    original = prepare.export_audio_native_onnx
    def changed(wrapper, features, path):
        original(wrapper, features, path)
        graph = onnx.load(path)
        final = graph.graph.output[-1].name
        for node in graph.graph.node:
            for index, name in enumerate(node.output):
                if name == final:
                    node.output[index] = "unmodified_final"
        graph.graph.node.extend([onnx.helper.make_node("Neg", ["unmodified_final"], [final])])
        onnx.save_model(graph, path)
    monkeypatch.setattr(prepare, "export_audio_native_onnx", changed)
    with pytest.raises(ValueError, match="trace stage final.*original"):
        run(snapshot, encoder, tmp_path / "stages", source_commit="b" * 40)
    assert not (tmp_path / "stages/audio-stage-validation.json").exists()
