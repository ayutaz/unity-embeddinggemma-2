import json

import onnx
import pytest
import torch

from embeddinggemma_tools import prepare, __main__ as cli
from test_audio_encoder_prepare import fixture


def corrupted_onnx(monkeypatch):
    original = prepare.export_audio_native_onnx
    def corrupt(wrapper, features, path):
        original(wrapper, features, path)
        model = onnx.load(path)
        output = model.graph.output[0].name
        for node in model.graph.node:
            for index, name in enumerate(node.output):
                if name == output: node.output[index] = "before_corruption"
        model.graph.node.extend([onnx.helper.make_node("Neg", ["before_corruption"], [output])])
        onnx.save_model(model, path)
    monkeypatch.setattr(prepare, "export_audio_native_onnx", corrupt)


def test_collects_all_six_native_references_without_turning_onnx_failures_into_success(tmp_path, monkeypatch):
    snapshot, inputs = fixture(tmp_path, monkeypatch)
    corrupted_onnx(monkeypatch)
    destination = tmp_path / "out"
    with pytest.raises(ValueError, match="ONNX diagnostic collection.*native reference complete"):
        prepare.prepare_audio_encoder(snapshot, inputs, destination, source_commit="b" * 40,
                                      native_silu=True, collect_onnx_failures=True)
    report = json.loads((destination / "audio-encoder-validation.json").read_text())
    reference = json.loads((destination / "audio-encoder-reference.json").read_text())
    assert report["success"] is False
    assert report["native_reference_complete"] is True
    assert report["onnx_numerically_validated"] is False
    assert report["case_count"] == report["onnx_case_count"] == report["onnx_failed_case_count"] == 6
    assert report["profile_count"] == 2
    assert len(reference["cases"]) == len({row["id"] for row in reference["cases"]}) == 6
    assert all(row["onnx_success"] is False for row in reference["cases"])
    failures = json.loads((destination / "audio-onnx-diagnostic-failures.json").read_text())
    assert len(failures["cases"]) == 6 and failures["source_commit"] == "b" * 40
    assert report["unity_validated"] is False and report["end_to_end_validated"] is False
    assert all(profile["import_format"] == "onnx" for profile in reference["profiles"].values())
    for name, spec in report["files"].items():
        assert (destination / name).stat().st_size == spec["bytes"]


def test_collect_mode_still_refuses_native_reference_mismatch(tmp_path, monkeypatch):
    snapshot, inputs = fixture(tmp_path, monkeypatch)
    def corrupt(wrapper, features, path):
        class Wrong(torch.nn.Module):
            def forward(self, inputs): return torch.zeros(wrapper.feature_count, wrapper.projection.text_hidden_size)
        torch.export.save(torch.export.export(Wrong(), (features,)), path)
    monkeypatch.setattr(prepare, "export_audio_pt2", corrupt)
    monkeypatch.setattr(prepare, "diagnose_audio_stages", lambda *args: {"diagnostic_only": True})
    destination = tmp_path / "out"
    with pytest.raises(ValueError, match="audio encoder export mismatch"):
        prepare.prepare_audio_encoder(snapshot, inputs, destination, source_commit="b" * 40,
                                      native_silu=True, collect_onnx_failures=True)
    assert not (destination / "audio-encoder-validation.json").exists()
    assert not (destination / "audio-encoder-reference.json").exists()


def test_collect_mode_requires_standard_onnx_route(tmp_path):
    with pytest.raises(ValueError, match="collect.*native-silu"):
        prepare.prepare_audio_encoder(tmp_path / "missing", tmp_path / "missing", tmp_path / "out",
                                      collect_onnx_failures=True)


def test_cli_collect_flag_is_explicit_and_required_workflow_keeps_strict_gate(tmp_path, monkeypatch):
    calls = []
    def generate(*args, **kwargs):
        calls.append(kwargs)
        return dict(success=True, stage="audio_encoder_projector_only", profile_count=2, case_count=6,
                    minimum_cosine=1.0, unity_validated=False, end_to_end_validated=False)
    monkeypatch.setattr(cli, "prepare_audio_encoder", generate)
    cli.main(["prepare-audio-encoder", "--snapshot", str(tmp_path), "--preprocessing", str(tmp_path),
              "--output", str(tmp_path / "out"), "--native-silu", "--collect-onnx-failures"])
    assert calls[0]["collect_onnx_failures"] is True
    from pathlib import Path
    workflow = Path(__file__).resolve().parents[2] / ".github/workflows/model-reference.yml"
    assert "--collect-onnx-failures" not in workflow.read_text(encoding="utf-8")


def test_runtime_diagnostic_workflow_is_opt_in_read_only_and_retains_failure():
    from pathlib import Path
    import yaml
    path = Path(__file__).resolve().parents[2] / ".github/workflows/audio-runtime-reference.yml"
    assert path.is_file(), "Separate opt-in native reference workflow is missing"
    workflow = yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    assert "workflow_dispatch" in workflow["on"] and "pull_request" in workflow["on"]
    assert "pull_request_target" not in workflow["on"]
    assert workflow["permissions"] == {"contents": "read"}
    job = workflow["jobs"]["native-reference"]
    assert "audio-runtime-diagnostics" in job["if"]
    assert "head.repo.full_name == github.repository" in job["if"]
    generation = [step for step in job["steps"] if "--collect-onnx-failures" in step.get("run", "")]
    assert len(generation) == 1
    assert "continue-on-error" not in generation[0], "Numeric failure must leave the workflow failed"
    artifacts = [step for step in job["steps"] if "upload-artifact" in step.get("uses", "")]
    assert artifacts and all(step["if"] == "always()" for step in artifacts)
