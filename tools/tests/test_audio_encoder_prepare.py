import hashlib
import json

import pytest
import torch
import yaml

from embeddinggemma_tools import prepare as preparation
from embeddinggemma_tools import __main__ as cli
from embeddinggemma_tools.audio import prepare_audio_preprocessor
from test_audio_encoder import audio_models
from test_audio_prepare import configuration


def factory():
    function = getattr(preparation, "prepare_audio_encoder", None)
    assert function is not None, "audio encoder reference preparation is missing"
    return function


def fixture(tmp_path, monkeypatch):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    processor = configuration(snapshot / "processor_config.json")
    (snapshot / "config.json").write_text('{}')
    (snapshot / "model.safetensors").write_bytes(b"tiny-audio-fixture-not-real-weights")
    inputs = tmp_path / "preprocessing"
    prepare_audio_preprocessor(processor, inputs, num_samples=1024, source_commit="b" * 40)
    monkeypatch.setattr(preparation, "load_audio_models", lambda path: audio_models(128), raising=False)
    return snapshot, inputs


def test_prepares_independent_official_audio_features_and_two_saved_profiles(tmp_path, monkeypatch):
    function = factory()
    snapshot, inputs = fixture(tmp_path, monkeypatch)
    output = tmp_path / "encoder"
    report = function(snapshot, inputs, output, source_commit="b" * 40)
    assert report["success"] is True
    assert report["stage"] == "audio_encoder_projector_only"
    assert report["unity_validated"] is False and report["end_to_end_validated"] is False
    assert report["profile_count"] == 2 and report["case_count"] == 6
    assert report["minimum_cosine"] >= 0.999999
    reference = json.loads((output / "audio-encoder-reference.json").read_text())
    assert reference["metadata"]["source_commit"] == "b" * 40
    assert reference["metadata"]["attention_implementation"] == "sdpa"
    assert reference["metadata"]["preprocessing_reference_sha256"] == hashlib.sha256((inputs / "audio-reference.json").read_bytes()).hexdigest()
    for item in reference["profiles"].values():
        assert item["feature_shape"] == [1, 6, 128]
        assert item["feature_width"] == 16
    for name, item in report["files"].items():
        assert (output / name).stat().st_size == item["bytes"]
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == item["sha256"]
    assert json.loads((output / "audio-encoder-validation.json").read_text()) == report


def test_encoder_failure_clears_stale_success(tmp_path):
    function = factory()
    output = tmp_path / "out"
    output.mkdir()
    (output / "audio-encoder-validation.json").write_text('{"success":true}')
    with pytest.raises(FileNotFoundError):
        function(tmp_path / "missing", tmp_path / "missing", output)
    assert not (output / "audio-encoder-validation.json").exists()


@pytest.mark.parametrize("problem", ["hash", "source", "processor", "incomplete"])
def test_preprocessing_must_be_audited_before_encoder_loading(tmp_path, monkeypatch, problem):
    function = factory()
    snapshot, inputs = fixture(tmp_path, monkeypatch)
    if problem == "hash":
        (inputs / "silence-features-fp32.bin").write_bytes(b"corrupt")
    elif problem == "source":
        source = "c" * 40
    elif problem == "processor":
        (snapshot / "processor_config.json").write_text('{}')
    else:
        report = json.loads((inputs / "audio-validation.json").read_text())
        report["success"] = False
        (inputs / "audio-validation.json").write_text(json.dumps(report))
    monkeypatch.setattr(preparation, "load_audio_models", lambda path: pytest.fail("must reject before weights are loaded"), raising=False)
    with pytest.raises(ValueError, match="preprocessing"):
        function(snapshot, inputs, tmp_path / "out", source_commit=source if problem == "source" else "b" * 40)
    assert not (tmp_path / "out/audio-encoder-validation.json").exists()


def test_wrong_saved_encoder_features_are_rejected(tmp_path, monkeypatch):
    function = factory()
    snapshot, inputs = fixture(tmp_path, monkeypatch)
    class Wrong(torch.nn.Module):
        def forward(self, features):
            return torch.zeros(2, 16)
    def corrupt(wrapper, features, output):
        torch.export.save(torch.export.export(Wrong(), (features,)), output)
    monkeypatch.setattr(preparation, "export_audio_pt2", corrupt, raising=False)
    with pytest.raises(ValueError, match="audio encoder export mismatch"):
        function(snapshot, inputs, tmp_path / "out", source_commit="b" * 40)
    assert not (tmp_path / "out/audio-encoder-validation.json").exists()


def test_cli_connects_pinned_snapshot_preprocessing_and_ci_source(tmp_path, monkeypatch, capsys):
    factory()
    calls, downloads = [], []
    def download(*args, **kwargs):
        downloads.append((args, kwargs))
        return str(tmp_path / "snapshot")
    def prepare(*args, **kwargs):
        calls.append((args, kwargs))
        return dict(success=True, stage="audio_encoder_projector_only", profile_count=2, case_count=6,
                    minimum_cosine=1.0, unity_validated=False, end_to_end_validated=False)
    monkeypatch.setattr(cli, "snapshot_download", download)
    monkeypatch.setattr(cli, "prepare_audio_encoder", prepare, raising=False)
    monkeypatch.setenv("GITHUB_SHA", "b" * 40)
    cli.main(["prepare-audio-encoder", "--preprocessing", str(tmp_path / "in"), "--output", str(tmp_path / "out")])
    assert downloads[0][1]["revision"] == preparation.MODEL_REVISION
    assert "processor_config.json" in downloads[0][1]["allow_patterns"]
    assert calls[0][0] == (tmp_path / "snapshot", tmp_path / "in", tmp_path / "out")
    assert calls[0][1]["source_commit"] == "b" * 40
    assert json.loads(capsys.readouterr().out)["end_to_end_validated"] is False


def test_required_reference_workflow_runs_audio_encoder_after_preprocessing():
    root = __import__("pathlib").Path(__file__).resolve().parents[2]
    workflow = yaml.safe_load((root / ".github/workflows/model-reference.yml").read_text(encoding="utf-8"))
    steps = workflow["jobs"]["reference"]["steps"]
    preprocess = next(i for i, step in enumerate(steps) if "embeddinggemma_tools.audio --output" in step.get("run", ""))
    encoders = [(i, step) for i, step in enumerate(steps) if "prepare-audio-encoder" in step.get("run", "")]
    assert len(encoders) == 1, "real audio encoder export is absent from the required reference workflow"
    assert encoders[0][0] > preprocess
    assert "--preprocessing ../artifacts/audio-preprocessing" in encoders[0][1]["run"]
    assert not encoders[0][1].get("continue-on-error", False)
