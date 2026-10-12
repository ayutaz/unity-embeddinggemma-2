import json
import importlib

from embeddinggemma_tools import export
from test_model import tiny_model


def test_preparation_validates_saved_backbone_without_claiming_encoder_or_unity_success(tmp_path, monkeypatch):
    from embeddinggemma_tools import prepare as existing
    factory = getattr(existing, "prepare_multimodal_backbone", None)
    assert factory is not None, "real-weight multimodal backbone preparation is missing"
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").write_text(json.dumps({"image_token_id": 70, "audio_token_id": 71}))
    (snapshot / "processor_config.json").write_text('{"processor_class":"EmbeddingGemma2Processor"}')
    (snapshot / "model.safetensors").write_bytes(b"mock weights for preparation orchestration only")
    monkeypatch.setattr(existing, "load_text_model", lambda _: tiny_model())
    output = tmp_path / "output"
    summary = factory(snapshot, output, sequence_length=10, image_feature_count=2,
                      audio_feature_count=1, revision="a" * 40, source_commit="b" * 40)
    assert summary["success"] is True
    assert summary["stage"] == "projected_features_backbone_only"
    assert summary["unity_validated"] is False
    assert summary["encoders_validated"] is False
    assert summary["case_count"] == 2
    assert summary["minimum_cosine"] >= 0.999999
    reference = json.loads((output / "backbone-reference.json").read_text())
    assert reference["metadata"]["sequence_length"] == 10
    assert reference["metadata"]["model_revision"] == "a" * 40
    assert reference["metadata"]["source_commit"] == "b" * 40
    assert set(summary["files"]) == {"backbone.pt2", "backbone-reference.json", "processor_config.json"}
    assert (output / "backbone.pt2").is_file()
    assert json.loads((output / "backbone-validation.json").read_text()) == summary


def test_failed_preparation_clears_stale_success_before_loading_model(tmp_path, monkeypatch):
    from embeddinggemma_tools import prepare as existing
    factory = getattr(existing, "prepare_multimodal_backbone", None)
    assert factory is not None, "failed preparation must not keep a stale pass"
    output = tmp_path / "out"
    output.mkdir()
    (output / "backbone-validation.json").write_text('{"success":true}')
    def fail(_):
        raise RuntimeError("missing projection weight")
    monkeypatch.setattr(existing, "load_text_model", fail)
    import pytest
    with pytest.raises(RuntimeError, match="missing projection"):
        factory(tmp_path, output, revision="a" * 40)
    assert not (output / "backbone-validation.json").exists()


def test_cli_downloads_pinned_processor_and_runs_backbone_gate(tmp_path, monkeypatch, capsys):
    cli = importlib.import_module("embeddinggemma_tools.__main__")
    prepare_calls = []
    downloads = []
    def download(model, **kwargs):
        downloads.append((model, kwargs))
        return str(tmp_path / "snapshot")
    def prepare(*args, **kwargs):
        prepare_calls.append((args, kwargs))
        return dict(success=True, stage="projected_features_backbone_only", case_count=2,
                    minimum_cosine=1.0, unity_validated=False, encoders_validated=False)
    monkeypatch.setattr(cli, "snapshot_download", download)
    monkeypatch.setattr(cli, "prepare_multimodal_backbone", prepare, raising=False)
    monkeypatch.setenv("GITHUB_SHA", "c" * 40)
    cli.main(["prepare-backbone", "--output", str(tmp_path / "output")])
    assert downloads[0][1]["revision"] == "914f7f89142e33e77833254d9c9b90c3cef7303b"
    assert "processor_config.json" in downloads[0][1]["allow_patterns"]
    assert prepare_calls[0][1]["source_commit"] == "c" * 40
    assert prepare_calls[0][1]["image_feature_count"] == 280
    assert prepare_calls[0][1]["audio_feature_count"] == 280
    assert prepare_calls[0][0][2] == 1024
    report = json.loads(capsys.readouterr().out)
    assert report["unity_validated"] is False
    assert report["encoders_validated"] is False
