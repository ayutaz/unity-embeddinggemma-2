import json
from pathlib import Path

import yaml
from workflow_privacy_assertions import assert_privacy_gated_upload

from embeddinggemma_tools import __main__ as cli
from embeddinggemma_tools.prepare import MODEL_REVISION


def test_cli_pins_stage_source_and_does_not_claim_encoder_acceptance(tmp_path, monkeypatch, capsys):
    calls, downloads = [], []
    def download(*args, **kwargs):
        downloads.append(kwargs)
        return str(tmp_path / "snapshot")
    def prepare(*args, **kwargs):
        calls.append((args, kwargs))
        return dict(success=True, stage="audio_runtime_stage_context_only", case_count=6, profile_count=2,
                    context_validated=True, full_encoder_validated=False, unity_validated=False,
                    end_to_end_validated=False)
    monkeypatch.setattr(cli, "snapshot_download", download)
    monkeypatch.setattr(cli, "prepare_audio_runtime_stages", prepare, raising=False)
    monkeypatch.setenv("GITHUB_SHA", "b" * 40)
    cli.main(["prepare-audio-runtime-stages", "--encoder-source", str(tmp_path / "encoder"),
              "--output", str(tmp_path / "stages")])
    assert downloads[0]["revision"] == MODEL_REVISION
    assert "processor_config.json" in downloads[0]["allow_patterns"]
    assert calls[0][0] == (tmp_path / "snapshot", tmp_path / "encoder", tmp_path / "stages")
    assert calls[0][1] == dict(revision=MODEL_REVISION, source_commit="b" * 40)
    report = json.loads(capsys.readouterr().out)
    assert report["context_validated"] is True
    assert report["full_encoder_validated"] is False and report["unity_validated"] is False


def test_opt_in_workflow_produces_stages_after_failed_native_collection_without_masking_failure():
    root = Path(__file__).resolve().parents[2]
    workflow = yaml.safe_load((root / ".github/workflows/audio-runtime-reference.yml").read_text(encoding="utf-8"))
    steps = workflow["jobs"]["native-reference"]["steps"]
    collection = next(i for i, row in enumerate(steps) if "--collect-onnx-failures" in row.get("run", ""))
    matches = [(i, row) for i, row in enumerate(steps) if "prepare-audio-runtime-stages" in row.get("run", "")]
    assert len(matches) == 1, "Native stage reference must be exported for actual Sentis diagnostics"
    index, step = matches[0]
    assert index > collection
    assert "always()" in step["if"] and "audio-encoder-validation.json" in step["if"]
    assert not steps[collection].get("continue-on-error", False)
    assert not step.get("continue-on-error", False)
    assert "--encoder-source ../artifacts/audio-runtime-reference" in step["run"]
    uploads = [row for row in steps if "actions/upload-artifact" in row.get("uses", "")
               and row.get("with", {}).get("name", "").startswith("audio-runtime-native-")]
    assert len(uploads) == 1, "Native stage evidence needs one dedicated artifact"
    upload = uploads[0]
    assert_privacy_gated_upload(upload)
    assert "artifacts/audio-runtime-reference/" in upload["with"]["path"]
    assert "artifacts/audio-runtime-stages/" in upload["with"]["path"]
