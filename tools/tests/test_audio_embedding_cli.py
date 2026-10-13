import json
from pathlib import Path

import yaml

from embeddinggemma_tools import __main__ as cli


def test_cli_connects_audio_source_and_preserves_native_only_scope(tmp_path, monkeypatch, capsys):
    assert callable(getattr(cli, "prepare_audio_embeddings", None)), "Final audio reference CLI is missing"
    calls = []
    def prepare(*args, **kwargs):
        calls.append((args, kwargs))
        return dict(success=True, stage="python_audio_embeddings_only", case_count=6, profile_count=2,
                    unity_validated=False, gpu_chain_validated=False, onnx_encoder_numerically_validated=False)
    monkeypatch.setattr(cli, "prepare_audio_embeddings", prepare)
    monkeypatch.setenv("GITHUB_SHA", "b" * 40)
    cli.main(["prepare-audio-embeddings", "--snapshot", str(tmp_path / "snapshot"),
              "--encoder-source", str(tmp_path / "encoder"), "--output", str(tmp_path / "out")])
    assert calls[0][0] == (tmp_path / "snapshot", tmp_path / "encoder", tmp_path / "out")
    assert calls[0][1]["source_commit"] == "b" * 40 and calls[0][1]["sequence_length"] == 1024
    assert json.loads(capsys.readouterr().out)["onnx_encoder_numerically_validated"] is False


def test_opt_in_workflow_runs_native_final_reference_without_masking_onnx_failure():
    root = Path(__file__).resolve().parents[2]
    workflow = yaml.safe_load((root / ".github/workflows/audio-runtime-reference.yml").read_text(encoding="utf-8"))
    steps = workflow["jobs"]["native-reference"]["steps"]
    collection = next(i for i, step in enumerate(steps) if "--collect-onnx-failures" in step.get("run", ""))
    final = [(i, step) for i, step in enumerate(steps) if "prepare-audio-embeddings" in step.get("run", "")]
    assert len(final) == 1, "Real audio final reference job is missing"
    index, step = final[0]
    assert index > collection and "always()" in step["if"]
    assert "audio-encoder-validation.json" in step["if"]
    assert "--encoder-source ../artifacts/audio-runtime-reference" in step["run"]
    assert not steps[collection].get("continue-on-error", False)
    assert not step.get("continue-on-error", False)
    uploads = [step for step in steps if "upload-artifact" in step.get("uses", "")]
    assert any("artifacts/audio-final-reference/" in step["with"]["path"] for step in uploads)
