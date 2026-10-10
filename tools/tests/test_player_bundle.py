import hashlib
import json

import pytest

from embeddinggemma_tools.player import stage_player_bundle, main
from test_stage import artifact, search_artifact  # Reuse the fixed fake artifact fixtures, not real-model evidence.

SOURCE = "b3b0d76e1d3b6f4e44af1e9ba810d5b3cb6b051d"
REVISION = "914f7f89142e33e77833254d9c9b90c3cef7303b"
MODEL_FILES = ["model-fp32.sentis", "model-float16.sentis", "tokenizer.json"]


@pytest.fixture
def prepared(search_artifact, tmp_path):
    directory = tmp_path / "prepared"
    directory.mkdir()
    (directory / MODEL_FILES[0]).write_bytes(b"fake fp32: staging test only")
    (directory / MODEL_FILES[1]).write_bytes(b"fake float16: staging test only")
    (directory / MODEL_FILES[2]).write_bytes((search_artifact / MODEL_FILES[2]).read_bytes())
    exported = json.loads((search_artifact / "export-validation.json").read_text())
    receipt = {"success": True, "sourceCommit": SOURCE, "searchSourceCommit": SOURCE,
               "unityVersion": "6000.3.16f1", "modelSha256": exported["sha256"]["model.pt2"],
               "files": [{"name": name, "bytes": (directory / name).stat().st_size,
                          "sha256": hashlib.sha256((directory / name).read_bytes()).hexdigest()} for name in MODEL_FILES]}
    (directory / "preparation.json").write_text(json.dumps(receipt))
    audit = {"success": True, "search_reference_staged": True, "model_revision": REVISION,
             "source_commit": SOURCE, "search_source_commit": SOURCE, "sha256": exported["sha256"]}
    audit_path = tmp_path / "stage.json"
    audit_path.write_text(json.dumps(audit))
    return directory, search_artifact, audit_path


def test_bundle_preserves_all_hashes_and_never_claims_player_or_gpu_execution(prepared, tmp_path):
    output = tmp_path / "bundle"
    report = stage_player_bundle(*prepared, output)
    assert report["success"] is True
    assert report["unity_executed"] is False
    assert report["gpu_verified"] is False
    assert report["model_revision"] == REVISION
    assert {file["name"] for file in report["files"]} == set(MODEL_FILES + ["reference.json", "search-reference.json"])
    for file in report["files"]:
        assert hashlib.sha256((output / file["name"]).read_bytes()).hexdigest() == file["sha256"]
    assert json.loads((output / "bundle.json").read_text()) == report
    assert not (output / "model.pt2").exists()


@pytest.mark.parametrize("fault", ["model_hash", "tokenizer_hash", "reference_hash", "receipt_failure",
                                 "source", "search_source", "model_origin_hash", "revision", "path_traversal"])
def test_bad_source_cannot_publish_a_success_bundle(prepared, tmp_path, fault):
    directory, reference, audit_path = prepared
    receipt = json.loads((directory / "preparation.json").read_text())
    audit = json.loads(audit_path.read_text())
    if fault == "model_hash":
        (directory / MODEL_FILES[0]).write_bytes(b"corrupt")
    elif fault == "tokenizer_hash":
        audit["sha256"]["tokenizer.json"] = "0" * 64
    elif fault == "reference_hash":
        (reference / "reference.json").write_text("{}")
    elif fault == "receipt_failure":
        receipt["success"] = False
    elif fault == "source":
        receipt["sourceCommit"] = "0" * 40
    elif fault == "search_source":
        receipt["searchSourceCommit"] = "0" * 40
    elif fault == "model_origin_hash":
        receipt["modelSha256"] = "0" * 64
    elif fault == "revision":
        audit["model_revision"] = "0" * 40
    elif fault == "path_traversal":
        receipt["files"][0]["name"] = "../outside.sentis"
    (directory / "preparation.json").write_text(json.dumps(receipt))
    audit_path.write_text(json.dumps(audit))
    output = tmp_path / "bundle"
    with pytest.raises((ValueError, OSError)):
        stage_player_bundle(*prepared, output)
    assert json.loads((output / "bundle.json").read_text())["success"] is False
    assert not list(output.glob("*.sentis"))


def test_existing_bundle_is_never_overwritten(prepared, tmp_path):
    output = tmp_path / "bundle"
    output.mkdir()
    sentinel = output / "model-fp32.sentis"
    sentinel.write_bytes(b"existing")
    with pytest.raises(ValueError, match="empty"):
        stage_player_bundle(*prepared, output)
    assert sentinel.read_bytes() == b"existing"


def test_source_changes_during_copy_cannot_publish_a_success_receipt(prepared, tmp_path, monkeypatch):
    import embeddinggemma_tools.player as player
    original = player.shutil.copyfile

    def changed(source, destination):
        original(source, destination)
        if source.name == "model-fp32.sentis":
            destination.write_bytes(b"changed during copy")

    monkeypatch.setattr(player.shutil, "copyfile", changed)
    output = tmp_path / "bundle"
    with pytest.raises(ValueError, match="changed during copy"):
        stage_player_bundle(*prepared, output)
    assert json.loads((output / "bundle.json").read_text())["success"] is False
    assert not (output / "model-fp32.sentis").exists()


def test_cli_reports_files_prepared_and_missing_input_without_execution_claims(prepared, tmp_path, capsys):
    directory, reference, audit = prepared
    args = ["--prepared", str(directory), "--reference", str(reference), "--audit", str(audit),
            "--output", str(tmp_path / "bundle")]
    assert main(args) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["unity_executed"] is False
    assert report["gpu_verified"] is False
    args[-1] = str(tmp_path / "failed-bundle")
    audit.unlink()
    assert main(args) == 1
    assert json.loads(capsys.readouterr().out)["success"] is False
