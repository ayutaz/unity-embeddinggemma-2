import hashlib
import json

import pytest

from embeddinggemma_tools.stage import stage_reference


SOURCE = "b3b0d76e1d3b6f4e44af1e9ba810d5b3cb6b051d"
REVISION = "914f7f89142e33e77833254d9c9b90c3cef7303b"


@pytest.fixture
def artifact(tmp_path):
    root = tmp_path / "reference"
    root.mkdir()
    metadata = dict(model_id="google/embeddinggemma-2", model_revision=REVISION,
                    source_commit=SOURCE, sequence_length=128, batch_size=1,
                    dtype="float32", embedding_dimension=768)
    cases = [dict(id=str(i), input_ids=[1] * 128, attention_mask=[1] * 128,
                  embedding=[1.0] + [0.0] * 767) for i in range(15)]
    (root / "reference.json").write_text(json.dumps(dict(schema_version=1, metadata=metadata, cases=cases)))
    (root / "tokenizer.json").write_text('{"model": {"type": "BPE"}}')
    (root / "model.pt2").write_bytes(b"model bytes")
    report = dict(metadata=metadata, case_count=15, minimum_cosine=1.0,
                  cases=[dict(id=str(i), cosine=1.0) for i in range(15)],
                  sha256={name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                          for name in ("model.pt2", "reference.json", "tokenizer.json")})
    (root / "export-validation.json").write_text(json.dumps(report))
    return root


def test_verified_artifact_is_staged_with_small_audit_report(artifact, tmp_path):
    project = tmp_path / "project"
    report = stage_reference(artifact, project, SOURCE)
    assert report["success"] is True
    assert report["source_commit"] == SOURCE
    assert report["m1_reference_passed"] is False
    assert (project / "Assets/M1Generated/model.pt2").read_bytes() == b"model bytes"
    assert (project / "artifacts/m1/reference.json").read_bytes() == (artifact / "reference.json").read_bytes()
    assert json.loads((project / "artifacts/m1-stage.json").read_text())["success"] is True


@pytest.mark.parametrize("fault", ["hash", "revision", "source", "cases", "cosine", "missing", "nan"])
def test_invalid_artifact_never_replaces_staged_model(artifact, tmp_path, fault):
    report_path = artifact / "export-validation.json"
    report = json.loads(report_path.read_text())
    if fault == "hash":
        (artifact / "model.pt2").write_bytes(b"corrupt")
    elif fault == "revision":
        report["metadata"]["model_revision"] = "0" * 40
    elif fault == "source":
        report["metadata"]["source_commit"] = "0" * 40
    elif fault == "cases":
        report["cases"].pop()
    elif fault == "cosine":
        report["cases"][0]["cosine"] = 0.9
    elif fault == "missing":
        (artifact / "tokenizer.json").unlink()
    elif fault == "nan":
        report["minimum_cosine"] = float("nan")
    report_path.write_text(json.dumps(report))
    project = tmp_path / "project"
    target = project / "Assets/M1Generated/model.pt2"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"previous model")
    (project / "artifacts").mkdir()
    (project / "artifacts/m1-stage.json").write_text('{"success": true}')
    with pytest.raises((ValueError, FileNotFoundError)):
        stage_reference(artifact, project, SOURCE)
    assert target.read_bytes() == b"previous model"
    assert json.loads((project / "artifacts/m1-stage.json").read_text())["success"] is False
