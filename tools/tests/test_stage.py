import hashlib
import json
import shutil
from pathlib import Path

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
    ids = [case["id"] for case in json.loads((Path(__file__).parents[1] / "cases/text.json").read_text(encoding="utf-8"))]
    cases = [dict(id=case_id, input_ids=[1] * 128, attention_mask=[1] * 128,
                  embedding=[1.0] + [0.0] * 767) for case_id in ids]
    (root / "reference.json").write_text(json.dumps(dict(schema_version=1, metadata=metadata, cases=cases)))
    (root / "tokenizer.json").write_text('{"model": {"type": "BPE"}}')
    (root / "model.pt2").write_bytes(b"model bytes")
    report = dict(metadata=metadata, case_count=15, minimum_cosine=1.0,
                  cases=[dict(id=case_id, cosine=1.0) for case_id in ids],
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


@pytest.mark.parametrize("fault", ["hash", "revision", "source", "cases", "cosine", "missing", "nan", "wrong_suite"])
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
    elif fault == "wrong_suite":
        reference_path = artifact / "reference.json"
        reference = json.loads(reference_path.read_text())
        reference["cases"][0]["id"] = "different-input-suite"
        report["cases"][0]["id"] = "different-input-suite"
        reference_path.write_text(json.dumps(reference))
        report["sha256"]["reference.json"] = hashlib.sha256(reference_path.read_bytes()).hexdigest()
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


@pytest.fixture
def search_artifact(artifact):
    from embeddinggemma_tools.search import SEARCH_CASES, load_search_cases, rank_documents
    from embeddinggemma_tools.text import format_text
    report = json.loads((artifact / "export-validation.json").read_text())
    suite = load_search_cases(SEARCH_CASES)
    def expand(row, role):
        return {**row, "role": role, "formatted_text": format_text(row["text"], role, row.get("title")),
                "input_ids": [1] * 128, "attention_mask": [1] * 128, "embedding": [1.0] + [0.0] * 767}
    docs = [expand(row, "document") for row in suite["documents"]]
    queries = [expand(row, "query") for row in suite["queries"]]
    for query in queries: query["ranking"] = rank_documents(query["embedding"], docs)
    reference = {"schema_version": 1, "metadata": report["metadata"], "documents": docs, "queries": queries}
    path = artifact / "search-reference.json"
    path.write_text(json.dumps(reference), encoding="utf-8")
    report["sha256"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    report["search_case_count"] = len(docs) + len(queries)
    report["search_cases"] = [{"id": row["id"], "cosine": 1.0} for row in docs + queries]
    report["search_minimum_cosine"] = 1.0
    (artifact / "export-validation.json").write_text(json.dumps(report))
    return artifact


def test_search_reference_is_audited_and_staged_with_the_model(search_artifact, tmp_path):
    project = tmp_path / "consumer"
    report = stage_reference(search_artifact, project, SOURCE, search=True)
    assert report["search_reference_staged"] is True
    assert report["search_case_count"] == 10
    assert (project / "artifacts/m1/search-reference.json").read_bytes() == (search_artifact / "search-reference.json").read_bytes()


@pytest.mark.parametrize("fault", ["missing", "hash", "ranking", "export"])
def test_search_audit_failure_never_replaces_existing_model(search_artifact, tmp_path, fault):
    path = search_artifact / "search-reference.json"
    report_path = search_artifact / "export-validation.json"
    report = json.loads(report_path.read_text())
    if fault == "missing": path.unlink()
    elif fault == "hash": path.write_text("corrupt")
    elif fault == "ranking":
        data = json.loads(path.read_text())
        data["queries"][0]["ranking"].reverse()
        path.write_text(json.dumps(data))
        report["sha256"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    elif fault == "export": report["search_cases"][0]["cosine"] = 0.5
    report_path.write_text(json.dumps(report))
    project = tmp_path / "consumer"
    model = project / "Assets/M1Generated/model.pt2"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"previous model")
    with pytest.raises((ValueError, FileNotFoundError)):
        stage_reference(search_artifact, project, SOURCE, search=True)
    assert model.read_bytes() == b"previous model"
    assert json.loads((project / "artifacts/m1-stage.json").read_text())["success"] is False


@pytest.mark.parametrize("fault", [None, "version", "tokenizer", "source", "rank"])
def test_cached_model_can_use_separately_audited_compatible_search_reference(search_artifact, tmp_path, fault):
    overlay = tmp_path / "search-only"
    overlay.mkdir()
    for name in ("search-reference.json", "export-validation.json"):
        shutil.copyfile(search_artifact / name, overlay / name)
    report_path = overlay / "export-validation.json"
    reference_path = overlay / "search-reference.json"
    report = json.loads(report_path.read_text())
    reference = json.loads(reference_path.read_text())
    report["metadata"]["source_commit"] = "c" * 40
    reference["metadata"]["source_commit"] = "c" * 40
    if fault == "version": report["metadata"]["versions"] = {"torch": "different"}
    elif fault == "tokenizer": report["sha256"]["tokenizer.json"] = "0" * 64
    elif fault == "source": reference["metadata"]["source_commit"] = "d" * 40
    elif fault == "rank": reference["queries"][0]["ranking"].reverse()
    reference_path.write_text(json.dumps(reference))
    report["sha256"]["search-reference.json"] = hashlib.sha256(reference_path.read_bytes()).hexdigest()
    report_path.write_text(json.dumps(report))
    project = tmp_path / "consumer"
    if fault:
        with pytest.raises(ValueError):
            stage_reference(search_artifact, project, SOURCE, search=True, search_source=overlay, search_source_commit="c" * 40)
        assert not (project / "Assets/M1Generated/model.pt2").exists()
    else:
        result = stage_reference(search_artifact, project, SOURCE, search=True, search_source=overlay, search_source_commit="c" * 40)
        assert result["source_commit"] == SOURCE
        assert result["search_source_commit"] == "c" * 40
        assert (project / "artifacts/m1/search-reference.json").read_bytes() == reference_path.read_bytes()
        assert (project / "artifacts/m1/search-export-validation.json").read_bytes() == report_path.read_bytes()
        assert result["sha256"]["model.pt2"] == hashlib.sha256((project / "Assets/M1Generated/model.pt2").read_bytes()).hexdigest()
