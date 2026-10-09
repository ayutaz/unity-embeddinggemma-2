"""Verify a successful pinned CI reference artifact before staging it for the Editor."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import shutil

from .prepare import MODEL_ID, MODEL_REVISION
from .search import SEARCH_CASES, load_search_cases, validate_search_reference


def stage_reference(source: Path, project: Path, source_commit: str, *, search=False,
                    search_source: Path | None = None, search_source_commit: str | None = None) -> dict:
    source, project = source.resolve(), project.resolve()
    audit_path = project / "artifacts/m1-stage.json"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit = {"success": False, "m1_reference_passed": False,
             "source": str(source), "source_commit": source_commit}

    def save():
        audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")

    save()
    try:
        if (search_source is not None or search_source_commit is not None) and not search:
            raise ValueError("Separate search reference requires --search")
        if (search_source is None) != (search_source_commit is None):
            raise ValueError("Provide both the search source directory and its exact checkout SHA")
        if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
            raise ValueError("Expected the exact CI checkout source SHA")
        report = json.loads((source / "export-validation.json").read_text(encoding="utf-8"))
        reference = json.loads((source / "reference.json").read_text(encoding="utf-8"))
        expected = dict(model_id=MODEL_ID, model_revision=MODEL_REVISION,
                        source_commit=source_commit, sequence_length=128,
                        batch_size=1, dtype="float32", embedding_dimension=768)
        for name, value in expected.items():
            if report["metadata"].get(name) != value or reference["metadata"].get(name) != value:
                raise ValueError(f"Artifact metadata mismatch: {name}")
        rows, cases = report["cases"], reference["cases"]
        if reference.get("schema_version") != 1 or report["case_count"] != 15 or len(rows) != 15 or len(cases) != 15:
            raise ValueError("All 15 reference and export cases are required")
        if len({case["id"] for case in cases}) != 15 or [row["id"] for row in rows] != [case["id"] for case in cases]:
            raise ValueError("Reference and export case IDs must match")
        fixed_cases = json.loads((Path(__file__).resolve().parents[1] / "cases/text.json").read_text(encoding="utf-8"))
        if [case["id"] for case in cases] != [case["id"] for case in fixed_cases]:
            raise ValueError("Artifact must use the fixed input suite in the same order")
        cosines = [row["cosine"] for row in rows]
        if not all(math.isfinite(value) and value >= 0.999999 for value in cosines):
            raise ValueError("Python export validation must pass every case")
        if not math.isfinite(report["minimum_cosine"]) or report["minimum_cosine"] != min(cosines):
            raise ValueError("Invalid minimum cosine")
        for case in cases:
            vector = case["embedding"]
            if len(case["input_ids"]) != 128 or len(case["attention_mask"]) != 128 or len(vector) != 768:
                raise ValueError(f"Invalid reference shape: {case['id']}")
            if not all(math.isfinite(value) for value in vector) or abs(sum(value * value for value in vector) - 1) > 0.002:
                raise ValueError(f"Reference must be finite and normalized: {case['id']}")
        names = ["reference.json", "tokenizer.json", "model.pt2"]
        if search:
            search_directory = search_source.resolve() if search_source is not None else source
            search_commit = search_source_commit or source_commit
            if not re.fullmatch(r"[0-9a-f]{40}", search_commit):
                raise ValueError("Expected the exact search CI checkout source SHA")
            search_report = json.loads((search_directory / "export-validation.json").read_text(encoding="utf-8"))
            search_metadata = {**report["metadata"], "source_commit": search_commit}
            if search_report["metadata"] != search_metadata:
                raise ValueError("Cached model and search reference must have identical model, export settings and dependency versions")
            if search_report["sha256"].get("tokenizer.json") != report["sha256"].get("tokenizer.json"):
                raise ValueError("Cached model and search reference must use the identical tokenizer hash")
            search_reference = json.loads((search_directory / "search-reference.json").read_text(encoding="utf-8"))
            validate_search_reference(search_reference, load_search_cases(SEARCH_CASES), search_metadata)
            search_rows = search_reference["documents"] + search_reference["queries"]
            comparisons = search_report["search_cases"]
            if (search_report["search_case_count"] != len(search_rows)
                    or [row["id"] for row in comparisons] != [row["id"] for row in search_rows]):
                raise ValueError("Search export cases must match the complete fixed suite")
            search_cosines = [row["cosine"] for row in comparisons]
            if (not all(math.isfinite(value) and value >= 0.999999 for value in search_cosines)
                    or search_report["search_minimum_cosine"] != min(search_cosines)):
                raise ValueError("Search export validation must pass every case")
        digests = {}
        for name in names:
            with (source / name).open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest != report["sha256"].get(name):
                raise ValueError(f"Artifact SHA-256 mismatch: {name}")
            digests[name] = digest
        if search:
            with (search_directory / "search-reference.json").open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest != search_report["sha256"].get("search-reference.json"):
                raise ValueError("Artifact SHA-256 mismatch: search-reference.json")
            digests["search-reference.json"] = digest
        destination = project / "artifacts/m1"
        destination.mkdir(parents=True, exist_ok=True)
        if source != destination:
            staged_names = ["reference.json", "tokenizer.json", "export-validation.json"]
            for name in staged_names:
                shutil.copyfile(source / name, destination / name)
        if search and search_directory != destination:
            shutil.copyfile(search_directory / "search-reference.json", destination / "search-reference.json")
            if search_source is not None:
                shutil.copyfile(search_directory / "export-validation.json", destination / "search-export-validation.json")
        model = project / "Assets/M1Generated/model.pt2"
        model.parent.mkdir(parents=True, exist_ok=True)
        pending = model.with_suffix(".pt2.pending")
        shutil.copyfile(source / "model.pt2", pending)
        pending.replace(model)
        audit.update(success=True, case_count=15, minimum_cosine=min(cosines), sha256=digests,
                     model_revision=MODEL_REVISION, staged_model=str(model))
        if search:
            audit.update(search_reference_staged=True, search_case_count=len(search_rows),
                         search_minimum_cosine=min(search_cosines), search_source_commit=search_commit,
                         cached_model_used=search_source is not None)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        audit["error"] = str(exc)
        save()
        raise
    save()
    return audit


def main(argv=None):
    project = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=project / "artifacts/m1")
    parser.add_argument("--project", type=Path, default=project)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--search", action="store_true", help="Require and stage the audited TextSearch reference")
    parser.add_argument("--search-source", type=Path, help="Separate compact search artifact; reuse a compatible audited model")
    parser.add_argument("--search-source-commit", help="Exact checkout SHA of the separate search artifact")
    args = parser.parse_args(argv)
    result = stage_reference(args.source, args.project, args.source_commit, search=args.search,
                             search_source=args.search_source, search_source_commit=args.search_source_commit)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
