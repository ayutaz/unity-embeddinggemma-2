"""Verify a successful pinned CI reference artifact before staging it for the Editor."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import shutil

from .prepare import MODEL_ID, MODEL_REVISION


def stage_reference(source: Path, project: Path, source_commit: str) -> dict:
    source, project = source.resolve(), project.resolve()
    audit_path = project / "artifacts/m1-stage.json"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit = {"success": False, "m1_reference_passed": False,
             "source": str(source), "source_commit": source_commit}

    def save():
        audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")

    save()
    try:
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
        digests = {}
        for name in ("reference.json", "tokenizer.json", "model.pt2"):
            with (source / name).open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest != report["sha256"].get(name):
                raise ValueError(f"Artifact SHA-256 mismatch: {name}")
            digests[name] = digest
        destination = project / "artifacts/m1"
        destination.mkdir(parents=True, exist_ok=True)
        if source != destination:
            for name in ("reference.json", "tokenizer.json", "export-validation.json"):
                shutil.copyfile(source / name, destination / name)
        model = project / "Assets/M1Generated/model.pt2"
        model.parent.mkdir(parents=True, exist_ok=True)
        pending = model.with_suffix(".pt2.pending")
        shutil.copyfile(source / "model.pt2", pending)
        pending.replace(model)
        audit.update(success=True, case_count=15, minimum_cosine=min(cosines), sha256=digests,
                     model_revision=MODEL_REVISION, staged_model=str(model))
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
    args = parser.parse_args(argv)
    result = stage_reference(args.source, args.project, args.source_commit)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
