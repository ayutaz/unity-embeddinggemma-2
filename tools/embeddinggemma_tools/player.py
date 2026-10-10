"""Stage an audited, portable validation bundle; never claim Player execution."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import shutil

MODEL_REVISION = "914f7f89142e33e77833254d9c9b90c3cef7303b"
MODEL_FILES = ("model-fp32.sentis", "model-float16.sentis", "tokenizer.json")


def _object(path):
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path.name}")
    return value


def _digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def stage_player_bundle(prepared, reference, audit_path, output):
    prepared, reference, audit_path, output = map(lambda path: Path(path).resolve(),
                                                 (prepared, reference, audit_path, output))
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Bundle output must be empty; existing files are never overwritten")
    output.mkdir(parents=True, exist_ok=True)
    report = {"success": False, "unity_executed": False, "gpu_verified": False, "files": []}
    receipt = output / "bundle.json"

    def save():
        receipt.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    save()
    try:
        audit = _object(audit_path)
        preparation = _object(prepared / "preparation.json")
        if (audit.get("success") is not True or audit.get("search_reference_staged") is not True
                or audit.get("model_revision") != MODEL_REVISION or preparation.get("success") is not True
                or preparation.get("unityVersion") != "6000.3.16f1"):
            raise ValueError("Successful pinned stage and Unity preparation receipts are required")
        for stage_name, prepared_name in (("source_commit", "sourceCommit"), ("search_source_commit", "searchSourceCommit")):
            commit = audit.get(stage_name)
            if not isinstance(commit, str) or not re.fullmatch(r"[a-f0-9]{40}", commit) or preparation.get(prepared_name) != commit:
                raise ValueError(f"Preparation source mismatch: {stage_name}")
        origin_hash = audit.get("sha256", {}).get("model.pt2")
        if not isinstance(origin_hash, str) or not re.fullmatch(r"[a-f0-9]{64}", origin_hash) or preparation.get("modelSha256") != origin_hash:
            raise ValueError("Prepared models must originate from the staged model hash")
        files = preparation.get("files")
        if not isinstance(files, list) or [entry.get("name") for entry in files if isinstance(entry, dict)] != list(MODEL_FILES) or len(files) != 3:
            raise ValueError("Preparation must declare exactly the two model files and tokenizer")
        sources = []
        for entry in files:
            path = prepared / entry["name"]
            digest = _digest(path)
            if path.stat().st_size != entry.get("bytes") or digest != entry.get("sha256"):
                raise ValueError(f"Prepared file hash or size mismatch: {path.name}")
            if path.name == "tokenizer.json" and digest != audit["sha256"].get(path.name):
                raise ValueError("Prepared tokenizer must match the staged reference tokenizer")
            sources.append((path, digest))
        references = {}
        for name, source in (("reference.json", "source_commit"), ("search-reference.json", "search_source_commit")):
            path = reference / name
            digest = _digest(path)
            if digest != audit["sha256"].get(name):
                raise ValueError(f"Reference hash mismatch: {name}")
            data = _object(path)
            metadata = data.get("metadata", {})
            expected = {"model_id": "google/embeddinggemma-2", "model_revision": MODEL_REVISION,
                        "source_commit": audit[source], "sequence_length": 128, "batch_size": 1,
                        "dtype": "float32", "embedding_dimension": 768}
            if data.get("schema_version") != 1 or any(metadata.get(key) != value for key, value in expected.items()):
                raise ValueError(f"Pinned reference metadata mismatch: {name}")
            groups = {"cases": 15} if name == "reference.json" else {"documents": 6, "queries": 4}
            rows = []
            for group, count in groups.items():
                items = data.get(group)
                if not isinstance(items, list) or len(items) != count:
                    raise ValueError(f"Reference requires {count} {group}")
                rows.extend(items)
            for row in rows:
                vector, ids, mask = row["embedding"], row["input_ids"], row["attention_mask"]
                if (len(vector) != 768 or len(ids) != 128 or len(mask) != 128
                        or any(type(value) is not int or value < 0 for value in ids)
                        or any(type(value) is not int or value not in (0, 1) for value in mask)
                        or any(type(value) not in (int, float) or not math.isfinite(value) for value in vector)
                        or abs(math.fsum(value * value for value in vector) - 1) > 0.002):
                    raise ValueError("Reference tokens and finite normalized vectors are required")
            references[name] = metadata
            sources.append((path, digest))
        if {key: value for key, value in references["reference.json"].items() if key != "source_commit"} != {
                key: value for key, value in references["search-reference.json"].items() if key != "source_commit"}:
            raise ValueError("M1 and search reference model settings and dependency versions must match")
        for source, digest in sources:
            target = output / source.name
            pending = target.with_suffix(target.suffix + ".pending")
            shutil.copyfile(source, pending)
            if _digest(pending) != digest:
                raise ValueError(f"File changed during copy: {source.name}")
            pending.replace(target)
            report["files"].append({"name": target.name, "bytes": target.stat().st_size, "sha256": digest})
        report.update(success=True, model_revision=MODEL_REVISION, model_source=audit["source_commit"],
                      search_source=audit["search_source_commit"], model_origin_sha256=origin_hash,
                      unity_preparation_version=preparation["unityVersion"])
        return report
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        report["error"] = str(exc)
        raise
    finally:
        save()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = stage_player_bundle(args.prepared, args.reference, args.audit, args.output)
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        report = {"success": False, "unity_executed": False, "gpu_verified": False, "error": str(exc)}
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
