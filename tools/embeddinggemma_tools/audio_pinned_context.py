"""Compare an audio diagnostic producer against the original frozen context."""

import numpy as np
import argparse
import hashlib
import json
from pathlib import Path

SIGNALS = {"silence", "impulse", "sine", "nyquist", "noise", "padded_noise"}


def compare_references(current, current_audit, pinned, pinned_audit):
    for reference in (current, pinned):
        rows = reference.get("cases", [])
        if len(rows) != 6 or {row.get("id") for row in rows} != SIGNALS:
            raise ValueError("pinned audio context requires six original signals")
        metadata = reference["metadata"]
        if (not metadata.get("versions") or not metadata.get("input_sha256")
                or not metadata.get("model_revision") or not metadata.get("source_commit")):
            raise ValueError("pinned audio context metadata is missing")
        if any(not spec.get("frame_mask") for spec in reference.get("profiles", {}).values()):
            raise ValueError("pinned audio context frame mask is missing")
        if metadata.get("feature_rtol") != 1e-4 or metadata.get("feature_atol") != 1e-5:
            raise ValueError("pinned audio context must keep the fixed tolerance")

    def sha(audit, name):
        if name is None:
            return None
        value = audit.get("files", {}).get(name, {}).get("sha256")
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError("pinned audio context omits an artifact hash")
        return value

    old_rows = {row["id"]: row for row in pinned["cases"]}
    old_metadata, new_metadata = pinned["metadata"], current["metadata"]
    model_equal = (new_metadata["model_revision"] == old_metadata["model_revision"]
                   and new_metadata["input_sha256"] == old_metadata["input_sha256"])
    versions_equal = new_metadata.get("versions") == old_metadata.get("versions")
    graph_matches, mask_matches, rows = {}, {}, []
    for row in current["cases"]:
        original = old_rows[row["id"]]
        if row["profile"] != original["profile"]:
            raise ValueError("pinned audio context profile mismatch")
        profile = row["profile"]
        spec, old_spec = current["profiles"][profile], pinned["profiles"][profile]
        if (spec["feature_count"], spec["feature_width"]) != (old_spec["feature_count"], old_spec["feature_width"]):
            raise ValueError("pinned audio context output profile mismatch")
        graph_matches[profile] = sha(current_audit, spec["graph"]) == sha(pinned_audit, old_spec["graph"])
        mask_matches[profile] = sha(current_audit, spec.get("frame_mask")) == sha(pinned_audit, old_spec.get("frame_mask"))
        actual = np.asarray(row["features"], dtype=np.float32)
        expected = np.asarray(original["features"], dtype=np.float32)
        if (actual.shape != (spec["feature_count"], spec["feature_width"]) or expected.shape != actual.shape
                or not np.isfinite(actual).all() or not np.isfinite(expected).all()):
            raise ValueError("pinned audio context shape/nonfinite mismatch")
        difference = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
        allowed = 1e-5 + 1e-4 * np.abs(expected.astype(np.float64))
        mismatches = int((difference > allowed).sum())
        rows.append(dict(id=row["id"], profile=profile,
                         input_sha_equal=sha(current_audit, row["input_features"]) == sha(pinned_audit, original["input_features"]),
                         native_exact_match=bool(np.array_equal(actual, expected)),
                         exact_difference_count=int((actual != expected).sum()),
                         native_fixed_matches=mismatches == 0, mismatched_elements=mismatches,
                         maximum_absolute_error=float(difference.max()), maximum_tolerance_ratio=float((difference / allowed).max())))
    validated = (model_equal and versions_equal and all(graph_matches.values()) and all(mask_matches.values())
                 and all(row["native_fixed_matches"] and row["input_sha_equal"] for row in rows))
    return dict(stage="audio_pinned_context_only", context_validated=validated, case_count=len(rows),
                model_inputs_equal=model_equal, library_versions_equal=versions_equal,
                original_graph_sha_equal=graph_matches, frame_mask_sha_equal=mask_matches,
                current_producer=new_metadata["source_commit"], pinned_producer=old_metadata["source_commit"],
                math_environment_comparison="unavailable" if not all(
                    value.get("cpu_math_environment") for value in (old_metadata, new_metadata)) else
                    ("equal" if old_metadata["cpu_math_environment"] == new_metadata["cpu_math_environment"] else "different"),
                feature_rtol=1e-4, feature_atol=1e-5, cases=rows,
                full_encoder_validated=False, unity_validated=False, end_to_end_validated=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--pinned", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    protected = {(root / name).resolve() for root in (args.current, args.pinned)
                 for name in ("audio-encoder-reference.json", "audio-encoder-validation.json")}
    if args.output.resolve() in protected:
        raise ValueError("context output must not overwrite a source reference")
    args.output.unlink(missing_ok=True)
    def read(directory):
        audit = json.loads((directory / "audio-encoder-validation.json").read_text(encoding="utf-8"))
        raw = (directory / "audio-encoder-reference.json").read_bytes()
        spec = audit.get("files", {}).get("audio-encoder-reference.json", {})
        if len(raw) != spec.get("bytes") or hashlib.sha256(raw).hexdigest() != spec.get("sha256"):
            raise ValueError("audio reference hash/size mismatch")
        return json.loads(raw), audit
    current, current_audit = read(args.current)
    pinned, pinned_audit = read(args.pinned)
    report = compare_references(current, current_audit, pinned, pinned_audit)
    report.update(reference_hashes_audited=True, graph_binaries_audited=False, stage_binaries_audited=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(dict(context_validated=report["context_validated"], case_count=report["case_count"],
                         full_encoder_validated=False, unity_validated=False)))
    if not report["context_validated"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
