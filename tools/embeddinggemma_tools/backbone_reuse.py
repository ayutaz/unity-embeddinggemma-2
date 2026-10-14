"""Audit existing backbone programs before using them for another prompt role."""

import hashlib
import json
from pathlib import Path
import re


def audit_backbone_source(directory, modality, *, revision, input_sha256, profiles):
    if modality not in ("audio", "image"):
        raise ValueError("backbone modality is invalid")
    root = Path(directory).resolve()
    receipt = json.loads((root / f"{modality}-embedding-validation.json").read_text(encoding="utf-8-sig"))
    if receipt.get("success") is not True or receipt.get("stage") != f"python_{modality}_embeddings_only":
        raise ValueError("backbone source is not a successful Python reference")
    def audited(name):
        if not isinstance(name, str):
            raise ValueError("backbone path must be a string")
        path = (root / name).resolve()
        metadata = receipt.get("files", {}).get(name, {})
        if not path.is_relative_to(root) or not path.is_file() or path.stat().st_size != metadata.get("bytes"):
            raise ValueError("backbone path or size mismatch")
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != metadata.get("sha256"):
            raise ValueError("backbone hash mismatch")
        return path
    reference_path = audited(f"{modality}-embedding-reference.json")
    reference = json.loads(reference_path.read_text(encoding="utf-8-sig"))
    context = reference.get("metadata", {})
    if (context != receipt.get("metadata") or context.get("model_revision") != revision
            or context.get("input_sha256") != input_sha256
            or re.fullmatch(r"[0-9a-f]{40}", context.get("source_commit", "")) is None
            or set(reference.get("profiles", {})) != set(profiles)):
        raise ValueError("backbone provenance or profile set mismatch")
    programs = {}
    for name, expected in profiles.items():
        spec = reference["profiles"][name]
        if any(spec.get(key) != value for key, value in expected.items()):
            raise ValueError("backbone fixed shape/profile mismatch")
        programs[name] = audited(spec.get("graph"))
    return dict(source_commit=context["source_commit"], reference_sha256=receipt["files"][reference_path.name]["sha256"],
                programs=programs)
