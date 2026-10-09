"""Generate pinned reference data and validate the saved export before Unity import."""

import hashlib
import importlib.metadata
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from .export import export_pt2
from .model import load_text_model
from .reference import build_reference, load_cases, write_reference
from .search import generate_search_reference

MODEL_ID = "google/embeddinggemma-2"
MODEL_REVISION = "914f7f89142e33e77833254d9c9b90c3cef7303b"


def prepare(
    snapshot: Path, cases_path: Path, destination: Path,
    sequence_length: int, revision: str, source_commit: str = "local", *, search_cases_path: Path | None = None,
) -> dict:
    destination.mkdir(parents=True, exist_ok=True)
    # Only the current completed generation may publish a success report.
    (destination / "export-validation.json").unlink(missing_ok=True)
    model = load_text_model(snapshot)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    cases = build_reference(model, tokenizer, load_cases(cases_path), sequence_length)
    metadata = {
        "model_id": MODEL_ID, "model_revision": revision, "source_commit": source_commit,
        "sequence_length": sequence_length, "batch_size": 1, "dtype": "float32",
        "embedding_dimension": model.config.embedding_dim,
        "pooling": "masked_mean_including_prompt", "normalization": "l2",
        "versions": {name: importlib.metadata.version(name)
                     for name in ("torch", "transformers", "sentence-transformers", "tokenizers")},
    }
    metadata["versions"]["torch"] = torch.__version__
    write_reference(destination / "reference.json", cases, metadata)
    search_cases = []
    if search_cases_path is not None:
        search = generate_search_reference(model, tokenizer, search_cases_path,
                                          destination / "search-reference.json", sequence_length, metadata)
        search_cases = search["documents"] + search["queries"]
    # The last reference call configures fixed-length truncation and padding in the backend.
    tokenizer.backend_tokenizer.save(str(destination / "tokenizer.json"))
    first = cases[0]
    export_pt2(model, torch.tensor([first["input_ids"]]), torch.tensor([first["attention_mask"]]),
               destination / "model.pt2")
    exported = torch.export.load(destination / "model.pt2").module()
    similarities = []
    with torch.no_grad():
        for case in cases + search_cases:
            actual = exported(torch.tensor([case["input_ids"]]), torch.tensor([case["attention_mask"]]))[0]
            expected = torch.tensor(case["embedding"])
            if not torch.isfinite(actual).all():
                raise ValueError(f"export embedding must be finite: {case['id']}")
            similarity = torch.nn.functional.cosine_similarity(actual, expected, dim=0).item()
            if similarity < 0.999999:
                raise ValueError(f"export differs from reference: {case['id']} cosine={similarity}")
            similarities.append({"id": case["id"], "cosine": similarity})
    digests = {}
    names = ["model.pt2", "reference.json", "tokenizer.json"]
    if search_cases_path is not None:
        names.append("search-reference.json")
    for name in names:
        with (destination / name).open("rb") as stream:
            digests[name] = hashlib.file_digest(stream, "sha256").hexdigest()
    base_similarities = similarities[:len(cases)]
    summary = {"case_count": len(cases), "minimum_cosine": min(row["cosine"] for row in base_similarities),
               "cases": base_similarities, "sha256": digests, "metadata": metadata}
    if search_cases_path is not None:
        search_similarities = similarities[len(cases):]
        summary.update(search_case_count=len(search_similarities), search_cases=search_similarities,
                       search_minimum_cosine=min(row["cosine"] for row in search_similarities))
    (destination / "export-validation.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return summary
