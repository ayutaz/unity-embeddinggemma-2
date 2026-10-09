"""Pinned search corpus, cosine ranking and reference audit; no downloads here."""

import json
import math
from pathlib import Path
import re

from .reference import build_reference

SEARCH_CASES = Path(__file__).resolve().parents[2] / "Packages/com.ayutaz.embeddinggemma/Samples~/TextSearch/Resources/EmbeddingGemmaTextSearch/corpus.json"


def load_search_cases(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("search suite must be an object")
    seen = set()
    for group in ("documents", "queries"):
        rows = data.get(group)
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"search suite requires nonempty {group}")
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("search case must be an object")
            case_id, text, title = row.get("id"), row.get("text"), row.get("title")
            if (not isinstance(case_id, str) or not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", case_id)
                    or case_id in seen):
                raise ValueError("search IDs must be unique lowercase ASCII identifiers")
            if not isinstance(text, str) or not text.strip():
                raise ValueError("search text must not be blank")
            if title is not None and not isinstance(title, str):
                raise ValueError("document title must be a string or null")
            seen.add(case_id)
    return data


def _norm(vector):
    if not isinstance(vector, (list, tuple)) or not vector:
        raise ValueError("embedding must be a nonempty vector")
    if any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in vector):
        raise ValueError("embedding must be finite numeric values")
    norm = math.sqrt(math.fsum(value * value for value in vector))
    if not math.isfinite(norm) or norm == 0:
        raise ValueError("embedding must have a finite positive norm")
    return norm


def rank_documents(query, documents):
    query_norm = _norm(query)
    ranking = []
    seen = set()
    if not documents:
        raise ValueError("documents must not be empty")
    for document in documents:
        vector = document["embedding"]
        if len(vector) != len(query):
            raise ValueError("embedding dimensions must match")
        if document["id"] in seen:
            raise ValueError("duplicate document ID")
        seen.add(document["id"])
        norm = _norm(vector)
        score = math.fsum(a * b for a, b in zip(query, vector)) / (query_norm * norm)
        ranking.append({"document_id": document["id"], "score": max(-1.0, min(1.0, score))})
    return sorted(ranking, key=lambda row: (-row["score"], row["document_id"]))


def generate_search_reference(model, tokenizer, cases_path, destination, sequence_length, metadata):
    suite = load_search_cases(cases_path)
    inputs = [{**row, "role": "document"} for row in suite["documents"]]
    inputs += [{**row, "role": "query"} for row in suite["queries"]]
    rows = build_reference(model, tokenizer, inputs, sequence_length)
    count = len(suite["documents"])
    documents, queries = rows[:count], rows[count:]
    for query in queries:
        query["ranking"] = rank_documents(query["embedding"], documents)
    reference = {"schema_version": 1, "metadata": dict(metadata), "documents": documents, "queries": queries}
    validate_search_reference(reference, suite, metadata)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(reference, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return reference


def validate_search_reference(reference, suite, expected_metadata):
    if reference.get("schema_version") != 1 or reference.get("metadata") != expected_metadata:
        raise ValueError("search reference metadata mismatch")
    dimension = expected_metadata["embedding_dimension"]
    for group, role in (("documents", "document"), ("queries", "query")):
        actual, expected = reference.get(group), suite[group]
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(f"search reference {group} count mismatch")
        for row, source in zip(actual, expected):
            if (any(row.get(key) != source.get(key) for key in ("id", "text", "title"))
                    or row.get("role") != role):
                raise ValueError("search reference input mismatch")
            vector = row["embedding"]
            if len(vector) != dimension or abs(_norm(vector) - 1) > 0.001:
                raise ValueError("search reference embedding must have the declared dimension and unit norm")
    for query in reference["queries"]:
        expected = rank_documents(query["embedding"], reference["documents"])
        if query.get("ranking") != expected:
            raise ValueError("search reference ranking or score mismatch")
