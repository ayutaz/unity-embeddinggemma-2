"""Reference vectors from Transformers and official sentence-transformers modules."""

import json
from pathlib import Path

import torch
from sentence_transformers.base.modules.normalize import Normalize
from sentence_transformers.sentence_transformer.modules.pooling import Pooling

from .text import format_text


def load_cases(path: Path) -> list[dict]:
    cases = json.loads(path.read_text(encoding="utf-8"))
    if not cases:
        raise ValueError("case suite must not be empty")
    seen = set()
    expanded = []
    for source in cases:
        row = dict(source)
        if row["id"] in seen:
            raise ValueError(f"duplicate case id: {row['id']}")
        seen.add(row["id"])
        repeat = row.pop("repeat", 1)
        if type(repeat) is not int or repeat < 1:
            raise ValueError("repeat must be a positive integer")
        row["text"] *= repeat
        format_text(row["text"], row["role"], row.get("title"))
        expanded.append(row)
    return expanded


@torch.inference_mode()
def build_reference(model, tokenizer, cases: list[dict], sequence_length: int) -> list[dict]:
    pooling = Pooling(model.config.embedding_dim, pooling_mode="mean", include_prompt=True)
    normalize = Normalize()
    rows = []
    for case in cases:
        text = format_text(case["text"], case["role"], case.get("title"))
        inputs = tokenizer(text, padding="max_length", truncation=True,
                           max_length=sequence_length, return_tensors="pt")
        hidden = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]).last_hidden_state
        features = pooling({"token_embeddings": hidden, "attention_mask": inputs["attention_mask"]})
        embedding = normalize(features)["sentence_embedding"][0]
        if not torch.isfinite(embedding).all():
            raise ValueError(f"reference embedding must be finite: {case['id']}")
        rows.append({**case, "formatted_text": text,
                     "input_ids": inputs["input_ids"][0].tolist(),
                     "attention_mask": inputs["attention_mask"][0].tolist(),
                     "embedding": embedding.tolist()})
    return rows


def write_reference(destination: Path, cases: list[dict], metadata: dict) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps({"schema_version": 1, "metadata": metadata, "cases": cases},
                   ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
