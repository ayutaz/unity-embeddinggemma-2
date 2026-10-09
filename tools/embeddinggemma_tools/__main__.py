"""CLI entry point. Use: uv run python -m embeddinggemma_tools prepare --output ..."""

import argparse
import json
import os
import re
from pathlib import Path

from huggingface_hub import snapshot_download

from .prepare import MODEL_ID, MODEL_REVISION, prepare
from .search import SEARCH_CASES


def sequence_length(value):
    parsed = int(value)
    if not 2 <= parsed <= 8192:
        raise argparse.ArgumentTypeError("sequence length must be between 2 and 8192")
    return parsed


def revision(value):
    if re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise argparse.ArgumentTypeError("revision must be an immutable 40-character commit SHA")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("prepare", help="Generate reference vectors and a validated .pt2 export")
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--snapshot", type=Path, help="Existing local snapshot; no download when specified")
    command.add_argument("--cases", type=Path, default=Path(__file__).resolve().parents[1] / "cases" / "text.json")
    command.add_argument("--sequence-length", type=sequence_length, default=128)
    command.add_argument("--revision", type=revision, default=MODEL_REVISION)
    command.add_argument("--search-cases", type=Path, default=SEARCH_CASES,
                         help="Fixed document/query suite included in the TextSearch sample")
    args = parser.parse_args(argv)
    snapshot = args.snapshot
    if snapshot is None:
        snapshot = Path(snapshot_download(MODEL_ID, revision=args.revision, allow_patterns=[
            "config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
            "config_sentence_transformers.json", "modules.json", "1_Pooling/config.json",
        ]))
    summary = prepare(snapshot, args.cases, args.output, args.sequence_length, args.revision,
                      source_commit=os.environ.get("GITHUB_SHA", "local"), search_cases_path=args.search_cases)
    print(json.dumps({name: summary[name] for name in (
        "case_count", "minimum_cosine", "search_case_count", "search_minimum_cosine")}))


if __name__ == "__main__":
    main()
