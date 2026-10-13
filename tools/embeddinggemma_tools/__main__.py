"""CLI entry point. Use: uv run python -m embeddinggemma_tools prepare --output ..."""

import argparse
import json
import os
import re
from pathlib import Path

from huggingface_hub import snapshot_download

from .prepare import MODEL_ID, MODEL_REVISION, prepare, prepare_multimodal_backbone, prepare_vision
from .search import SEARCH_CASES
from .prompt_reference import prepare_prompt_reference
from .native_silu_probe import prepare_silu_probe


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
    backbone = commands.add_parser("prepare-backbone", help="Validate projected-feature backbone; not encoder or Unity proof")
    backbone.add_argument("--output", type=Path, required=True)
    backbone.add_argument("--snapshot", type=Path)
    backbone.add_argument("--sequence-length", type=sequence_length, default=1024)
    backbone.add_argument("--revision", type=revision, default=MODEL_REVISION)
    backbone.add_argument("--image-feature-count", type=int, default=280)
    backbone.add_argument("--audio-feature-count", type=int, default=280)
    vision = commands.add_parser("prepare-vision", help="Official image processor / vision export gate; not Unity proof")
    vision.add_argument("--output", type=Path, required=True)
    vision.add_argument("--snapshot", type=Path)
    vision.add_argument("--revision", type=revision, default=MODEL_REVISION)
    prompts = commands.add_parser("prepare-prompts", help="Weights-free official multimodal prompt/token references")
    prompts.add_argument("--output", type=Path, required=True)
    prompts.add_argument("--snapshot", type=Path)
    prompts.add_argument("--revision", type=revision, default=MODEL_REVISION)
    silu=commands.add_parser("prepare-silu-probe",help="Weights-free native SiLU and ONNX Swish probe")
    silu.add_argument("--output",type=Path,required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare-silu-probe":
        summary=prepare_silu_probe(args.output,source_commit=os.environ.get("GITHUB_SHA","local"))
        print(json.dumps({name:summary[name] for name in ("success","stage","case_count","unity_validated")}))
        return
    snapshot = args.snapshot
    if snapshot is None:
        patterns = [
            "config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
            "config_sentence_transformers.json", "modules.json", "1_Pooling/config.json",
        ]
        if args.command in ("prepare-backbone", "prepare-vision"):
            patterns.append("processor_config.json")
        if args.command == "prepare-prompts":
            patterns = ["config.json", "tokenizer.json", "tokenizer_config.json", "config_sentence_transformers.json"]
        snapshot = Path(snapshot_download(MODEL_ID, revision=args.revision, allow_patterns=patterns))
    if args.command == "prepare-prompts":
        summary = prepare_prompt_reference(snapshot, args.output, revision=args.revision,
                                           source_commit=os.environ.get("GITHUB_SHA", "local"))
        print(json.dumps({name: summary[name] for name in ("success", "stage", "case_count", "unity_validated", "end_to_end_validated")}))
        return
    if args.command == "prepare-vision":
        summary = prepare_vision(snapshot, args.output, revision=args.revision,
                                 source_commit=os.environ.get("GITHUB_SHA", "local"))
        print(json.dumps({name: summary[name] for name in (
            "success", "stage", "profile_count", "case_count", "minimum_cosine", "unity_validated", "end_to_end_validated")}))
        return
    if args.command == "prepare-backbone":
        summary = prepare_multimodal_backbone(snapshot, args.output, args.sequence_length,
                    image_feature_count=args.image_feature_count, audio_feature_count=args.audio_feature_count,
                    revision=args.revision, source_commit=os.environ.get("GITHUB_SHA", "local"))
        print(json.dumps({name: summary[name] for name in (
            "success", "stage", "case_count", "minimum_cosine", "unity_validated", "encoders_validated")}))
        return
    summary = prepare(snapshot, args.cases, args.output, args.sequence_length, args.revision,
                      source_commit=os.environ.get("GITHUB_SHA", "local"), search_cases_path=args.search_cases)
    print(json.dumps({name: summary[name] for name in (
        "case_count", "minimum_cosine", "search_case_count", "search_minimum_cosine")}))


if __name__ == "__main__":
    main()
