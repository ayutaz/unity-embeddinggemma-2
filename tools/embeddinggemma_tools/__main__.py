"""CLI entry point. Use: uv run python -m embeddinggemma_tools prepare --output ..."""

import argparse
import json
import os
import re
from pathlib import Path

from huggingface_hub import snapshot_download

from .prepare import MODEL_ID, MODEL_REVISION, prepare, prepare_audio_encoder, prepare_audio_runtime_stages, prepare_multimodal_backbone, prepare_vision
from .search import SEARCH_CASES
from .prompt_reference import prepare_prompt_reference
from .image_embeddings import prepare_image_embeddings
from .audio_embeddings import prepare_audio_embeddings
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
    audio = commands.add_parser("prepare-audio-encoder", help="Official audio encoder export gate; not Unity or final embedding proof")
    audio.add_argument("--output", type=Path, required=True)
    audio.add_argument("--preprocessing", type=Path, required=True)
    audio.add_argument("--snapshot", type=Path)
    audio.add_argument("--revision", type=revision, default=MODEL_REVISION)
    audio.add_argument("--native-silu",action="store_true",help="Keep native SiLU math and create standard Swish ONNX companions")
    audio.add_argument("--collect-onnx-failures", action="store_true",
                       help="Finish native references while recording ONNX failures; still exits with failure")
    stages = commands.add_parser("prepare-audio-runtime-stages", help="Pinned native stages; context only, not encoder acceptance")
    stages.add_argument("--output", type=Path, required=True)
    stages.add_argument("--encoder-source", type=Path, required=True)
    stages.add_argument("--snapshot", type=Path)
    stages.add_argument("--revision", type=revision, default=MODEL_REVISION)
    prompts = commands.add_parser("prepare-prompts", help="Weights-free official multimodal prompt/token references")
    prompts.add_argument("--output", type=Path, required=True)
    prompts.add_argument("--snapshot", type=Path)
    prompts.add_argument("--revision", type=revision, default=MODEL_REVISION)
    images = commands.add_parser("prepare-image-embeddings", help="Official image final vectors and fixed backbone profiles")
    images.add_argument("--output", type=Path, required=True)
    images.add_argument("--snapshot", type=Path)
    images.add_argument("--vision-source", type=Path, required=True)
    images.add_argument("--sequence-length", type=sequence_length, default=1024)
    images.add_argument("--revision", type=revision, default=MODEL_REVISION)
    images.add_argument("--role", choices=("query", "document", "raw"), default="document")
    images.add_argument("--vision-source-commit", type=revision)
    final_audio = commands.add_parser("prepare-audio-embeddings", help="Official audio final vectors and native-only saved-chain gate")
    final_audio.add_argument("--output", type=Path, required=True)
    final_audio.add_argument("--snapshot", type=Path)
    final_audio.add_argument("--encoder-source", type=Path, required=True)
    final_audio.add_argument("--sequence-length", type=sequence_length, default=1024)
    final_audio.add_argument("--revision", type=revision, default=MODEL_REVISION)
    final_audio.add_argument("--role", choices=("query", "document", "raw"), default="document")
    final_audio.add_argument("--encoder-source-commit", type=revision)
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
        if args.command in ("prepare-backbone", "prepare-vision", "prepare-audio-encoder", "prepare-image-embeddings", "prepare-audio-runtime-stages", "prepare-audio-embeddings"):
            patterns.append("processor_config.json")
        if args.command == "prepare-prompts":
            patterns = ["config.json", "tokenizer.json", "tokenizer_config.json", "config_sentence_transformers.json"]
        snapshot = Path(snapshot_download(MODEL_ID, revision=args.revision, allow_patterns=patterns))
    if args.command == "prepare-audio-embeddings":
        summary = prepare_audio_embeddings(snapshot, args.encoder_source, args.output, sequence_length=args.sequence_length,
                                           revision=args.revision, source_commit=os.environ.get("GITHUB_SHA", "local"),
                                           role=args.role, encoder_source_commit=args.encoder_source_commit)
        print(json.dumps({name: summary[name] for name in (
            "success", "stage", "case_count", "profile_count", "unity_validated", "gpu_chain_validated", "onnx_encoder_numerically_validated")}))
        return
    if args.command == "prepare-audio-runtime-stages":
        summary = prepare_audio_runtime_stages(snapshot, args.encoder_source, args.output, revision=args.revision,
                                               source_commit=os.environ.get("GITHUB_SHA", "local"))
        print(json.dumps({name: summary[name] for name in (
            "success", "stage", "profile_count", "case_count", "context_validated", "full_encoder_validated",
            "unity_validated", "end_to_end_validated")}))
        return
    if args.command == "prepare-audio-encoder":
        summary = prepare_audio_encoder(snapshot, args.preprocessing, args.output, revision=args.revision,
                                         source_commit=os.environ.get("GITHUB_SHA", "local"),native_silu=args.native_silu,
                                         collect_onnx_failures=args.collect_onnx_failures)
        print(json.dumps({name: summary[name] for name in (
            "success", "stage", "profile_count", "case_count", "minimum_cosine", "unity_validated", "end_to_end_validated")}))
        return
    if args.command == "prepare-image-embeddings":
        summary = prepare_image_embeddings(snapshot,args.vision_source,args.output,sequence_length=args.sequence_length,
                                           revision=args.revision,source_commit=os.environ.get("GITHUB_SHA","local"),
                                           role=args.role,vision_source_commit=args.vision_source_commit)
        print(json.dumps({name:summary[name] for name in ("success","stage","case_count","profile_count","unity_validated","gpu_chain_validated")}))
        return
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
