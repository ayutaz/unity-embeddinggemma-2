"""Replay one official audio layer against immutable native stage input/output."""

import json
import argparse
import hashlib
import math
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from safetensors import safe_open
from transformers import Gemma4AudioConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4AudioLayer
from transformers.models.gemma4.modeling_gemma4 import (
    Gemma4AudioModel, Gemma4AudioRelPositionalEncoding, sliding_window_mask_function,
)
from transformers.masking_utils import create_bidirectional_mask
from huggingface_hub import snapshot_download

PARTS = ("feed_forward1", "norm_pre_attn", "self_attn", "norm_post_attn", "lconv1d", "feed_forward2", "norm_out")


def load_layer(snapshot, *, index=0):
    snapshot = Path(snapshot)
    configuration = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    config = Gemma4AudioConfig.from_dict(configuration["audio_config"])
    if type(index) is not int or not 0 <= index < config.num_hidden_layers:
        raise ValueError("select an existing audio layer index")
    config._attn_implementation = "sdpa"
    layer = Gemma4AudioLayer(config, index).float()
    prefix = f"audio_tower.layers.{index}."
    with safe_open(snapshot / "model.safetensors", framework="pt", device="cpu") as weights:
        state = {key.removeprefix(prefix): weights.get_tensor(key).to(torch.float32)
                 for key in weights.keys() if key.startswith(prefix)}
    layer.load_state_dict(state, strict=True)
    return layer.eval(), config


def capture_parts(layer, hidden, blocked_mask, positions, pinned_final):
    for value in (hidden, pinned_final, positions):
        if value.dtype != torch.float32 or not torch.isfinite(value).all():
            raise ValueError("native replay requires finite FP32 context")
    if hidden.ndim != 3 or hidden.shape != pinned_final.shape or blocked_mask.dtype != torch.bool:
        raise ValueError("native replay shape or boolean mask mismatch")
    values, handles = {}, []
    def observe(name):
        def callback(module, args, output):
            tensor = output[0] if name == "self_attn" else output
            values[name] = tensor.detach().clone()
        return callback
    try:
        for name in PARTS:
            handles.append(getattr(layer, name).register_forward_hook(observe(name)))
        with torch.no_grad():
            actual = layer(hidden.clone(), attention_mask=blocked_mask, position_embeddings=positions)
    finally:
        for handle in handles:
            handle.remove()
    if not torch.equal(actual, pinned_final):
        raise ValueError("native replay does not match pinned layer final")
    if tuple(values) != PARTS or any(not torch.isfinite(value).all() for value in values.values()):
        raise ValueError("native replay parts are incomplete or nonfinite")
    return values


def _identity(path):
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"bytes": path.stat().st_size, "sha256": digest}


def prepare(snapshot, context, output, *, reference_source, diagnostic_source):
    """Produce internal references for the frozen, fully valid silence layer 0."""
    snapshot, context, output = map(Path, (snapshot, context, output))
    if any(re.fullmatch(r"[0-9a-f]{40}", value) is None
           for value in (reference_source, diagnostic_source)):
        raise ValueError("source must be an immutable SHA")
    reference = json.loads((context / "audio-stage-reference.json").read_text(encoding="utf-8"))
    audit = json.loads((context / "audio-stage-validation.json").read_text(encoding="utf-8"))
    metadata = reference["metadata"]
    if (audit.get("success") is not True or audit.get("context_validated") is not True
            or audit.get("stage") != "audio_runtime_stage_context_only"
            or audit.get("metadata") != metadata
            or metadata.get("source_commit") != reference_source
            or metadata.get("feature_rtol") != 1e-4 or metadata.get("feature_atol") != 1e-5
            or re.fullmatch(r"[0-9a-f]{40}", metadata.get("model_revision", "")) is None):
        raise ValueError("pinned context metadata mismatch")
    cases = [case for case in reference["cases"] if case["id"] == "silence"
             and case["profile"].split("-")[1] == case["profile"].split("-")[3]]
    if len(cases) != 1:
        raise ValueError("select one fully valid silence context")
    case = cases[0]
    stages = {value["name"]: value for value in case["stages"]}
    tensors = []
    input_files = {}
    for name in ("subsample", "layer_0"):
        stage = stages[name]
        path = (context / stage["expected"]).resolve()
        if not path.is_relative_to(context.resolve()):
            raise ValueError("context file path escapes root")
        identity = _identity(path)
        if identity != audit["files"][stage["expected"]]:
            raise ValueError("context binary hash or size mismatch")
        shape = stage["shape"]
        if (len(shape) != 3 or any(type(size) is not int or size <= 0 for size in shape)
                or identity["bytes"] != math.prod(shape) * 4):
            raise ValueError("context tensor size mismatch")
        tensors.append(torch.from_numpy(np.fromfile(path, dtype="<f4").reshape(shape)))
        input_files[stage["expected"]] = identity
    for name in ("config.json", "model.safetensors"):
        if _identity(snapshot / name)["sha256"] != metadata["input_sha256"][name]:
            raise ValueError("snapshot hash mismatch")
    layer, config = load_layer(snapshot)
    hidden, pinned_final = tensors
    frames = reference["profiles"][case["profile"]]["feature_shape"][1]
    if tuple(hidden.shape) != (1, (frames + 3) // 4, config.hidden_size):
        raise ValueError("context hidden shape mismatch")
    output_mask = torch.ones(hidden.shape[:2], dtype=torch.bool)
    attention_mask = create_bidirectional_mask(
        config=config, inputs_embeds=hidden, attention_mask=output_mask,
        and_mask_function=sliding_window_mask_function(
            (config.attention_context_left - 1, config.attention_context_right)))
    blocked = Gemma4AudioModel._convert_4d_mask_to_blocked_5d(
        SimpleNamespace(config=config), attention_mask)
    positions = Gemma4AudioRelPositionalEncoding(config)(hidden)
    parts = capture_parts(layer, hidden, blocked, positions, pinned_final)
    output.mkdir(parents=True, exist_ok=False)
    files = {}
    for name, tensor in parts.items():
        path = output / f"{name}-fp32.bin"
        tensor.numpy().astype("<f4").tofile(path)
        files[name] = dict(file=path.name, shape=list(tensor.shape), **_identity(path))
    report = dict(context_validated=True, native_final_exact_match=True,
                  reference_source=reference_source, diagnostic_source=diagnostic_source,
                  model_revision=metadata["model_revision"], profile=case["profile"],
                  case=case["id"], input_files=input_files, parts=files,
                  full_encoder_validated=False, unity_validated=False)
    (output / "layer-replay.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--context", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--reference-source", required=True)
    parser.add_argument("--diagnostic-source", required=True)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.report.exists():
        raise ValueError("refusing to overwrite replay report")
    report = dict(success=False, context_validated=False, native_final_exact_match=False,
                  full_encoder_validated=False, unity_validated=False,
                  reference_source=args.reference_source, diagnostic_source=args.diagnostic_source,
                  torch_version=torch.__version__, torch_threads=torch.get_num_threads())
    try:
        snapshot = args.snapshot
        if snapshot is None:
            from .prepare import MODEL_ID
            metadata = json.loads((args.context / "audio-stage-reference.json").read_text(encoding="utf-8"))["metadata"]
            revision = metadata["model_revision"]
            if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
                raise ValueError("model revision must be immutable")
            snapshot = Path(snapshot_download(MODEL_ID, revision=revision,
                                             allow_patterns=["config.json", "model.safetensors"]))
        report.update(prepare(snapshot, args.context, args.output,
                              reference_source=args.reference_source, diagnostic_source=args.diagnostic_source))
        report["success"] = True
    except Exception as error:
        report.update(error_type=type(error).__name__, error=str(error))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
