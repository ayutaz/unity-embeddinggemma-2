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
FEED_FORWARD_PARTS = ("pre_layer_norm", "ffw_layer_1", "act_fn", "ffw_layer_2", "post_layer_norm")
FEED_FORWARD_PATHS = tuple("feed_forward1." + name for name in FEED_FORWARD_PARTS)


def linear_fp64(module, hidden):
    """Isolated CPU oracle: exact FP32 inputs/weights, FP64 accumulation, FP32 output."""
    weight = module.linear.weight
    if (hidden.dtype != torch.float32 or weight.dtype != torch.float32
            or hidden.device.type != "cpu" or weight.device.type != "cpu"
            or module.linear.bias is not None or hidden.ndim < 2
            or hidden.shape[-1] != weight.shape[1]
            or not torch.isfinite(hidden).all() or not torch.isfinite(weight).all()):
        raise ValueError("projection oracle requires finite unbiased CPU FP32 context")
    with torch.no_grad():
        working = hidden.to(torch.float64)
        if module.use_clipped_linears:
            working = working.clamp(module.input_min, module.input_max)
        result = torch.nn.functional.linear(working, weight.to(torch.float64)).to(torch.float32)
        if module.use_clipped_linears:
            result = result.clamp(module.output_min, module.output_max)
    if not torch.isfinite(result).all():
        raise ValueError("projection oracle produced nonfinite values")
    return result


def _fixed_metrics(actual, expected):
    if actual.shape != expected.shape or not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise ValueError("projection comparison shape or finite-value mismatch")
    error = (actual.double() - expected.double()).abs()
    allowed = 1e-5 + 1e-4 * expected.double().abs()
    mismatches = int((error > allowed).sum())
    return dict(elementwise_passed=mismatches == 0, mismatches=mismatches,
                maximum_absolute_error=float(error.max()), maximum_tolerance_ratio=float((error / allowed).max()))


def projection_oracle(layer, parts):
    """Compare independent exact-input projections; never replace frozen FP32 outputs."""
    ff = layer.feed_forward1
    targets = {"linear1": parts["feed_forward1.ffw_layer_1"],
               "linear2": parts["feed_forward1.ffw_layer_2"],
               "linear2_post_norm": parts["feed_forward1.post_layer_norm"]}
    with torch.no_grad():
        first = parts["feed_forward1.pre_layer_norm"]
        activation = parts["feed_forward1.act_fn"]
        baseline1, baseline2 = ff.ffw_layer_1(first), ff.ffw_layer_2(activation)
        baseline_post = ff.post_layer_norm(baseline2.clamp(-ff.gradient_clipping, ff.gradient_clipping))
        if any(not torch.equal(value, targets[name]) for name, value in
               (("linear1", baseline1), ("linear2", baseline2), ("linear2_post_norm", baseline_post))):
            raise ValueError("pinned FP32 projection baseline mismatch")
        outputs = dict(linear1=linear_fp64(ff.ffw_layer_1, first), linear2=linear_fp64(ff.ffw_layer_2, activation))
        outputs["linear2_post_norm"] = ff.post_layer_norm(outputs["linear2"].clamp(-ff.gradient_clipping, ff.gradient_clipping))
    metrics = {name: _fixed_metrics(value, targets[name]) for name, value in outputs.items()}
    return dict(scope="exact_native_inputs_fp64_linears_only_then_fp32_post_norm",
                fp32_baselines_exact_match=True, metrics=metrics,
                fp64_projection_gate_passed=all(value["elementwise_passed"] for value in metrics.values())), outputs


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


def _capture_named(layer, hidden, blocked_mask, positions, pinned_final, names):
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
        for name in names:
            handles.append(layer.get_submodule(name).register_forward_hook(observe(name)))
        with torch.no_grad():
            actual = layer(hidden.clone(), attention_mask=blocked_mask, position_embeddings=positions)
    finally:
        for handle in handles:
            handle.remove()
    if not torch.equal(actual, pinned_final):
        raise ValueError("native replay does not match pinned layer final")
    if set(values) != set(names) or any(not torch.isfinite(value).all() for value in values.values()):
        raise ValueError("native replay parts are incomplete or nonfinite")
    return {name: values[name] for name in names}


def capture_parts(layer, hidden, blocked_mask, positions, pinned_final):
    return _capture_named(layer, hidden, blocked_mask, positions, pinned_final, PARTS)


def capture_feed_forward_parts(layer, hidden, blocked_mask, positions, pinned_final):
    values = _capture_named(layer, hidden, blocked_mask, positions, pinned_final, FEED_FORWARD_PATHS)
    return {name: values["feed_forward1." + name] for name in FEED_FORWARD_PARTS}


def _identity(path):
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"bytes": path.stat().st_size, "sha256": digest}


def prepare(snapshot, context, output, *, reference_source, diagnostic_source, detail_feed_forward=False, with_projection_oracle=False):
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
    detail_feed_forward = detail_feed_forward or with_projection_oracle
    parts = _capture_named(layer, hidden, blocked, positions, pinned_final,
                           (*PARTS, *FEED_FORWARD_PATHS) if detail_feed_forward else PARTS)
    oracle_report, oracle_values = projection_oracle(layer, parts) if with_projection_oracle else (None, {})
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
                  detail_feed_forward=detail_feed_forward,
                  full_encoder_validated=False, unity_validated=False)
    if oracle_report is not None:
        oracle_report["files"] = {}
        for name, tensor in oracle_values.items():
            path = output / f"candidate-{name}-fp32.bin"
            tensor.numpy().astype("<f4").tofile(path)
            oracle_report["files"][name] = dict(file=path.name, shape=list(tensor.shape), **_identity(path))
        report.update(fp64_projection_oracle=oracle_report, fp64_projection_gate_passed=oracle_report["fp64_projection_gate_passed"])
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
    parser.add_argument("--detail-feed-forward", action="store_true")
    parser.add_argument("--projection-oracle", action="store_true")
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
                              reference_source=args.reference_source, diagnostic_source=args.diagnostic_source,
                              detail_feed_forward=args.detail_feed_forward, with_projection_oracle=args.projection_oracle))
        report["success"] = report.get("fp64_projection_gate_passed", True)
    except Exception as error:
        report.update(error_type=type(error).__name__, error=str(error))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
