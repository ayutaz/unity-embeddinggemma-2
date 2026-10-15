"""Replay one official audio layer against immutable native stage input/output."""

import json
import argparse
import hashlib
import math
import re
import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from safetensors import safe_open
from transformers import Gemma4AudioConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4AudioLayer
from transformers.models.gemma4.modeling_gemma4 import (
    Gemma4AudioModel, Gemma4AudioRelPositionalEncoding, sliding_window_mask_function,
    Gemma4AudioSubSampleConvProjection,
)
from transformers.masking_utils import create_bidirectional_mask
from huggingface_hub import snapshot_download

PARTS = ("feed_forward1", "norm_pre_attn", "self_attn", "norm_post_attn", "lconv1d", "feed_forward2", "norm_out")
FEED_FORWARD_PARTS = ("pre_layer_norm", "ffw_layer_1", "act_fn", "ffw_layer_2", "post_layer_norm")
FEED_FORWARD_PATHS = tuple("feed_forward1." + name for name in FEED_FORWARD_PARTS)
ATTENTION_PROJECTION_PARTS = ("q_proj", "k_proj", "v_proj", "relative_k_proj", "post")
ATTENTION_PROJECTION_PATHS = tuple("self_attn." + name for name in ATTENTION_PROJECTION_PARTS)
CONVOLUTION_PARTS = ("pre_layer_norm", "linear_start", "depthwise_conv1d", "conv_norm", "act_fn", "linear_end")
SUBSAMPLE_PARTS = ("layer0.conv", "layer0.norm", "layer0.act", "layer1.conv",
                   "layer1.norm", "layer1.act", "input_proj_linear")


class CompensatedClippedLinear(torch.nn.Module):
    """Exportable FP32 candidate, borrowing weights and preserving official clips."""

    def __init__(self, source, *, chunk_size=64, include_bias=False):
        super().__init__()
        if type(chunk_size) is not int or chunk_size <= 0:
            raise ValueError("chunk size must be a positive integer")
        if (source.linear.weight.dtype != torch.float32 or (source.linear.bias is not None and not include_bias)
                or not torch.isfinite(source.linear.weight).all()):
            raise ValueError("candidate requires finite unbiased FP32 weights")
        if source.linear.bias is not None and (source.linear.bias.dtype != torch.float32
                or not torch.isfinite(source.linear.bias).all()):
            raise ValueError("candidate requires finite FP32 bias")
        self.linear = source.linear
        self.output_bias = source.linear.bias if include_bias else None
        self.chunk_size = chunk_size
        self.use_clipped_linears = source.use_clipped_linears
        if self.use_clipped_linears:
            for name in ("input_min", "input_max", "output_min", "output_max"):
                self.register_buffer(name, getattr(source, name).detach().clone())

    def forward(self, hidden):
        if not torch.compiler.is_compiling() and not torch.jit.is_tracing():
            if (hidden.dtype != torch.float32 or hidden.device != self.linear.weight.device
                    or hidden.ndim < 2 or hidden.shape[-1] != self.linear.in_features
                    or not torch.isfinite(hidden).all()):
                raise ValueError("candidate requires finite matching FP32 input")
        working = hidden.clamp(self.input_min, self.input_max) if self.use_clipped_linears else hidden
        total = hidden.new_zeros((*hidden.shape[:-1], self.linear.out_features))
        correction = torch.zeros_like(total)
        for start in range(0, self.linear.in_features, self.chunk_size):
            part = torch.nn.functional.linear(working[..., start:start + self.chunk_size],
                                              self.linear.weight[:, start:start + self.chunk_size])
            updated = total + part
            correction = correction + torch.where(total.abs() >= part.abs(),
                (total - updated) + part, (part - updated) + total)
            total = updated
        output = total + correction
        if self.output_bias is not None:
            output = output + self.output_bias
        return output.clamp(self.output_min, self.output_max) if self.use_clipped_linears else output


def linear_fp32_compensated(module, hidden, *, chunk_size=64):
    """FP32 block products with Neumaier compensation; preserves official clipping."""
    if type(chunk_size) is not int or chunk_size <= 0:
        raise ValueError("chunk size must be a positive integer")
    weight = module.linear.weight
    if (hidden.dtype != torch.float32 or weight.dtype != torch.float32
            or hidden.device != weight.device or module.linear.bias is not None
            or hidden.ndim < 2 or hidden.shape[-1] != weight.shape[1]
            or not torch.isfinite(hidden).all() or not torch.isfinite(weight).all()):
        raise ValueError("compensated projection requires finite unbiased FP32 context")
    with torch.no_grad():
        working = hidden.clamp(module.input_min, module.input_max) if module.use_clipped_linears else hidden
        total = torch.zeros((*hidden.shape[:-1], weight.shape[0]), dtype=torch.float32, device=hidden.device)
        correction = torch.zeros_like(total)
        for start in range(0, weight.shape[1], chunk_size):
            part = torch.nn.functional.linear(working[..., start:start + chunk_size], weight[:, start:start + chunk_size])
            updated = total + part
            correction = correction + torch.where(total.abs() >= part.abs(),
                (total - updated) + part, (part - updated) + total)
            total = updated
        result = total + correction
        if module.use_clipped_linears:
            result = result.clamp(module.output_min, module.output_max)
    if not torch.isfinite(result).all():
        raise ValueError("compensated projection produced nonfinite values")
    return result


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


def _projection_context(layer, parts):
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
    return ff, first, activation, targets


def compensated_projection_comparison(layer, parts, *, chunk_size=64):
    ff, first, activation, targets = _projection_context(layer, parts)
    with torch.no_grad():
        outputs = dict(linear1=linear_fp32_compensated(ff.ffw_layer_1, first, chunk_size=chunk_size),
                       linear2=linear_fp32_compensated(ff.ffw_layer_2, activation, chunk_size=chunk_size))
        outputs["linear2_post_norm"] = ff.post_layer_norm(outputs["linear2"].clamp(-ff.gradient_clipping, ff.gradient_clipping))
    metrics = {name: _fixed_metrics(value, targets[name]) for name, value in outputs.items()}
    return dict(scope="exact_native_inputs_fp32_chunked_compensated_linears_then_fp32_post_norm",
                fp32_baselines_exact_match=True, chunk_size=chunk_size, metrics=metrics,
                fp32_compensated_gate_passed=all(value["elementwise_passed"] for value in metrics.values())), outputs


def projection_oracle(layer, parts):
    """Compare independent exact-input projections; never replace frozen FP32 outputs."""
    ff, first, activation, targets = _projection_context(layer, parts)
    with torch.no_grad():
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


def _capture_named(layer, hidden, blocked_mask, positions, pinned_final, names, *, capture_inputs=False):
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
            if capture_inputs:values[name+'.input']=args[0].detach().clone()
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
    expected_names=set(names)|({name+'.input' for name in names} if capture_inputs else set())
    if set(values) != expected_names or any(not torch.isfinite(value).all() for value in values.values()):
        raise ValueError("native replay parts are incomplete or nonfinite")
    return {name: values[name] for name in (*names, *(name+'.input' for name in names))} if capture_inputs else {name: values[name] for name in names}


def capture_parts(layer, hidden, blocked_mask, positions, pinned_final):
    return _capture_named(layer, hidden, blocked_mask, positions, pinned_final, PARTS)


def capture_feed_forward_parts(layer, hidden, blocked_mask, positions, pinned_final):
    values = _capture_named(layer, hidden, blocked_mask, positions, pinned_final, FEED_FORWARD_PATHS)
    return {name: values["feed_forward1." + name] for name in FEED_FORWARD_PARTS}


def capture_subsample_parts(module, features, frame_mask, pinned_final):
    if (features.dtype != torch.float32 or pinned_final.dtype != torch.float32
            or features.ndim != 3 or pinned_final.ndim != 3
            or frame_mask.dtype != torch.bool or frame_mask.shape != features.shape[:2]
            or not torch.isfinite(features).all() or not torch.isfinite(pinned_final).all()):
        raise ValueError("subsample replay requires finite FP32 input and matching boolean mask")
    values, handles = {}, []
    def observe(name):
        def callback(_module, _args, output):
            values[name] = output.detach().clone()
        return callback
    try:
        for name in SUBSAMPLE_PARTS:
            handles.append(module.get_submodule(name).register_forward_hook(observe(name)))
        with torch.no_grad():
            actual, _ = module(features.clone(), frame_mask.clone())
    finally:
        for handle in handles:
            handle.remove()
    if not torch.equal(actual, pinned_final):
        error=ValueError("native replay does not match pinned subsample final")
        try:error.context_diagnostics=diagnose_subsample_context(module,features,frame_mask,pinned_final)
        except Exception as diagnostic_error:error.context_diagnostics=dict(scope='subsample_cpu_context_diagnosis_only',error=str(diagnostic_error))
        raise error
    if set(values) != set(SUBSAMPLE_PARTS) or any(not torch.isfinite(value).all() for value in values.values()):
        raise ValueError("subsample replay parts are incomplete or nonfinite")
    return {name: values[name] for name in SUBSAMPLE_PARTS}


def diagnose_subsample_context(module,features,frame_mask,expected):
    """Failure-only finite grid; never change the pinned output or accept a probe."""
    from .conv_reference_context import _runtime
    from .reference_context_failure import feature_error_metrics
    initial_threads=torch.get_num_threads();initial_mkldnn=torch.backends.mkldnn.enabled
    rows=[];runtime=_runtime()
    try:
        for weight_layout in ('contiguous','channels_last'):
            candidate=copy.deepcopy(module)
            candidate.to(memory_format=torch.contiguous_format if weight_layout=='contiguous' else torch.channels_last)
            for feature_layout in ('contiguous','time_contiguous'):
                values=features.detach().clone().contiguous()
                if feature_layout=='time_contiguous':values=values.transpose(1,2).contiguous().transpose(1,2)
                for enabled in (True,False):
                    for requested in (1,2,4):
                        row=dict(weight_layout=weight_layout,feature_layout=feature_layout,
                                 feature_strides=list(values.stride()),mkldnn_enabled=enabled,
                                 requested_threads=requested,executed=False)
                        rows.append(row)
                        try:
                            torch.backends.mkldnn.enabled=enabled;torch.set_num_threads(requested)
                            row['actual_threads']=torch.get_num_threads()
                            with torch.no_grad():actual,_=candidate(values.clone(),frame_mask.clone())
                            row.update(feature_error_metrics(actual,expected))
                            if actual.shape==expected.shape:
                                row['bit_differences']=int(torch.count_nonzero(actual.contiguous().view(torch.int32)!=expected.contiguous().view(torch.int32)))
                            row['executed']=True
                        except Exception as error:row['error']=f'{type(error).__name__}: {error}'
    finally:
        torch.set_num_threads(initial_threads);torch.backends.mkldnn.enabled=initial_mkldnn
    return dict(scope='subsample_cpu_context_diagnosis_only',runtime=runtime,conditions=rows,
                full_encoder_validated=False,unity_validated=False,end_to_end_validated=False)


def _identity(path):
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"bytes": path.stat().st_size, "sha256": digest}


def capture_attention_projections(layer, hidden, blocked_mask, positions, pinned_final):
    values = _capture_named(layer, hidden, blocked_mask, positions, pinned_final, ATTENTION_PROJECTION_PATHS)
    return {name: values["self_attn." + name] for name in ATTENTION_PROJECTION_PARTS}


def capture_convolution_parts(layer, hidden, blocked_mask, positions, pinned_final):
    """Observe unchanged native convolution modules and their borrowed inputs."""
    names=tuple('lconv1d.'+name for name in CONVOLUTION_PARTS)
    captured=_capture_named(layer,hidden,blocked_mask,positions,pinned_final,names,capture_inputs=True)
    values={name+'.'+kind:captured['lconv1d.'+name+('.input' if kind=='input' else '')]
            for name in CONVOLUTION_PARTS for kind in ('input','output')}
    values['depthwise_conv1d.weight']=layer.lconv1d.depthwise_conv1d.weight.detach().clone()
    values['glu.input']=values['linear_start.output'].clone()
    values['glu.output']=values['depthwise_conv1d.input'].transpose(1,2).contiguous().clone()
    if len(values)!=15 or any(v.dtype!=torch.float32 or not torch.isfinite(v).all() for v in values.values()):
        raise ValueError('Pinned convolution capture is incomplete or not finite FP32')
    return values


def capture_attention_operations(layer, hidden, blocked_mask, positions, pinned_final):
    """Observe native attention dispatch without replacing its operators."""
    from torch.utils._python_dispatch import TorchDispatchMode
    class Capture(TorchDispatchMode):
        def __init__(self):
            super().__init__();self.active=False;self.products=0;self.values={}
        def __torch_dispatch__(self,func,types,args=(),kwargs=None):
            result=func(*args,**(kwargs or {}))
            if self.active:
                if func==torch.ops.aten.bmm.default:
                    if self.products>=3:raise ValueError('Pinned attention matrix structure changed')
                    name=('matrix_ac','matrix_bd','weighted_values')[self.products];self.products+=1
                    self.values[name+'.left']=args[0].detach().clone();self.values[name+'.right']=args[1].detach().clone()
                    self.values[name+'.output']=result.detach().clone()
                elif func in (torch.ops.aten.tanh.default,torch.ops.aten._softmax.default):
                    name='tanh' if func==torch.ops.aten.tanh.default else 'softmax'
                    if name+'.input' in self.values:raise ValueError('Pinned attention activation structure changed')
                    self.values[name+'.input']=args[0].detach().clone();self.values[name+'.output']=result.detach().clone()
            return result
    mode=Capture()
    def begin(*args):mode.active=True
    def end(*args):mode.active=False
    handles=[layer.self_attn.register_forward_pre_hook(begin),layer.self_attn.register_forward_hook(end)]
    try:
        with mode:_capture_named(layer,hidden,blocked_mask,positions,pinned_final,('self_attn',))
    finally:
        for handle in handles:handle.remove()
    if mode.products!=3 or len(mode.values)!=13 or any(v.dtype!=torch.float32 or not torch.isfinite(v).all() for v in mode.values.values()):
        raise ValueError('Pinned attention operation capture is incomplete')
    return mode.values


def prepare(snapshot, context, output, *, reference_source, diagnostic_source, detail_feed_forward=False, detail_attention_projections=False, detail_attention_ops=False, detail_convolution=False,
            with_projection_oracle=False, with_compensated_projection=False, chunk_size=64,
            subsample_input=None, subsample_input_sha256=None, layer_index=0, case_id='silence'):
    """Produce internal references against one original frozen signal/layer pair."""
    snapshot, context, output = map(Path, (snapshot, context, output))
    if type(layer_index) is not int or layer_index<0:raise ValueError('Select a nonnegative layer index')
    if layer_index!=0 and (subsample_input is not None or subsample_input_sha256 is not None):
        raise ValueError('subsample replay requires layer 0')
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
    cases = [case for case in reference["cases"] if case["id"] == case_id
             and case["profile"].split("-")[1] == case["profile"].split("-")[3]]
    if len(cases) != 1:
        raise ValueError("select one fully valid original signal context")
    case = cases[0]
    stages = {value["name"]: value for value in case["stages"]}
    input_stage='subsample' if layer_index==0 else f'layer_{layer_index-1}'
    output_stage=f'layer_{layer_index}'
    if any(name not in stages for name in (input_stage,output_stage)):
        raise ValueError('Selected layer requires original input and final context stages')
    tensors = []
    input_files = {}
    for name in (input_stage, output_stage):
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
    features = None
    if subsample_input is not None:
        shape = reference["profiles"][case["profile"]]["feature_shape"]
        if (re.fullmatch(r"[0-9a-f]{64}", subsample_input_sha256 or "") is None
                or _identity(Path(subsample_input)) != dict(bytes=math.prod(shape) * 4, sha256=subsample_input_sha256)):
            raise ValueError("pinned subsample input hash or size mismatch")
        features = torch.from_numpy(np.fromfile(subsample_input, dtype="<f4").reshape(shape))
    elif subsample_input_sha256 is not None:
        raise ValueError("subsample input hash requires an explicit input path")
    layer, config = load_layer(snapshot,index=layer_index)
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
    detail_feed_forward = detail_feed_forward or with_projection_oracle or with_compensated_projection
    names = (*PARTS, *(FEED_FORWARD_PATHS if detail_feed_forward else ()),
             *(ATTENTION_PROJECTION_PATHS if detail_attention_projections else ()))
    parts = _capture_named(layer, hidden, blocked, positions, pinned_final, names)
    if detail_attention_ops:
        parts.update({'self_attn.'+name:value for name,value in capture_attention_operations(layer,hidden,blocked,positions,pinned_final).items()})
    if detail_convolution:
        parts.update({'lconv1d.'+name:value for name,value in capture_convolution_parts(layer,hidden,blocked,positions,pinned_final).items()})
    if features is not None:
        module = Gemma4AudioSubSampleConvProjection(config).float().eval()
        prefix = "audio_tower.subsample_conv_projection."
        with safe_open(snapshot / "model.safetensors", framework="pt", device="cpu") as weights:
            state = {key.removeprefix(prefix): weights.get_tensor(key).to(torch.float32)
                     for key in weights.keys() if key.startswith(prefix)}
        module.load_state_dict(state, strict=True)
        captured = capture_subsample_parts(module, features, torch.ones(features.shape[:2], dtype=torch.bool), hidden)
        parts.update({"subsample." + name: tensor for name, tensor in captured.items()})
    oracle_report, oracle_values = projection_oracle(layer, parts) if with_projection_oracle else (None, {})
    compensated_report, compensated_values = (compensated_projection_comparison(layer, parts, chunk_size=chunk_size)
                                               if with_compensated_projection else (None, {}))
    output.mkdir(parents=True, exist_ok=False)
    files = {}
    for name, tensor in parts.items():
        path = output / f"{name}-fp32.bin"
        tensor.numpy().astype("<f4").tofile(path)
        files[name] = dict(file=path.name, shape=list(tensor.shape), **_identity(path))
    report = dict(context_validated=True, native_final_exact_match=True,
                  layer_index=layer_index,input_stage=input_stage,output_stage=output_stage,
                  reference_source=reference_source, diagnostic_source=diagnostic_source,
                  model_revision=metadata["model_revision"], profile=case["profile"],
                  case=case["id"], input_files=input_files, parts=files,
                  detail_feed_forward=detail_feed_forward,
                  detail_attention_projections=detail_attention_projections,
                  detail_attention_ops=detail_attention_ops,
                  detail_convolution=detail_convolution,
                  full_encoder_validated=False, unity_validated=False)
    if features is not None:
        report.update(subsample_native_final_exact_match=True, subsample_input=_identity(Path(subsample_input)))
    if oracle_report is not None:
        oracle_report["files"] = {}
        for name, tensor in oracle_values.items():
            path = output / f"candidate-{name}-fp32.bin"
            tensor.numpy().astype("<f4").tofile(path)
            oracle_report["files"][name] = dict(file=path.name, shape=list(tensor.shape), **_identity(path))
        report.update(fp64_projection_oracle=oracle_report, fp64_projection_gate_passed=oracle_report["fp64_projection_gate_passed"])
    if compensated_report is not None:
        compensated_report["files"] = {}
        for name, tensor in compensated_values.items():
            path = output / f"candidate-compensated-{name}-fp32.bin"
            tensor.numpy().astype("<f4").tofile(path)
            compensated_report["files"][name] = dict(file=path.name, shape=list(tensor.shape), **_identity(path))
        report.update(fp32_compensated_projection=compensated_report,
                      fp32_compensated_gate_passed=compensated_report["fp32_compensated_gate_passed"])
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
    parser.add_argument("--detail-attention-projections", action="store_true")
    parser.add_argument("--detail-attention-ops", action="store_true")
    parser.add_argument("--detail-convolution", action="store_true")
    parser.add_argument("--layer-index",type=int,default=0)
    parser.add_argument("--case-id",choices=('silence','impulse','sine','nyquist','noise'),default='silence')
    parser.add_argument("--projection-oracle", action="store_true")
    parser.add_argument("--compensated-projection", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--subsample-input", type=Path)
    parser.add_argument("--subsample-input-sha256")
    args = parser.parse_args(argv)
    if args.report.exists():
        raise ValueError("refusing to overwrite replay report")
    report = dict(success=False, context_validated=False, native_final_exact_match=False,
                  layer_index=args.layer_index,case=args.case_id,
                  full_encoder_validated=False, unity_validated=False,
                  reference_source=args.reference_source, diagnostic_source=args.diagnostic_source,
                  torch_version=torch.__version__, torch_threads=torch.get_num_threads())
    try:
        if args.layer_index<0:raise ValueError('Select a nonnegative layer index')
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
                              detail_feed_forward=args.detail_feed_forward, with_projection_oracle=args.projection_oracle,
                              detail_attention_projections=args.detail_attention_projections,
                              detail_attention_ops=args.detail_attention_ops,
                              detail_convolution=args.detail_convolution,
                              layer_index=args.layer_index,
                              case_id=args.case_id,
                              with_compensated_projection=args.compensated_projection, chunk_size=args.chunk_size,
                              subsample_input=args.subsample_input, subsample_input_sha256=args.subsample_input_sha256))
        report["success"] = (report.get("fp64_projection_gate_passed", True)
                             and report.get("fp32_compensated_gate_passed", True))
    except Exception as error:
        report.update(error_type=type(error).__name__, error=str(error))
        if hasattr(error,'context_diagnostics'):report['subsample_context_diagnostics']=error.context_diagnostics
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
