"""Original-context-qualified first audio projection ONNX diagnosis only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
from dataclasses import dataclass

import numpy as np
import torch
from onnx.reference import ReferenceEvaluator

from .audio_native_norm_replay import PRODUCER, REVISION, REFERENCE_SHA, candidate_output, checked_normalizations, load_frozen_input
from .export import export_audio_native_onnx
from .export import fp32_cascade_mean1024, fp32_cascade_mean1536, fp32_outer_cascade_mean1024, fp32_refined_rstd
from .reference_context_failure import feature_error_metrics
from .static_block192_onnx import audit_static_linear_nodes, static_block192_export, static_block192_addmm_export
from .streaming_reference import run_flat_reference
from .static_stem_conv_onnx import static_stem_conv_export

VARIANTS = ("native", "refined_moment", "refined_mean_moment")
EXPERIMENTAL_VARIANTS = (*VARIANTS, "refined_mean_moment_rms", "ordered_static_linear_refined", "ordered_original_stem")
STAGE_REFERENCE_SHA = "6620f335563a91ccc484df7d9b89d4d889e58dab4ec3d7d0389b36abe36b454a"
STAGE_AUDIT_SHA = "3edee8acdd9ebebb819ae488b753a3a94949b953aa0906995e516764932b0f9b"


@dataclass(frozen=True)
class FrozenStageReference:
    expected: torch.Tensor
    provenance: dict


def load_frozen_stage_reference(directory, stage, *, case_id='silence'):
    if stage not in STAGES:
        raise ValueError("Unknown frozen reference stage")
    root = Path(directory).resolve()
    for name, digest in (("audio-stage-reference.json", STAGE_REFERENCE_SHA), ("audio-stage-validation.json", STAGE_AUDIT_SHA)):
        path = root / name
        if not path.is_file() or file_digest(path) != digest:
            raise ValueError("Frozen stage reference/audit hash mismatch")
    reference = json.loads((root / "audio-stage-reference.json").read_text(encoding="utf-8"))
    audit = json.loads((root / "audio-stage-validation.json").read_text(encoding="utf-8"))
    metadata = reference["metadata"]
    if (not audit["success"] or not audit["context_validated"] or metadata != audit["metadata"]
            or metadata["source_commit"] != PRODUCER or metadata["model_revision"] != REVISION
            or metadata["encoder_reference_sha256"] != REFERENCE_SHA or metadata["feature_atol"] != 1e-5
            or metadata["feature_rtol"] != 1e-4):
        raise ValueError("Frozen stage provenance, context or tolerance differs")
    rows = [row for row in reference["cases"] if row["id"] == case_id]
    profiles = {'frames-1119-valid-1119': 280, 'frames-1119-valid-839': 210}
    if (len(rows) != 1 or not rows[0]["native_final_matches_original"] or not rows[0]["trace_numpy_matches_original"]
            or rows[0]["profile"] not in profiles
            or rows[0]["input_features"] != f"{rows[0]['profile']}/{case_id}-features-fp32.bin"):
        raise ValueError("Frozen case input/final context differs")
    profile = rows[0]['profile']
    entries = [entry for entry in rows[0]["stages"] if entry["name"] == stage]
    shape = [profiles[profile], 512] if stage == "projected_features" else [1, 280, 1024]
    if len(entries) != 1 or entries[0]["shape"] != shape:
        raise ValueError("Frozen stage output shape differs")
    entry = entries[0]
    path = (root / entry["expected"]).resolve()
    spec = audit["files"].get(entry["expected"], {})
    if (not path.is_relative_to(root) or not path.is_file() or path.stat().st_size != int(np.prod(shape)) * 4
            or path.stat().st_size != spec.get("bytes") or file_digest(path) != spec.get("sha256")):
        raise ValueError("Frozen stage tensor path, size or hash mismatch")
    expected = tensor(torch.from_numpy(np.fromfile(path, dtype="<f4").reshape(shape)))
    return FrozenStageReference(expected, dict(stage=stage, case_id=case_id, profile=profile, reference_sha256=STAGE_REFERENCE_SHA,
        audit_sha256=STAGE_AUDIT_SHA, expected_sha256=spec["sha256"], reference_source=PRODUCER,
        model_revision=REVISION, encoder_reference_sha256=REFERENCE_SHA))
# Names verified against b85a2ad... serialized schema in the pinned graph.
STAGES = {"subsample": ("view_2", torch.ops.aten.view.default),
          **{f"layer_{index}": (f"mul_{27 + 25 * index}", torch.ops.aten.mul.Tensor) for index in range(12)},
          "projected_features": ("select", torch.ops.aten.select.int)}


def boundary_for_stage(program, stage, *, profile='frames-1119-valid-1119'):
    if stage not in STAGES:
        raise ValueError("Unknown fixed audio stage")
    if profile not in ('frames-1119-valid-1119', 'frames-1119-valid-839'):
        raise ValueError('Unknown fixed audio profile boundary')
    name, target = STAGES[stage]
    padded = stage == 'projected_features' and profile == 'frames-1119-valid-839'
    if padded:
        name, target = 'slice_51', torch.ops.aten.slice.Tensor
    matches = [node for node in program.graph.nodes if node.name == name]
    if len(matches) != 1 or matches[0].op != "call_function" or matches[0].target != target:
        raise ValueError("Frozen audio stage boundary differs")
    if padded:
        node = matches[0]
        if (node.kwargs or len(node.args) not in (4, 5) or node.args[1:4] != (0, 0, 210)
                or len(node.args) == 5 and node.args[4] != 1):
            raise ValueError('Pinned padded final crop boundary differs')
    if stage == 'projected_features':
        outputs = [node for node in program.graph.nodes if node.op == 'output']
        final = outputs[0].args[0] if len(outputs) == 1 else None
        if isinstance(final, tuple) and len(final) == 1:
            final = final[0]
        if final is not matches[0]:
            raise ValueError('Pinned final boundary does not produce the original graph output')
    return name


def extract_prefix(program, boundary):
    matches = [node for node in program.graph.nodes if node.name == boundary and node.op == "call_function"]
    if len(matches) != 1:
        raise ValueError("Prefix boundary must be one existing operation")
    selected = matches[0]
    needed = {selected}
    pending = [selected]
    while pending:
        for dependency in pending.pop().all_input_nodes:
            if dependency not in needed:
                needed.add(dependency)
                pending.append(dependency)
    graph = torch.fx.Graph()
    copied = {}
    for node in program.graph.nodes:
        if node in needed:
            copied[node] = graph.node_copy(node, lambda old: copied[old])
    graph.output(copied[selected])
    graph.lint()
    return torch.fx.GraphModule(program, graph)


def expected_rms_count(stage):
    if stage == "subsample":
        return 0
    if stage == "projected_features":
        return 109
    if stage in STAGES and stage.startswith("layer_"):
        return 9 * (int(stage.removeprefix("layer_")) + 1)
    raise ValueError("Unknown fixed RMS stage")


def expected_linear_count(stage):
    if stage == "subsample":
        return 1
    if stage == "projected_features":
        return 111
    if stage in STAGES and stage.startswith("layer_"):
        return 1 + 9 * (int(stage.removeprefix("layer_")) + 1)
    raise ValueError("Unknown fixed linear stage")


def audit_rms_nodes(prefix):
    means, inverses = set(), set()
    for mean in prefix.graph.nodes:
        if mean.op != "call_function" or mean.target != torch.ops.aten.mean.dim:
            continue
        square = mean.args[0]
        if (len(mean.args) != 3 or mean.kwargs or tuple(mean.args[1]) != (-1,) or mean.args[2] is not True
                or not isinstance(square, torch.fx.Node) or square.target != torch.ops.aten.pow.Tensor_Scalar
                or len(square.args) != 2 or square.args[1] != 2 or len(mean.users) != 1):
            raise ValueError("RMS mean must be an unambiguous squared last-axis FP32 reduction")
        addition = next(iter(mean.users))
        if (addition.target != torch.ops.aten.add.Tensor or addition.args != (mean, 1e-6)
                or addition.kwargs or len(addition.users) != 1):
            raise ValueError("RMS epsilon must remain exactly 1e-6")
        inverse = next(iter(addition.users))
        if inverse.target != torch.ops.aten.pow.Tensor_Scalar or inverse.args != (addition, -0.5) or inverse.kwargs:
            raise ValueError("RMS inverse must retain the original negative-half power")
        means.add(mean)
        inverses.add(inverse)
    return means, inverses


def ordered_rms_mean(values):
    width = values.shape[-1]
    if values.dtype != torch.float32 or values.device.type != "cpu":
        raise ValueError("RMS candidate requires original CPU FP32 statistics")
    if values.is_contiguous() and width == 1024:
        return fp32_cascade_mean1024(values)
    if values.is_contiguous() and width == 1536:
        return fp32_cascade_mean1536(values)
    if tuple(values.shape) == (1, 280, 1024) and values.stride() == (286720, 1, 280):
        return fp32_outer_cascade_mean1024(values)
    raise ValueError("RMS candidate has an unqualified width or reduction layout")


def audit_stem_convolutions(prefix):
    selected, shapes = set(), []
    for node in prefix.graph.nodes:
        if node.op != 'call_function' or node.target != torch.ops.aten.convolution.default:
            continue
        if (len(node.args) != 9 or node.kwargs or node.args[2] is not None
                or tuple(node.args[3]) != (2, 2) or tuple(node.args[4]) != (1, 1)
                or tuple(node.args[5]) != (1, 1) or node.args[6] is not False
                or tuple(node.args[7]) != (0, 0) or node.args[8] != 1):
            raise ValueError('Unqualified original stem convolution contract')
        left, right = node.args[:2]
        values = [arg.meta.get('val') for arg in (left, right)]
        if (right.op != 'get_attr' or not all(isinstance(v, torch.Tensor) and v.dtype == torch.float32
                and v.device.type == 'cpu' and v.ndim == 4 for v in values)
                or values[0].shape[1] != values[1].shape[1]):
            raise ValueError('Unqualified original stem convolution tensors')
        selected.add(node); shapes.append(tuple(values[1].shape))
    if sorted(shapes) != sorted([(128, 1, 3, 3), (32, 128, 3, 3)]):
        raise ValueError('Original stem convolution coverage differs')
    return selected


class PrefixExport(torch.nn.Module):
    def __init__(self, prefix, variant):
        super().__init__()
        self.prefix = prefix
        self.variant = variant
        self.selected = checked_normalizations(prefix)
        self.rms_means, self.rms_inverses = audit_rms_nodes(prefix) if variant == "refined_mean_moment_rms" else (set(), set())
        ordered = variant in ("ordered_static_linear_refined", "ordered_original_stem")
        self.linear_nodes = audit_static_linear_nodes(prefix) if ordered else set()
        if ordered and not self.linear_nodes:
            raise ValueError("Ordered prefix requires original 280-row parameter projections")
        self.linear_candidate = static_block192_export
        self.biased_linear_candidate = static_block192_addmm_export
        self.convolution_nodes = audit_stem_convolutions(prefix) if variant == 'ordered_original_stem' else set()
        if variant == 'ordered_original_stem' and (len(self.selected) != 2 or len(self.linear_nodes) != 1):
            raise ValueError('Original stem requires2normalizations and1projection')
        self.convolution_candidate = static_stem_conv_export

    def forward(self, values):
        selected, variant, rms_means, rms_inverses = self.selected, self.variant, self.rms_means, self.rms_inverses
        silu_nodes = getattr(self, "silu_nodes", set())
        silu_candidate = getattr(self, "silu_candidate", None)
        silu_observations = getattr(self, "silu_observations", None)
        linear_nodes = getattr(self, "linear_nodes", set())
        linear_candidate = getattr(self, "linear_candidate", None)
        biased_linear_candidate = getattr(self, "biased_linear_candidate", None)
        bmm_nodes = getattr(self, "bmm_nodes", set())
        bmm_candidate = getattr(self, "bmm_candidate", None)
        softmax_nodes = getattr(self, 'softmax_nodes', set())
        sigmoid_nodes = getattr(self, 'sigmoid_nodes', set())
        convolution_nodes = getattr(self, 'convolution_nodes', set())
        tanh_nodes = getattr(self, 'tanh_nodes', set())
        softmax_candidate = getattr(self, 'softmax_candidate', None)
        sigmoid_candidate = getattr(self, 'sigmoid_candidate', None)
        convolution_candidate = getattr(self, 'convolution_candidate', None)
        tanh_candidate = getattr(self, 'tanh_candidate', None)

        class Replay(torch.fx.Interpreter):
            def run_node(self, node):
                if node in softmax_nodes or node in sigmoid_nodes or node in convolution_nodes or node in tanh_nodes:
                    args, kwargs = self.fetch_args_kwargs_from_env(node)
                    if kwargs:
                        raise ValueError('Only the original audited audio leaf contracts are supported')
                    if node in softmax_nodes:
                        return softmax_candidate(args[0])
                    if node in sigmoid_nodes:
                        return sigmoid_candidate(args[0])
                    if node in tanh_nodes:
                        return tanh_candidate(args[0])
                    return convolution_candidate(args[0], args[1])
                if node in bmm_nodes:
                    args, kwargs = self.fetch_args_kwargs_from_env(node)
                    if len(args) != 2 or kwargs:
                        raise ValueError("Only original audited attention BMM contracts are supported")
                    return bmm_candidate(*args)
                if node in linear_nodes:
                    args, kwargs = self.fetch_args_kwargs_from_env(node)
                    if kwargs:
                        raise ValueError("Only original default MatMul/addmm contracts are audited")
                    return (linear_candidate(*args) if node.target == torch.ops.aten.mm.default
                            else biased_linear_candidate(*args))
                if node in silu_nodes:
                    args, kwargs = self.fetch_args_kwargs_from_env(node)
                    if len(args) != 1 or kwargs:
                        raise ValueError("SiLU candidate requires the audited unary operation")
                    if silu_observations is not None and not torch.jit.is_tracing():
                        value = args[0]
                        silu_observations.append(dict(node=node.name, shape=list(value.shape),
                            minimum=float(value.min()), maximum=float(value.max()),
                            positive_tail_elements=int((value > 80).sum()),
                            negative_tail_elements=int((value < -80).sum())))
                    return silu_candidate(args[0])
                if node in rms_means:
                    args, kwargs = self.fetch_args_kwargs_from_env(node)
                    return ordered_rms_mean(args[0])
                if node in rms_inverses:
                    args, kwargs = self.fetch_args_kwargs_from_env(node)
                    return fp32_refined_rstd(args[0])
                if node.op == "call_function" and node.target == torch.ops.aten._to_copy.default:
                    args, kwargs = self.fetch_args_kwargs_from_env(node)
                    if (len(args) != 1 or kwargs != {"dtype": torch.bool} or args[0].dtype != torch.int64
                            or args[0].device.type != "cpu"):
                        raise ValueError("Only the audited int64 mask to bool copy is supported")
                    # Same copy semantics; Tensor.to traces as supported ONNX Cast.
                    return args[0].to(dtype=torch.bool, copy=True)
                if node.op == "call_function" and node.target == torch.ops.aten._softmax.default:
                    args, kwargs = self.fetch_args_kwargs_from_env(node)
                    if (len(args) != 3 or kwargs or args[1] != -1 or args[2] is not False
                            or args[0].dtype != torch.float32 or args[0].device.type != "cpu"):
                        raise ValueError("Only the audited FP32 last-axis softmax is supported")
                    return torch.softmax(args[0], dim=-1)
                if variant != "native" and node in selected:
                    args, kwargs = self.fetch_args_kwargs_from_env(node)
                    return candidate_output(*args, "refined_mean_moment" if variant in ("refined_mean_moment_rms", "ordered_static_linear_refined", "ordered_original_stem") else variant)
                return super().run_node(node)

        return Replay(self.prefix).run(values)


def tensor(value):
    if isinstance(value, tuple) and len(value) == 1:
        value = value[0]
    if (not isinstance(value, torch.Tensor) or value.dtype != torch.float32 or value.device.type != "cpu"
            or not value.numel() or not torch.isfinite(value).all()):
        raise ValueError("Prefix requires finite nonempty CPU FP32 tensor outputs")
    return value.detach().contiguous()


def exact_difference(actual, expected):
    if actual.shape != expected.shape:
        raise ValueError("Prefix context output shape differs")
    return int((actual.contiguous().view(torch.int32) != expected.contiguous().view(torch.int32)).sum())


def file_digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def weight_digests(prefix):
    return {name: dict(dtype=str(value.dtype), shape=list(value.shape),
                       sha256=hashlib.sha256(value.detach().contiguous().numpy().tobytes()).hexdigest())
            for name, value in prefix.state_dict().items()}


def probe_prefix(program, values, expected, output, *, boundary, provenance=None, frozen_stage=None, variants=VARIANTS):
    if (not isinstance(variants, (tuple, list)) or not variants or any(not isinstance(value, str) or value not in EXPERIMENTAL_VARIANTS for value in variants)
            or len(set(variants)) != len(variants)):
        raise ValueError("Prefix variants must be unique supported choices")
    output = Path(output)
    receipt = output / "frozen-audio-prefix-onnx.json"
    if receipt.exists() or any((output / (variant + ".onnx")).exists() for variant in EXPERIMENTAL_VARIANTS):
        raise ValueError("Refusing to overwrite prefix diagnostic evidence")
    output.mkdir(parents=True, exist_ok=True)
    report = dict(scope="original_context_qualified_audio_prefix_onnx_only", context_qualified=False,
                  reference_qualified=False,
                  complete=False, cases=[], full_encoder_validated=False, gpu_validated=False,
                  end_to_end_validated=False, boundary=boundary, provenance=provenance)

    def save():
        receipt.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    with torch.no_grad():
        actual_whole, expected = tensor(program(values)), tensor(expected)
        whole_metrics = feature_error_metrics(actual_whole, expected)
        exact = exact_difference(actual_whole, expected)
        whole_metrics["exact_difference_count"] = exact
        report["native_whole_encoder_metrics"] = whole_metrics
        save()
        if exact and frozen_stage is None:
            return report
        report["context_qualified"] = exact == 0
        report["reference_qualified"] = True
        prefix = extract_prefix(program, boundary)
        before = weight_digests(prefix)
        native = tensor(prefix(values))
        control = tensor(PrefixExport(prefix, "native")(values))
        report["native_wrapper_control_bits_equal"] = exact_difference(control, native) == 0
        save()
        if not report["native_wrapper_control_bits_equal"]:
            raise ValueError("Native prefix wrapper control differs from original prefix bits")
        reference_prefix = native
        if frozen_stage is not None:
            if not isinstance(frozen_stage, FrozenStageReference):
                raise ValueError("Explicit frozen stage requires an audited stage reference")
            reference_prefix = tensor(frozen_stage.expected)
            if reference_prefix.shape != native.shape:
                raise ValueError("Frozen stage and prefix output shapes differ")
            report.update(scope="audited_original_fixed_stage_onnx_diagnosis_only",
                          frozen_stage_provenance=frozen_stage.provenance)
        reference_prefix.numpy().astype("<f4").tofile(output / "reference-prefix-fp32.bin")
        report["reference_prefix_sha256"] = file_digest(output / "reference-prefix-fp32.bin")
        report["native_prefix_vs_reference_metrics"] = feature_error_metrics(native, reference_prefix)
        report["native_prefix_vs_reference_metrics"]["exact_difference_count"] = exact_difference(native, reference_prefix)
        native.numpy().astype("<f4").tofile(output / "native-prefix-fp32.bin")
        report.update(prefix_shape=list(native.shape), original_prefix_weights=before,
                      native_prefix_sha256=file_digest(output / "native-prefix-fp32.bin"))
        save()
        for variant in variants:
            wrapper = PrefixExport(prefix, variant)
            wrapped = tensor(wrapper(values))
            same_native = exact_difference(wrapped, native) == 0
            if variant == "native" and not same_native:
                raise ValueError("Native prefix wrapper differs from original prefix bits")
            path = output / (variant + ".onnx")
            export_audio_native_onnx(wrapper, values, path)
            evaluator = ReferenceEvaluator(str(path))
            # Keep default qualification unchanged. This explicit large-graph
            # diagnostic uses the same kernels/order with last-consumer release
            # for both its same-host baseline and candidate.
            streaming = any(v in ("ordered_static_linear_refined", "ordered_original_stem") for v in variants)
            values_onnx = (run_flat_reference(evaluator, {"input": values.numpy()})[0] if streaming
                           else evaluator.run(None, {"input": values.numpy()})[0])
            actual = tensor(torch.from_numpy(values_onnx))
            if actual.shape != native.shape:
                raise ValueError("Prefix ONNX output shape differs")
            metrics = feature_error_metrics(actual, reference_prefix)
            metrics["exact_difference_count"] = exact_difference(actual, reference_prefix)
            if weight_digests(prefix) != before:
                raise ValueError("Original prefix coefficient bytes changed")
            capture = output / (variant + "-fp32.bin")
            actual.numpy().astype("<f4").tofile(capture)
            report["cases"].append(dict(variant=variant, graph=path.name, graph_sha256=file_digest(path),
                capture=capture.name, capture_sha256=file_digest(capture), metrics=metrics,
                native_wrapper_bits_equal=same_native, native_whole_encoder_exact_difference_count=exact,
                coefficients_unchanged=True, rms_mean_replacements=len(wrapper.rms_means), rms_inverse_replacements=len(wrapper.rms_inverses),
                normalization_replacements=len(wrapper.selected) if variant != "native" else 0,
                linear_replacements=len(wrapper.linear_nodes), ordered_linear_nodes=sorted(node.name for node in wrapper.linear_nodes),
                ordered_linear_native_execution="original_MatMul_addmm_passthrough_not_fixed_order_execution" if wrapper.linear_nodes else None,
                convolution_replacements=len(wrapper.convolution_nodes),
                convolution_native_execution='original_conv2d_passthrough_not_fixed_order_execution' if wrapper.convolution_nodes else None,
                reference_execution="same_kernels_original_order_last_consumer_release" if streaming else "default_ReferenceEvaluator"))
            save()
        report["complete"] = True
        save()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=tuple(STAGES), default="subsample")
    parser.add_argument("--frozen-stage-source", type=Path)
    parser.add_argument("--variant", choices=("all", *EXPERIMENTAL_VARIANTS), default="all")
    parser.add_argument('--case', choices=('silence', 'impulse', 'sine', 'nyquist', 'noise', 'padded_noise'), default='silence')
    args = parser.parse_args(argv)
    if args.variant == 'ordered_original_stem' and args.stage != 'subsample':
        raise ValueError('Original stem candidate requires the subsample boundary')
    source = os.environ.get("GITHUB_SHA", "")
    if re.fullmatch("[0-9a-f]{40}", source) is None:
        raise ValueError("Frozen prefix needs immutable job provenance")
    case_options = {'case_id': args.case} if args.case != 'silence' else {}
    program, values, expected, provenance = load_frozen_input(args.input, **case_options)
    provenance.update(job_source=source, torch_version=str(torch.__version__), threads=torch.get_num_threads(),
                      cpu_capability=torch.backends.cpu.get_cpu_capability())
    profile = provenance.get('profile', 'frames-1119-valid-1119')
    boundary_options = {'profile': profile} if profile != 'frames-1119-valid-1119' else {}
    boundary = boundary_for_stage(program, args.stage, **boundary_options)
    provenance["stage"] = args.stage
    frozen = load_frozen_stage_reference(args.frozen_stage_source, args.stage, **case_options) if args.frozen_stage_source else None
    variants = (VARIANTS if args.variant == "all" else
                ("native", args.variant) if args.variant in ("ordered_static_linear_refined", "ordered_original_stem") else (args.variant,))
    if any(v in ("ordered_static_linear_refined", "ordered_original_stem") for v in variants):
        audited = PrefixExport(extract_prefix(program, boundary), args.variant)
        count = expected_linear_count(args.stage)
        if len(audited.linear_nodes) != count:
            raise ValueError("Pinned280-row prefix linear replacement count differs")
        provenance["ordered_linear_expected_count"] = count
    if "refined_mean_moment_rms" in variants:
        audited = PrefixExport(extract_prefix(program, boundary), "refined_mean_moment_rms")
        count = expected_rms_count(args.stage)
        if len(audited.rms_means) != count or len(audited.rms_inverses) != count:
            raise ValueError("Pinned RMS replacement count differs")
        provenance["rms_expected_count"] = count
    report = probe_prefix(program, values, expected, args.output, boundary=boundary, provenance=provenance, frozen_stage=frozen, variants=variants)
    print(json.dumps(report))
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
