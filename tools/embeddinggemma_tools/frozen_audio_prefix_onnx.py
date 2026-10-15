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

VARIANTS = ("native", "refined_moment", "refined_mean_moment")
EXPERIMENTAL_VARIANTS = (*VARIANTS, "refined_mean_moment_rms")
STAGE_REFERENCE_SHA = "6620f335563a91ccc484df7d9b89d4d889e58dab4ec3d7d0389b36abe36b454a"
STAGE_AUDIT_SHA = "3edee8acdd9ebebb819ae488b753a3a94949b953aa0906995e516764932b0f9b"


@dataclass(frozen=True)
class FrozenStageReference:
    expected: torch.Tensor
    provenance: dict


def load_frozen_stage_reference(directory, stage):
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
    rows = [row for row in reference["cases"] if row["id"] == "silence"]
    if (len(rows) != 1 or not rows[0]["native_final_matches_original"] or not rows[0]["trace_numpy_matches_original"]
            or rows[0]["profile"] != "frames-1119-valid-1119"
            or rows[0]["input_features"] != "frames-1119-valid-1119/silence-features-fp32.bin"):
        raise ValueError("Frozen silence input/final context differs")
    entries = [entry for entry in rows[0]["stages"] if entry["name"] == stage]
    shape = [280, 512] if stage == "projected_features" else [1, 280, 1024]
    if len(entries) != 1 or entries[0]["shape"] != shape:
        raise ValueError("Frozen stage output shape differs")
    entry = entries[0]
    path = (root / entry["expected"]).resolve()
    spec = audit["files"].get(entry["expected"], {})
    if (not path.is_relative_to(root) or not path.is_file() or path.stat().st_size != int(np.prod(shape)) * 4
            or path.stat().st_size != spec.get("bytes") or file_digest(path) != spec.get("sha256")):
        raise ValueError("Frozen stage tensor path, size or hash mismatch")
    expected = tensor(torch.from_numpy(np.fromfile(path, dtype="<f4").reshape(shape)))
    return FrozenStageReference(expected, dict(stage=stage, reference_sha256=STAGE_REFERENCE_SHA,
        audit_sha256=STAGE_AUDIT_SHA, expected_sha256=spec["sha256"], reference_source=PRODUCER,
        model_revision=REVISION, encoder_reference_sha256=REFERENCE_SHA))
# Names verified against b85a2ad... serialized schema in the pinned graph.
STAGES = {"subsample": ("view_2", torch.ops.aten.view.default),
          **{f"layer_{index}": (f"mul_{27 + 25 * index}", torch.ops.aten.mul.Tensor) for index in range(12)},
          "projected_features": ("select", torch.ops.aten.select.int)}


def boundary_for_stage(program, stage):
    if stage not in STAGES:
        raise ValueError("Unknown fixed audio stage")
    name, target = STAGES[stage]
    matches = [node for node in program.graph.nodes if node.name == name]
    if len(matches) != 1 or matches[0].op != "call_function" or matches[0].target != target:
        raise ValueError("Frozen audio stage boundary differs")
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


class PrefixExport(torch.nn.Module):
    def __init__(self, prefix, variant):
        super().__init__()
        self.prefix = prefix
        self.variant = variant
        self.selected = checked_normalizations(prefix)
        self.rms_means, self.rms_inverses = audit_rms_nodes(prefix) if variant == "refined_mean_moment_rms" else (set(), set())

    def forward(self, values):
        selected, variant, rms_means, rms_inverses = self.selected, self.variant, self.rms_means, self.rms_inverses
        silu_nodes = getattr(self, "silu_nodes", set())
        silu_candidate = getattr(self, "silu_candidate", None)
        silu_observations = getattr(self, "silu_observations", None)

        class Replay(torch.fx.Interpreter):
            def run_node(self, node):
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
                    return candidate_output(*args, "refined_mean_moment" if variant == "refined_mean_moment_rms" else variant)
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
    if receipt.exists() or any((output / (variant + ".onnx")).exists() for variant in VARIANTS):
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
            values_onnx = ReferenceEvaluator(str(path)).run(None, {"input": values.numpy()})[0]
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
                coefficients_unchanged=True, rms_mean_replacements=len(wrapper.rms_means), rms_inverse_replacements=len(wrapper.rms_inverses)))
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
    args = parser.parse_args(argv)
    source = os.environ.get("GITHUB_SHA", "")
    if re.fullmatch("[0-9a-f]{40}", source) is None:
        raise ValueError("Frozen prefix needs immutable job provenance")
    program, values, expected, provenance = load_frozen_input(args.input)
    provenance.update(job_source=source, torch_version=str(torch.__version__), threads=torch.get_num_threads(),
                      cpu_capability=torch.backends.cpu.get_cpu_capability())
    boundary = boundary_for_stage(program, args.stage)
    provenance["stage"] = args.stage
    frozen = load_frozen_stage_reference(args.frozen_stage_source, args.stage) if args.frozen_stage_source else None
    variants = VARIANTS if args.variant == "all" else (args.variant,)
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
