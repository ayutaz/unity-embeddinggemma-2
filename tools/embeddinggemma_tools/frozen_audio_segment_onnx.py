"""Opt-in fixed-input segment diagnosis; never a whole-encoder acceptance gate."""

import argparse
import json
import os
from pathlib import Path
import re

import torch
from onnx.reference import ReferenceEvaluator

from .frozen_audio_prefix_onnx import (
    FrozenStageReference, PrefixExport, STAGES, audit_rms_nodes, boundary_for_stage,
    exact_difference, export_audio_native_onnx, file_digest, load_frozen_input,
    load_frozen_stage_reference, tensor, weight_digests,
)
from .reference_context_failure import feature_error_metrics
from .native_silu_onnx import fp32_native_silu
from .streaming_reference import run_flat_reference

VARIANTS = ("native", "refined_mean_moment_rms", "native_silu")


def extract_segment(program, from_boundary, to_boundary):
    nodes = list(program.graph.nodes)
    matches = [[node for node in nodes if node.name == name and node.op == "call_function"]
               for name in (from_boundary, to_boundary)]
    if any(len(found) != 1 for found in matches):
        raise ValueError("Segment boundary must be one existing operation")
    cut, end = (found[0] for found in matches)
    if nodes.index(cut) >= nodes.index(end):
        raise ValueError("Segment input boundary must precede its output boundary")
    needed, pending = {end}, [end]
    while pending:
        node = pending.pop()
        if node is cut:
            continue
        if node.op == "placeholder":
            raise ValueError("Segment has an original input dependency bypassing its cut")
        for dependency in node.all_input_nodes:
            if dependency not in needed:
                needed.add(dependency)
                pending.append(dependency)
    if cut not in needed:
        raise ValueError("Segment boundary is not an ancestor of the output")
    graph = torch.fx.Graph()
    copied = {cut: graph.placeholder("frozen_stage_input")}
    for node in nodes:
        if node in needed and node is not cut:
            copied[node] = graph.node_copy(node, lambda old: copied[old])
    graph.output(copied[end])
    graph.lint()
    return torch.fx.GraphModule(program, graph)


class SegmentExport(PrefixExport):
    def __init__(self, segment, variant):
        # Only post-subsample segments qualify. The original two LayerNorms
        # remain mandatory in the independent full-program loader and prefix path.
        torch.nn.Module.__init__(self)
        if variant not in VARIANTS:
            raise ValueError("Unknown segment variant")
        if any(node.target == torch.ops.aten.native_layer_norm.default for node in segment.graph.nodes):
            raise ValueError("Segment must start after both original native normalizations")
        self.prefix, self.variant, self.selected = segment, variant, []
        self.rms_means, self.rms_inverses = audit_rms_nodes(segment) if variant == "refined_mean_moment_rms" else (set(), set())
        self.silu_nodes = {node for node in segment.graph.nodes
                           if node.op == "call_function" and node.target == torch.ops.aten.silu.default} if variant == "native_silu" else set()
        if variant == "native_silu" and not self.silu_nodes:
            raise ValueError("Explicit SiLU candidate requires original SiLU nodes")
        self.silu_candidate = fp32_native_silu
        self.silu_observations = []


def probe_segment(program, frozen_input, frozen_output, output, *, from_boundary, to_boundary,
                  variants=("native",), provenance=None):
    if not all(isinstance(value, FrozenStageReference) for value in (frozen_input, frozen_output)):
        raise ValueError("Segment requires both explicitly audited frozen stage references")
    if (not isinstance(variants, (tuple, list)) or not variants or len(set(variants)) != len(variants)
            or any(variant not in VARIANTS for variant in variants)):
        raise ValueError("Segment variants must be unique supported choices")
    output = Path(output)
    receipt = output / "frozen-audio-segment-onnx.json"
    if receipt.exists() or any((output / (variant + ".onnx")).exists() for variant in variants):
        raise ValueError("Refusing to overwrite segment diagnostic evidence")
    segment = extract_segment(program, from_boundary, to_boundary)
    before = weight_digests(segment)
    values, expected = tensor(frozen_input.expected), tensor(frozen_output.expected)
    output.mkdir(parents=True, exist_ok=True)
    report = dict(scope="audited_frozen_input_audio_segment_onnx_diagnosis_only", complete=False,
                  from_boundary=from_boundary, to_boundary=to_boundary, provenance=provenance,
                  input_provenance=frozen_input.provenance, target_provenance=frozen_output.provenance,
                  original_segment_weights=before, cases=[], full_encoder_validated=False,
                  gpu_validated=False, end_to_end_validated=False,
                  reference_execution="onnx.reference.ReferenceEvaluator kernels, original node order, release after last explicit consumer")

    def save():
        receipt.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    def capture(name, value):
        path = output / name
        value.numpy().astype("<f4").tofile(path)
        return file_digest(path)

    def metrics(actual):
        result = feature_error_metrics(actual, expected)
        result["exact_difference_count"] = exact_difference(actual, expected)
        return result

    save()
    with torch.no_grad():
        native = tensor(segment(values))
        control = tensor(SegmentExport(segment, "native")(values))
        report["native_wrapper_control_bits_equal"] = exact_difference(control, native) == 0
        report["native_segment_vs_reference_metrics"] = metrics(native)
        report["input_sha256"] = capture("frozen-input-fp32.bin", values)
        report["reference_sha256"] = capture("reference-output-fp32.bin", expected)
        report["native_sha256"] = capture("native-output-fp32.bin", native)
        save()
        if not report["native_wrapper_control_bits_equal"]:
            raise ValueError("Native segment bridge differs from original operation bits")
        for variant in variants:
            wrapper = SegmentExport(segment, variant)
            wrapped = tensor(wrapper(values))
            graph = output / (variant + ".onnx")
            export_audio_native_onnx(wrapper, values, graph)
            array = run_flat_reference(ReferenceEvaluator(str(graph)), {"input": values.numpy()})[0]
            actual = tensor(torch.from_numpy(array))
            if weight_digests(segment) != before:
                raise ValueError("Original segment coefficient bytes changed")
            name = variant + "-fp32.bin"
            report["cases"].append(dict(variant=variant, graph=graph.name, graph_sha256=file_digest(graph),
                capture=name, capture_sha256=capture(name, actual), metrics=metrics(actual),
                native_wrapper_bits_equal=exact_difference(wrapped, native) == 0,
                coefficients_unchanged=True, rms_mean_replacements=len(wrapper.rms_means),
                rms_inverse_replacements=len(wrapper.rms_inverses), silu_replacements=len(wrapper.silu_nodes),
                silu_input_domains=wrapper.silu_observations))
            save()
    report["complete"] = True
    save()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frozen-stage-source", type=Path, required=True)
    parser.add_argument("--from-stage", choices=tuple(STAGES)[:-1], required=True)
    parser.add_argument("--stage", choices=tuple(STAGES), required=True)
    parser.add_argument("--variant", choices=VARIANTS, default="native")
    args = parser.parse_args(argv)
    source = os.environ.get("GITHUB_SHA", "")
    if re.fullmatch("[0-9a-f]{40}", source) is None:
        raise ValueError("Frozen segment needs immutable job provenance")
    if tuple(STAGES).index(args.from_stage) >= tuple(STAGES).index(args.stage):
        raise ValueError("Segment input boundary must precede output boundary")
    program, _, _, provenance = load_frozen_input(args.input)
    provenance.update(job_source=source, torch_version=str(torch.__version__), threads=torch.get_num_threads(),
                      cpu_capability=torch.backends.cpu.get_cpu_capability())
    start = load_frozen_stage_reference(args.frozen_stage_source, args.from_stage)
    end = load_frozen_stage_reference(args.frozen_stage_source, args.stage)
    start_boundary, end_boundary = (boundary_for_stage(program, stage) for stage in (args.from_stage, args.stage))
    if args.variant == "native_silu":
        audited = SegmentExport(extract_segment(program, start_boundary, end_boundary), args.variant)
        layer_count = min(tuple(STAGES).index(args.stage), 12) - tuple(STAGES).index(args.from_stage)
        if len(audited.silu_nodes) != layer_count * 3:
            raise ValueError("Pinned segment SiLU replacement count differs")
        provenance["silu_expected_count"] = layer_count * 3
    report = probe_segment(program, start, end, args.output,
        from_boundary=start_boundary, to_boundary=end_boundary, variants=(args.variant,), provenance=provenance)
    print(json.dumps(dict(complete=report["complete"], cases=report["cases"], full_encoder_validated=False)))
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
