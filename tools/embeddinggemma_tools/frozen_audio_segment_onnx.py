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
    exact_difference, expected_rms_count, export_audio_native_onnx, file_digest, load_frozen_input,
    load_frozen_stage_reference, tensor, weight_digests,
)
from .reference_context_failure import feature_error_metrics
from .native_silu_onnx import fp32_native_silu
from .native_softmax_onnx import fp32_native_softmax24
from .native_audio_primitives_onnx import fp32_native_sigmoid, fp32_native_depthwise5
from .native_tanh_onnx import fp32_native_tanh
from .midpoint_fp32_fma import midpoint_query_key_bmm_export
from .streaming_reference import run_flat_reference
from .static_block192_onnx import audit_static_linear_nodes, static_block192_export, static_block192_addmm_export
from .static_block192_onnx import audit_attention_bmm_nodes, audit_parameter_bmm_nodes, static_block192_bmm_export
from .static_block192_onnx import audit_relative_position_linear_nodes

PRECISE = "ordered_original_primitives"
PRECISE_POSITIONS = "ordered_original_positions"
PRECISE_TANH = "ordered_original_tanh"
PRECISE_MIDPOINT = 'ordered_original_midpoint_bmm'
VARIANTS = ("native", "refined_mean_moment_rms", "native_silu", "refined_mean_moment_rms_silu", "ordered_static_linear", "ordered_static_linear_attention", "ordered_static_linear_all_bmm", PRECISE, PRECISE_POSITIONS, PRECISE_TANH, PRECISE_MIDPOINT)


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
    copied[cut].meta = dict(cut.meta)
    for node in nodes:
        if node in needed and node is not cut:
            copied[node] = graph.node_copy(node, lambda old: copied[old])
    graph.output(copied[end])
    graph.lint()
    return torch.fx.GraphModule(program, graph)


def audit_audio_leaf_nodes(segment):
    softmax, sigmoid, convolution = set(), set(), set()
    def value(node, shape):
        actual = node.meta.get('val') if isinstance(node, torch.fx.Node) else None
        return (isinstance(actual, torch.Tensor) and actual.dtype == torch.float32
                and actual.device.type == 'cpu' and tuple(actual.shape) == shape)
    for node in segment.graph.nodes:
        if node.op != 'call_function':
            continue
        if node.target == torch.ops.aten._softmax.default:
            if (len(node.args) != 3 or node.kwargs or node.args[1:] != (-1, False)
                    or not value(node.args[0], (1, 8, 24, 12, 24))):
                raise ValueError('Unqualified original audio Softmax contract')
            softmax.add(node)
        elif node.target == torch.ops.aten.sigmoid.default:
            if len(node.args) != 1 or node.kwargs or not value(node.args[0], (1, 280, 1024)):
                raise ValueError('Unqualified original audio sigmoid contract')
            sigmoid.add(node)
        elif node.target == torch.ops.aten.convolution.default:
            if (len(node.args) != 9 or node.kwargs or node.args[2] is not None
                    or tuple(node.args[3]) != (1,) or tuple(node.args[4]) != (0,)
                    or tuple(node.args[5]) != (1,) or node.args[6] is not False
                    or tuple(node.args[7]) != (0,) or node.args[8] != 1024
                    or not value(node.args[0], (1, 1024, 284)) or not value(node.args[1], (1024, 1, 5))
                    or node.args[1].op != 'get_attr'):
                raise ValueError('Unqualified original Depthwise convolution contract')
            convolution.add(node)
    if not all((softmax, sigmoid, convolution)):
        raise ValueError('Precise candidate requires all original audio leaf families')
    return softmax, sigmoid, convolution


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
        precise = variant in (PRECISE, PRECISE_POSITIONS, PRECISE_TANH, PRECISE_MIDPOINT)
        self.rms_means, self.rms_inverses = audit_rms_nodes(segment) if variant.startswith("refined_mean_moment_rms") or precise else (set(), set())
        self.silu_nodes = {node for node in segment.graph.nodes
                           if node.op == "call_function" and node.target == torch.ops.aten.silu.default} if variant.endswith("_silu") or precise else set()
        if (variant.endswith("_silu") or precise) and not self.silu_nodes:
            raise ValueError("Explicit SiLU candidate requires original SiLU nodes")
        self.silu_candidate = fp32_native_silu
        self.silu_observations = []
        self.linear_nodes = audit_static_linear_nodes(segment) if variant.startswith("ordered_static_linear") or precise else set()
        if (variant.startswith("ordered_static_linear") or precise) and not self.linear_nodes:
            raise ValueError("Ordered static linear candidate requires original280-row parameter projections")
        self.position_nodes = audit_relative_position_linear_nodes(segment) if variant in (PRECISE_POSITIONS, PRECISE_TANH, PRECISE_MIDPOINT) else set()
        if variant in (PRECISE_POSITIONS, PRECISE_TANH, PRECISE_MIDPOINT) and not self.position_nodes:
            raise ValueError('Ordered position candidate requires original13-row parameter projections')
        self.linear_nodes |= self.position_nodes
        self.linear_candidate = static_block192_export
        self.biased_linear_candidate = static_block192_addmm_export
        batched = variant in ("ordered_static_linear_attention", "ordered_static_linear_all_bmm", PRECISE, PRECISE_POSITIONS, PRECISE_TANH, PRECISE_MIDPOINT)
        self.bmm_nodes = audit_attention_bmm_nodes(segment) if batched else set()
        if batched and not self.bmm_nodes:
            raise ValueError("Ordered attention candidate requires qualified original attention BMM nodes")
        self.parameter_bmm_nodes = audit_parameter_bmm_nodes(segment) if variant == "ordered_static_linear_all_bmm" or precise else set()
        if (variant == "ordered_static_linear_all_bmm" or precise) and not self.parameter_bmm_nodes:
            raise ValueError("Ordered full BMM candidate requires original parameter-backed projection")
        self.bmm_nodes |= self.parameter_bmm_nodes
        self.bmm_candidate = static_block192_bmm_export
        self.softmax_nodes, self.sigmoid_nodes, self.convolution_nodes = audit_audio_leaf_nodes(segment) if precise else (set(), set(), set())
        self.softmax_candidate, self.sigmoid_candidate, self.convolution_candidate = fp32_native_softmax24, fp32_native_sigmoid, fp32_native_depthwise5
        self.tanh_nodes = set()
        if variant in (PRECISE_TANH, PRECISE_MIDPOINT):
            for node in segment.graph.nodes:
                if node.op != 'call_function' or node.target != torch.ops.aten.tanh.default:
                    continue
                value = node.args[0].meta.get('val') if len(node.args) == 1 and isinstance(node.args[0], torch.fx.Node) else None
                if (node.kwargs or not isinstance(value, torch.Tensor) or value.dtype != torch.float32
                        or value.device.type != 'cpu' or tuple(value.shape) != (1, 8, 24, 12, 24)):
                    raise ValueError('Unqualified original attention Tanh contract')
                self.tanh_nodes.add(node)
            if not self.tanh_nodes:
                raise ValueError('Ordered Tanh candidate requires original attention Tanh nodes')
        self.tanh_candidate = fp32_native_tanh
        self.midpoint_bmm_nodes = set()
        if variant == PRECISE_MIDPOINT:
            self.midpoint_bmm_nodes = {n for n in self.bmm_nodes-self.parameter_bmm_nodes
                if tuple(n.args[0].meta['val'].shape)==(192,12,128)
                and tuple(n.args[1].meta['val'].shape)==(192,128,24)}
            if len(self.midpoint_bmm_nodes) != len(self.tanh_nodes):
                raise ValueError('Pinned query-key midpoint attention BMM coverage differs')
            self.bmm_candidate = midpoint_query_key_bmm_export


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
                relative_position_replacements=len(wrapper.position_nodes),
                tanh_replacements=len(wrapper.tanh_nodes),
                rms_inverse_replacements=len(wrapper.rms_inverses), silu_replacements=len(wrapper.silu_nodes),
                silu_input_domains=wrapper.silu_observations, linear_replacements=len(wrapper.linear_nodes),
                ordered_linear_native_execution="original_MatMul_addmm_passthrough_not_fixed_order_execution" if wrapper.linear_nodes else None,
                ordered_linear_nodes=sorted(node.name for node in wrapper.linear_nodes),
                attention_bmm_replacements=len(wrapper.bmm_nodes - wrapper.parameter_bmm_nodes),
                attention_bmm_nodes=sorted(node.name for node in wrapper.bmm_nodes - wrapper.parameter_bmm_nodes),
                parameter_bmm_replacements=len(wrapper.parameter_bmm_nodes),
                parameter_bmm_nodes=sorted(node.name for node in wrapper.parameter_bmm_nodes),
                softmax_replacements=len(wrapper.softmax_nodes), sigmoid_replacements=len(wrapper.sigmoid_nodes),
                depthwise_replacements=len(wrapper.convolution_nodes),
                attention_bmm_native_execution="original_bmm_passthrough_not_fixed_order_execution" if wrapper.bmm_nodes else None))
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
    if args.variant != "native":
        audited = SegmentExport(extract_segment(program, start_boundary, end_boundary), args.variant)
    if args.variant.endswith("_silu") or args.variant in (PRECISE, PRECISE_POSITIONS, PRECISE_TANH):
        layer_count = min(tuple(STAGES).index(args.stage), 12) - tuple(STAGES).index(args.from_stage)
        if len(audited.silu_nodes) != layer_count * 3:
            raise ValueError("Pinned segment SiLU replacement count differs")
        provenance["silu_expected_count"] = layer_count * 3
    if args.variant.startswith("refined_mean_moment_rms") or args.variant in (PRECISE, PRECISE_POSITIONS, PRECISE_TANH):
        rms_count = expected_rms_count(args.stage) - expected_rms_count(args.from_stage)
        if len(audited.rms_means) != rms_count or len(audited.rms_inverses) != rms_count:
            raise ValueError("Pinned segment RMS replacement count differs")
        provenance["rms_expected_count"] = rms_count
    if args.variant.startswith("ordered_static_linear") or args.variant in (PRECISE, PRECISE_POSITIONS, PRECISE_TANH):
        layer_count = min(tuple(STAGES).index(args.stage), 12) - tuple(STAGES).index(args.from_stage)
        count = layer_count * (10 if args.variant in (PRECISE_POSITIONS, PRECISE_TANH) else 9) + (2 if args.stage == "projected_features" else 0)
        if len(audited.linear_nodes) != count:
            raise ValueError("Pinned280-row static linear replacement count differs")
        provenance["ordered_linear_expected_count"] = count
        if args.variant in ("ordered_static_linear_attention", "ordered_static_linear_all_bmm", PRECISE, PRECISE_POSITIONS, PRECISE_TANH):
            if len(audited.bmm_nodes - audited.parameter_bmm_nodes) != layer_count * 3:
                raise ValueError("Pinned attention BMM replacement count differs")
            provenance["ordered_attention_bmm_expected_count"] = layer_count * 3
            if args.variant in ("ordered_static_linear_all_bmm", PRECISE, PRECISE_POSITIONS, PRECISE_TANH):
                if len(audited.parameter_bmm_nodes) != layer_count:
                    raise ValueError("Pinned parameter-backed BMM replacement count differs")
                provenance["ordered_parameter_bmm_expected_count"] = layer_count
    if args.variant in (PRECISE, PRECISE_POSITIONS, PRECISE_TANH):
        if any(len(nodes) != layer_count for nodes in (audited.softmax_nodes, audited.sigmoid_nodes, audited.convolution_nodes)):
            raise ValueError('Pinned original audio leaf replacement count differs')
        provenance['audio_leaf_expected_count_per_family'] = layer_count
        if args.variant == PRECISE_TANH and len(audited.tanh_nodes) != layer_count:
            raise ValueError('Pinned original attention Tanh replacement count differs')
    # The joint candidate carries a same-host ONNX baseline; comparisons from
    # earlier jobs alone cannot establish the source of a small drift.
    variants = (("native", "ordered_static_linear", args.variant) if args.variant == "ordered_static_linear_attention" else
                ("native", args.variant) if args.variant in ("refined_mean_moment_rms_silu", "ordered_static_linear", "ordered_static_linear_all_bmm", PRECISE, PRECISE_POSITIONS, PRECISE_TANH) else (args.variant,))
    report = probe_segment(program, start, end, args.output,
        from_boundary=start_boundary, to_boundary=end_boundary, variants=variants, provenance=provenance)
    print(json.dumps(dict(complete=report["complete"], cases=report["cases"], full_encoder_validated=False)))
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
