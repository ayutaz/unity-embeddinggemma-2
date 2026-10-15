"""Compare isolated ONNX operators on native inputs; never inject their outputs."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re

import numpy as np
import onnx
from onnx import helper, TensorProto
from onnx.reference import ReferenceEvaluator
import torch

from .frozen_audio_prefix_onnx import (FrozenStageReference, STAGES, boundary_for_stage,
    exact_difference, file_digest, load_frozen_input, load_frozen_stage_reference, tensor, weight_digests)
from .frozen_audio_segment_onnx import extract_segment
from .reference_context_failure import feature_error_metrics

OPERATORS = {
    torch.ops.aten.mm.default: "MatMul", torch.ops.aten.bmm.default: "MatMul",
    torch.ops.aten._softmax.default: "Softmax", torch.ops.aten.silu.default: "SiLU",
    torch.ops.aten.convolution.default: "Conv", torch.ops.aten.mean.dim: "ReduceMean",
    torch.ops.aten.pow.Tensor_Scalar: "Pow",
}


def onnx_run(kind, arrays, attributes):
    names = [f"input_{i}" for i in range(len(arrays))]
    inputs = [helper.make_tensor_value_info(name, TensorProto.INT64 if a.dtype == np.int64 else TensorProto.FLOAT,
                                           list(a.shape)) for name, a in zip(names, arrays, strict=True)]
    if kind == "SiLU":
        nodes = [helper.make_node("Neg", [names[0]], ["negative"]),
                 helper.make_node("Exp", ["negative"], ["exponential"]),
                 helper.make_node("Add", ["exponential", "one"], ["denominator"]),
                 helper.make_node("Div", [names[0], "denominator"], ["output"])]
        initializers = [helper.make_tensor("one", TensorProto.FLOAT, [], [1.0])]
    else:
        nodes, initializers = [helper.make_node(kind, names, ["output"], **attributes)], []
    graph = helper.make_graph(nodes, "isolated_native_input_operator", inputs,
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [None] * arrays[0].ndim)], initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 24)])
    onnx.checker.check_model(model)
    # The standalone graph has no custom operators or coefficient conversion.
    with np.errstate(over="ignore"):
        result = ReferenceEvaluator(model).run(None, dict(zip(names, arrays, strict=True)))[0]
    return tensor(torch.from_numpy(result))


def primitive_outputs(target, args, kwargs):
    if target not in OPERATORS or kwargs:
        raise ValueError("Unaudited primitive target or keyword arguments")
    tensors = [value for value in args if isinstance(value, torch.Tensor)]
    if not tensors or any(value.dtype != torch.float32 or value.device.type != "cpu" for value in tensors):
        raise ValueError("Primitive inputs must retain CPU FP32 dtype")
    kind = OPERATORS[target]
    arrays = [value.detach().numpy() for value in tensors]
    if kind == "MatMul":
        rank = 2 if target == torch.ops.aten.mm.default else 3
        if len(args) != 2 or len(tensors) != 2 or any(value.ndim != rank for value in tensors):
            raise ValueError("MatMul requires two original matrices")
        return [("native_layout", onnx_run(kind, arrays, {}), {}),
                ("contiguous_inputs", onnx_run(kind, [np.ascontiguousarray(a) for a in arrays], {}), {})]
    if kind == "Softmax":
        if len(args) != 3 or args[1] != -1 or args[2] is not False:
            raise ValueError("Only original last-axis FP32 softmax is audited")
        attributes = dict(axis=-1)
    elif kind == "ReduceMean":
        if len(args) != 3 or tuple(args[1]) != (-1,) or args[2] is not True:
            raise ValueError("Only original squared last-axis mean is audited")
        arrays.append(np.asarray([-1], dtype=np.int64))
        attributes = dict(keepdims=1)
    elif kind == "Pow":
        if len(args) != 2 or args[1] not in (2, -0.5):
            raise ValueError("Only original square and inverse-square-root powers are audited")
        arrays.append(np.asarray(args[1], dtype=np.float32))
        attributes = dict(exponent=args[1])
        return [("standard_onnx", onnx_run(kind, arrays, {}), attributes)]
    elif kind == "Conv":
        if (len(args) != 9 or args[6] is not False or any(args[7])
                or type(args[8]) is not int or args[8] < 1 or tensors[0].ndim not in (3, 4)):
            raise ValueError("Only original non-transposed 1D/2D convolution is audited")
        attributes = dict(strides=list(args[3]), pads=list(args[4]) * 2,
                          dilations=list(args[5]), group=args[8])
    else:
        if len(args) != 1:
            raise ValueError("SiLU must be the original unary operation")
        attributes = {}
    return [("standard_onnx", onnx_run(kind, arrays, attributes), attributes)]


def metrics(actual, expected):
    result = feature_error_metrics(actual, expected)
    result["exact_difference_count"] = exact_difference(actual, expected)
    return result


def probe_primitives(program, frozen_input, frozen_output, output, *, from_boundary, to_boundary, provenance=None,
                     variant='standard'):
    from .frozen_audio_segment_onnx import PRECISE
    if variant not in ('standard', PRECISE):
        raise ValueError('Unknown isolated primitive variant')
    if not all(isinstance(value, FrozenStageReference) for value in (frozen_input, frozen_output)):
        raise ValueError("Primitive replay requires both audited frozen stage references")
    output = Path(output)
    receipt = output / "audio-primitive-replay.json"
    if receipt.exists():
        raise ValueError("Refusing to overwrite primitive replay evidence")
    segment = extract_segment(program, from_boundary, to_boundary)
    candidate_nodes = {}
    if variant == PRECISE:
        from .audio_candidate_primitives import audited_replacements, run_candidate
        candidate_nodes = audited_replacements(segment)
    before = weight_digests(segment)
    values, expected = tensor(frozen_input.expected), tensor(frozen_output.expected)
    output.mkdir(parents=True, exist_ok=True)
    report = dict(scope="noninjecting_isolated_onnx_operators_on_audited_frozen_input_segment", complete=False,
        provenance=provenance, from_boundary=from_boundary, to_boundary=to_boundary,
        input_provenance=frozen_input.provenance, target_provenance=frozen_output.provenance,
        cases=[], worst_operator_captures={}, contract_captures={}, variant=variant,
        candidate_nodes=sorted(node.name for node in candidate_nodes), full_encoder_validated=False, gpu_validated=False,
        end_to_end_validated=False, all_replayed_values_remain_original_native_operations=True)

    def save():
        receipt.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    def capture(kind, row, args, native, actual, *, table='worst_operator_captures', prefix='worst'):
        prior = report[table].get(kind)
        if prior and (prior["maximum_tolerance_ratio"], prior['exact_difference_count']) >= (
                row["metrics"]["maximum_tolerance_ratio"], row['metrics']['exact_difference_count']):
            return
        files = []
        arrays = [(f"input{i}", value) for i, value in enumerate(args) if isinstance(value, torch.Tensor)]
        arrays.extend((("native", native), ("onnx", actual)))
        for name, value in arrays:
            path = output / f"{prefix}-{kind}-{name}-fp32.bin"
            value.detach().contiguous().numpy().astype("<f4").tofile(path)
            files.append(dict(file=path.name, shape=list(value.shape), native_strides=list(value.stride()),
                              sha256=file_digest(path)))
        report[table][kind] = dict(node=row["node"], variant=row["variant"], operator=row['operator'],
            attributes=row["attributes"], maximum_tolerance_ratio=row["metrics"]["maximum_tolerance_ratio"],
            exact_difference_count=row['metrics']['exact_difference_count'], files=files)

    save()
    with torch.no_grad():
        native = tensor(segment(values))
        report["native_segment_original_metrics"] = metrics(native, expected)
        report["original_context_qualified"] = exact_difference(native, expected) == 0
        save()

        class Replay(torch.fx.Interpreter):
            def run_node(self, node):
                selected = node.op == "call_function" and (node.target in OPERATORS or node in candidate_nodes)
                if selected:
                    args, kwargs = self.fetch_args_kwargs_from_env(node)
                result = super().run_node(node)
                if selected:
                    target = tensor(result)
                    changed = node in candidate_nodes
                    if changed:
                        if kwargs:
                            raise ValueError('Isolated candidate requires original positional arguments')
                        actual, attrs = run_candidate(candidate_nodes[node], tuple(x for x in args if isinstance(x, torch.Tensor)), output)
                        results = [(variant, actual, attrs)]
                    else:
                        results = primitive_outputs(node.target, args, kwargs)
                    for label, actual, attrs in results:
                        kind = OPERATORS.get(node.target, {torch.ops.aten.addmm.default: 'Gemm',
                            torch.ops.aten.sigmoid.default: 'Sigmoid'}.get(node.target))
                        if kind is None:
                            raise ValueError('Missing audited isolated operator label')
                        row = dict(node=node.name, target=str(node.target), operator=kind,
                            variant=label, candidate_replaced=changed, attributes=attrs, metrics=metrics(actual, target),
                            inputs=[dict(shape=list(value.shape), strides=list(value.stride()),
                                sha256=hashlib.sha256(value.detach().contiguous().numpy().tobytes()).hexdigest())
                                for value in args if isinstance(value, torch.Tensor)])
                        report["cases"].append(row)
                        capture(row["operator"], row, args, target, actual)
                        if variant == PRECISE:
                            contract = dict(target=row['target'], variant=label,
                                inputs=[dict(shape=spec['shape'], strides=spec['strides']) for spec in row['inputs']])
                            key = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()[:20]
                            capture(key, row, args, target, actual, table='contract_captures', prefix='contract')
                        save()
                # Isolated ONNX results are never fed into the native replay.
                return result

        try:
            replayed = tensor(Replay(segment).run(values))
            report["native_interpreter_control_bits_equal"] = exact_difference(replayed, native) == 0
            if not report["native_interpreter_control_bits_equal"] or weight_digests(segment) != before:
                raise ValueError("Primitive replay altered the native control or original coefficients")
        except Exception as error:
            report["diagnostic_error"] = dict(type=type(error).__name__, message=str(error))
            save()
            raise
    report["complete"] = True
    report["coefficients_unchanged"] = True
    save()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frozen-stage-source", type=Path, required=True)
    parser.add_argument("--from-stage", choices=tuple(STAGES)[:-1], required=True)
    parser.add_argument("--stage", choices=tuple(STAGES), required=True)
    parser.add_argument('--variant', choices=('standard', 'ordered_original_primitives'), default='standard')
    args = parser.parse_args(argv)
    source = os.environ.get("GITHUB_SHA", "")
    if re.fullmatch("[0-9a-f]{40}", source) is None:
        raise ValueError("Primitive replay needs immutable job provenance")
    if tuple(STAGES).index(args.from_stage) >= tuple(STAGES).index(args.stage):
        raise ValueError("Primitive input boundary must precede output boundary")
    program, _, _, provenance = load_frozen_input(args.input)
    provenance.update(job_source=source, torch_version=str(torch.__version__), numpy_version=np.__version__,
                      cpu_capability=torch.backends.cpu.get_cpu_capability(), threads=torch.get_num_threads())
    result = probe_primitives(program, load_frozen_stage_reference(args.frozen_stage_source, args.from_stage),
        load_frozen_stage_reference(args.frozen_stage_source, args.stage), args.output,
        from_boundary=boundary_for_stage(program, args.from_stage), to_boundary=boundary_for_stage(program, args.stage),
        provenance=provenance, variant=args.variant)
    print(json.dumps(dict(complete=result["complete"], operator_comparisons=len(result["cases"]),
        local_failure_cases=sum(row["metrics"]["mismatched_elements"] > 0 for row in result["cases"]),
        native_segment_original_metrics=result["native_segment_original_metrics"], full_encoder_validated=False)))
    return 0 if result["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
