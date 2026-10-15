"""Isolated actual ONNX execution of the audited connected candidate's operators.

Native forwards of static matrix helpers are passthrough controls only. This
diagnostic always evaluates the exported standard FP32 graph independently.
"""
import tempfile
from pathlib import Path

import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator
import torch

from .export import fp32_refined_rstd
from .frozen_audio_prefix_onnx import file_digest, ordered_rms_mean, tensor
from .frozen_audio_segment_onnx import PRECISE, SegmentExport
from .streaming_reference import run_flat_reference


def replacements(wrapper):
    selected = {node: (wrapper.biased_linear_candidate if node.target == torch.ops.aten.addmm.default
                       else wrapper.linear_candidate) for node in wrapper.linear_nodes}
    for nodes, function in ((wrapper.bmm_nodes, wrapper.bmm_candidate),
            (wrapper.rms_means, ordered_rms_mean), (wrapper.rms_inverses, fp32_refined_rstd),
            (wrapper.silu_nodes, wrapper.silu_candidate), (wrapper.softmax_nodes, wrapper.softmax_candidate),
            (wrapper.sigmoid_nodes, wrapper.sigmoid_candidate), (wrapper.convolution_nodes, wrapper.convolution_candidate)):
        if selected.keys() & nodes:
            raise ValueError('Candidate node belongs to multiple replacement families')
        selected.update(dict.fromkeys(nodes, function))
    return selected


def audited_replacements(segment):
    return replacements(SegmentExport(segment, PRECISE))


def run_candidate(function, inputs, output):
    if (not inputs or any(not isinstance(x, torch.Tensor) or x.device.type != 'cpu'
                         or x.dtype != torch.float32 or not x.numel() for x in inputs)):
        raise ValueError('Isolated candidate requires nonempty CPU FP32 tensor inputs')
    inputs = tuple(x.detach() for x in inputs)  # Retain original physical layout.
    class Operator(torch.nn.Module):
        def forward(self, *values):
            return function(*values)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='isolated-candidate-', dir=output) as directory:
        path = Path(directory) / 'operator.onnx'
        names = [f'input_{index}' for index in range(len(inputs))]
        with torch.no_grad():
            torch.onnx.export(Operator(), inputs, path, dynamo=False, opset_version=20,
                input_names=names, output_names=['output'], do_constant_folding=False)
        model = onnx.version_converter.convert_version(onnx.load(path), 24)
        if any(node.domain not in ('', 'ai.onnx') for node in model.graph.node):
            raise ValueError('Isolated candidate retained a nonstandard operator')
        constants = list(model.graph.initializer)
        constants.extend(attr.t for node in model.graph.node for attr in node.attribute
                         if attr.type == onnx.AttributeProto.TENSOR)
        if (any(value.data_type == onnx.TensorProto.DOUBLE for value in constants)
                or any(node.op_type == 'Cast' and attr.name == 'to' and attr.i == onnx.TensorProto.DOUBLE
                       for node in model.graph.node for attr in node.attribute)):
            raise ValueError('Isolated candidate introduced FP64 operations')
        onnx.checker.check_model(model)
        onnx.save(model, path)
        digest = file_digest(path)
        feeds = {name: value.numpy() for name, value in zip(names, inputs, strict=True)}
        # ReferenceEvaluator supplies the actual candidate, never native forward.
        with np.errstate(over='ignore'):
            actual = run_flat_reference(ReferenceEvaluator(model), feeds)[0]
        result = tensor(torch.from_numpy(actual))
    return result, dict(execution='actual_standard_fp32_onnx_on_same_native_inputs',
        graph_sha256=digest, candidate_function=getattr(function, '__name__', type(function).__name__),
        native_forward_used_as_candidate_evidence=False)
