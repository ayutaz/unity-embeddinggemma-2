import pytest
import numpy as np
import torch

from embeddinggemma_tools import frozen_audio_segment_onnx as probe
from test_ordered_audio_attention_segment import fixture


def complete_fixture():
    segment, values = fixture()
    segment.register_buffer('stats', torch.ones(1, 280, 1024) * 0.1)
    segment.register_buffer('scores', torch.ones(1, 8, 24, 12, 24) * 0.1)
    segment.register_buffer('conv_input', torch.ones(1, 1024, 284) * 0.1)
    segment.register_buffer('conv_weight', torch.ones(1024, 1, 5) * 0.02)
    graph = segment.graph
    output = next(n for n in graph.nodes if n.op == 'output')
    placeholder = next(n for n in graph.nodes if n.op == 'placeholder')
    with graph.inserting_before(output):
        scalar = graph.call_function(torch.ops.aten.mean.default, (placeholder,))
        x = graph.call_function(torch.ops.aten.add.Tensor, (graph.get_attr('stats'), scalar))
        squared = graph.call_function(torch.ops.aten.pow.Tensor_Scalar, (x, 2))
        mean = graph.call_function(torch.ops.aten.mean.dim, (squared, [-1], True))
        inverse = graph.call_function(torch.ops.aten.pow.Tensor_Scalar,
            (graph.call_function(torch.ops.aten.add.Tensor, (mean, 1e-6)), -0.5))
        normalized = graph.call_function(torch.ops.aten.mul.Tensor, (x, inverse))
        activated = graph.call_function(torch.ops.aten.silu.default, (normalized,))
        gate = graph.call_function(torch.ops.aten.sigmoid.default, (activated,))
        scores = graph.call_function(torch.ops.aten.add.Tensor, (graph.get_attr('scores'), scalar))
        softmax = graph.call_function(torch.ops.aten._softmax.default, (scores, -1, False))
        conv_input = graph.call_function(torch.ops.aten.add.Tensor, (graph.get_attr('conv_input'), scalar))
        convolution = graph.call_function(torch.ops.aten.convolution.default,
            (conv_input, graph.get_attr('conv_weight'), None, [1], [0], [1], False, [0], 1024))
        result = output.args[0]
        for node in (gate, softmax, convolution):
            result = graph.call_function(torch.ops.aten.add.Tensor,
                (result, graph.call_function(torch.ops.aten.mean.default, (node,))))
    output.args = (result,)
    graph.lint(); segment.recompile()
    class Metadata(torch.fx.Interpreter):
        def run_node(self, node):
            value = super().run_node(node)
            if isinstance(value, torch.Tensor): node.meta['val'] = value.detach()
            return value
    Metadata(segment).run(values)
    return segment, values


def test_explicit_precise_candidate_covers_qualified_families_and_keeps_default_untouched(tmp_path):
    segment, values = complete_fixture()
    original = str(segment.graph)
    wrapper = probe.SegmentExport(segment, 'ordered_original_primitives')
    assert len(wrapper.linear_nodes) == 1 and len(wrapper.bmm_nodes) == 4
    assert len(wrapper.rms_means) == len(wrapper.rms_inverses) == len(wrapper.silu_nodes) == 1
    assert len(wrapper.softmax_nodes) == len(wrapper.sigmoid_nodes) == len(wrapper.convolution_nodes) == 1
    actual, expected = probe.tensor(wrapper(values)), probe.tensor(segment(values))
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
    assert str(segment.graph) == original
    plain = probe.SegmentExport(segment, 'ordered_static_linear_all_bmm')
    assert not plain.rms_means and not plain.silu_nodes
    assert not getattr(plain, 'softmax_nodes', set())
    graph = tmp_path / 'precise.onnx'
    probe.export_audio_native_onnx(wrapper, values, graph)
    exported = probe.run_flat_reference(probe.ReferenceEvaluator(str(graph)), {'input': values.numpy()})[0]
    np.testing.assert_allclose(exported, expected.numpy(), atol=1e-5, rtol=1e-4)


def test_precise_candidate_fails_closed_on_unqualified_depthwise_contract():
    segment, _ = complete_fixture()
    node = next(n for n in segment.graph.nodes if n.target == torch.ops.aten.convolution.default)
    node.args = (*node.args[:-1], 512)
    with pytest.raises(ValueError, match='convolution|Depthwise'):
        probe.SegmentExport(segment, 'ordered_original_primitives')
