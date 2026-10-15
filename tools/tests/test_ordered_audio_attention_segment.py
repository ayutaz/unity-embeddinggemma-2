import numpy as np
import onnx
import pytest
import torch

from embeddinggemma_tools import frozen_audio_segment_onnx as probe


VARIANT = "ordered_static_linear_attention"


def fixture():
    class Module(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("weight", torch.ones(4, 4) * 0.01)
            self.register_buffer("keys", torch.ones(192, 128, 24) * 0.01)
            self.register_buffer("positions", torch.ones(8, 128, 13) * 0.01)
            self.register_buffer("values", torch.ones(192, 24, 128) * 0.01)
            self.register_buffer("unqualified_left", torch.ones(1, 280, 1024) * 0.01)
            self.register_buffer("unqualified_right", torch.eye(1024).unsqueeze(0))
        def forward(self, x):
            cut = x * 1.0
            seed = cut.mean()
            query = (self.keys.transpose(1, 2)[:, :12, :] + seed).contiguous()
            ac = torch.bmm(query, self.keys)
            bd = torch.bmm(query.reshape(8, 288, 128), self.positions)
            weighted = torch.bmm(ac, self.values)
            untouched = torch.bmm(self.unqualified_left + seed, self.unqualified_right)
            return torch.mm(cut, self.weight) + ac.mean() + bd.mean() + weighted.mean() + untouched.mean()
    values = torch.randn(280, 4, generator=torch.Generator().manual_seed(340)) * 0.01
    exported = torch.export.export(Module(), (values,)).run_decompositions({}).module()
    source = torch.fx.GraphModule(exported, exported.graph)
    cut = next(n for n in source.graph.nodes if n.target == torch.ops.aten.mul.Tensor)
    end = next(n for n in source.graph.nodes if n.op == "output").args[0][0]
    return probe.extract_segment(source, cut.name, end.name), values


def test_attention_lowering_selects_only_three_qualified_dynamic_shapes_and_keeps_native_passthrough(tmp_path):
    segment, values = fixture()
    before = str(segment.graph)
    wrapper = probe.SegmentExport(segment, VARIANT)
    assert len(wrapper.linear_nodes) == 1 and len(wrapper.bmm_nodes) == 3
    assert {tuple(n.args[0].meta['val'].shape) for n in wrapper.bmm_nodes} == {
        (192, 12, 128), (8, 288, 128), (192, 12, 24)}
    calls = []
    def observe(left, right):
        calls.append((tuple(left.shape), tuple(right.shape)))
        return torch.bmm(left, right)
    wrapper.bmm_candidate = observe
    expected = probe.tensor(segment(values))
    actual = probe.tensor(wrapper(values))
    assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))
    assert len(calls) == 3 and str(segment.graph) == before
    assert not wrapper.rms_means and not wrapper.silu_nodes
    exported = probe.SegmentExport(segment, VARIANT)
    graph = tmp_path / 'connected.onnx'
    probe.export_audio_native_onnx(exported, values, graph)
    model = onnx.load(graph)
    assert sum(n.op_type == 'MatMul' for n in model.graph.node) == 1
    result = probe.run_flat_reference(probe.ReferenceEvaluator(model), {'input': values.numpy()})[0]
    np.testing.assert_allclose(result, expected.numpy(), atol=1e-5, rtol=1e-4)


def test_previous_linear_variant_does_not_implicitly_enable_attention_rewrite():
    segment, _ = fixture()
    wrapper = probe.SegmentExport(segment, "ordered_static_linear")
    assert not getattr(wrapper, "bmm_nodes", set())


def test_new_attention_candidate_requires_both_audited_families():
    segment, _ = fixture()
    for node in segment.graph.nodes:
        if node.target == torch.ops.aten.bmm.default:
            node.args[0].meta = {}
    with pytest.raises(ValueError, match="attention"):
        probe.SegmentExport(segment, VARIANT)
