from pathlib import Path

import numpy as np
import onnx
import pytest
import torch

from embeddinggemma_tools import frozen_audio_segment_onnx as probe
from embeddinggemma_tools import static_block192_onnx as linear
from test_ordered_audio_precise_segment import complete_fixture

VARIANT = 'ordered_original_positions'


def fixture():
    segment, values = complete_fixture()
    segment.register_buffer('relative_input', torch.ones(13, 1024) * .01)
    segment.register_buffer('relative_weight', torch.ones(1024, 1024) * .001)
    graph = segment.graph
    output = next(n for n in graph.nodes if n.op == 'output')
    placeholder = next(n for n in graph.nodes if n.op == 'placeholder')
    with graph.inserting_before(output):
        seed = graph.call_function(torch.ops.aten.mean.default, (placeholder,))
        left = graph.call_function(torch.ops.aten.add.Tensor, (graph.get_attr('relative_input'), seed))
        right = graph.call_function(torch.ops.aten.permute.default, (graph.get_attr('relative_weight'), [1, 0]))
        mm = graph.call_function(torch.ops.aten.mm.default, (left, right))
        result = graph.call_function(torch.ops.aten.add.Tensor,
            (output.args[0], graph.call_function(torch.ops.aten.mean.default, (mm,))))
    output.args = (result,); graph.lint(); segment.recompile()
    class Metadata(torch.fx.Interpreter):
        def run_node(self, node):
            value = super().run_node(node)
            if isinstance(value, torch.Tensor): node.meta['val'] = value.detach()
            return value
    Metadata(segment).run(values)
    return segment, values, mm


def test_new_candidate_adds_only_audited13row_projection_and_preserves_previous_candidate(tmp_path):
    segment, values, mm = fixture()
    old = probe.SegmentExport(segment, probe.PRECISE)
    assert mm not in old.linear_nodes
    new = probe.SegmentExport(segment, VARIANT)
    assert new.linear_nodes == old.linear_nodes | {mm}
    before = str(segment.graph)
    native = probe.tensor(segment(values))
    assert probe.exact_difference(probe.tensor(new(values)), native) == 0
    graph = tmp_path / 'positions.onnx'
    probe.export_audio_native_onnx(new, values, graph)
    model = onnx.load(graph)
    assert not any(node.op_type == 'MatMul' for node in model.graph.node)
    actual = probe.run_flat_reference(probe.ReferenceEvaluator(model), {'input': values.numpy()})[0]
    np.testing.assert_allclose(actual, native.numpy(), atol=1e-5, rtol=1e-4)
    assert str(segment.graph) == before


def test_position_audit_rejects_dynamic_coefficients_and_wrong_width():
    audit = getattr(linear, 'audit_relative_position_linear_nodes', None)
    assert callable(audit), 'Original13row position projection audit is missing'
    segment, _, mm = fixture()
    assert audit(segment) == {mm}
    mm.args[1].args[0].op = 'placeholder'
    assert audit(segment) == set()


def test_workflow_and_chained_cli_support_explicit_positions_candidate():
    root = Path(__file__).resolve().parents[2]
    workflow = (root / '.github/workflows/frozen-audio-prefix-onnx.yml').read_text()
    chain = (root / 'tools/embeddinggemma_tools/frozen_audio_chained_onnx.py').read_text()
    assert VARIANT in workflow and 'PRECISE_POSITIONS' in chain
