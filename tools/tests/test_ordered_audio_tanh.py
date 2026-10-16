from pathlib import Path

import numpy as np
import onnx
import pytest
import torch

from embeddinggemma_tools import frozen_audio_segment_onnx as probe
from embeddinggemma_tools import frozen_audio_chained_onnx as chain
from test_ordered_audio_positions import fixture as position_fixture

VARIANT = 'ordered_original_tanh'


def fixture():
    segment, values, _ = position_fixture()
    graph = segment.graph
    softmax = next(n for n in graph.nodes if n.target == torch.ops.aten._softmax.default)
    scores = softmax.args[0]
    with graph.inserting_before(softmax):
        tanh = graph.call_function(torch.ops.aten.tanh.default, (scores,))
    tanh.meta['val'] = torch.tanh(scores.meta['val'])
    softmax.args = (tanh, -1, False); graph.lint(); segment.recompile()
    return segment, values, tanh


def test_explicit_tanh_candidate_keeps_previous_positions_mode_and_evaluates_actual_connected_onnx(tmp_path):
    segment, values, tanh = fixture()
    previous = probe.SegmentExport(segment, probe.PRECISE_POSITIONS)
    new = probe.SegmentExport(segment, VARIANT)
    assert not getattr(previous, 'tanh_nodes', set())
    assert new.tanh_nodes == {tanh} and new.position_nodes == previous.position_nodes
    assert new.linear_nodes == previous.linear_nodes and new.bmm_nodes == previous.bmm_nodes
    graph = tmp_path / 'tanh.onnx'
    probe.export_audio_native_onnx(new, values, graph)
    model = onnx.load(graph)
    assert not any(n.op_type == 'Tanh' for n in model.graph.node)
    actual = probe.run_flat_reference(probe.ReferenceEvaluator(model), {'input': values.numpy()})[0]
    np.testing.assert_allclose(actual, probe.tensor(segment(values)).numpy(), atol=1e-5, rtol=1e-4)


def test_tanh_candidate_fails_closed_on_wrong_original_attention_shape():
    segment, _, tanh = fixture()
    tanh.args[0].meta['val'] = torch.zeros(1, 8, 24, 12, 23)
    with pytest.raises(ValueError, match='Tanh'):
        probe.SegmentExport(segment, VARIANT)


def test_new_chained_cli_candidate_requires_immutable_source_and_is_explicit_in_workflow(tmp_path, monkeypatch):
    monkeypatch.delenv('GITHUB_SHA', raising=False)
    with pytest.raises(ValueError, match='provenance'):
        chain.main(['--input', str(tmp_path), '--output', str(tmp_path / 'out'),
            '--frozen-stage-source', str(tmp_path), '--variant', VARIANT])
    text = (Path(__file__).resolve().parents[2] / '.github/workflows/frozen-audio-prefix-onnx.yml').read_text()
    assert VARIANT in text
