import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnx
import pytest
import torch

from embeddinggemma_tools import audio_primitive_replay as replay
from embeddinggemma_tools.frozen_audio_segment_onnx import PRECISE
from embeddinggemma_tools.static_block192_onnx import static_block192_export, static_block192_addmm_export
from test_audio_primitive_replay import frozen, program


def candidate_module():
    from embeddinggemma_tools import audio_candidate_primitives
    return audio_candidate_primitives


@pytest.mark.parametrize('biased', [False, True])
def test_actual_exported_matrix_candidate_is_evaluated_and_native_passthrough_is_not_evidence(tmp_path, biased):
    candidate = candidate_module()
    a = torch.randn(3, 13, generator=torch.Generator().manual_seed(193)) * .1
    b = torch.randn(5, 13, generator=torch.Generator().manual_seed(194)).T * .1
    bias = torch.randn(5, generator=torch.Generator().manual_seed(195)) * .1
    args = (bias, a, b) if biased else (a, b)
    fn = static_block192_addmm_export if biased else static_block192_export
    actual, metadata = candidate.run_candidate(fn, args, tmp_path)
    expected = torch.addmm(bias, a, b) if biased else torch.mm(a, b)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
    assert metadata['execution'] == 'actual_standard_fp32_onnx_on_same_native_inputs'
    assert len(metadata['graph_sha256']) == 64
    assert not metadata['native_forward_used_as_candidate_evidence']
    assert not list(tmp_path.glob('*.onnx')), 'Large per-node graphs must be temporary'


def test_evaluator_output_is_used_even_if_native_forward_returns_a_different_answer(tmp_path):
    candidate = candidate_module()
    class DeliberateDifference(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            return x + 10
        @staticmethod
        def symbolic(g, x):
            return g.op('Identity', x)
    value = torch.tensor([[1., -2., 0.]])
    actual, _ = candidate.run_candidate(DeliberateDifference.apply, (value,), tmp_path)
    assert torch.equal(actual, value)
    assert not torch.equal(actual, DeliberateDifference.apply(value))


def test_rejects_non_fp32_input_before_export(tmp_path):
    with pytest.raises(ValueError, match='CPU FP32'):
        candidate_module().run_candidate(torch.neg, (torch.ones(2, dtype=torch.float64),), tmp_path)


def test_candidate_mapping_uses_audited_nodes_and_the_same_replacement_functions():
    candidate = candidate_module()
    graph = torch.fx.Graph()
    x = graph.placeholder('x')
    nodes = [graph.call_function(torch.ops.aten.neg.default, (x,)) for _ in range(8)]
    graph.output(nodes[-1])
    funcs = [torch.neg, torch.abs, torch.sin, torch.cos, torch.tanh, torch.relu, torch.sigmoid, torch.exp]
    wrapper = SimpleNamespace(linear_nodes={nodes[0]}, bmm_nodes={nodes[1]}, rms_means={nodes[2]},
        rms_inverses={nodes[3]}, silu_nodes={nodes[4]}, softmax_nodes={nodes[5]},
        sigmoid_nodes={nodes[6]}, convolution_nodes={nodes[7]},
        linear_candidate=funcs[0], biased_linear_candidate=funcs[0], bmm_candidate=funcs[1],
        silu_candidate=funcs[4], softmax_candidate=funcs[5], sigmoid_candidate=funcs[6], convolution_candidate=funcs[7])
    mapping = candidate.replacements(wrapper)
    assert set(mapping) == set(nodes)
    assert mapping[nodes[0]] is funcs[0] and mapping[nodes[1]] is funcs[1]
    assert all(mapping[nodes[i]] is funcs[i] for i in range(4, 8))
    assert mapping[nodes[2]].__name__ == 'ordered_rms_mean'
    assert mapping[nodes[3]].__name__ == 'fp32_refined_rstd'


def test_explicit_candidate_mode_preserves_native_control_and_captures_each_contract(tmp_path, monkeypatch):
    candidate = candidate_module()
    source, start, end = program()
    x = torch.randn(4, 8, generator=torch.Generator().manual_seed(27))
    original_graph = str(source.graph)
    original_weights = source.weight.clone()
    # Exercise collection and noninjection with a deliberately failing isolated
    # candidate. Real-model family auditing is performed by SegmentExport.
    def mapping(segment):
        return {node: torch.neg for node in segment.graph.nodes
                if node.target == torch.ops.aten.silu.default}
    monkeypatch.setattr(candidate, 'audited_replacements', mapping)
    r = replay.probe_primitives(source, frozen(x * 2, 'subsample'), frozen(source(x), 'layer_0'),
        tmp_path, from_boundary=start, to_boundary=end, variant=PRECISE)
    assert r['complete'] and r['original_context_qualified'] and r['native_interpreter_control_bits_equal']
    changed = [row for row in r['cases'] if row['candidate_replaced']]
    assert len(changed) == 1 and changed[0]['metrics']['mismatched_elements'] > 0
    assert changed[0]['attributes']['execution'] == 'actual_standard_fp32_onnx_on_same_native_inputs'
    assert r['contract_captures'] and not r['full_encoder_validated']
    assert str(source.graph) == original_graph and torch.equal(source.weight, original_weights)
    for capture in r['contract_captures'].values():
        for spec in capture['files']:
            path = tmp_path / spec['file']
            assert hashlib.sha256(path.read_bytes()).hexdigest() == spec['sha256']
    assert all(row['candidate_replaced'] is False for row in r['cases'] if row['operator'] == 'MatMul')


def test_candidate_mode_rejects_unknown_variant_and_does_not_change_default_selection(tmp_path):
    source, start, end = program()
    x = torch.zeros(4, 8)
    with pytest.raises(ValueError, match='variant'):
        replay.probe_primitives(source, frozen(x, 'subsample'), frozen(source(x), 'layer_0'),
            tmp_path, from_boundary=start, to_boundary=end, variant='unqualified')
    assert torch.ops.aten.addmm.default not in replay.OPERATORS
    assert torch.ops.aten.sigmoid.default not in replay.OPERATORS


def test_workflow_wires_only_explicit_candidate_mode_and_retains_boundaries_and_timeout():
    text = (Path(__file__).resolve().parents[2] / '.github/workflows/frozen-audio-prefix-onnx.yml').read_text()
    assert 'candidate_args' in text and '--variant "$PROBE_VARIANT"' in text
    assert 'ordered_original_primitives' in text and 'timeout-minutes: 20' in text
