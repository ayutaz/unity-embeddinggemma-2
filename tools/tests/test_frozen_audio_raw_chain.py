import operator
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from embeddinggemma_tools import frozen_audio_chained_onnx as probe


def fixture():
    root = torch.nn.Module()
    root.register_buffer('scale128', torch.ones(128)); root.register_buffer('scale32', torch.ones(32))
    root.register_buffer('stem_weight', torch.randn(32, 4, generator=torch.Generator().manual_seed(162)) * .1)
    root.register_buffer('w1', torch.eye(4) * 2); root.register_buffer('w2', torch.eye(4) * 3)
    graph = torch.fx.Graph(); x = graph.placeholder('features')
    norm0 = graph.call_function(torch.ops.aten.native_layer_norm.default,
        (x, [128], graph.get_attr('scale128'), None, 1e-6))
    n0 = graph.call_function(operator.getitem, (norm0, 0))
    sliced = graph.call_function(torch.ops.aten.slice.Tensor, (n0, 1, 0, 32))
    norm1 = graph.call_function(torch.ops.aten.native_layer_norm.default,
        (sliced, [32], graph.get_attr('scale32'), None, 1e-6))
    n1 = graph.call_function(operator.getitem, (norm1, 0))
    stem = graph.call_function(torch.ops.aten.mm.default, (n1, graph.get_attr('stem_weight')))
    middle = graph.call_function(torch.ops.aten.mm.default, (stem, graph.get_attr('w1')))
    final = graph.call_function(torch.ops.aten.mm.default, (middle, graph.get_attr('w2')))
    graph.output((final,))
    source = torch.fx.GraphModule(root, graph)
    values = torch.randn(4, 128, generator=torch.Generator().manual_seed(163))
    return source, values, [stem.name, middle.name, final.name]


def case(source, values):
    cls = getattr(probe, 'RawAudioCase', None)
    assert cls is not None, 'Raw original input contract is missing'
    return cls(values, source(values)[0].detach(), {'scope': 'synthetic_unit_only'})


def test_raw_features_run_through_actual_prefix_and_body_with_no_frozen_stage_input(tmp_path):
    source, values, boundaries = fixture()
    r = probe.probe_raw_chained_segments(source, case(source, values), tmp_path, boundaries=boundaries)
    assert r['complete'] and r['raw_original_input_executed']
    assert r['native_chain_control_bits_equal'] and r['context_qualified']
    assert not r['whole_encoder_validated'] and not r['gpu_validated']
    row = r['cases'][0]
    assert len(row['parts']) == len(boundaries) == 3
    assert row['parts'][0]['from_boundary'] == 'raw_features'
    assert row['metrics']['mismatched_elements'] == 0
    np.testing.assert_array_equal(np.fromfile(tmp_path / 'raw-input-fp32.bin', dtype='<f4'), values.numpy().ravel())
    assert not (tmp_path / 'frozen-input-fp32.bin').exists()
    for first, second in zip(row['parts'], row['parts'][1:]):
        assert first['output_sha256'] == second['input_sha256']


def test_wrong_actual_raw_prefix_output_propagates_through_entire_body_without_native_reset(tmp_path, monkeypatch):
    source, values, boundaries = fixture(); real = probe.run_flat_reference; feeds = []
    def run(evaluator, inputs):
        feeds.append(inputs['input'].copy())
        return [np.zeros((4, 4), dtype=np.float32)] if len(feeds) == 1 else real(evaluator, inputs)
    monkeypatch.setattr(probe, 'run_flat_reference', run)
    r = probe.probe_raw_chained_segments(source, case(source, values), tmp_path, boundaries=boundaries)
    assert r['complete'] and r['native_chain_control_bits_equal']
    np.testing.assert_array_equal(feeds[0], values.numpy())
    assert not np.count_nonzero(feeds[1]) and not np.count_nonzero(feeds[2])
    assert r['cases'][0]['metrics']['mismatched_elements'] == 16


def test_raw_chain_rejects_a_cut_that_drops_the_original_final_operation(tmp_path):
    source, values, boundaries = fixture()
    with pytest.raises(ValueError, match='final boundary'):
        probe.probe_raw_chained_segments(source, case(source, values), tmp_path, boundaries=boundaries[:-1])


def test_raw_cli_never_loads_an_intermediate_frozen_stage_and_forwards_original_input(tmp_path, monkeypatch):
    source, values, _ = fixture(); calls = []
    monkeypatch.setenv('GITHUB_SHA', 'a' * 40)
    monkeypatch.setattr(probe, 'load_frozen_input', lambda path: (source, values, source(values)[0], {}))
    monkeypatch.setattr(probe, 'boundary_for_stage', lambda program, stage: stage)
    def forbidden(*args, **kwargs):
        pytest.fail('Raw chain must not load any intermediate frozen reference')
    monkeypatch.setattr(probe, 'load_frozen_stage_reference', forbidden)
    def run(program, raw, output, **kwargs):
        calls.append((program, raw, kwargs)); return dict(complete=True, context_qualified=False, cases=[])
    monkeypatch.setattr(probe, 'probe_raw_chained_segments', run, raising=False)
    assert probe.main(['--input', str(tmp_path), '--output', str(tmp_path / 'out'),
        '--variant', 'native', '--raw-input']) == 0
    assert calls[0][0] is source and torch.equal(calls[0][1].values, values)
    assert calls[0][2]['boundaries'] == ['subsample', 'layer_2', 'layer_5', 'layer_8', 'projected_features']


def test_workflow_raw_mode_does_not_download_intermediate_stage_goldens():
    p = Path(__file__).resolve().parents[2] / '.github/workflows/frozen-audio-prefix-onnx.yml'
    text = p.read_text(encoding='utf-8'); workflow = yaml.safe_load(text)
    inputs = workflow.get('on', workflow.get(True))['workflow_dispatch']['inputs']
    assert inputs['raw_input']['default'] is False
    steps = workflow['jobs']['probe']['steps']
    frozen = next(s for s in steps if s.get('name', '').startswith('Fetch independently audited original stage'))
    assert "env.RAW_INPUT != 'true'" in frozen['if']
    assert '--raw-input' in text and 'RAW_INPUT' in workflow['jobs']['probe']['env']


def test_frozen_mode_still_requires_audited_stage_source_before_loading_model(tmp_path, monkeypatch):
    monkeypatch.setenv('GITHUB_SHA', 'a' * 40)
    def forbidden(*args, **kwargs):
        pytest.fail('Missing frozen source must reject before graph/model loading')
    monkeypatch.setattr(probe, 'load_frozen_input', forbidden)
    with pytest.raises(ValueError, match='source'):
        probe.main(['--input', str(tmp_path), '--output', str(tmp_path / 'out'), '--variant', 'native'])


@pytest.mark.parametrize('failure_part', [0, 1])
def test_partial_raw_failure_distinguishes_requested_mode_from_actual_prefix_execution(tmp_path, monkeypatch, failure_part):
    source, values, boundaries = fixture(); real = probe.run_flat_reference; calls = []
    def run(evaluator, inputs):
        index = len(calls); calls.append(index)
        if index == failure_part: raise ValueError('unit-test raw partition failure')
        return real(evaluator, inputs)
    monkeypatch.setattr(probe, 'run_flat_reference', run)
    with pytest.raises(ValueError, match='raw partition failure'):
        probe.probe_raw_chained_segments(source, case(source, values), tmp_path, boundaries=boundaries)
    r = json.loads((tmp_path / 'frozen-audio-chained-onnx.json').read_text())
    assert r['raw_input_mode'] and not r['complete']
    assert r['raw_original_input_executed'] is (failure_part > 0)
    assert not r['whole_encoder_validated'] and not r['gpu_validated']


def test_actual_candidate_couples_two_conv_stem_to_audited_tanh_body_without_reset(tmp_path):
    from test_ordered_original_stem import fixture as stem_fixture
    from test_ordered_audio_tanh import fixture as body_fixture
    stem, values, stem_name = stem_fixture(); body, _, _ = body_fixture()
    root = torch.nn.Module(); root.add_module('stem', stem); root.add_module('body', body)
    graph = torch.fx.Graph(); copied = {}; stem_output = None
    for module_name, module in (('stem', stem), ('body', body)):
        for node in module.graph.nodes:
            if node.op == 'output':
                result = node.args[0]
                if isinstance(result, tuple): result = result[0]
                if module_name == 'stem': stem_output = copied[result]
                else: graph.output((copied[result],))
                continue
            if node.op == 'placeholder':
                copied[node] = graph.placeholder('raw_features') if module_name == 'stem' else stem_output
            elif node.op == 'get_attr':
                copied[node] = graph.get_attr(module_name + '.' + node.target)
            else:
                copied[node] = graph.node_copy(node, lambda prior: copied[prior])
                if node.op == 'call_module': copied[node].target = module_name + '.' + node.target
            copied[node].meta = dict(node.meta)
    graph.lint(); source = torch.fx.GraphModule(root, graph)
    final = next(n for n in graph.nodes if n.op == 'output').args[0][0]
    raw = probe.RawAudioCase(values, source(values)[0].detach(), {'scope': 'synthetic_candidate_unit_only'})
    r = probe.probe_raw_chained_segments(source, raw, tmp_path,
        boundaries=[stem_output.name, final.name], variants=(probe.PRECISE_TANH,))
    assert r['complete'] and r['raw_original_input_executed'] and r['native_chain_control_bits_equal']
    assert r['cases'][0]['metrics']['mismatched_elements'] == 0
    first, second = r['cases'][0]['parts']
    assert first['stem_convolution_replacements'] == first['normalization_replacements'] == 2
    assert second['tanh_replacements'] == 1 and first['output_sha256'] == second['input_sha256']
