import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from embeddinggemma_tools import audio_primitive_replay as replay
from embeddinggemma_tools import audio_candidate_primitives as candidates
from embeddinggemma_tools.frozen_audio_segment_onnx import PRECISE_TANH
from embeddinggemma_tools.native_tanh_onnx import fp32_native_tanh
from test_audio_primitive_replay import frozen


def test_explicit_tanh_map_uses_same_audited_positions_and_tanh_as_connected_candidate():
    from test_ordered_audio_tanh import fixture
    source,_,_=fixture()
    mapping=candidates.audited_replacements(source,variant=PRECISE_TANH)
    tanh=[n for n in source.graph.nodes if n.target==torch.ops.aten.tanh.default]
    assert len(tanh)==1 and mapping[tanh[0]] is fp32_native_tanh
    positions=[n for n in mapping if n.target==torch.ops.aten.mm.default and n.args[0].meta['val'].shape[0]==13]
    assert len(positions)==1


@pytest.mark.parametrize('kind',['tanh','div'])
def test_remaining_unmodified_tanh_and_scalar_division_are_independently_replayed(kind):
    values=torch.tensor([[-.54,.9197,1.5]],dtype=torch.float32)
    target=torch.ops.aten.tanh.default if kind=='tanh' else torch.ops.aten.div.Tensor
    args=(values,) if kind=='tanh' else (values,torch.tensor(50.,dtype=torch.float32))
    rows=replay.primitive_outputs(target,args,{})
    assert len(rows)==1 and rows[0][0]=='standard_onnx'
    np.testing.assert_allclose(rows[0][1].numpy(),target(*args).numpy(),atol=1e-5,rtol=1e-4)


def test_new_candidate_reports_tanh_and_div_without_injecting_wrong_isolated_candidate(tmp_path,monkeypatch):
    graph=torch.fx.Graph();raw=graph.placeholder('raw');start=graph.call_function(torch.ops.aten.mul.Tensor,(raw,2.))
    divisor=graph.get_attr('divisor');division=graph.call_function(torch.ops.aten.div.Tensor,(start,divisor))
    tanh=graph.call_function(torch.ops.aten.tanh.default,(division,));graph.output(tanh)
    root=torch.nn.Module();root.register_buffer('divisor',torch.tensor(50.))
    source=torch.fx.GraphModule(root,graph);values=torch.tensor([[-2.,1.,3.]])
    seen=[]
    def mapping(segment,*,variant):
        seen.append(variant);return {n:torch.neg for n in segment.graph.nodes if n.target==torch.ops.aten.tanh.default}
    monkeypatch.setattr(candidates,'audited_replacements',mapping)
    r=replay.probe_primitives(source,frozen(values*2,'subsample'),frozen(source(values),'layer_0'),
        tmp_path,from_boundary=start.name,to_boundary=tanh.name,variant=PRECISE_TANH)
    assert seen==[PRECISE_TANH] and r['complete'] and r['original_context_qualified']
    assert r['native_interpreter_control_bits_equal'] and r['all_replayed_values_remain_original_native_operations']
    assert {c['operator'] for c in r['cases']}=={'Tanh','Div'}
    assert next(c for c in r['cases'] if c['operator']=='Tanh')['metrics']['mismatched_elements']>0
    assert r['contract_captures'] and not r['gpu_validated']


def test_primitive_cli_selects_sine_original_input_and_both_frozen_endpoints(tmp_path,monkeypatch):
    monkeypatch.setenv('GITHUB_SHA','a'*40);calls=[]
    program=object()
    def load(path,*,case_id):
        calls.append(('input',case_id));return program,None,None,{'case_id':case_id,'profile':'frames-1119-valid-1119'}
    def stage(path,name,*,case_id):
        calls.append(('stage',case_id,name));return name
    def run(source,a,b,output,**kwargs):
        assert source is program and a=='subsample' and b=='layer_2' and kwargs['variant']==PRECISE_TANH
        return dict(complete=True,cases=[],native_segment_original_metrics={})
    monkeypatch.setattr(replay,'load_frozen_input',load)
    monkeypatch.setattr(replay,'load_frozen_stage_reference',stage)
    monkeypatch.setattr(replay,'boundary_for_stage',lambda source,name:name)
    monkeypatch.setattr(replay,'probe_primitives',run)
    assert replay.main(['--input',str(tmp_path),'--output',str(tmp_path/'out'),
        '--frozen-stage-source',str(tmp_path/'stage'),'--from-stage','subsample','--stage','layer_2',
        '--case','sine','--variant',PRECISE_TANH])==0
    assert calls==[('input','sine'),('stage','sine','subsample'),('stage','sine','layer_2')]


def test_workflow_passes_non_silence_case_to_explicit_tanh_primitive_mode():
    text=(Path(__file__).resolve().parents[2]/'.github/workflows/frozen-audio-prefix-onnx.yml').read_text()
    workflow=yaml.safe_load(text);steps=workflow['jobs']['probe']['steps']
    select=next(s['run'] for s in steps if s.get('name','').startswith('Select pinned original case'))
    assert 'Non-silence primitive replay requires an explicit audited candidate' in select
    command=next(s['run'] for s in steps if s.get('name','').startswith('Diagnose fixed boundary'))
    primitive=command.split('if [ "$PRIMITIVE_REPLAY" = true ]; then',1)[1].split('\n          fi',1)[0]
    assert 'ordered_original_tanh' in primitive and '--case "$PROBE_CASE"' in primitive
