import json
from pathlib import Path
import shutil

import numpy as np
import pytest
import torch
import yaml

from embeddinggemma_tools.export import export_audio_native_onnx
from embeddinggemma_tools.frozen_audio_prefix_onnx import extract_prefix, file_digest, weight_digests
from embeddinggemma_tools.frozen_audio_segment_onnx import extract_segment
from embeddinggemma_tools.frozen_audio_chained_onnx import _replacement_counts


def module():
    from embeddinggemma_tools import audio_raw_resume
    return audio_raw_resume


@pytest.fixture
def checkpoint(tmp_path, monkeypatch):
    probe = module()
    class Five(torch.nn.Module):
        def __init__(self):
            super().__init__()
            for i in range(5):
                self.register_buffer(f's{i}', torch.tensor(float(i+2)))
        def forward(self, x):
            return ((((x*self.s0)*self.s1)*self.s2)*self.s3)*self.s4
    program = torch.fx.symbolic_trace(Five())
    cuts = [n.name for n in program.graph.nodes if n.op == 'call_function']
    bounds = ['raw_features', *cuts]
    monkeypatch.setattr(probe,'BOUNDARIES',bounds)
    values = torch.arange(1,9,dtype=torch.float32).reshape(2,4)
    expected = program(values).detach()
    root = tmp_path/'previous';root.mkdir()
    def capture(name, a):
        a.detach().numpy().astype('<f4').tofile(root/name)
        return file_digest(root/name)
    pieces = [extract_prefix(program,cuts[0]), *[extract_segment(program,a,b) for a,b in zip(cuts,cuts[1:])]]
    provenance = dict(case_id='noise',profile='frames-1119-valid-1119',job_source='a'*40)
    report = dict(scope='raw_original_feature_input_connected_onnx_partitions_only',complete=False,
        raw_input_mode=True,raw_original_input_executed=True,native_chain_control_bits_equal=True,
        whole_encoder_validated=False,gpu_validated=False,end_to_end_validated=False,
        provenance=provenance,boundaries=bounds,original_weights=weight_digests(program),
        input_sha256=capture('raw-input-fp32.bin',values),
        reference_sha256=capture('reference-output-fp32.bin',expected),
        native_sha256=capture('native-output-fp32.bin',expected),
        phase=dict(name='evaluating',part=3,variant='ordered_original_midpoint_bmm'),cases=[])
    for variant,count in [('native',5),('ordered_original_midpoint_bmm',3)]:
        row=dict(variant=variant,parts=[],actual_carry='previous_onnx_output_without_frozen_stage_reset')
        actual=values
        for i,piece in enumerate(pieces[:count]):
            inp=actual;actual=piece(inp).detach();piece.variant='native'
            graph=root/f'{variant}-part{i}.onnx';export_audio_native_onnx(piece,inp,graph)
            row['parts'].append(dict(index=i,from_boundary=bounds[i],to_boundary=bounds[i+1],
                graph_sha256=file_digest(graph),native_input_strides=list(inp.stride()),
                input_capture=f'{variant}-part{i}-input-fp32.bin',output_capture=f'{variant}-part{i}-output-fp32.bin',
                input_sha256=capture(f'{variant}-part{i}-input-fp32.bin',inp),
                output_sha256=capture(f'{variant}-part{i}-output-fp32.bin',actual),
                **_replacement_counts(piece,is_stem=i==0)))
        report['cases'].append(row)
    f=root/'frozen-audio-chained-onnx.json';f.write_text(json.dumps(report))
    spec=dict(feature_shape=[2,4],feature_count=2,feature_width=4)
    monkeypatch.setattr(probe,'PREDECESSOR_REPORT_SHA',file_digest(f))
    monkeypatch.setattr(probe,'PREDECESSOR_SOURCE','a'*40)
    monkeypatch.setattr(probe,'INTERMEDIATE_SHAPE',(2,4))
    monkeypatch.setattr(probe,'original_case',lambda _:(values.numpy(),expected.numpy(),spec))
    monkeypatch.setattr(probe,'verify_math_source',lambda:'unit-test-only')
    def wrap(piece,index):
        piece.variant='native'
        return piece
    monkeypatch.setattr(probe,'wrap_piece',wrap)
    monkeypatch.setattr(probe,'validate_suffix_counts',lambda counts,index:None)
    return probe,program,root,tmp_path/'resumed',expected


def test_actual_onnx_continuation_uses_saved_prediction_and_verifies_all_prefix_graphs(checkpoint):
    p,program,root,out,expected=checkpoint
    c=p.load_checkpoint(root,root)
    r=p.resume(program,c,out,job_source='b'*40)
    assert r['complete'] and r['original_tolerance_passed']
    assert len(r['verified_prefix_graphs'])==3 and len(r['parts'])==2
    assert [s['index'] for s in r['parts']]==[3,4]
    assert r['parts'][0]['input_sha256']==c['report']['cases'][1]['parts'][-1]['output_sha256']
    assert r['parts'][1]['input_sha256']==r['parts'][0]['output_sha256']
    assert r['predecessor_source']=='a'*40 and r['job_source']=='b'*40
    np.testing.assert_array_equal(np.fromfile(out/r['parts'][-1]['output_capture'],dtype='<f4'),expected.numpy().ravel())
    assert r['coefficients_unchanged'] and not r['gpu_validated']


def test_wrong_continuation_output_propagates_and_fails_final_reference(checkpoint,monkeypatch):
    p,program,root,out,_=checkpoint;real=p.run_flat_reference;feeds=[]
    def evaluate(evaluator,inputs):
        feeds.append(inputs['input'].copy())
        return [np.zeros((2,4),dtype=np.float32)] if len(feeds)==1 else real(evaluator,inputs)
    monkeypatch.setattr(p,'run_flat_reference',evaluate)
    r=p.resume(program,p.load_checkpoint(root,root),out,job_source='b'*40)
    assert len(feeds)==2 and not np.count_nonzero(feeds[1])
    assert r['complete'] and not r['original_tolerance_passed']


@pytest.mark.parametrize('change',['capture','report','source','carry'])
def test_corrupt_or_substituted_prefix_rejected_before_export(checkpoint,change,monkeypatch):
    p,_,root,_,_=checkpoint;f=root/'frozen-audio-chained-onnx.json';d=json.loads(f.read_text())
    if change=='capture':
        path=root/d['cases'][1]['parts'][-1]['output_capture'];path.write_bytes(b'corrupted')
    else:
        if change=='report': d['complete']=True
        if change=='source': d['provenance']['job_source']='c'*40
        if change=='carry': d['cases'][1]['parts'][1]['input_sha256']='0'*64
        f.write_text(json.dumps(d))
        if change!='report':monkeypatch.setattr(p,'PREDECESSOR_REPORT_SHA',file_digest(f))
    with pytest.raises(ValueError):p.load_checkpoint(root,root)


def test_regenerated_prefix_graph_or_coefficient_difference_is_not_accepted(checkpoint,monkeypatch):
    p,program,root,out,_=checkpoint;c=p.load_checkpoint(root,root)
    real=p.export_audio_native_onnx
    def changed(wrapper,example,path):
        real(wrapper,example,path);Path(path).write_bytes(Path(path).read_bytes()+b'changed')
    monkeypatch.setattr(p,'export_audio_native_onnx',changed)
    with pytest.raises(ValueError,match='graph'):
        p.resume(program,c,out,job_source='b'*40)
    assert not json.loads((out/'audio-raw-resume.json').read_text())['complete']


def test_completed_evidence_is_not_overwritten(checkpoint):
    p,program,root,out,_=checkpoint;c=p.load_checkpoint(root,root)
    p.resume(program,c,out,job_source='b'*40)
    with pytest.raises(ValueError,match='overwrite'):p.resume(program,c,out,job_source='b'*40)


def test_changed_original_weights_rejected_before_any_graph_export(checkpoint):
    p,program,root,out,_=checkpoint;c=p.load_checkpoint(root,root)
    program.s0.add_(1)
    with pytest.raises(ValueError,match='coefficients'):
        p.resume(program,c,out,job_source='b'*40)
    assert not out.exists()


def test_resumed_outputs_and_lineage_are_independently_audited(checkpoint):
    p,program,root,out,_=checkpoint
    p.resume(program,p.load_checkpoint(root,root),out,job_source='b'*40)
    r=p.audit_resume(out/'audio-raw-resume.json',root,root)
    assert r['original_tolerance_passed'] and r['values_compared']==8
    assert r['execution_sources']==['a'*40,'b'*40]
    assert r['prefix_graphs_reexported_and_bit_identity_verified']


@pytest.mark.parametrize('change',['summary','carry','prefix_graph','job_source'])
def test_resumed_auditor_rejects_forged_metrics_and_broken_lineage(checkpoint,change):
    p,program,root,out,_=checkpoint
    p.resume(program,p.load_checkpoint(root,root),out,job_source='b'*40)
    f=out/'audio-raw-resume.json';r=json.loads(f.read_text())
    if change=='summary':r['metrics']['mismatched_elements']=123
    if change=='carry':r['parts'][0]['input_sha256']='0'*64
    if change=='prefix_graph':r['verified_prefix_graphs'][0]['sha256']='0'*64
    if change=='job_source':
        r['job_source']='a'*40
        for part in r['parts']:part['execution_source']='a'*40
    f.write_text(json.dumps(r))
    with pytest.raises(ValueError):p.audit_resume(f,root,root)


def test_explicit_resume_job_fetches_real_prediction_and_runs_only_one_bounded_job():
    path=Path(__file__).resolve().parents[2]/'.github/workflows/frozen-audio-prefix-onnx.yml'
    w=yaml.safe_load(path.read_text());inputs=w.get('on',w.get(True))['workflow_dispatch']['inputs']
    assert inputs['resume_noise']['default'] is False
    job=w['jobs']['resume_noise'];assert job['timeout-minutes']==20 and 'strategy' not in job
    text=json.dumps(job)
    assert '38825780838' in text and 'e99c44214ab7d7f526b0fede5b561862460f1d66-noise' in text
    assert '11814567883' in text and 'audio_raw_resume' in text
    assert 'frozen-stage-reference' not in text
    for name in ('probe','matrix_probe','native_context_probe'):
        assert '!inputs.resume_noise' in w['jobs'][name]['if']
