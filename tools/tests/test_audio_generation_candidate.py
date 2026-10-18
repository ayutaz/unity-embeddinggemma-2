import copy
import hashlib
import importlib
import json

import numpy as np
import pytest
import torch
from test_audit_audio_raw_chain import evidence


def module():
    return importlib.import_module('embeddinggemma_tools.audio_generation_candidate')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def candidate(tmp_path, monkeypatch):
    m=module()
    original=tmp_path/'original';current=tmp_path/'candidate'
    original.mkdir();current.mkdir()
    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.wide=torch.nn.LayerNorm(128,eps=1e-6,bias=False)
            self.narrow=torch.nn.LayerNorm(32,eps=1e-6,bias=False)
        def forward(self,x):
            wide=torch.ops.aten.native_layer_norm.default(x,[128],self.wide.weight,None,1e-6)[0]
            return torch.ops.aten.native_layer_norm.default(wide[...,:32],[32],self.narrow.weight,None,1e-6)[0]
    inputs=torch.randn(2,128,generator=torch.Generator().manual_seed(19))
    graph=current/'audio.pt2'
    torch.export.save(torch.export.export(Encoder(),(inputs,)),graph)
    raw=current/'input.bin';inputs.numpy().astype('<f4').tofile(raw)
    metadata={'source_commit':m.PRODUCER,'model_revision':m.REVISION,'feature_atol':1e-5,'feature_rtol':1e-4,
              'versions':{'torch':str(torch.__version__).split('+')[0]},'input_sha256':{'model.safetensors':'d'*64}}
    reference={'metadata':metadata,'cases':[{'id':'silence','profile':'full','input_features':'input.bin','features':np.ones((2,32)).tolist()}],
               'profiles':{'full':{'native_graph':'audio.pt2','feature_shape':[2,128],'feature_count':2,'feature_width':32}}}
    audit={'metadata':metadata,'native_reference_complete':True,'files':{'input.bin':{'bytes':raw.stat().st_size,'sha256':sha(raw)},
           'audio.pt2':{'bytes':graph.stat().st_size,'sha256':'b'*64}}}
    for name,value in [('audio-encoder-reference.json',reference),('audio-encoder-validation.json',audit)]:
        (original/name).write_text(json.dumps(value),encoding='utf-8')
    fresh=copy.deepcopy(audit);fresh['metadata']['source_commit']='a'*40
    fresh['files']['audio.pt2']['sha256']=sha(graph)
    (current/'audio-encoder-validation.json').write_text(json.dumps(fresh),encoding='utf-8')
    new_reference=copy.deepcopy(reference);new_reference['metadata']=fresh['metadata']
    (current/'audio-encoder-reference.json').write_text(json.dumps(new_reference),encoding='utf-8')
    contract={'schema':1,'generation':'audio-test','producer_source':'a'*40,'producer_run':1,'artifact_id':1,'artifact_sha256':'e'*64,
              'model_revision':m.REVISION,'source_tree':'a'*40,'dependency_lock_blob':'c'*40,'qualified':False,
              'audit_sha256':sha(current/'audio-encoder-validation.json'),'reference_sha256':sha(current/'audio-encoder-reference.json'),
              'graphs':{'audio.pt2':fresh['files']['audio.pt2']}}
    path=tmp_path/'contract.json';path.write_text(json.dumps(contract),encoding='utf-8')
    monkeypatch.setattr(m,'CONTRACT_PATH',path);monkeypatch.setattr(m,'CONTRACT_SHA',sha(path))
    monkeypatch.setattr(m,'REFERENCE_SHA',sha(original/'audio-encoder-reference.json'))
    monkeypatch.setattr(m,'AUDIT_SHA',sha(original/'audio-encoder-validation.json'))
    return m,current,original,contract,path


def test_candidate_loads_actual_graph_but_keeps_original_golden(candidate):
    m,current,original,contract,_=candidate
    graph,values,expected,provenance=m.load_candidate_input(current,original,case_id='silence')
    assert torch.equal(expected,torch.ones(2,32))
    assert not torch.equal(graph(values),expected)
    assert provenance['model_generation']=='audio-test'
    assert provenance['graph_sha256']==contract['graphs']['audio.pt2']['sha256']
    assert provenance['reference_source']==m.PRODUCER
    assert provenance['graph_producer']=='a'*40
    assert provenance['original_graph_bytes'] is False


@pytest.mark.parametrize('target',['audio.pt2','input.bin','audio-encoder-validation.json','audio-encoder-reference.json'])
def test_changed_candidate_bytes_are_rejected_before_deserialization(candidate,monkeypatch,target):
    m,current,original,_,_=candidate
    with (current/target).open('ab') as f:f.write(b'changed')
    monkeypatch.setattr(torch.export,'load',lambda *a:pytest.fail('untrusted graph was deserialized'))
    with pytest.raises(ValueError):m.load_candidate_input(current,original,case_id='silence')


def test_changed_original_is_not_replaced_by_candidate_reference(candidate,monkeypatch):
    m,current,original,_,_=candidate
    (original/'audio-encoder-reference.json').write_text('{}',encoding='utf-8')
    monkeypatch.setattr(torch.export,'load',lambda *a:pytest.fail('changed golden reached deserialization'))
    with pytest.raises(ValueError):m.load_candidate_input(current,original,case_id='silence')


def test_unknown_case_never_falls_back_to_another_signal(candidate):
    m,current,original,_,_=candidate
    with pytest.raises(ValueError):m.load_candidate_input(current,original,case_id='missing')


def test_changed_contract_does_not_authorize_a_new_model(candidate):
    m,current,original,contract,path=candidate
    contract['graphs']['audio.pt2']['sha256']='f'*64
    path.write_text(json.dumps(contract),encoding='utf-8')
    with pytest.raises(ValueError):m.load_candidate_input(current,original,case_id='silence')


def test_candidate_cli_is_explicit_raw_input_only(monkeypatch,tmp_path):
    from embeddinggemma_tools import frozen_audio_chained_onnx as chain
    monkeypatch.setenv('GITHUB_SHA','a'*40)
    with pytest.raises(ValueError,match='raw'):
        chain.main(['--input',str(tmp_path),'--candidate-directory',str(tmp_path),
                    '--output',str(tmp_path/'out'),'--frozen-stage-source',str(tmp_path)])


@pytest.fixture
def numeric_candidate(evidence,monkeypatch):
    audit,original,path,report,expected=evidence
    m=module();contract={'generation':'audio-test','producer_source':'a'*40,'graphs':{'audio.pt2':{'sha256':'e'*64}}}
    monkeypatch.setattr(m,'candidate_contract',lambda:contract)
    monkeypatch.setattr(m,'CONTRACT_SHA','f'*64)
    for key in ('provenance','input_provenance','target_provenance'):
        report[key]=dict(report[key],graph_sha256='e'*64,model_generation='audio-test',
            candidate_contract_sha256='f'*64,graph_producer='a'*40,original_graph_bytes=False)
    report['cases'][1]['variant']='ordered_original_midpoint_bmm'
    report['native_control_context']={'mkl_cbwr':'COMPATIBLE','mkl_dynamic':'FALSE','threads':4}
    chain=path.parent/'native-chain-output-fp32.bin';expected.tofile(chain)
    report['native_chain_control_sha256']=sha(chain)
    path.write_text(json.dumps(report),encoding='utf-8')
    return m,audit,original,path,report


def test_candidate_audit_retains_every_original_numeric_check_without_promoting(numeric_candidate):
    m,audit,original,path,_=numeric_candidate
    with pytest.raises(ValueError):audit.audit_report(path,original,expected_variant='ordered_original_midpoint_bmm')
    result=m.audit_candidate_report(path,original)
    assert result['original_tolerance_passed'] and result['all_capture_and_carry_hashes_verified']
    assert result['model_generation']=='audio-test'
    assert result['native_chain_control_independently_verified'] is True
    assert result['generation_qualified'] is False and result['gpu_validated'] is False


@pytest.mark.parametrize('field,value',[('graph_sha256','b'*64),('model_generation','audio-other'),
    ('candidate_contract_sha256','0'*64),('graph_producer','0'*40),('original_graph_bytes',True)])
def test_candidate_audit_rejects_mixed_or_forged_generation(numeric_candidate,field,value):
    m,_,original,path,report=numeric_candidate
    report['target_provenance'][field]=value
    path.write_text(json.dumps(report),encoding='utf-8')
    with pytest.raises(ValueError):m.audit_candidate_report(path,original)


def test_candidate_full_vs_cut_control_is_checked_from_actual_bytes(numeric_candidate):
    m,_,original,path,report=numeric_candidate
    control=path.parent/'native-chain-output-fp32.bin'
    np.zeros((2,3),dtype='<f4').tofile(control)
    report['native_chain_control_sha256']=sha(control)
    path.write_text(json.dumps(report),encoding='utf-8')
    with pytest.raises(ValueError,match='control'):m.audit_candidate_report(path,original)


@pytest.mark.parametrize('key,value',[('mkl_cbwr',None),('mkl_dynamic','TRUE'),('threads',1)])
def test_candidate_requires_the_existing_native_execution_controls(numeric_candidate,key,value):
    m,_,original,path,report=numeric_candidate
    report['native_control_context'][key]=value
    path.write_text(json.dumps(report),encoding='utf-8')
    with pytest.raises(ValueError,match='control'):m.audit_candidate_report(path,original)


@pytest.mark.parametrize('problem',[None,'missing','duplicate','source','values','generation'])
def test_candidate_summary_requires_the_full_original_matrix(monkeypatch,problem):
    m=module();names=['silence','impulse','sine','nyquist','noise','padded_noise']
    rows=[dict(case_id=name,profile='frames-1119-valid-839' if name=='padded_noise' else 'frames-1119-valid-1119',
        source='a'*40,original_tolerance_passed=True,values_compared=(210 if name=='padded_noise' else 280)*512,
        candidate_metrics={'mismatched_elements':0,'exact_difference_count':3},native_context_qualified=False,
        model_generation='audio-test',candidate_contract_sha256='f'*64) for name in names]
    if problem=='missing':rows.pop()
    if problem=='duplicate':rows[-1]=rows[0]
    if problem=='source':rows[-1]['source']='b'*40
    if problem=='values':rows[-1]['values_compared']+=1
    if problem=='generation':rows[-1]['model_generation']='audio-other'
    monkeypatch.setattr(m,'candidate_contract',lambda:{'generation':'audio-test'})
    monkeypatch.setattr(m,'CONTRACT_SHA','f'*64)
    monkeypatch.setattr(m,'audit_candidate_report',lambda index,original:rows[index])
    if problem:
        with pytest.raises(ValueError):m.audit_candidate_reports(list(range(len(rows))),'unused',source='a'*40)
    else:
        report=m.audit_candidate_reports(list(range(6)),'unused',source='a'*40)
        assert report['success'] and report['values_compared']==824320
        assert not report['generation_qualified'] and not report['gpu_validated']


def test_candidate_fetch_selects_one_profile_and_original_bytes_only(monkeypatch,tmp_path):
    m=module();contract=m.candidate_contract();calls=[]
    from embeddinggemma_tools import artifact_zip
    class Reader:
        def __enter__(self):return self
        def __exit__(self,*args):pass
    def reader(repo,artifact):
        assert repo=='ayutaz/unity-embeddinggemma-2'
        return Reader(),{'id':artifact,'digest':'sha256:'+contract['artifact_sha256'],
                         'workflow_run':{'id':contract['producer_run'],'head_sha':contract['producer_source'],
                                         'repository_id':1,'head_repository_id':1}}
    monkeypatch.setattr(artifact_zip,'github_artifact_reader',reader)
    def extract(reader,out,members,**kwargs):
        out.mkdir(parents=True,exist_ok=True);calls.append(members)
        return {'selected_count':len(members)}
    monkeypatch.setattr(artifact_zip,'extract_members',extract)
    monkeypatch.setattr(m,'fetch_original',lambda out:calls.append(['original_golden_only']))
    monkeypatch.setattr(m,'_candidate_input_files',lambda *args,**kw:None)
    m.fetch_candidate(tmp_path/'candidate',tmp_path/'original',case_id='padded_noise')
    assert calls[0]==['original_golden_only']
    selected=calls[1]
    assert len(selected)==4
    assert 'audio-runtime-reference/frames-1119-valid-839/audio.pt2' in selected
    assert 'audio-runtime-reference/frames-1119-valid-839/padded_noise-features-fp32.bin' in selected
    assert not any('1119-valid-1119' in item or item.endswith('.onnx') for item in selected)


@pytest.mark.parametrize('problem',['digest','run','source','fork'])
def test_candidate_fetch_rejects_wrong_remote_provenance(monkeypatch,tmp_path,problem):
    m=module();contract=m.candidate_contract()
    from embeddinggemma_tools import artifact_zip
    class Reader:
        def __enter__(self):return self
        def __exit__(self,*args):pass
    metadata={'id':contract['artifact_id'],'digest':'sha256:'+contract['artifact_sha256'],
              'workflow_run':{'id':contract['producer_run'],'head_sha':contract['producer_source'],
                              'repository_id':1,'head_repository_id':1}}
    if problem=='digest':metadata['digest']='sha256:'+'0'*64
    if problem=='run':metadata['workflow_run']['id']+=1
    if problem=='source':metadata['workflow_run']['head_sha']='0'*40
    if problem=='fork':metadata['workflow_run']['head_repository_id']=2
    monkeypatch.setattr(artifact_zip,'github_artifact_reader',lambda *a:(Reader(),metadata))
    monkeypatch.setattr(artifact_zip,'extract_members',lambda *a,**kw:pytest.fail('untrusted artifact was extracted'))
    monkeypatch.setattr(m,'fetch_original',lambda out:None)
    with pytest.raises(ValueError):m.fetch_candidate(tmp_path/'candidate',tmp_path/'original',case_id='silence')
