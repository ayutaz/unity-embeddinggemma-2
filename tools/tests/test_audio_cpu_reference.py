import hashlib
import importlib
import importlib.metadata

import numpy as np
import pytest
import torch

from test_audio_cpu_cohort import cohort

pytest.importorskip('onnxruntime', reason='Install the cpu-comparison group for audio CPU reference execution')
SOURCE = 'a'*40


@pytest.fixture
def prepared(cohort, monkeypatch, tmp_path):
    driver = importlib.import_module('embeddinggemma_tools.audio_cpu_reference')
    _,pre,_,encoder,_,document,save = cohort
    snapshot = tmp_path/'snapshot'
    snapshot.mkdir()
    hashes = {}
    for name in document['metadata']['input_sha256']:
        raw = name.encode()
        (snapshot/name).write_bytes(raw)
        hashes[name] = hashlib.sha256(raw).hexdigest()
    encoder['metadata']['input_sha256'] = {name:hashes[name] for name in encoder['metadata']['input_sha256']}
    document['metadata']['input_sha256'] = hashes
    pre['metadata']['processor_sha256'] = hashes['processor_config.json']
    for row in encoder['cases']:
        row['features'] = [[1.0]*512 for _ in row['features']]
    roots = save()
    monkeypatch.setattr(driver.protocol,'AUDIO_REFERENCE',driver.references.DOCUMENT_REFERENCE_SHA256)
    monkeypatch.setattr(driver,'_validate_execution_source',lambda _: {'synthetic.py':'0'*64})
    monkeypatch.setattr(driver,'_admit_chain',lambda *_: None,raising=False)
    monkeypatch.setattr(driver,'cpu_execution_context',lambda:dict(CPU_capability='synthetic',torch_num_threads=4))
    monkeypatch.setattr(driver,'_peak_rss_bytes',lambda:1024)
    state = dict(pre_calls=0, encoder_calls=0, backbone_calls=0, closes=0,
                 preprocessing_offset=0.0, invalid_frame_mask=False, backbone_error=False)
    class Preprocessor(torch.nn.Module):
        def forward(self,wave,mask):
            state['pre_calls'] += 1
            valid = (int(mask.sum())-161)//160+1
            features = (wave.mean()+state['preprocessing_offset']).expand(1,1119,128).contiguous()
            actual_mask = (torch.arange(1119)[None] < valid).int()
            if state['invalid_frame_mask']: actual_mask[0,0]=0
            return features,actual_mask
    class Encoder(torch.nn.Module):
        def __init__(self,case):
            super().__init__()
            self.case=case
            self.configuration=dict(source=SOURCE,case_id=case.id,profile=case.profile,
                references_used_as_inference_inputs=False,chain_report_sha256='d'*64,
                original_reference_audit_passed=True,complete_graph_hashes_checked_at_load=True,
                execution_schedule='chain-major',maximum_owned_sessions=5,
                CPU_reference_comparison_qualified=False,GPU_validated=False,complete_performance_accepted=False,
                sessions=[dict(providers=['CPUExecutionProvider'],graph_sha256='e'*64,
                    node_execution_order='PRIORITY_BASED',graph_optimizations='disabled',execution_mode='sequential',
                    intra_op_threads=4,inter_op_threads=1,fallback_enabled=False,
                    cpu_memory_arena='disabled',memory_pattern='disabled',constant_prepacking='disabled',
                    onnxruntime_version=importlib.metadata.version('onnxruntime'))]*5)
        def forward(self,value):
            state['encoder_calls'] += 1
            return (value.mean()+1).expand(self.case.encoder_features.shape).contiguous()
        def close(self): state['closes'] += 1
    class Backbone(torch.nn.Module):
        def forward(self,ids,mask,image,audio):
            state['backbone_calls'] += 1
            if state['backbone_error']: raise RuntimeError('synthetic downstream failure')
            value = audio.new_zeros(768)
            value[0]=audio.mean()
            return value/value.norm()
    def modules(snapshot,encoder_root,chain_report,source,case,final_metadata,final_profile):
        return Preprocessor(),Encoder(case),Backbone()
    monkeypatch.setattr(driver,'_load_modules',modules)
    output=tmp_path/'output'
    def run(case_id='silence'):
        return driver.measure_audio_cpu_reference(snapshot,*roots,tmp_path/'chain.json',output,
            source=SOURCE,case_id=case_id)
    return driver,state,output,run,snapshot


@pytest.mark.parametrize('case_id',['silence','padded_noise'])
def test_actual_stage_values_are_carried_four_times_and_saved_after_stage_timers(prepared,case_id):
    driver,state,output,run,_=prepared
    result=run(case_id)
    assert result['success'] and result['phase']=='complete' and result['source']==SOURCE
    assert result['actual_backend']=='CPU' and result['precision']=='fp32'
    assert result['CPU_comparison_qualified'] is result['complete_performance_accepted'] is False
    assert state==dict(pre_calls=4,encoder_calls=4,backbone_calls=4,closes=1,
        preprocessing_offset=0.0,invalid_frame_mask=False,backbone_error=False)
    assert len(result['rounds'])==len(result['intermediate_inspections'])==4
    for row in result['rounds']:
        assert set(row['captures'])=={'features','frame_mask','encoder'}
        assert all((output/v['file']).is_file() for v in row['captures'].values())
    assert all(v['preprocessing_tolerance_failures']==v['encoder_tolerance_failures']==0
        and v['frame_mask_exact'] for v in result['intermediate_inspections'])
    journal=driver._json(output/'audio-cpu-source.json')
    assert journal['model_execution_complete'] and journal['phase']=='complete'
    pipeline=driver._json(output/'pipeline.json')
    assert pipeline['stage_observer_present'] and pipeline['stage_observer_time_excluded']
    assert all(v['stage_observer_wall_seconds']>=0 for v in pipeline['rounds'])


def test_final_vector_agreement_does_not_hide_invalid_actual_intermediates(prepared):
    driver,state,output,run,_=prepared
    state['preprocessing_offset']=1.0
    with pytest.raises(ValueError): run()
    result=driver._json(output/'audio-cpu-reference.json')
    assert result['success'] is False and result['final_vector_gate_passed'] is True
    assert all(v['preprocessing_tolerance_failures']>0 and v['encoder_tolerance_failures']>0
        for v in result['intermediate_inspections'])
    actual=np.fromfile(output/result['rounds'][0]['captures']['features']['file'],dtype='<f4')
    assert np.all(actual==1) and state['closes']==1


def test_actual_frame_mask_failure_preserves_partial_preprocessing_evidence(prepared):
    driver,state,output,run,_=prepared
    state['invalid_frame_mask']=True
    with pytest.raises(ValueError): run()
    result=driver._json(output/'audio-cpu-reference.json')
    assert not result['success'] and not result['model_execution_complete']
    assert set(result['rounds'][0]['captures'])=={'features','frame_mask'}
    assert state['encoder_calls']==state['backbone_calls']==0 and state['closes']==1


def test_downstream_failure_closes_encoder_and_preserves_failed_pipeline(prepared):
    driver,state,output,run,_=prepared
    state['backbone_error']=True
    with pytest.raises(ValueError): run()
    result=driver._json(output/'audio-cpu-reference.json')
    assert not result['success'] and result['error_class']=='RuntimeError'
    assert driver._json(output/'pipeline.json')['phase']=='failed' and state['closes']==1


def test_missing_or_changed_snapshot_is_rejected_before_model_load(prepared):
    _,state,_,run,snapshot=prepared
    (snapshot/'model.safetensors').write_bytes(b'changed')
    with pytest.raises(ValueError): run()
    assert state['pre_calls']==state['encoder_calls']==0


def test_context_drift_cannot_qualify_a_numerically_matching_measurement(prepared,monkeypatch):
    driver,state,output,run,_=prepared
    contexts=iter([dict(CPU_capability='before'),dict(CPU_capability='after')])
    monkeypatch.setattr(driver,'cpu_execution_context',lambda:next(contexts))
    with pytest.raises(ValueError): run()
    result=driver._json(output/'audio-cpu-reference.json')
    assert result['final_vector_gate_passed'] and not result['execution_context_stable'] and not result['success']
    assert state['closes']==1


def test_existing_output_is_not_overwritten_or_reexecuted(prepared):
    _,state,output,run,_=prepared
    run()
    original=(output/'audio-cpu-reference.json').read_bytes()
    with pytest.raises(ValueError): run()
    assert (output/'audio-cpu-reference.json').read_bytes()==original
    assert state['pre_calls']==4


@pytest.mark.parametrize('case_id',['silence','padded_noise'])
def test_separate_capture_audit_reloads_originals_and_recomputes_all_rounds(prepared,case_id):
    driver,_,output,run,snapshot=prepared
    run(case_id)
    base=snapshot.parent
    audit=driver.audit_audio_cpu_reference(output,base/'pre',base/'encoder',base/'document.json',source=SOURCE)
    assert audit['all_four_chains_match_originals'] is True
    assert audit['CPU_comparison_qualified'] is audit['complete_performance_accepted'] is False
    assert len(audit['intermediate_inspections'])==len(audit['final_inspections'])==4


@pytest.mark.parametrize('mutation',['encoder','features','frame_mask','final','journal','provider','source','missing'])
def test_capture_audit_rejects_tampering_even_with_updated_capture_hashes(prepared,mutation):
    import json
    driver,_,output,run,snapshot=prepared
    run()
    report=driver._json(output/'audio-cpu-reference.json')
    pipeline=driver._json(output/'pipeline.json')
    if mutation in ('encoder','features','frame_mask'):
        entry=report['rounds'][0]['captures'][mutation]
        path=output/entry['file']
        values=np.fromfile(path,dtype=entry['encoding'])
        values[0]=0 if mutation=='frame_mask' else 2
        path.write_bytes(values.tobytes())
        entry['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    if mutation=='final':
        path=output/pipeline['rounds'][0]['capture']
        values=np.zeros(768,dtype='<f4')
        values[1]=1
        path.write_bytes(values.tobytes())
        pipeline['rounds'][0]['capture_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
        (output/'pipeline.json').write_text(json.dumps(pipeline),encoding='utf-8')
    if mutation=='journal':
        journal=driver._json(output/'audio-cpu-source.json')
        journal['source']='b'*40
        (output/'audio-cpu-source.json').write_text(json.dumps(journal),encoding='utf-8')
    if mutation=='provider': report['encoder_configuration']['sessions'][0]['providers']=['CUDAExecutionProvider']
    if mutation=='source': report['source']='b'*40
    if mutation=='missing': (output/report['rounds'][0]['captures']['encoder']['file']).unlink()
    (output/'audio-cpu-reference.json').write_text(json.dumps(report),encoding='utf-8')
    base=snapshot.parent
    with pytest.raises(ValueError):
        driver.audit_audio_cpu_reference(output,base/'pre',base/'encoder',base/'document.json',source=SOURCE)


def cli_arguments(snapshot,output,command='measure'):
    base=snapshot.parent
    return [command,'--preprocessing',str(base/'pre'),'--encoder-reference',str(base/'encoder'),
        '--document-reference',str(base/'document.json'),'--chain-report',str(base/'chain.json'),
        '--case','silence','--output',str(output),'--snapshot',str(snapshot),'--source',SOURCE]


def test_cli_measure_then_separate_audit_has_path_free_output_and_no_overwrite(prepared,monkeypatch,capsys):
    driver,_,output,_,snapshot=prepared
    monkeypatch.setattr(driver.torch,'set_num_interop_threads',lambda _:None)
    assert driver.main(cli_arguments(snapshot,output))==0
    assert driver.main(cli_arguments(snapshot,output,'audit'))==0
    path=output/'independent-audio-CPU-audit.json'
    original=path.read_bytes()
    assert driver.main(cli_arguments(snapshot,output,'audit'))==1
    assert path.read_bytes()==original
    text=capsys.readouterr().out
    assert str(snapshot.parent) not in text


def test_original_chain_failure_prevents_model_execution(prepared,monkeypatch):
    driver,state,_,run,_=prepared
    def fail(*_): raise ValueError('synthetic audit rejection')
    monkeypatch.setattr(driver,'_admit_chain',fail,raising=False)
    with pytest.raises(ValueError):run()
    assert state['pre_calls']==state['encoder_calls']==state['closes']==0


def test_cli_invalid_original_reference_fails_before_any_snapshot_download(prepared,monkeypatch,capsys):
    driver,_,output,_,snapshot=prepared
    (snapshot.parent/'document.json').write_bytes(b'changed')
    def unexpected(*_): pytest.fail('snapshot requested before original input admission')
    monkeypatch.setattr(driver,'_download_snapshot',unexpected,raising=False)
    args=cli_arguments(snapshot,output)
    snapshot_flag=args.index('--snapshot')
    del args[snapshot_flag:snapshot_flag+2]
    assert driver.main(args)==1
    assert str(snapshot.parent) not in capsys.readouterr().out
