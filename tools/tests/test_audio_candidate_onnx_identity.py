import importlib
import json

import pytest
from test_onnx_computation_identity import model,write
from embeddinggemma_tools.onnx_computation_identity import fingerprint

PROFILE='frames-1119-valid-1119'


def module():
    return importlib.import_module('embeddinggemma_tools.audio_candidate_onnx_identity')


@pytest.fixture
def baseline(tmp_path,monkeypatch):
    path=tmp_path/'original.onnx';sha=write(path,model());original=fingerprint(path,expected_sha256=sha)
    m=module();monkeypatch.setattr(m,'baseline',lambda:dict(profiles={PROFILE:original}))
    return m,path,sha,original


def test_exact_computation_is_separate_from_numeric_and_gpu_acceptance(baseline):
    m,path,sha,_=baseline
    report=m.inspect_native(path,expected_sha256=sha,profile=PROFILE,source='a'*40)
    assert report['complete'] is True and report['computation_identity_matches_original'] is True
    assert report['changed_components']==[]
    assert report['model_executed'] is False and report['gpu_validated'] is False
    assert report['generation_qualified'] is False


def test_changed_weight_is_classified_without_accepting_the_candidate(baseline,tmp_path):
    m,_,_,_=baseline;value=model();value.graph.initializer[0].float_data[0]=0.5
    path=tmp_path/'changed.onnx';sha=write(path,value)
    report=m.inspect_native(path,expected_sha256=sha,profile=PROFILE,source='a'*40)
    assert report['computation_identity_matches_original'] is False
    assert 'initializer_values_sha256' in report['changed_components']
    assert report['component_identity']['model_header_sha256'] is True


def test_matching_values_with_changed_names_and_edges_do_not_count_as_identity(baseline,tmp_path):
    m,_,_,_=baseline;value=model();value.graph.initializer[0].name='renamed';value.graph.node[0].input[1]='renamed'
    path=tmp_path/'renamed.onnx';sha=write(path,value)
    report=m.inspect_native(path,expected_sha256=sha,profile=PROFILE,source='a'*40)
    assert report['component_identity']['initializer_values_sha256'] is True
    assert report['computation_identity_matches_original'] is False


def test_mismatched_model_is_a_failed_cli_with_an_honest_completed_report(baseline,tmp_path):
    m,_,_,_=baseline;value=model();value.graph.node[0].op_type='Sub'
    path=tmp_path/'changed.onnx';sha=write(path,value);output=tmp_path/'result.json'
    result=m.main(['--input',str(path),'--model-sha256',sha,'--profile',PROFILE,'--source','a'*40,'--output',str(output)])
    report=json.loads(output.read_text())
    assert result==1 and report['complete'] is True and report['computation_identity_matches_original'] is False
    assert report['gpu_validated'] is False


def test_cli_never_overwrites_model_or_existing_report(baseline):
    m,path,sha,_=baseline;before=path.read_bytes()
    result=m.main(['--input',str(path),'--model-sha256',sha,'--profile',PROFILE,'--source','a'*40,'--output',str(path)])
    assert result==1 and path.read_bytes()==before


def test_changed_baseline_file_fails_closed(tmp_path,monkeypatch):
    m=module();path=tmp_path/'baseline.json';path.write_text('{}')
    monkeypatch.setattr(m,'BASELINE_PATH',path)
    with pytest.raises(ValueError,match='baseline'):m.baseline()


@pytest.mark.parametrize('source',['main','a'*39,'A'*40])
def test_execution_source_is_fixed_before_reading_models(baseline,source):
    m,path,sha,_=baseline
    with pytest.raises(ValueError,match='source'):m.inspect_native(path,expected_sha256=sha,profile=PROFILE,source=source)


@pytest.mark.parametrize('problem',[None,'digest','source','missing','audit','graph'])
def test_fetch_binds_actual_graph_to_reviewed_producer_audit(tmp_path,monkeypatch,problem):
    import hashlib,io,zipfile
    from embeddinggemma_tools import artifact_zip
    m=module();path=tmp_path/'tiny.onnx';sha=write(path,model());graph=path.read_bytes()
    audit=json.dumps(dict(metadata=dict(source_commit='a'*40,model_revision='c'*40),native_reference_complete=True,
        files={PROFILE+'/audio.onnx':dict(bytes=len(graph),sha256=sha)})).encode()
    contract=dict(artifact_id=1,artifact_sha256='b'*64,producer_run=2,producer_source='a'*40,
        model_revision='c'*40,audit_sha256=hashlib.sha256(audit).hexdigest())
    if problem=='source':contract['producer_source']='0'*40
    if problem=='audit':audit+=b'changed'
    if problem=='graph':graph+=b'changed'
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w') as archive:
        archive.writestr('audio-runtime-reference/audio-encoder-validation.json',audit)
        archive.writestr('audio-runtime-reference/'+PROFILE+'/audio.onnx',graph)
        archive.writestr('audio-runtime-reference/unrequested.pt2',b'not selected')
    data=buffer.getvalue();selected=[];real=artifact_zip.extract_members
    source=tmp_path/'source.zip';source.write_bytes(data)
    contract['artifact_sha256']=hashlib.sha256(data).hexdigest()
    if problem=='digest':contract['artifact_sha256']='0'*64
    if problem=='missing':source.unlink()
    def extract(reader,output,names,**kwargs):
        assert problem not in ('digest','missing'),'Unverified archive reached extraction'
        selected.extend(names);return real(reader,output,names,**kwargs)
    monkeypatch.setattr(m.candidate,'candidate_contract',lambda:contract)
    monkeypatch.setattr(artifact_zip,'github_artifact_reader',lambda *a:pytest.fail('Retired hosted source was used'))
    monkeypatch.setattr(artifact_zip,'extract_members',extract)
    if problem:
        with pytest.raises(ValueError):m.fetch_native(tmp_path/'download',profile=PROFILE,source_archive=source)
    else:
        result,actual,provenance=m.fetch_native(tmp_path/'download',profile=PROFILE,source_archive=source)
        assert result.read_bytes()==graph and actual==sha and provenance['candidate_producer_source']=='a'*40
        assert provenance['source_archive_transport']=='verified_local_archive' and not provenance['provider_metadata_revalidated']
        assert len(selected)==2 and not any(name.endswith('.pt2') for name in selected)
