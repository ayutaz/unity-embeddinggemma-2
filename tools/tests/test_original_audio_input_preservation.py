import importlib
import json

import pytest
import torch
from test_audio_generation_candidate import candidate
from test_audio_candidate_distribution import inputs, sha


def module():
    return importlib.import_module('embeddinggemma_tools.original_audio_input_preservation')


def test_preserves_exact_goldens_and_all_six_inputs_without_loading_any_model(inputs,tmp_path,monkeypatch):
    current,original=inputs
    for path in current.rglob('*.pt2'): path.unlink()
    monkeypatch.setattr(torch.export,'load',lambda *_:pytest.fail('Preservation must not load models'))
    result=module().preserve(current,original,tmp_path/'out',source='c'*40)
    assert result['case_count']==6 and result['profile_count']==2
    assert len(result['files'])==8
    assert result['models_included'] is False and result['model_executed'] is False
    assert result['gpu_validated'] is False and result['generation_qualified'] is False
    for name,spec in result['files'].items():
        assert sha(tmp_path/'out'/name)==spec['sha256']
        source=original/name if name.endswith('.json') else current/name
        assert (tmp_path/'out'/name).read_bytes()==source.read_bytes()
    assert json.loads((tmp_path/'out/input-preservation.json').read_text())==result
    assert not list((tmp_path/'out').rglob('*.pt2'))


@pytest.mark.parametrize('name',['noise.bin','padded_noise.bin'])
def test_modified_raw_input_prevents_manifest_publication(inputs,tmp_path,name):
    current,original=inputs
    with (current/name).open('ab') as stream: stream.write(b'changed')
    with pytest.raises(ValueError): module().preserve(current,original,tmp_path/'out',source='c'*40)
    assert not (tmp_path/'out/input-preservation.json').exists()


def test_original_golden_is_not_replaced_by_a_fresh_reference(inputs,tmp_path):
    current,original=inputs
    (original/'audio-encoder-reference.json').write_text('{}',encoding='utf-8')
    with pytest.raises(ValueError): module().preserve(current,original,tmp_path/'out',source='c'*40)
    assert not (tmp_path/'out/input-preservation.json').exists()


def test_existing_output_is_not_overwritten(inputs,tmp_path):
    output=tmp_path/'out';output.mkdir();(output/'keep').write_text('existing')
    with pytest.raises(ValueError,match='empty'): module().preserve(*inputs,output,source='c'*40)
    assert (output/'keep').read_text()=='existing'


@pytest.mark.parametrize('source',['main','c'*39,'C'*40])
def test_source_identity_must_be_immutable(inputs,tmp_path,source):
    with pytest.raises(ValueError,match='source'): module().preserve(*inputs,tmp_path/'out',source=source)


def test_fetch_selects_only_six_original_raw_inputs_and_checks_artifact_identity(inputs,tmp_path,monkeypatch):
    import io
    import zipfile
    from embeddinggemma_tools import artifact_zip
    m=module();current,original=inputs
    reference=json.loads((original/'audio-encoder-reference.json').read_text())
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w') as archive:
        for row in reference['cases']:
            archive.writestr('audio-runtime-reference/'+row['input_features'],(current/row['input_features']).read_bytes())
        archive.writestr('audio-runtime-reference/audio.pt2',b'unrequested-model')
    data=buffer.getvalue();selected=[]
    def reader(repo,artifact_id):
        assert repo=='ayutaz/unity-embeddinggemma-2' and artifact_id==1
        return artifact_zip.RangeReader(len(data),lambda start,stop:data[start:stop]),dict(
            digest='sha256:'+'e'*64,workflow_run=dict(id=1,head_sha='a'*40,repository_id=7,head_repository_id=7))
    real=artifact_zip.extract_members
    def extract(reader,output,names,**kwargs):
        selected.extend(names);return real(reader,output,names,**kwargs)
    monkeypatch.setattr(m.candidate,'fetch_original',lambda _:None)
    monkeypatch.setattr(artifact_zip,'github_artifact_reader',reader)
    monkeypatch.setattr(artifact_zip,'extract_members',extract)
    output=tmp_path/'download';m.fetch_inputs(output,original)
    assert len(selected)==6 and all(name.endswith('.bin') for name in selected)
    assert not list(output.rglob('*.pt2'))
    for row in reference['cases']:
        assert sha(output/'audio-runtime-reference'/row['input_features'])==sha(current/row['input_features'])


def test_fetch_rejects_mismatched_source_before_extracting_any_input(inputs,tmp_path,monkeypatch):
    from embeddinggemma_tools import artifact_zip
    m=module();_,original=inputs
    monkeypatch.setattr(m.candidate,'fetch_original',lambda _:None)
    monkeypatch.setattr(artifact_zip,'github_artifact_reader',lambda *_:(
        artifact_zip.RangeReader(1,lambda start,stop:b'x'[start:stop]),
        dict(digest='sha256:'+'e'*64,workflow_run=dict(id=1,head_sha='b'*40,repository_id=7,head_repository_id=7))))
    monkeypatch.setattr(artifact_zip,'extract_members',lambda *a,**k:pytest.fail('Mismatched artifact reached extraction'))
    with pytest.raises(ValueError,match='source'):m.fetch_inputs(tmp_path/'download',original)
