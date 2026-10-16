import hashlib
import json
from pathlib import Path

import pytest
import yaml

from embeddinggemma_tools import pinned_reference_archive as archive


def artifact(identifier, *, backup=True, expired=False, expires='2027-01-01T00:00:00Z'):
    return dict(id=identifier, name=archive.BACKUP_NAME if backup else 'original',
        expired=expired, expires_at=expires, created_at=f'2026-10-{identifier:02d}T00:00:00Z',
        workflow_run=dict(id=100+identifier,repository_id=1,head_repository_id=1))


def test_verified_backup_is_preferred_even_before_original_expires():
    old=artifact(1,backup=False)
    assert archive.select_artifact([artifact(2),artifact(3)],old)['id']==3


def test_original_fallback_and_expiry_fail_closed():
    old=artifact(1,backup=False)
    assert archive.select_artifact([artifact(2,expired=True)],old)['id']==1
    with pytest.raises(ValueError):
        archive.select_artifact([artifact(2,expired=True)],dict(old,expired=True))


@pytest.mark.parametrize('change',['wrong_name','fork','expired_timestamp'])
def test_untrusted_or_stale_backup_is_not_selected(change):
    bad=artifact(2)
    if change=='wrong_name':bad['name']='unrelated-model'
    if change=='fork':bad['workflow_run']['head_repository_id']=2
    if change=='expired_timestamp':bad['expires_at']='2026-01-01T00:00:00Z'
    assert archive.select_artifact([bad],artifact(1,backup=False))['id']==1


@pytest.fixture
def original(tmp_path,monkeypatch):
    metadata=dict(source_commit=archive.PRODUCER,model_revision=archive.REVISION,
                  feature_atol=1e-5,feature_rtol=1e-4)
    (tmp_path/'graph.pt2').write_bytes(b'unchanged-frozen-original-graph')
    (tmp_path/'input.bin').write_bytes(b'unchanged-input')
    def description(p):return dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    reference=dict(metadata=metadata,profiles={name:{} for name in archive.PROFILES},
                   cases=[dict(id=name) for name in archive.CASES])
    ref=tmp_path/'audio-encoder-reference.json';ref.write_text(json.dumps(reference),encoding='utf-8')
    audit=dict(metadata=metadata,native_reference_complete=True,
               files={name:description(tmp_path/name) for name in ('graph.pt2','input.bin','audio-encoder-reference.json')})
    val=tmp_path/'audio-encoder-validation.json';val.write_text(json.dumps(audit),encoding='utf-8')
    monkeypatch.setattr(archive,'REFERENCE_SHA',description(ref)['sha256'])
    monkeypatch.setattr(archive,'AUDIT_SHA',description(val)['sha256'])
    return tmp_path


def test_copy_verification_checks_original_bytes_without_model_execution(original):
    result=archive.verify_original(original)
    assert result['success'] and result['audited_file_count']==3
    assert result['reference_sha256']==archive.REFERENCE_SHA
    assert result['gpu_validated'] is False and result['models_executed'] is False


@pytest.mark.parametrize('name',['graph.pt2','input.bin','audio-encoder-reference.json','audio-encoder-validation.json'])
def test_corrupt_reference_graph_or_input_cannot_be_preserved_as_valid(original,name):
    p=original/name;p.write_bytes(p.read_bytes()+b'corrupt')
    with pytest.raises(ValueError):archive.verify_original(original)


def test_resolve_cli_records_immutable_id_and_run_for_actions(tmp_path,monkeypatch):
    monkeypatch.setattr(archive,'resolve_artifact',lambda:artifact(3))
    output=tmp_path/'selection.json';github_output=tmp_path/'output.txt'
    monkeypatch.setenv('GITHUB_OUTPUT',str(github_output))
    assert archive.main(['resolve','--output',str(output)])==0
    assert json.loads(output.read_text())['id']==3
    assert github_output.read_text()=='artifact-id=3\nrun-id=103\n'


def test_preservation_workflow_uploads_only_after_original_hash_verification():
    root=Path(__file__).resolve().parents[2]/'.github/workflows'
    w=yaml.load((root/'frozen-audio-prefix-onnx.yml').read_text(encoding='utf-8'),Loader=yaml.BaseLoader)
    assert w['on']['workflow_dispatch']['inputs']['preserve_original']['default']=='false'
    job=w['jobs']['preserve_original'];steps=job['steps']
    assert 'inputs.preserve_original' in job['if']
    verify=next(i for i,s in enumerate(steps) if 'pinned_reference_archive verify' in s.get('run',''))
    upload=next(i for i,s in enumerate(steps) if 'actions/upload-artifact@' in s.get('uses',''))
    assert verify<upload and steps[upload]['with']['retention-days']=='90'
    assert not steps[upload].get('if') and not job.get('continue-on-error')
    for name in ('probe','resume_noise','native_context_probe','matrix_probe'):
        assert '!inputs.preserve_original' in w['jobs'][name]['if']
