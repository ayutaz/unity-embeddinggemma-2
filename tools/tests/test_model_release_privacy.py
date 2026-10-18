"""Known model distributions require exact producer/review bindings, not a suffix exception."""
import hashlib
import importlib
import json
from pathlib import Path
import struct
import subprocess

import pytest


def module(): return importlib.import_module('embeddinggemma_tools.model_release_privacy')
def sha(raw): return hashlib.sha256(raw).hexdigest()
def canonical(value): return sha(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode())


def reviewed(tmp_path, monkeypatch, *, text='public tensor name', large=False, suffix='sentis'):
    root=tmp_path/'distribution'; (root/'objects').mkdir(parents=True)
    if suffix=='sentis':
        raw=text.encode(); description=struct.pack('<I',len(raw))+raw+b'\0'
        payload=struct.pack('<i',len(description))+description+b'\0'*16
        if large: payload+=b'\0'*(17*1024*1024)
    elif suffix=='bin': payload=struct.pack('<2f',.25,-.5)
    else: payload=json.dumps({'public':'tokenizer'}).encode()
    name='objects/'+sha(payload)+'.'+suffix; path=root/name; path.write_bytes(payload)
    descriptor=dict(schema_version=1,backend='GPUCompute',model_revision='914f7f89142e33e77833254d9c9b90c3cef7303b',
                    source_commit='a'*40,object_layout='flat',files={name:dict(bytes=len(payload),sha256=sha(payload))})
    raw=json.dumps(descriptor).encode(); (root/'bundle.json').write_bytes(raw)
    inventory=dict(schema=1,producer='a'*40,descriptor_sha256=sha(raw),object_layout='flat',
                   objects=[dict(path=name,asset_name=path.name,bytes=len(payload),sha256=sha(payload))])
    raw_inventory=json.dumps(inventory).encode(); (root/'release-assets.json').write_bytes(raw_inventory)
    proof=dict(schema=1,review='model-description+source-provenance',success=True,
               descriptor_sha256=sha(raw),inventory_sha256=sha(raw_inventory),object_count=1,
               all_source_and_destination_hashes_verified=True,all_selected_module_bindings_unchanged=True,
               contains_private_metadata=False,images_in_distribution=False)
    record=dict(inventory_sha256=sha(raw_inventory),review_path='docs/results/reviewed-model-test.json',
                review_canonical_sha256=canonical(proof),producer='a'*40)
    policy=dict(schema=1,distributions={sha(raw):record})
    documents={'.privacy-reviewed-model-releases.json':policy,record['review_path']:proof}
    m=module(); monkeypatch.setattr(m,'_versioned_json',lambda cwd,name:documents[name])
    return m,path,documents,root


@pytest.mark.parametrize('suffix',['sentis','bin','json'])
def test_exact_reviewed_model_or_coefficient_asset_can_be_checked_without_general_binary_approval(tmp_path,monkeypatch,suffix):
    m,path,_,_=reviewed(tmp_path,monkeypatch,suffix=suffix)
    assert m.reviewed_model_asset_safe(path,tmp_path)


def test_large_reviewed_model_is_stream_hashed_and_still_runs_its_description_privacy_check(tmp_path,monkeypatch):
    m,path,_,_=reviewed(tmp_path,monkeypatch,large=True)
    assert path.stat().st_size>16*1024**2
    assert m.reviewed_model_asset_safe(path,tmp_path)


@pytest.mark.parametrize('problem',['no-policy','no-proof','bad-proof','bad-inventory','modified-bytes','wrong-producer','unknown-file'])
def test_missing_changed_or_unreviewed_payloads_fail_closed(tmp_path,monkeypatch,problem):
    m,path,documents,root=reviewed(tmp_path,monkeypatch)
    policy=documents['.privacy-reviewed-model-releases.json']; record=next(iter(policy['distributions'].values()))
    if problem=='no-policy': documents.pop('.privacy-reviewed-model-releases.json')
    elif problem=='no-proof': documents.pop(record['review_path'])
    elif problem=='bad-proof': documents[record['review_path']]['contains_private_metadata']=True
    elif problem=='bad-inventory': (root/'release-assets.json').write_text('{}')
    elif problem=='modified-bytes': path.write_bytes(path.read_bytes()+b'\0')
    elif problem=='wrong-producer': record['producer']='b'*40
    elif problem=='unknown-file': path=root/'objects'/('0'*64+'.sentis'); path.write_bytes(b'unreviewed\0')
    assert not m.reviewed_model_asset_safe(path,tmp_path)


@pytest.mark.parametrize('kind',['path','email','credential'])
def test_even_a_hash_registered_model_with_private_description_is_rejected(tmp_path,monkeypatch,kind):
    text=('C:'+'/'+'Users'+'/'+'fixture-user'+'/private') if kind=='path' else ('fixture'+'@'+'private-domain.dev') if kind=='email' else ('ghp_'+'A'*36)
    m,path,_,_=reviewed(tmp_path,monkeypatch,text=text)
    assert not m.reviewed_model_asset_safe(path,tmp_path)


def test_model_review_never_allows_unreviewed_image_or_archive_bytes(tmp_path,monkeypatch):
    m,path,_,root=reviewed(tmp_path,monkeypatch)
    for name in ('capture.png','archive.zip'):
        foreign=root/name;foreign.write_bytes(path.read_bytes())
        assert not m.reviewed_model_asset_safe(foreign,tmp_path)


@pytest.mark.parametrize('large',[False,True])
def test_release_upload_hook_requires_the_exact_reviewed_model_payload(tmp_path,monkeypatch,large):
    from embeddinggemma_tools.privacy_hook import decide
    m,path,_,_=reviewed(tmp_path,monkeypatch,large=large)
    event=dict(tool_name='Bash',cwd=str(tmp_path),tool_input=dict(command='gh release upload models-preview '+path.relative_to(tmp_path).as_posix()))
    assert decide(event,lambda:True)['hookSpecificOutput']['permissionDecision']=='allow'
    path.write_bytes(path.read_bytes()+b'\0')
    assert decide(event,lambda:True)['hookSpecificOutput']['permissionDecision']=='deny'


def test_review_authority_is_committed_head_not_untracked_or_working_tree_policy(tmp_path,monkeypatch):
    m=module(); versioned=m._versioned_json
    _,path,documents,_=reviewed(tmp_path,monkeypatch)
    monkeypatch.setattr(m,'_versioned_json',versioned)
    repo=tmp_path/'review-repo';repo.mkdir()
    def git(*args):
        return subprocess.run(['git','-C',str(repo),*args],capture_output=True,check=True)
    git('init','-q')
    for name,value in documents.items():
        target=repo/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(value))
    assert not m.reviewed_model_asset_safe(path,repo)
    git('add','--','.privacy-reviewed-model-releases.json','docs/results/reviewed-model-test.json')
    git('-c','user.name=fixture','-c','user.email=123+fixture@users.noreply.github.com',
        'commit','-q','-m','Record synthetic model metadata review')
    assert m.reviewed_model_asset_safe(path,repo)
    (repo/'.privacy-reviewed-model-releases.json').write_text('{}')
    (repo/'docs/results/reviewed-model-test.json').write_text('{}')
    assert m.reviewed_model_asset_safe(path,repo)
    path.write_bytes(path.read_bytes()+b'\0')
    assert not m.reviewed_model_asset_safe(path,repo)
