"""Exact native-regression assets need committed reviews and independent metadata checks."""
import hashlib
import importlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import zipfile

import pytest


def sha(raw): return hashlib.sha256(raw).hexdigest()
def canonical(value): return sha(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode())
def module(): return importlib.import_module('embeddinggemma_tools.native_regression_release_privacy')
NAMES = ['candidate-1119.pt2', 'candidate-839.pt2', 'original-reference.json', 'original-audit.json',
         'candidate-reference.json', 'candidate-audit.json'] + ['input-'+s+'.bin' for s in ('silence','impulse','sine','nyquist','noise','padded_noise')]


def reviewed(tmp_path, monkeypatch, *, model_extra=None, other_text='1', duplicate=False, json_value='reference'):
    root=tmp_path/'release';root.mkdir()
    sources=Path(__file__).parents[1]/'references'
    contract=json.loads((sources/'audio-candidate-e9de158.json').read_text())
    selection=json.loads((sources/'selected-audio-generation.json').read_text())
    files={}
    for name in NAMES:
        path=root/name
        if name.endswith('.pt2'):
            document={'graph_module':{'graph':{'nodes':[{'target':'aten.mul.Tensor','metadata':model_extra or {'val':'f32[2]'}}]}}}
            with zipfile.ZipFile(path,'w') as archive:
                archive.writestr('audio/models/model.json',json.dumps(document))
                archive.writestr('audio/data/weights/weight_0',b'\0'*8)
                archive.writestr('audio/version',other_text)
                if duplicate:
                    with pytest.warns(UserWarning):archive.writestr('audio/version','1')
            members={}
            with zipfile.ZipFile(path) as archive:
                for item in archive.infolist():
                    if item.filename.endswith('/models/model.json'):continue
                    members[item.filename]=dict(bytes=item.file_size,sha256=sha(archive.read(item)))
            profile='1119' if '1119' in name else '839'
            graph=next(v for k,v in contract['graphs'].items() if k.startswith('frames-1119-valid-'+profile+'/'))
            extra=dict(kind='pt2',source_graph_sha256=graph['sha256'],semantic_model_sha256=sha(json.dumps(document,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()),non_model_member_manifest_sha256=canonical(members),member_count=3)
        elif name.endswith('.bin'):
            signal=name.removeprefix('input-').removesuffix('.bin');profile='839' if signal=='padded_noise' else '1119'
            path.write_bytes(struct.pack('<2f',.25,-.5));extra=dict(kind='original-input',original_member='frames-1119-valid-'+profile+'/'+signal+'-features-fp32.bin')
        else:
            path.write_text(json.dumps({'public':json_value,'values':[.25,-.5]}));extra=dict(kind='json')
        raw=path.read_bytes();files[name]=dict(bytes=len(raw),sha256=sha(raw),**extra)
    inventory=dict(schema=1,scope='selected-native-regression-exact-assets',candidate_contract_sha256=sha((sources/'audio-candidate-e9de158.json').read_bytes()),selected_generation_canonical_sha256=canonical(selection),trace_removal_producer='6bfc2232412f88a78dc23227b5d57ee5a3c7c05c',files=files)
    raw=json.dumps(inventory).encode();(root/'native-regression-assets.json').write_bytes(raw)
    proof=dict(schema=1,review='native-graph-metadata+exact-numeric-source',success=True,inventory_sha256=sha(raw),asset_count=12,all_original_numeric_bytes_verified=True,all_source_and_distribution_graph_hashes_verified=True,all_non_model_members_unchanged=True,all_graph_semantics_unchanged=True,all_textual_metadata_checked=True,contains_private_metadata=False,images_in_distribution=False,opaque_tensor_privacy_certified=False,original_native_pt2_recovered=False,GPU_or_release_acceptance=False)
    record=dict(review_path='docs/results/native-release-test.json',review_canonical_sha256=canonical(proof))
    docs={'.privacy-reviewed-native-regressions.json':dict(schema=1,inventories={sha(raw):record}),record['review_path']:proof,'tools/references/audio-candidate-e9de158.json':contract,'tools/references/selected-audio-generation.json':selection}
    m=module();monkeypatch.setattr(m,'_head_json',lambda cwd,name:docs[name])
    return m,root,docs


@pytest.mark.parametrize('name',NAMES)
def test_exact_reviewed_asset_cohort_is_accepted(tmp_path,monkeypatch,name):
    m,root,_=reviewed(tmp_path,monkeypatch)
    assert m.reviewed_native_asset_safe(root/name,tmp_path)


@pytest.mark.parametrize('problem',['policy','proof','proof-flags','inventory','bytes','cohort','source','selection','link'])
def test_missing_changed_or_untrusted_authority_fails_closed(tmp_path,monkeypatch,problem):
    m,root,docs=reviewed(tmp_path,monkeypatch);path=root/'candidate-1119.pt2'
    policy=docs['.privacy-reviewed-native-regressions.json'];record=next(iter(policy['inventories'].values()))
    if problem=='policy':docs.pop('.privacy-reviewed-native-regressions.json')
    elif problem=='proof':docs.pop(record['review_path'])
    elif problem=='proof-flags':docs[record['review_path']]['opaque_tensor_privacy_certified']=True
    elif problem=='inventory':(root/'native-regression-assets.json').write_text('{}')
    elif problem=='bytes':path.write_bytes(path.read_bytes()+b'changed')
    elif problem=='source':docs['tools/references/audio-candidate-e9de158.json']['graphs'].clear()
    elif problem=='selection':docs['tools/references/selected-audio-generation.json']['sdk_runtime_tree']='0'*40
    elif problem=='cohort':
        path=root/'foreign.pt2';path.write_bytes((root/'candidate-1119.pt2').read_bytes())
    else:
        target=root/'real.pt2';path.rename(target)
        try:path.symlink_to(target)
        except OSError:pytest.skip('Host does not permit synthetic symlinks')
    assert not m.reviewed_native_asset_safe(path,tmp_path)


@pytest.mark.parametrize('kind',['trace','path','email','credential','duplicate','other-text'])
def test_registered_hash_is_not_enough_to_approve_private_or_ambiguous_graph_metadata(tmp_path,monkeypatch,kind):
    extra={'val':'f32[2]'};other='1'
    if kind=='trace':extra['stack_trace']='optional public trace'
    elif kind=='path':extra['note']='/'+'home'+'/'+'synthetic'+'/private'
    elif kind=='email':extra['note']='fixture'+'@'+'private-domain.dev'
    elif kind=='credential':extra['note']='ghp_'+'A'*36
    elif kind=='other-text':other='fixture'+'@'+'private-domain.dev'
    m,root,_=reviewed(tmp_path,monkeypatch,model_extra=extra,other_text=other,duplicate=kind=='duplicate')
    assert not m.reviewed_native_asset_safe(root/'candidate-1119.pt2',tmp_path)


@pytest.mark.parametrize('name',['capture.png','archive.zip','unknown.bin'])
def test_review_never_approves_other_files_or_formats(tmp_path,monkeypatch,name):
    m,root,_=reviewed(tmp_path,monkeypatch);path=root/name;path.write_bytes(b'unreviewed\0')
    assert not m.reviewed_native_asset_safe(path,tmp_path)


def test_authority_comes_from_committed_HEAD(tmp_path,monkeypatch):
    m=module();head_json=m._head_json;m,root,docs=reviewed(tmp_path,monkeypatch)
    monkeypatch.setattr(m,'_head_json',head_json);repo=tmp_path/'repo';repo.mkdir()
    def git(*args):return subprocess.run(['git','-C',str(repo),*args],capture_output=True,check=True)
    git('init','-q')
    for name,value in docs.items():
        path=repo/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
    assert not m.reviewed_native_asset_safe(root/'candidate-1119.pt2',repo)
    git('add','.')
    git('-c','user.name=fixture','-c','user.email=123+fixture@users.noreply.github.com','commit','-q','-m','Record synthetic native review')
    assert m.reviewed_native_asset_safe(root/'candidate-1119.pt2',repo)
    (repo/'.privacy-reviewed-native-regressions.json').write_text('{}')
    assert m.reviewed_native_asset_safe(root/'candidate-1119.pt2',repo)


@pytest.mark.parametrize('verb',['create','upload'])
def test_release_hook_uses_exact_native_asset_review_and_rejects_changed_bytes(tmp_path,monkeypatch,verb):
    from embeddinggemma_tools.privacy_hook import decide
    _,root,_=reviewed(tmp_path,monkeypatch)
    request=dict(tool_name='Bash',cwd=str(tmp_path),tool_input=dict(command=f'gh release {verb} preview release/candidate-1119.pt2'))
    assert decide(request,lambda:True)['hookSpecificOutput']['permissionDecision']=='allow'
    path=root/'candidate-1119.pt2';path.write_bytes(path.read_bytes()+b'changed')
    assert decide(request,lambda:True)['hookSpecificOutput']['permissionDecision']=='deny'


def test_exact_reviewed_reference_above_generic_upload_limit_is_still_metadata_checked(tmp_path,monkeypatch):
    m,root,_=reviewed(tmp_path,monkeypatch,json_value='x'*(17*1024**2))
    assert m.reviewed_native_asset_safe(root/'original-reference.json',tmp_path)


def test_registered_json_with_private_text_is_rejected(tmp_path,monkeypatch):
    m,root,_=reviewed(tmp_path,monkeypatch,json_value='fixture'+'@'+'private-domain.dev')
    assert not m.reviewed_native_asset_safe(root/'original-reference.json',tmp_path)


def test_privacy_guard_does_not_import_torch_or_deserialize_programs():
    result=subprocess.run([sys.executable,'-c',
        'import sys; import embeddinggemma_tools.native_regression_release_privacy; assert "torch" not in sys.modules'],
        capture_output=True)
    assert result.returncode==0
