import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from embeddinggemma_tools.original_audio_recovery import AUDIT_SHA256, pinned_files, verify


def contract(tmp_path):
    values={'profile/audio.pt2':b'original graph','reference.json':b'original reference','audit.json':b'original audit'}
    expected={}
    for name,data in values.items():
        path=tmp_path/name;path.parent.mkdir(exist_ok=True);path.write_bytes(data)
        expected[name]={'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
    return expected


def test_only_byte_identical_files_count_as_recovered(tmp_path):
    expected=contract(tmp_path)
    result=verify(tmp_path,expected_files=expected,graph_paths=['profile/audio.pt2'])
    assert result['graphs_byte_identical'] and result['complete_archive_byte_identical']
    assert result['gpu_validated'] is False
    (tmp_path/'profile/audio.pt2').write_bytes(b'regenerated different graph')
    result=verify(tmp_path,expected_files=expected,graph_paths=['profile/audio.pt2'])
    assert not result['graphs_byte_identical'] and not result['complete_archive_byte_identical']


def test_reference_drift_cannot_be_hidden_by_matching_graphs(tmp_path):
    expected=contract(tmp_path)
    (tmp_path/'reference.json').write_bytes(b'new reference')
    result=verify(tmp_path,expected_files=expected,graph_paths=['profile/audio.pt2'])
    assert result['graphs_byte_identical'] and not result['complete_archive_byte_identical']
    assert result['scope']=='original_graph_bytes_only'


def test_extra_files_cannot_be_published_as_the_complete_original_archive(tmp_path):
    expected=contract(tmp_path)
    (tmp_path/'extra.bin').write_bytes(b'unqualified data')
    result=verify(tmp_path,expected_files=expected,graph_paths=['profile/audio.pt2'])
    assert result['graphs_byte_identical'] and not result['complete_archive_byte_identical']


def test_missing_graph_and_outside_contract_fail_closed(tmp_path):
    expected=contract(tmp_path)
    (tmp_path/'profile/audio.pt2').unlink()
    assert not verify(tmp_path,expected_files=expected,graph_paths=['profile/audio.pt2'])['graphs_byte_identical']
    with pytest.raises(ValueError):
        verify(tmp_path,expected_files={'../outside':expected['reference.json']},graph_paths=['../outside'])


def test_pinned_contract_is_immutable_and_includes_its_own_audit(tmp_path):
    files=pinned_files()
    assert files['audio-encoder-validation.json']=={'bytes':3735,'sha256':AUDIT_SHA256}
    assert len([name for name in files if name.endswith('.pt2')])==2
    fake=tmp_path/'altered.json'
    fake.write_text('{}')
    with pytest.raises(ValueError):
        pinned_files(fake)


def test_cli_reports_missing_originals_as_failure_and_never_gpu_pass(tmp_path):
    report=tmp_path/'report.json';output=tmp_path/'github-output'
    environment=dict(os.environ,GITHUB_OUTPUT=str(output),RECOVERY_CONTROLLER_SHA='a'*40)
    result=subprocess.run([sys.executable,'-m','embeddinggemma_tools.original_audio_recovery',
                           '--input',str(tmp_path/'missing'),'--output',str(report),
                           '--collection-outcome','failure'],env=environment,capture_output=True)
    assert result.returncode==1
    saved=json.loads(report.read_text())
    assert saved['scope']=='not_recovered' and saved['gpu_validated'] is False
    assert saved['collection_outcome']=='failure'
    assert str(tmp_path) not in report.read_text()
    assert output.read_text().splitlines()==['graphs_match=false','complete_match=false']


def test_recovery_workflow_is_manual_pinned_and_never_qualifies_gpu():
    root=Path(__file__).resolve().parents[2]
    workflow=yaml.load((root/'.github/workflows/audio-runtime-reference.yml').read_text(),Loader=yaml.BaseLoader)
    assert workflow['on']['workflow_dispatch']['inputs']['recover_original_bytes']['default']=='false'
    assert '!inputs.recover_original_bytes' in workflow['jobs']['native-reference']['if']
    assert workflow['jobs']['recover']['permissions']=={'contents':'read'}
    assert "github.event_name == 'workflow_dispatch'" in workflow['jobs']['recover']['if']
    steps=workflow['jobs']['recover']['steps']
    assert any(s.get('with',{}).get('ref')=='d2ad81dfdf42d057fb4b7f30dd59d557f8a6dd75' for s in steps)
    assert any('1abe24067230501283fe73e1f20e4edd5eb9f830' in s.get('run','') for s in steps)
    verify_step=next(s for s in steps if s.get('id')=='verify')
    assert 'original_audio_recovery.py' in verify_step['run']
    complete=next(s for s in steps if s.get('with',{}).get('name','').startswith('pinned-audio-original-'))
    assert "steps.verify.outputs.complete_match == 'true'" in complete['if']
    assert 'outcome' in complete['if']
