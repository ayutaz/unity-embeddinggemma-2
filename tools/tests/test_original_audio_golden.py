import copy
import hashlib
import json
import struct
from pathlib import Path

import pytest
import yaml

from embeddinggemma_tools.original_audio_golden import reconstruct


def fixture(tmp_path):
    original={'metadata':{'source':'immutable fixture'},'cases':[{'id':'signal','features':[[0.0,1.0],[-0.0,0.5]]}]}
    raw=(json.dumps(original,indent=2,allow_nan=False)+'\n').encode()
    capture=struct.pack('<4f',0.0,1.0,-0.0,0.5)
    path=tmp_path/'signal/reference.bin';path.parent.mkdir();path.write_bytes(capture)
    template=copy.deepcopy(original);template['cases'][0]['features']=None
    return {'schema':1,'template':template,'captures':[{'case':'signal','member':'reference.bin','bytes':16,
            'sha256':hashlib.sha256(capture).hexdigest(),'rows':2,'columns':2}]},raw


def test_only_exact_original_json_bytes_are_recovered(tmp_path):
    contract,raw=fixture(tmp_path)
    assert reconstruct(contract,tmp_path,expected_sha=hashlib.sha256(raw).hexdigest())==raw


def test_changed_capture_or_template_is_not_a_new_golden(tmp_path):
    contract,raw=fixture(tmp_path);expected=hashlib.sha256(raw).hexdigest()
    (tmp_path/'signal/reference.bin').write_bytes(struct.pack('<4f',0,1,0,0.5))
    with pytest.raises(ValueError):reconstruct(contract,tmp_path,expected_sha=expected)
    (tmp_path/'signal/reference.bin').write_bytes(struct.pack('<4f',0,1,-0.0,0.5))
    contract['template']['metadata']['source']='different source'
    with pytest.raises(ValueError):reconstruct(contract,tmp_path,expected_sha=expected)


def test_coverage_and_capture_paths_fail_closed(tmp_path):
    contract,raw=fixture(tmp_path);expected=hashlib.sha256(raw).hexdigest()
    contract['captures'][0]['member']='../outside.bin'
    with pytest.raises(ValueError):reconstruct(contract,tmp_path,expected_sha=expected)
    contract['captures']=[]
    with pytest.raises(ValueError):reconstruct(contract,tmp_path,expected_sha=expected)


def test_manual_golden_preservation_never_claims_original_models():
    root=Path(__file__).resolve().parents[2]
    workflow=yaml.load((root/'.github/workflows/audio-runtime-reference.yml').read_text(encoding='utf-8'),Loader=yaml.BaseLoader)
    assert workflow['on']['workflow_dispatch']['inputs']['recover_golden_only']['default']=='false'
    job=workflow['jobs']['recover-golden']
    assert job['permissions']=={'contents':'read','actions':'read'}
    assert "github.event_name == 'workflow_dispatch'" in job['if']
    assert '!inputs.recover_golden_only' in workflow['jobs']['recover']['if']
    assert '!inputs.recover_golden_only' in workflow['jobs']['native-reference']['if']
    steps=job['steps']
    assert any('original_audio_golden' in s.get('run','') for s in steps)
    upload=next(s for s in steps if s.get('uses','').startswith('actions/upload-artifact@'))
    assert upload['with']['name'].startswith('pinned-audio-golden-')
    assert "outcome == 'success'" in upload['if']


def test_native_diagnostics_use_recovered_immutable_golden_without_missing_models():
    root=Path(__file__).resolve().parents[2]
    workflow=yaml.load((root/'.github/workflows/audio-runtime-reference.yml').read_text(encoding='utf-8'),Loader=yaml.BaseLoader)
    steps=workflow['jobs']['native-reference']['steps']
    fetch=next(s['run'] for s in steps if 'embeddinggemma_tools.artifact_zip' in s.get('run',''))
    assert '--artifact-id 12127104206' in fetch
    assert '--member audio-encoder-reference.json' in fetch
    assert '--member audio-encoder-validation.json' in fetch
    assert 'ac23f39904b8149dd5ccb6485afcc93d96047f0977b5faf1d640c8bd018c13c9  ../artifacts/pinned-audio/audio-encoder-reference.json' in fetch
    assert '68d29c96012ef0a44114f2d3edbbff8d14e116ac697abee1a36d0b212155cfa1  ../artifacts/pinned-audio/audio-encoder-validation.json' in fetch
    assert 'audio.pt2' not in fetch and '11814567883' not in fetch
    # Restoring the independent golden must never replace the required model gate.
    required=yaml.load((root/'.github/workflows/model-reference.yml').read_text(encoding='utf-8'),Loader=yaml.BaseLoader)
    assert required['jobs']['pinned-audio']['uses']=='./.github/workflows/pinned-audio-acceptance.yml'
    assert not required['jobs']['pinned-audio'].get('continue-on-error')
