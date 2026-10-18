import copy
import hashlib
import importlib
import json
import shutil

import numpy as np
import pytest
import torch
from test_audio_generation_candidate import candidate


CASES = ('silence', 'impulse', 'sine', 'nyquist', 'noise', 'padded_noise')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def inputs(candidate, monkeypatch):
    m, current, original, contract, path = candidate
    reference = json.loads((original/'audio-encoder-reference.json').read_text())
    audit = json.loads((original/'audio-encoder-validation.json').read_text())
    template = copy.deepcopy(reference['cases'][0])
    reference['cases'] = []
    padded = current/'padded/audio.pt2'; padded.parent.mkdir()
    shutil.copyfile(current/'audio.pt2', padded)
    reference['profiles']['padded'] = dict(reference['profiles']['full'], native_graph='padded/audio.pt2')
    contract['graphs']['padded/audio.pt2'] = dict(bytes=padded.stat().st_size, sha256=sha(padded))
    for index, name in enumerate(CASES):
        filename = name+'.bin'
        values = np.arange(256, dtype='<f4').reshape(2,128)/256+index
        values.tofile(current/filename)
        audit['files'][filename] = dict(bytes=(current/filename).stat().st_size, sha256=sha(current/filename))
        reference['cases'].append(dict(template, id=name, input_features=filename,
                                       profile='padded' if name=='padded_noise' else 'full'))
    audit['files']['padded/audio.pt2'] = dict(contract['graphs']['padded/audio.pt2'], sha256='b'*64)
    for name, value in [('audio-encoder-reference.json',reference), ('audio-encoder-validation.json',audit)]:
        (original/name).write_text(json.dumps(value), encoding='utf-8')
    fresh_reference = copy.deepcopy(reference); fresh_audit = copy.deepcopy(audit)
    for value in (fresh_reference, fresh_audit): value['metadata']['source_commit'] = 'a'*40
    fresh_audit['files'].update(contract['graphs'])
    for name, value in [('audio-encoder-reference.json',fresh_reference), ('audio-encoder-validation.json',fresh_audit)]:
        (current/name).write_text(json.dumps(value), encoding='utf-8')
    contract['reference_sha256'] = sha(current/'audio-encoder-reference.json')
    contract['audit_sha256'] = sha(current/'audio-encoder-validation.json')
    path.write_text(json.dumps(contract),encoding='utf-8')
    monkeypatch.setattr(m, 'CONTRACT_SHA', sha(path))
    monkeypatch.setattr(m, 'REFERENCE_SHA', sha(original/'audio-encoder-reference.json'))
    monkeypatch.setattr(m, 'AUDIT_SHA', sha(original/'audio-encoder-validation.json'))
    for key, value in [('OMP_NUM_THREADS','4'),('MKL_NUM_THREADS','4'),('MKL_CBWR','COMPATIBLE'),('MKL_DYNAMIC','FALSE')]:
        monkeypatch.setenv(key, value)
    return current, original


def module():
    return importlib.import_module('embeddinggemma_tools.audio_candidate_distribution')


def test_all_six_actual_native_outputs_remain_bit_identical_without_qualification(inputs,tmp_path):
    current, original = inputs; output = tmp_path/'distribution'
    before = {p:sha(p) for root in (current,original) for p in root.rglob('*') if p.is_file()}
    result = module().prepare(current, original, output, source='c'*40)
    assert result['case_count']==6 and result['profile_count']==2
    assert {r['case_id'] for r in result['native_outputs']} == set(CASES)
    assert all(r['bit_identical'] and r['source_sha256']==r['distribution_sha256'] for r in result['native_outputs'])
    assert result['native_outputs_bit_identical'] is True
    assert result['generation_qualified'] is False and result['gpu_validated'] is False
    assert result['original_graph_bytes'] is False
    assert result['source']=='c'*40
    assert len(result['graphs']) == 2
    assert all(r['computational_structure_unchanged'] for r in result['graphs'].values())
    assert all(sha(path)==digest for path,digest in before.items())
    assert json.loads((output/'distribution.json').read_text())==result
    assert not (output/'audio-encoder-reference.json').exists()


def test_mutated_distribution_cannot_be_published_as_bit_identical(inputs,tmp_path,monkeypatch):
    m=module(); real=m.native_outputs
    def changed(paths, cases):
        result=real(paths,cases)
        if next(iter(paths.values())).parent.name == 'distribution':
            result['silence']=result['silence'].copy()
            result['silence'].view(np.uint32).flat[0] ^= 1
        return result
    monkeypatch.setattr(m,'native_outputs',changed)
    output=tmp_path/'distribution'
    with pytest.raises(ValueError,match='bit'):
        m.prepare(*inputs,output,source='c'*40)
    assert not (output/'distribution.json').exists()


def test_changed_raw_input_is_rejected_before_any_native_load(inputs,tmp_path,monkeypatch):
    m=module(); current,original=inputs
    with (current/'noise.bin').open('ab') as stream: stream.write(b'changed')
    monkeypatch.setattr(torch.export,'load',lambda *args:pytest.fail('Unverified input reached Torch'))
    with pytest.raises(ValueError): m.prepare(current,original,tmp_path/'out',source='c'*40)


@pytest.mark.parametrize('key,value',[('MKL_CBWR','AUTO'),('MKL_DYNAMIC','TRUE'),('OMP_NUM_THREADS','1'),('MKL_NUM_THREADS','1')])
def test_native_comparison_requires_recorded_execution_controls(inputs,tmp_path,monkeypatch,key,value):
    monkeypatch.setenv(key,value)
    with pytest.raises(ValueError,match='control'):
        module().prepare(*inputs,tmp_path/'out',source='c'*40)


def test_preservation_never_overwrites_an_existing_output(inputs,tmp_path):
    output=tmp_path/'out';output.mkdir();(output/'keep').write_text('existing')
    with pytest.raises(ValueError,match='empty'):
        module().prepare(*inputs,output,source='c'*40)
    assert (output/'keep').read_text()=='existing'
