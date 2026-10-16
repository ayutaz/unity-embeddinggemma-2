import hashlib
import json

import numpy as np
import pytest

from embeddinggemma_tools import pinned_audio_onnx


def fixture(tmp_path):
    roots = [tmp_path/'current', tmp_path/'pinned']
    signals = ['silence','impulse','sine','nyquist','noise','padded_noise']
    for root in roots:
        root.mkdir()
        files = {}
        def binary(name, value):
            raw = value if isinstance(value, bytes) else np.asarray(value,dtype='<f4').tobytes()
            (root/name).write_bytes(raw)
            files[name] = dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
        rows, profiles = [], {}
        for index, signal in enumerate(signals):
            profile = 'padded' if signal == 'padded_noise' else 'full'
            binary(signal+'-input.bin',[1,2,3,4])
            output = np.array([[1,2,3]],dtype=np.float32)
            rows.append(dict(id=signal,profile=profile,input_features=signal+'-input.bin',features=output.tolist()))
            if root == roots[0]:
                binary(signal+'-actual.bin',output)
                rows[-1]['onnx_features'] = signal+'-actual.bin'
            if profile not in profiles:
                binary(profile+'-mask.bin',[1,0] if profile == 'padded' else [1,1])
                binary(profile+'.onnx',b'new graph' if root == roots[0] else b'old graph')
                profiles[profile]=dict(feature_count=1,feature_width=3,frame_mask=profile+'-mask.bin',graph=profile+'.onnx')
        metadata=dict(source_commit=('a' if root == roots[0] else 'b')*40,model_revision='c'*40,
            input_sha256={'weights':'d'*64}, versions={'torch':'pinned'},feature_atol=1e-5,feature_rtol=1e-4)
        reference=dict(metadata=metadata,cases=rows,profiles=profiles)
        raw=json.dumps(reference).encode()
        binary('audio-encoder-reference.json',raw)
        (root/'audio-encoder-validation.json').write_text(json.dumps(dict(
            files=files,metadata=metadata,case_count=6,profile_count=2,native_reference_complete=True)))
    return roots


def test_all_six_saved_onnx_outputs_are_compared_to_unchanged_pinned_values(tmp_path):
    current,pinned = fixture(tmp_path)
    report = pinned_audio_onnx.compare(current,pinned)
    assert report['success'] and report['case_count']==6 and report['profile_count']==2
    assert report['native_context_bit_matches_original']
    assert report['unity_validated'] is False and report['end_to_end_validated'] is False


def test_one_numeric_failure_is_retained_while_all_cases_are_measured(tmp_path):
    current,pinned = fixture(tmp_path)
    path=current/'silence-actual.bin'
    raw=np.array([[1.1,2,3]],dtype='<f4').tobytes(); path.write_bytes(raw)
    audit=json.loads((current/'audio-encoder-validation.json').read_text())
    audit['files'][path.name]['sha256']=hashlib.sha256(raw).hexdigest()
    (current/'audio-encoder-validation.json').write_text(json.dumps(audit))
    report=pinned_audio_onnx.compare(current,pinned)
    assert not report['success'] and report['failed_case_count']==1 and report['case_count']==6


def test_changed_fresh_native_values_never_replace_the_pinned_onnx_expectation(tmp_path):
    current,pinned=fixture(tmp_path)
    path=current/'audio-encoder-reference.json'
    reference=json.loads(path.read_text())
    reference['cases'][0]['features']=[[1.1,2,3]]
    raw=json.dumps(reference).encode(); path.write_bytes(raw)
    audit=json.loads((current/'audio-encoder-validation.json').read_text())
    audit['files'][path.name]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    (current/'audio-encoder-validation.json').write_text(json.dumps(audit))
    report=pinned_audio_onnx.compare(current,pinned)
    assert report['success'], 'actual outputs still match the immutable original golden'
    assert not report['native_context_fixed_matches_original']
    assert not report['native_context_bit_matches_original']
    assert report['native_context_required_for_localization']


@pytest.mark.parametrize('problem',['tamper','input','version','missing_case'])
def test_bad_provenance_cannot_be_a_numeric_pass(tmp_path,problem):
    current,pinned=fixture(tmp_path)
    if problem=='tamper':
        (current/'silence-actual.bin').write_bytes(b'invalid')
    elif problem=='input':
        audit=json.loads((current/'audio-encoder-validation.json').read_text())
        audit['files']['silence-input.bin']['sha256']='0'*64
        (current/'audio-encoder-validation.json').write_text(json.dumps(audit))
    else:
        path=current/'audio-encoder-reference.json'; reference=json.loads(path.read_text())
        if problem=='version': reference['metadata']['versions']['torch']='different'
        else: reference['cases'].pop()
        raw=json.dumps(reference).encode(); path.write_bytes(raw)
        audit=json.loads((current/'audio-encoder-validation.json').read_text())
        audit['metadata']=reference['metadata']
        audit['files'][path.name]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
        (current/'audio-encoder-validation.json').write_text(json.dumps(audit))
    with pytest.raises(ValueError): pinned_audio_onnx.compare(current,pinned)
