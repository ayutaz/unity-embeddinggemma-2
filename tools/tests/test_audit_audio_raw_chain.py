import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from embeddinggemma_tools.reference_context_failure import feature_error_metrics


def module():
    from embeddinggemma_tools import audit_audio_raw_chain
    return audit_audio_raw_chain


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(values, expected):
    r = feature_error_metrics(torch.from_numpy(values), torch.from_numpy(expected))
    r['exact_difference_count'] = int(np.count_nonzero(values.view('u4') != expected.view('u4')))
    return r


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    audit = module(); original = tmp_path / 'original'; original.mkdir()
    directory = tmp_path / 'captures'; directory.mkdir()
    profile = 'frames-1119-valid-1119'
    raw = np.ones((1, 3, 128), dtype='<f4')
    middle = np.ones((1, 2, 4), dtype='<f4')
    expected = np.arange(1, 7, dtype='<f4').reshape(2, 3)
    def save(name, array):
        path = directory / name; array.tofile(path); return digest(path)
    metadata = dict(source_commit=audit.PRODUCER, model_revision=audit.REVISION,
                    feature_atol=1e-5, feature_rtol=1e-4)
    reference = dict(metadata=metadata, cases=[dict(id='silence', profile=profile,
        input_features='silence.bin', features=expected.tolist())], profiles={profile:dict(
        feature_shape=list(raw.shape), feature_count=2, feature_width=3, native_graph='audio.pt2')})
    reference_path = original / 'audio-encoder-reference.json'
    reference_path.write_text(json.dumps(reference))
    input_sha = save('raw-input-fp32.bin', raw)
    manifest = dict(metadata=metadata, native_reference_complete=True, files={
        'silence.bin':dict(bytes=raw.nbytes, sha256=input_sha),
        'audio.pt2':dict(sha256='b'*64)})
    audit_path = original / 'audio-encoder-validation.json'; audit_path.write_text(json.dumps(manifest))
    monkeypatch.setattr(audit, 'REFERENCE_SHA', digest(reference_path))
    monkeypatch.setattr(audit, 'AUDIT_SHA', digest(audit_path))
    monkeypatch.setattr(audit, 'INTERMEDIATE_SHAPE', middle.shape)
    provenance = dict(reference_source=audit.PRODUCER, model_revision=audit.REVISION, case_id='silence',
        profile=profile, reference_sha256=audit.REFERENCE_SHA, audit_sha256=audit.AUDIT_SHA,
        input_sha256=input_sha, graph_sha256='b'*64, job_source='a'*40)
    boundaries = ['raw_features','view_2','mul_77','mul_152','mul_227','select']
    report = dict(scope='raw_original_feature_input_connected_onnx_partitions_only', complete=True,
        raw_input_mode=True, raw_original_input_executed=True, native_chain_control_bits_equal=True,
        context_qualified=True, whole_encoder_validated=False, gpu_validated=False, end_to_end_validated=False,
        provenance=provenance, input_provenance=provenance, target_provenance=provenance,
        input_sha256=input_sha, reference_sha256=save('reference-output-fp32.bin',expected),
        native_sha256=save('native-output-fp32.bin',expected), native_whole_metrics=metrics(expected,expected),
        boundaries=boundaries, cases=[])
    for variant in ('native','ordered_original_tanh'):
        row = dict(variant=variant, coefficients_unchanged=True,
            actual_carry='previous_onnx_output_without_frozen_stage_reset', parts=[], metrics=metrics(expected,expected))
        for index in range(5):
            input_name=f'{variant}-part{index}-input-fp32.bin'; output_name=f'{variant}-part{index}-output-fp32.bin'
            row['parts'].append(dict(index=index,from_boundary=boundaries[index],to_boundary=boundaries[index+1],
                input_capture=input_name,output_capture=output_name,
                input_sha256=save(input_name,raw if index==0 else middle),
                output_sha256=save(output_name,expected if index==4 else middle),graph_sha256='c'*64))
        report['cases'].append(row)
    path=directory/'frozen-audio-chained-onnx.json'; path.write_text(json.dumps(report))
    return audit, original, path, report, expected


def test_independent_raw_audit_recomputes_final_metrics_and_all_actual_carry_hashes(evidence):
    audit, original, path, _, _ = evidence
    r = audit.audit_report(path, original)
    assert r['original_tolerance_passed'] and r['candidate_metrics']['exact_difference_count']==0
    assert r['native_context_qualified'] and r['all_capture_and_carry_hashes_verified']
    assert not r['gpu_validated'] and not r['default_export_promoted']


def test_audit_rejects_relabelled_frozen_or_incomplete_execution(evidence):
    audit, original, path, report, _ = evidence
    report['raw_original_input_executed']=False; path.write_text(json.dumps(report))
    with pytest.raises(ValueError,match='raw'): audit.audit_report(path,original)


def test_audit_detects_wrong_carry_even_if_each_individual_file_hash_is_honest(evidence):
    audit, original, path, report, _ = evidence
    part=report['cases'][1]['parts'][2]; p=path.parent/part['input_capture']
    np.zeros((1,2,4),dtype='<f4').tofile(p); part['input_sha256']=digest(p)
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError,match='carry'): audit.audit_report(path,original)


def test_audit_rejects_a_forged_zero_mismatch_summary(evidence):
    audit, original, path, report, expected = evidence
    part=report['cases'][1]['parts'][-1]; p=path.parent/part['output_capture']
    np.zeros_like(expected).tofile(p); part['output_sha256']=digest(p)
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError,match='metrics'): audit.audit_report(path,original)


def test_honest_numerical_failure_is_collected_without_becoming_a_pass(evidence):
    audit, original, path, report, expected = evidence
    actual=np.zeros_like(expected); part=report['cases'][1]['parts'][-1];p=path.parent/part['output_capture']
    actual.tofile(p);part['output_sha256']=digest(p);report['cases'][1]['metrics']=metrics(actual,expected)
    path.write_text(json.dumps(report));r=audit.audit_report(path,original)
    assert not r['original_tolerance_passed'] and r['candidate_metrics']['mismatched_elements']==6


def test_native_drift_is_retained_separately_from_candidate_acceptance(evidence):
    audit, original, path, report, expected = evidence
    actual=expected+1; p=path.parent/'native-output-fp32.bin';actual.tofile(p)
    report.update(native_sha256=digest(p),native_whole_metrics=metrics(actual,expected),context_qualified=False)
    path.write_text(json.dumps(report));r=audit.audit_report(path,original)
    assert r['original_tolerance_passed'] and not r['native_context_qualified']
    assert r['native_whole_metrics']['mismatched_elements']==6
    assert r['candidate_vs_current_native']['mismatched_elements']==6


def test_audit_requires_independent_original_bytes_and_rejects_escaped_captures(evidence):
    audit, original, path, report, _ = evidence
    report['cases'][1]['parts'][0]['output_capture']='../outside.bin';path.write_text(json.dumps(report))
    with pytest.raises(ValueError,match='capture'):audit.audit_report(path,original)
    (original/'audio-encoder-reference.json').write_text('{}')
    with pytest.raises(ValueError,match='hash'):audit.audit_report(path,original)


@pytest.mark.parametrize('problem', ['missing','duplicate','profile','mixed_source','none'])
def test_all_case_audit_requires_exact_six_case_two_profile_matrix_and_retains_failure(problem,monkeypatch):
    audit=module(); names=['silence','impulse','sine','nyquist','noise','padded_noise']
    rows=[dict(case_id=name,profile='frames-1119-valid-839' if name=='padded_noise' else 'frames-1119-valid-1119',
        source='a'*40,original_tolerance_passed=name!='noise',values_compared=6,
        candidate_metrics={'mismatched_elements':1 if name=='noise' else 0,'exact_difference_count':1 if name=='noise' else 0},
        native_context_qualified=False) for name in names]
    if problem=='missing':rows.pop()
    if problem=='duplicate':rows[-1]['case_id']='silence'
    if problem=='profile':rows[-1]['profile']='frames-1119-valid-1119'
    if problem=='mixed_source':rows[-1]['source']='b'*40
    monkeypatch.setattr(audit,'audit_report',lambda path,original:rows[path])
    if problem!='none':
        with pytest.raises(ValueError,match='six|profile|source'):audit.audit_reports(list(range(len(rows))),Path('original'))
    else:
        r=audit.audit_reports(list(range(6)),Path('original'))
        assert not r['success'] and r['case_count']==6 and r['values_compared']==36
        assert r['candidate_tolerance_failures']==1 and r['candidate_bit_differences']==1
        assert not r['gpu_validated'] and not r['default_export_promoted']
