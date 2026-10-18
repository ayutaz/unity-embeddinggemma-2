import json
from pathlib import Path

import pytest
import torch
import yaml
from workflow_privacy_assertions import assert_privacy_gated_upload


def module():
    from embeddinggemma_tools import audio_native_context
    return audio_native_context


def fixture(monkeypatch, probe, *, fail_case=None, wrong_output=False):
    calls = []
    def load(directory, *, case_id):
        calls.append(case_id)
        if case_id == fail_case:
            raise ValueError('injected audited input failure')
        values = torch.full((2, 3), float(len(calls)))
        expected = values + 1
        return (lambda x: (x + (0 if wrong_output else 1),)), values, expected, dict(
            case_id=case_id, profile='padded' if case_id == 'padded_noise' else 'full',
            graph_sha256='b'*64, input_sha256='c'*64)
    monkeypatch.setattr(probe, 'load_frozen_input', load)
    monkeypatch.setenv('GITHUB_SHA', 'a'*40)
    return calls


def test_native_all_six_executes_inputs_and_retains_drift_without_becoming_onnx_or_gpu_pass(tmp_path, monkeypatch):
    probe = module(); calls = fixture(monkeypatch, probe, wrong_output=True)
    r = probe.probe(tmp_path/'input', tmp_path/'out')
    assert calls == ['silence', 'impulse', 'sine', 'nyquist', 'noise', 'padded_noise']
    assert r['complete'] and not r['original_tolerance_passed']
    assert not r['onnx_validated'] and not r['gpu_validated'] and not r['fresh_official_validated']
    assert r['source'] == 'a'*40
    for row in r['cases']:
        assert row['metrics']['mismatched_elements'] == 6
        capture = tmp_path/'out'/row['capture']
        assert capture.stat().st_size == 24
        assert probe.digest(capture) == row['capture_sha256']


def test_original_tolerance_success_requires_all_six_complete_and_no_graph_intervention(tmp_path, monkeypatch):
    probe = module(); fixture(monkeypatch, probe)
    r = probe.probe(tmp_path/'input', tmp_path/'out')
    assert r['complete'] and r['original_tolerance_passed']
    assert len(r['cases']) == 6 and r['values_compared'] == 36
    assert all(row['metrics']['exact_difference_count'] == 0 for row in r['cases'])
    assert not r['fresh_official_validated'] and not r['onnx_validated']


def test_partial_failure_keeps_completed_captures_and_incomplete_receipt(tmp_path, monkeypatch):
    probe = module(); fixture(monkeypatch, probe, fail_case='sine')
    with pytest.raises(ValueError, match='audited input'):
        probe.probe(tmp_path/'input', tmp_path/'out')
    r = json.loads((tmp_path/'out/audio-native-context.json').read_text())
    assert not r['complete'] and not r['original_tolerance_passed']
    assert [row['case_id'] for row in r['cases']] == ['silence', 'impulse']
    assert r['diagnostic_error']['type'] == 'ValueError'
    with pytest.raises(ValueError, match='overwrite'):
        probe.probe(tmp_path/'input', tmp_path/'out')


def test_job_source_is_required_before_loading_any_model(tmp_path, monkeypatch):
    probe = module(); calls = fixture(monkeypatch, probe)
    monkeypatch.setenv('GITHUB_SHA', 'local')
    with pytest.raises(ValueError, match='source'):
        probe.probe(tmp_path/'input', tmp_path/'out')
    assert not calls


def test_workflow_is_explicit_single_native_job_with_original_six_inputs_and_numeric_only_upload():
    p = Path(__file__).resolve().parents[2]/'.github/workflows/frozen-audio-prefix-onnx.yml'
    w = yaml.safe_load(p.read_text())
    inputs = w.get('on', w.get(True))['workflow_dispatch']['inputs']
    assert inputs['native_context_probe']['default'] is False
    job = w['jobs']['native_context_probe']
    assert job['timeout-minutes'] == 10 and 'matrix' not in job
    assert 'inputs.native_context_probe' in job['if']
    assert '!inputs.native_context_probe' in w['jobs']['probe']['if']
    assert '!inputs.native_context_probe' in w['jobs']['matrix_probe']['if']
    assert job['runs-on'] == "${{ inputs.matrix_runner || 'windows-2022' }}"
    fetch = next(s['run'] for s in job['steps'] if s.get('name', '').startswith('Fetch original'))
    assert '11814567883' in fetch and fetch.count('audio.pt2') == 2
    for case in ('silence','impulse','sine','nyquist','noise','padded_noise'):
        assert case+'-features-fp32.bin' in fetch
    execute = next(s['run'] for s in job['steps'] if 'audio_native_context' in s.get('run',''))
    assert 'embeddinggemma_tools.audio_native_context' in execute
    upload = next(s for s in job['steps'] if s.get('uses','').startswith('actions/upload-artifact'))
    assert_privacy_gated_upload(upload)
    assert '.pt2' not in upload['with']['path']
