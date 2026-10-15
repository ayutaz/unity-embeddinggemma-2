import json
import os
from pathlib import Path
import subprocess

import pytest
import torch


def module():
    from embeddinggemma_tools import frozen_matmul_native_context
    return frozen_matmul_native_context


def test_two_contexts_set_mkl_before_fresh_interpreter_without_changing_aten_or_parent():
    source = {'MKL_ENABLE_INSTRUCTIONS': 'AVX512', 'ATEN_CPU_CAPABILITY': 'avx512', 'MKL_CBWR': 'AUTO'}
    before = dict(source)
    contexts = module().context_environments(source)
    assert set(contexts) == {'default', 'avx2'}
    assert 'MKL_ENABLE_INSTRUCTIONS' not in contexts['default']
    assert contexts['avx2']['MKL_ENABLE_INSTRUCTIONS'] == 'AVX2'
    assert all(env['ATEN_CPU_CAPABILITY'] == 'avx512' and env['MKL_CBWR'] == 'AUTO' for env in contexts.values())
    assert all(env['MKL_VERBOSE'] == '1' for env in contexts.values())
    assert source == before


def test_native_worker_retains_original_target_failure_and_does_not_claim_candidate_or_encoder(tmp_path, monkeypatch):
    probe = module()
    a, b = torch.ones(3, 4), torch.ones(4, 2)
    monkeypatch.delenv('MKL_ENABLE_INSTRUCTIONS', raising=False)
    monkeypatch.setenv('GITHUB_SHA', 'a' * 40)
    monkeypatch.setattr(probe, 'load_capture', lambda _: (a, b, torch.zeros(3, 2), {'immutable': 'unit-test-only'}))
    r = probe.worker(tmp_path / 'input', tmp_path / 'out', 'default')
    assert r['complete'] and r['metrics']['mismatched_elements'] == 6
    assert not r['full_encoder_validated'] and not r['gpu_validated']
    assert r['execution'] == 'native_torch_mm_only_not_onnx_candidate'
    assert r['capture_provenance'] == {'immutable': 'unit-test-only'}


def test_parent_uses_fresh_child_and_same_fixture_for_both_contexts(tmp_path, monkeypatch):
    probe = module(); calls = []
    def run(command, **kwargs):
        calls.append((command, kwargs))
        context = command[command.index('--context') + 1]
        output = Path(command[command.index('--output') + 1]); output.mkdir(parents=True)
        (output / 'native-context.json').write_text(json.dumps(dict(complete=True, context=context,
            capture_provenance={'same_original_fixture': True}, metrics={'mismatched_elements': 1})))
        return subprocess.CompletedProcess(command, 0, stdout='MKL test verbose output\n', stderr='')
    monkeypatch.setattr(probe.subprocess, 'run', run)
    r = probe.probe_native_contexts(tmp_path / 'input', tmp_path / 'out')
    assert r['complete'] and len(r['cases']) == len(calls) == 2
    assert r['same_original_fixture'] and not r['full_encoder_validated']
    assert all('--worker' in command and kwargs['check'] and kwargs['timeout'] == 120 for command, kwargs in calls)
    assert 'MKL_ENABLE_INSTRUCTIONS' not in calls[0][1]['env']
    assert calls[1][1]['env']['MKL_ENABLE_INSTRUCTIONS'] == 'AVX2'
    assert all((tmp_path / 'out' / row['log']).is_file() for row in r['cases'])


def test_failed_child_keeps_incomplete_receipt_and_refuses_overwrite(tmp_path, monkeypatch):
    probe = module()
    def fail(*args, **kwargs):
        raise subprocess.TimeoutExpired('unit-test-worker', 120)
    monkeypatch.setattr(probe.subprocess, 'run', fail)
    with pytest.raises(subprocess.TimeoutExpired):
        probe.probe_native_contexts(tmp_path / 'input', tmp_path / 'out')
    receipt = tmp_path / 'out/native-matmul-contexts.json'
    r = json.loads(receipt.read_text())
    assert not r['complete'] and r['diagnostic_error']['type'] == 'TimeoutExpired'
    with pytest.raises(ValueError, match='overwrite'):
        probe.probe_native_contexts(tmp_path / 'input', tmp_path / 'out')


def test_workflow_adds_only_to_small_opt_in_matrix_job():
    path = Path(__file__).resolve().parents[2] / '.github/workflows/frozen-audio-prefix-onnx.yml'
    text = path.read_text(); before, matrix = text.split('\n  matrix_probe:\n', 1)
    assert 'frozen_matmul_native_context' in matrix
    assert 'frozen_matmul_native_context' not in before
    assert 'timeout-minutes: 10' in matrix
