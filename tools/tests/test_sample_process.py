import json

import pytest

from embeddinggemma_tools.sample_process import audit_sample_process, main


def fixture(tmp_path):
    report = {'schema_version': 1, 'run_id': '1' * 32, 'build': {'codeCommit': 'a' * 40},
              'started_utc': '2026-10-10T08:54:20Z', 'completed_utc': '2026-10-10T08:54:28Z',
              'success': True, 'gpu_verified': True, 'phase': 'completed',
              'validation_mode': 'real_model', 'is_editor': False, 'batch_mode': False,
              'worker_released': True, 'font_destroyed': True, 'blank_rejected': True,
              'missing_model_rejected': True, 'console': [],
              'gpu_execution': {'success': True, 'worker_backend': 'GPUCompute',
                                'computed_layer_output_count': 1,
                                'computation_cpu_fallback_indices': [], 'intermediate_flush_indices': []}}
    receipt = {'runId': '1' * 32, 'source': 'a' * 40, 'exitCode': 0,
               'startedUtc': '2026-10-10T08:54:08Z', 'completedUtc': '2026-10-10T08:54:33Z',
               'seconds': 32425}  # Old launcher bug: derive timing from UTC timestamps instead.
    paths = tmp_path / 'report.json', tmp_path / 'process.json'
    for path, value in zip(paths, (report, receipt)):
        path.write_text(json.dumps(value), encoding='utf-8')
    return paths


def change(path, key, value):
    data = json.loads(path.read_text())
    data[key] = value
    path.write_text(json.dumps(data), encoding='utf-8')


@pytest.mark.parametrize('problem', ['missing', 'cpu_backend', 'computation_fallback', 'flush',
                                   'empty_graph', 'failed_audit', 'invalid_indices', 'boolean_count'])
def test_current_acceptance_requires_explicit_worker_gpu_execution_evidence(tmp_path, problem):
    paths = fixture(tmp_path)
    report = json.loads(paths[0].read_text())
    profile = report['gpu_execution']
    if problem == 'missing': report.pop('gpu_execution')
    if problem == 'cpu_backend': profile['worker_backend'] = 'CPU'
    if problem == 'computation_fallback': profile['computation_cpu_fallback_indices'] = [5]
    if problem == 'flush': profile['intermediate_flush_indices'] = [5]
    if problem == 'empty_graph': profile['computed_layer_output_count'] = 0
    if problem == 'failed_audit': profile['success'] = False
    if problem == 'invalid_indices': profile['computation_cpu_fallback_indices'] = None
    if problem == 'boolean_count': profile['computed_layer_output_count'] = True
    paths[0].write_text(json.dumps(report))
    assert audit_sample_process(*paths)['process_gate_passed'] is False


def test_successful_report_with_crashed_process_is_rejected(tmp_path):
    paths = fixture(tmp_path)
    change(paths[1], 'exitCode', -1073741819)
    result = audit_sample_process(*paths)
    assert result['runtime_report_success'] is True
    assert result['runtime_gpu_verified'] is True
    assert result['process_gate_passed'] is False
    assert result['reason'] == 'nonzero_process_exit'
    assert result['process_seconds'] == 25


def test_zero_exit_completes_only_the_process_gate(tmp_path):
    result = audit_sample_process(*fixture(tmp_path))
    assert result['process_gate_passed'] is True
    assert result['reference_reaudited'] is False
    assert result['execution_performed_by_auditor'] is False
    assert result['m2_release_verified'] is False


@pytest.mark.parametrize('field,value', [
    ('success', False), ('success', 1), ('gpu_verified', False), ('phase', 'running'),
    ('validation_mode', 'injected'), ('is_editor', True), ('batch_mode', True),
    ('worker_released', False), ('font_destroyed', False), ('blank_rejected', False),
    ('missing_model_rejected', False), ('console', [{'type': 'Error'}]),
])
def test_zero_exit_cannot_override_incomplete_runtime(tmp_path, field, value):
    paths = fixture(tmp_path)
    change(paths[0], field, value)
    assert audit_sample_process(*paths)['process_gate_passed'] is False


@pytest.mark.parametrize('field,value', [
    ('runId', '2' * 32), ('source', 'b' * 40), ('exitCode', False),
    ('exitCode', None), ('completedUtc', None),
    ('completedUtc', '2026-10-10T08:54:21Z'),
    ('startedUtc', '2026-10-10T08:54:21Z'),
    ('startedUtc', '2026-10-10T17:54:08+09:00'),
])
def test_mixed_or_unfinished_process_receipt_is_invalid(tmp_path, field, value):
    paths = fixture(tmp_path)
    change(paths[1], field, value)
    with pytest.raises(ValueError):
        audit_sample_process(*paths)


def test_cli_crash_exit_and_no_private_paths(tmp_path, capsys):
    paths = fixture(tmp_path)
    change(paths[1], 'exitCode', -1073741819)
    assert main(['--report', str(paths[0]), '--process', str(paths[1])]) == 1
    result = json.loads(capsys.readouterr().out)
    assert result['process_gate_passed'] is False
    assert str(tmp_path) not in json.dumps(result)
