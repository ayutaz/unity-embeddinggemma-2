"""Windows must execute the same bounded original-PCM comparison protocol."""
from pathlib import Path
import fnmatch

import pytest
import yaml

import test_audio_cpu_workflow as linux_contracts
from workflow_privacy_assertions import assert_privacy_gated_upload


def workflow():
    path = Path(__file__).parents[2] / '.github/workflows/audio-cpu-windows.yml'
    return yaml.safe_load(path.read_text(encoding='utf-8'))


def audio_job():
    return workflow()['jobs']['audio_cpu_windows']


def test_windows_comparison_is_bounded_and_independent_of_other_model_modes():
    value = workflow()
    events = value.get('on', value.get(True))
    assert set(events) == {'workflow_dispatch', 'pull_request'}
    assert events['pull_request']['branches'] == ['main']
    assert value['permissions'] == {'contents': 'read', 'actions': 'read'}
    job = audio_job()
    assert job['runs-on'] == 'windows-latest'
    assert 0 < job['timeout-minutes'] <= 90
    assert job['strategy']['matrix'] == {'case_id': ['silence', 'padded_noise']}
    assert job['strategy']['fail-fast'] is False
    assert not job.get('continue-on-error')
    assert not any(step.get('continue-on-error') for step in job['steps'])


@pytest.mark.parametrize('name,should_run', [
    ('.github/workflows/audio-cpu-windows.yml', True),
    ('tools/tests/test_audio_cpu_windows_workflow.py', True),
    ('tools/embeddinggemma_tools/audio_cpu_reference.py', True),
    ('tools/embeddinggemma_tools/cpu_progress.py', True),
    ('tools/uv.lock', True),
    ('docs/release-plan.md', False),
    ('README.md', False),
    ('Packages/com.ayutaz.embeddinggemma/Runtime/EmbeddingGemma.cs', False),
])
def test_real_model_comparison_is_automatically_scoped_to_its_protocol(name, should_run):
    value = workflow()
    events = value.get('on', value.get(True))
    patterns = events['pull_request']['paths']
    assert any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns) is should_run


def test_windows_uses_the_reviewed_cpu_math_controls_and_an_explicit_shell():
    job = audio_job()
    assert job['defaults']['run'] == {'shell': 'bash', 'working-directory': 'tools'}
    env = job['env']
    assert env['OMP_NUM_THREADS'] == env['MKL_NUM_THREADS'] == env['OPENBLAS_NUM_THREADS'] == '4'
    assert env['MKL_CBWR'] == 'COMPATIBLE' and env['MKL_DYNAMIC'] == 'FALSE'
    assert env['CASE_ID'] == '${{ matrix.case_id }}'
    assert env['HF_HUB_DISABLE_TELEMETRY'] == '1'


def test_windows_runs_original_gate_before_actual_PCM_measurement_and_saved_capture_audit():
    steps = audio_job()['steps']
    modules = [
        'embeddinggemma_tools.selected_audio_generation fetch-distribution',
        'embeddinggemma_tools.audio_cpu_inputs',
        'embeddinggemma_tools.frozen_audio_chained_onnx',
        'embeddinggemma_tools.audio_cpu_reference measure',
        'embeddinggemma_tools.audio_cpu_reference audit',
    ]
    positions = [next(i for i, step in enumerate(steps) if name in step.get('run', '')) for name in modules]
    assert positions == sorted(set(positions))
    materialize, gate, measure, audit = [steps[i]['run'] for i in positions[1:]]
    assert '--fetch-missing-inputs' in materialize
    assert '--raw-input --variant ordered_original_midpoint_bmm' in gate
    assert '--case "$CASE_ID"' in gate and '--case "$CASE_ID"' in measure
    for command in (materialize, measure, audit):
        assert '--source "$GITHUB_SHA"' in command
    assert '--chain-report ../artifacts/audio-cpu-pinned/frozen-audio-chained-onnx.json' in measure
    assert not any('|| true' in step.get('run', '') for step in steps)


def test_windows_partial_evidence_is_guarded_and_cannot_be_confused_with_linux_artifacts():
    steps = audio_job()['steps']
    uploads = [(i, step) for i, step in enumerate(steps) if 'actions/upload-artifact@' in step.get('uses', '')]
    assert len(uploads) == 1
    index, upload = uploads[0]
    assert_privacy_gated_upload(upload)
    guard = steps[index - 1]
    assert guard['uses'] == './.github/actions/privacy-artifact-guard' and guard['if'] == 'always()'
    paths = upload['with']['path']
    assert paths == guard['with']['paths']
    assert upload['with']['name'] == 'audio-cpu-windows-full-${{ matrix.case_id }}-${{ github.sha }}'
    assert {'artifacts/audio-cpu-full/*.json', 'artifacts/audio-cpu-full/*.bin',
            'artifacts/audio-cpu-pinned/*.json', 'artifacts/audio-cpu-pinned/*.bin',
            'artifacts/audio-cpu-inputs/audio-cpu-inputs.json'} <= set(paths.splitlines())
    assert 'artifacts/audio-cpu-inputs/*' not in paths
    assert not any(token in paths for token in ('snapshot', '*.onnx', '*.pt2', '*.safetensors', '*.log'))


@pytest.mark.parametrize('marker', [
    'selected_audio_generation fetch-distribution',
    'embeddinggemma_tools.audio_cpu_inputs',
    'embeddinggemma_tools.frozen_audio_chained_onnx',
    'embeddinggemma_tools.audio_cpu_reference measure',
    'embeddinggemma_tools.audio_cpu_reference audit',
])
@pytest.mark.parametrize('command_status', [0, 7])
def test_windows_shell_keeps_raw_diagnostics_private_and_preserves_failure_status(tmp_path, monkeypatch, marker, command_status):
    monkeypatch.setattr(linux_contracts, 'audio_job', audio_job)
    linux_contracts.test_actual_shell_preserves_command_status_without_echoing_raw_diagnostics(
        tmp_path, marker, command_status)
