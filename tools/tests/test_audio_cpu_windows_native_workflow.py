"""The PR baseline must not repeat the slow legacy ONNX execution route."""
from pathlib import Path
import fnmatch
import pytest
import yaml
from test_audio_cpu_windows_workflow import workflow,audio_job
import test_audio_cpu_workflow as shell_contract


def test_native_is_explicit_default_and_legacy_remains_a_manual_diagnostic():
    data=workflow();events=data.get('on',data.get(True))
    selection=events['workflow_dispatch']['inputs']['reference_executor']
    assert selection['type']=='choice' and selection['default']=='native-fp32'
    assert selection['options']==['native-fp32','onnx-cpu']
    steps=audio_job()['steps']
    native=next(step for step in steps if '--native-candidate' in step.get('run',''))
    legacy=next(step for step in steps if 'embeddinggemma_tools.frozen_audio_chained_onnx' in step.get('run',''))
    legacy_measure=next(step for step in steps if '--chain-report' in step.get('run',''))
    assert native['if']=="github.event_name != 'workflow_dispatch' || inputs.reference_executor == 'native-fp32'"
    assert legacy['if']==legacy_measure['if']=="github.event_name == 'workflow_dispatch' && inputs.reference_executor == 'onnx-cpu'"
    assert '--native-build ../artifacts/private-native-build' in native['run']
    assert '--snapshot ../artifacts/audio-cpu-snapshot' in native['run']
    assert '--source "$GITHUB_SHA"' in native['run'] and '--case "$CASE_ID"' in native['run']
    assert '--chain-report' not in native['run']


def test_native_snapshot_is_pinned_and_raw_logs_or_libraries_are_never_uploaded():
    steps=audio_job()['steps']
    snapshot=next(step for step in steps if 'snapshot_download' in step.get('run',''))
    assert snapshot['if']==next(step['if'] for step in steps if '--native-candidate' in step.get('run',''))
    assert 'revision=MODEL_REVISION' in snapshot['run'] and 'local_dir="../artifacts/audio-cpu-snapshot"' in snapshot['run']
    assert 'cohort.snapshot_input_sha256' in snapshot['run']
    uploads=[step for step in steps if 'actions/upload-artifact@' in step.get('uses','')]
    assert len(uploads)==1
    assert not any(token in uploads[0]['with']['path'] for token in ('snapshot','private-native-build','*.dll','*.so','*.log'))


@pytest.mark.parametrize('name',['tools/embeddinggemma_tools/native_audio_cpu_encoder.py','tools/embeddinggemma_tools/vision_cpu_simd_reference.py','tools/native_reference/block192_fma.cpp'])
def test_native_execution_inputs_trigger_the_windows_check(name):
    data=workflow();events=data.get('on',data.get(True))
    assert any(fnmatch.fnmatchcase(name,pattern) for pattern in events['pull_request']['paths'])


def test_required_model_numerical_workflow_is_still_required():
    root=Path(__file__).parents[2]
    required=yaml.safe_load((root/'.github/workflows/ci.yml').read_text(encoding='utf-8'))
    model=yaml.safe_load((root/'.github/workflows/model-reference.yml').read_text(encoding='utf-8'))
    assert 'model-reference' in required['jobs']['required']['needs']
    assert model['jobs']['pinned-audio']['uses']=='./.github/workflows/pinned-audio-acceptance.yml'


@pytest.mark.parametrize('marker',['snapshot_download','--native-candidate'])
@pytest.mark.parametrize('status',[0,7])
def test_new_actual_shell_steps_preserve_failures_and_keep_diagnostics_private(tmp_path,monkeypatch,marker,status):
    monkeypatch.setattr(shell_contract,'audio_job',audio_job)
    shell_contract.test_actual_shell_preserves_command_status_without_echoing_raw_diagnostics(tmp_path,marker,status)
