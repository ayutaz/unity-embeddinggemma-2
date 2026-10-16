import copy
import json
from pathlib import Path

import pytest
import yaml

from embeddinggemma_tools import pinned_audio_acceptance as gate


@pytest.fixture
def measured(monkeypatch):
    row = dict(success=True, source='a'*40, case_count=6, profile_count=2,
               values_compared=824320, candidate_tolerance_failures=0,
               native_all_six_context_qualified=False, gpu_validated=False,
               default_export_promoted=False)
    monkeypatch.setattr(gate, 'audit_reports', lambda *a, **kw: copy.deepcopy(row))
    return row


def test_original_gate_accepts_native_drift_without_claiming_gpu(measured):
    result = gate.validate([], 'original', source='a'*40)
    assert result['success'] and not result['native_all_six_context_qualified']
    assert not result['gpu_validated'] and not result['default_export_promoted']


@pytest.mark.parametrize('field,value', [('success',False), ('source','b'*40),
    ('case_count',5), ('profile_count',1), ('values_compared',824319),
    ('candidate_tolerance_failures',1)])
def test_missing_wrong_source_or_failed_original_comparison_blocks(measured,field,value):
    measured[field] = value
    with pytest.raises(ValueError):
        gate.validate([], 'original', source='a'*40)


def test_cli_writes_actual_audit_and_fails_on_numerical_failure(measured,tmp_path):
    output=tmp_path/'audit.json'
    assert gate.main(['--original','original','--reports',str(tmp_path),
                      '--source','a'*40,'--output',str(output)]) == 0
    assert json.loads(output.read_text())['values_compared']==824320
    measured['success']=False
    with pytest.raises(ValueError):
        gate.main(['--original','original','--reports',str(tmp_path),
                   '--source','a'*40,'--output',str(output)])


def test_required_ci_has_real_pinned_six_case_execution_and_fresh_cpu_is_diagnostic():
    root=Path(__file__).resolve().parents[2]/'.github/workflows'
    model=yaml.load((root/'model-reference.yml').read_text(encoding='utf-8'),Loader=yaml.BaseLoader)
    pinned=model['jobs']['pinned-audio']
    assert pinned['uses']=='./.github/workflows/pinned-audio-acceptance.yml'
    assert not pinned.get('if') and not pinned.get('continue-on-error')
    diagnostic=next(s for s in model['jobs']['reference']['steps']
                    if 'prepare-audio-encoder' in s.get('run',''))
    assert diagnostic['continue-on-error']=='true'
    assert '--collect-onnx-failures' in diagnostic['run']
    workflow=yaml.load((root/'pinned-audio-acceptance.yml').read_text(encoding='utf-8'),Loader=yaml.BaseLoader)
    cases=workflow['jobs']['execute']
    assert cases['strategy']['matrix']['case_id']==['silence','impulse','sine','nyquist','noise','padded_noise']
    assert cases['strategy']['fail-fast']=='false'
    commands='\n'.join(s.get('run','') for s in cases['steps'])
    assert '--raw-input --variant ordered_original_midpoint_bmm' in commands
    assert 'pinned_reference_archive resolve' in commands
    assert '--artifact-id ${{ steps.original.outputs.artifact-id }}' in commands
    assert 'frozen-stage-source' not in commands
    audit=workflow['jobs']['audit']
    assert audit['needs']==['execute'] and not audit.get('continue-on-error')
    assert 'embeddinggemma_tools.pinned_audio_acceptance' in '\n'.join(s.get('run','') for s in audit['steps'])
