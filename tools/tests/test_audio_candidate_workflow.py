from pathlib import Path

import yaml

ROOT=Path(__file__).resolve().parents[2]


def workflow():
    return yaml.load((ROOT/'.github/workflows/audio-runtime-reference.yml').read_text(encoding='utf-8'),Loader=yaml.BaseLoader)


def test_candidate_qualification_is_explicit_manual_and_keeps_original_modes_separate():
    w=workflow();option=w['on']['workflow_dispatch']['inputs']['qualify_candidate_generation']
    assert option['type']=='boolean' and option['default']=='false'
    for name in ('recover-golden','recover','native-reference'):
        assert '!inputs.qualify_candidate_generation' in w['jobs'][name]['if']
    job=w['jobs']['candidate-execute']
    assert "github.event_name == 'workflow_dispatch'" in job['if']
    assert 'inputs.qualify_candidate_generation' in job['if']
    assert job['permissions']=={'contents':'read','actions':'read'}
    assert job['strategy']['matrix']['case_id']==['silence','impulse','sine','nyquist','noise','padded_noise']
    assert job['timeout-minutes']=='40'
    assert job['env']['MKL_CBWR']=='COMPATIBLE' and job['env']['MKL_DYNAMIC']=='FALSE'
    assert job['env']['OMP_NUM_THREADS']==job['env']['MKL_NUM_THREADS']=='4'
    commands='\n'.join(step.get('run','') for step in job['steps'])
    assert 'embeddinggemma_tools.audio_generation_candidate fetch' in commands
    assert '--candidate-directory ../artifacts/candidate/audio-runtime-reference' in commands
    assert '--raw-input --variant ordered_original_midpoint_bmm' in commands
    assert '--case "$CASE_ID"' in commands
    assert not job.get('continue-on-error')
    assert not any(step.get('continue-on-error') for step in job['steps'])


def test_candidate_audit_requires_all_actual_captures_without_promoting_required_gate():
    w=workflow();job=w['jobs']['candidate-audit']
    assert job['needs']==['candidate-execute']
    assert 'always()' in job['if'] and 'inputs.qualify_candidate_generation' in job['if']
    commands='\n'.join(step.get('run','') for step in job['steps'])
    assert 'embeddinggemma_tools.audio_generation_candidate fetch-original' in commands
    assert 'embeddinggemma_tools.audio_generation_candidate audit' in commands
    assert '--source "$EXECUTION_SOURCE"' in commands
    assert '--reports ../artifacts/candidate-cases' in commands
    for name in ('candidate-execute','candidate-audit'):
        steps=w['jobs'][name]['steps']
        for index,step in enumerate(steps):
            if 'actions/upload-artifact@' not in step.get('uses',''):continue
            guard=steps[index-1]
            assert guard['uses']=='./.github/actions/privacy-artifact-guard'
            assert step['with']['path']==guard['with']['paths']
            assert 'always()' in step['if'] and "outcome == 'success'" in step['if']
    required=yaml.load((ROOT/'.github/workflows/model-reference.yml').read_text(encoding='utf-8'),Loader=yaml.BaseLoader)
    assert required['jobs']['pinned-audio']['uses']=='./.github/workflows/pinned-audio-acceptance.yml'
    assert not required['jobs']['pinned-audio'].get('if')
    assert not required['jobs']['pinned-audio'].get('continue-on-error')


def test_tools_working_directory_is_created_before_any_shell_step_uses_it():
    for job in workflow()['jobs'].values():
        if job.get('defaults',{}).get('run',{}).get('working-directory')!='tools':continue
        checkout=next(i for i,step in enumerate(job['steps']) if step.get('uses','').startswith('actions/checkout@'))
        for index,step in enumerate(job['steps']):
            if 'run' in step and step.get('working-directory','tools')=='tools':
                assert index>checkout,'A shell step used tools before checkout created it'


def test_candidate_preservation_is_manual_exclusive_and_uploads_only_verified_distribution():
    w=workflow(); option=w['on']['workflow_dispatch']['inputs']['preserve_candidate_distribution']
    assert option['type']=='boolean' and option['default']=='false'
    for name in ('candidate-execute','candidate-audit','recover-golden','recover','native-reference'):
        assert '!inputs.preserve_candidate_distribution' in w['jobs'][name]['if']
    job=w['jobs']['preserve-candidate']
    assert "github.event_name == 'workflow_dispatch'" in job['if']
    for flag in ('qualify_candidate_generation','recover_golden_only','recover_original_bytes'):
        assert '!inputs.'+flag in job['if']
    assert job['permissions']=={'contents':'read','actions':'read'}
    assert job['timeout-minutes']=='40' and not job.get('continue-on-error')
    assert job['env']['MKL_CBWR']=='COMPATIBLE' and job['env']['MKL_DYNAMIC']=='FALSE'
    assert job['env']['OMP_NUM_THREADS']==job['env']['MKL_NUM_THREADS']=='4'
    commands='\n'.join(step.get('run','') for step in job['steps'])
    assert 'embeddinggemma_tools.audio_candidate_distribution' in commands
    assert '--fetch' in commands and '--source "$EXECUTION_SOURCE"' in commands
    assert 'frozen_audio_chained_onnx' not in commands
    upload=job['steps'][-1]; guard=job['steps'][-2]
    assert guard['uses']=='./.github/actions/privacy-artifact-guard'
    assert upload['with']['path']==guard['with']['paths']=='artifacts/candidate-distribution/**'
    assert upload['with']['retention-days']=='90'
    assert upload['with']['if-no-files-found']=='error'
    assert 'success()' in upload['if'] and "outcome == 'success'" in upload['if']
    assert not any(step.get('continue-on-error') for step in job['steps'])


def test_original_input_preservation_is_exclusive_and_does_not_download_or_execute_graphs():
    w=workflow();option=w['on']['workflow_dispatch']['inputs']['preserve_original_inputs']
    assert option['type']=='boolean' and option['default']=='false'
    for name in ('preserve-candidate','candidate-execute','candidate-audit','recover-golden','recover','native-reference'):
        assert '!inputs.preserve_original_inputs' in w['jobs'][name]['if']
    job=w['jobs']['preserve-inputs']
    assert "github.event_name == 'workflow_dispatch'" in job['if']
    assert 'inputs.preserve_original_inputs' in job['if']
    for name in ('preserve_candidate_distribution','qualify_candidate_generation','recover_golden_only','recover_original_bytes'):
        assert '!inputs.'+name in job['if']
    assert job['permissions']=={'contents':'read','actions':'read'}
    assert job['timeout-minutes']=='10'
    commands='\n'.join(step.get('run','') for step in job['steps'])
    assert 'embeddinggemma_tools.original_audio_input_preservation' in commands and '--fetch' in commands
    assert 'audio_candidate_distribution' not in commands and 'frozen_audio_chained_onnx' not in commands
    guard,upload=job['steps'][-2:]
    assert guard['uses']=='./.github/actions/privacy-artifact-guard'
    assert guard['with']['paths']==upload['with']['path']=='artifacts/original-audio-inputs/**'
    assert upload['with']['retention-days']=='90' and upload['with']['if-no-files-found']=='error'
    assert 'success()' in upload['if'] and "outcome == 'success'" in upload['if']
    assert not job.get('continue-on-error') and not any(s.get('continue-on-error') for s in job['steps'])


def test_native_onnx_identity_diagnostic_keeps_mismatches_failed_and_uploads_no_models():
    w=workflow();options=w['on']['workflow_dispatch']['inputs']
    assert len(options)<=10 and options['diagnose_candidate_onnx_identity']['default']=='false'
    for name in ('preserve-inputs','preserve-candidate','candidate-execute','candidate-audit','recover-golden','recover','native-reference'):
        assert '!inputs.diagnose_candidate_onnx_identity' in w['jobs'][name]['if']
    job=w['jobs']['candidate-native-identity']
    assert "github.event_name == 'workflow_dispatch'" in job['if'] and 'inputs.diagnose_candidate_onnx_identity' in job['if']
    for name in ('preserve_original_inputs','preserve_candidate_distribution','qualify_candidate_generation','recover_golden_only','recover_original_bytes'):
        assert '!inputs.'+name in job['if']
    assert job['strategy']['matrix']['profile']==['frames-1119-valid-1119','frames-1119-valid-839']
    assert job['strategy']['fail-fast']=='false' and job['strategy']['max-parallel']=='2'
    assert job['permissions']=={'contents':'read','actions':'read'}
    assert job['timeout-minutes']=='20' and not job.get('continue-on-error')
    commands='\n'.join(step.get('run','') for step in job['steps'])
    assert 'embeddinggemma_tools.audio_candidate_onnx_identity' in commands and '--fetch' in commands
    assert '--profile "$PROFILE"' in commands and '--source "$EXECUTION_SOURCE"' in commands
    assert not any(step.get('continue-on-error') for step in job['steps'])
    guard,upload=job['steps'][-2:]
    assert guard['uses']=='./.github/actions/privacy-artifact-guard'
    assert guard['with']['paths']==upload['with']['path']=='artifacts/candidate-native-identity/identity.json'
    assert 'always()' in upload['if'] and "outcome == 'success'" in upload['if']
    assert upload['with']['retention-days']=='90' and upload['with']['if-no-files-found']=='error'
