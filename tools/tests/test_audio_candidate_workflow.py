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
    assert 'embeddinggemma_tools.native_regression_distribution fetch' in commands
    assert '--candidate-distribution-directory ../artifacts/candidate/audio-runtime-reference' in commands
    assert 'embeddinggemma_tools.audio_generation_candidate fetch' not in commands
    assert '--candidate-directory ' not in commands
    assert '--raw-input --variant ordered_original_midpoint_bmm' in commands
    assert '--case "$CASE_ID"' in commands
    assert not job.get('continue-on-error')
    assert not any(step.get('continue-on-error') for step in job['steps'])


def test_candidate_audit_requires_all_actual_captures_without_promoting_required_gate():
    w=workflow();job=w['jobs']['candidate-audit']
    assert job['needs']==['candidate-execute']
    assert 'always()' in job['if'] and 'inputs.qualify_candidate_generation' in job['if']
    commands='\n'.join(step.get('run','') for step in job['steps'])
    assert 'embeddinggemma_tools.native_regression_distribution fetch-original' in commands
    assert 'embeddinggemma_tools.native_regression_distribution audit' in commands
    assert 'embeddinggemma_tools.audio_generation_candidate' not in commands
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


def test_candidate_preservation_is_retired_without_fetch_or_publication():
    w=workflow(); option=w['on']['workflow_dispatch']['inputs']['preserve_candidate_distribution']
    assert option['type']=='boolean' and option['default']=='false' and 'Retired' in option['description']
    for name in ('candidate-execute','candidate-audit','recover-golden','recover','native-reference'):
        assert '!inputs.preserve_candidate_distribution' in w['jobs'][name]['if']
    job=w['jobs']['retired-raw-source']
    assert "github.event_name == 'workflow_dispatch'" in job['if'] and 'inputs.preserve_candidate_distribution' in job['if']
    assert len(job['steps'])==1 and 'exit 1' in job['steps'][0]['run']
    assert '--source-archive' in job['steps'][0]['run'] and not any('uses' in step for step in job['steps'])
    assert not job.get('continue-on-error') and 'preserve-candidate' not in w['jobs']


def test_original_input_preservation_is_exclusive_and_does_not_download_or_execute_graphs():
    w=workflow();option=w['on']['workflow_dispatch']['inputs']['preserve_original_inputs']
    assert option['type']=='boolean' and option['default']=='false'
    for name in ('candidate-execute','candidate-audit','recover-golden','recover','native-reference'):
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


def test_native_onnx_identity_hosted_route_is_retired_without_model_download_or_upload():
    w=workflow();options=w['on']['workflow_dispatch']['inputs']
    assert len(options)<=10 and options['diagnose_candidate_onnx_identity']['default']=='false'
    assert 'Retired' in options['diagnose_candidate_onnx_identity']['description']
    for name in ('preserve-inputs','candidate-execute','candidate-audit','recover-golden','recover','native-reference'):
        assert '!inputs.diagnose_candidate_onnx_identity' in w['jobs'][name]['if']
    job=w['jobs']['retired-raw-source']
    assert "github.event_name == 'workflow_dispatch'" in job['if'] and 'inputs.diagnose_candidate_onnx_identity' in job['if']
    assert len(job['steps'])==1 and 'exit 1' in job['steps'][0]['run']
    assert not any('uses' in step or step.get('continue-on-error') for step in job['steps'])
    assert not job.get('continue-on-error') and 'candidate-native-identity' not in w['jobs']
