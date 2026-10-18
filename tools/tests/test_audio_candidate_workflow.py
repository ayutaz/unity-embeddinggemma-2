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
