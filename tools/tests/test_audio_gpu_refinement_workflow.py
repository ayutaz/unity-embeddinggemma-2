from pathlib import Path

import yaml


def workflow():
    path=Path(__file__).parents[2]/'.github/workflows/audio-native-norm-replay.yml'
    return yaml.load(path.read_text(),Loader=yaml.BaseLoader)


def test_registered_manual_workflow_keeps_seed_mode_and_selects_refinement_explicitly():
    value=workflow();choice=value['on']['workflow_dispatch']['inputs']['preparation_stage']
    assert choice['type']=='choice' and choice['default']=='seed'
    assert choice['options']==['seed','refined-partitions']
    assert "inputs.preparation_stage == 'seed'" in value['jobs']['candidate-gpu-seed']['if']
    assert "inputs.preparation_stage == 'refined-partitions'" in value['jobs']['candidate-gpu-refinement']['if']
    assert 'inputs.prepare_gpu_candidate' in value['jobs']['candidate-gpu-refinement']['if']


def test_refinement_job_covers_two_fixed_profiles_with_read_only_pinned_execution():
    value=workflow();job=value['jobs']['candidate-gpu-refinement']
    assert value['permissions']=={'contents':'read','actions':'read'}
    assert job['strategy']['matrix']['case']==['silence','padded_noise']
    assert job['strategy']['max-parallel']=='2' and job['strategy']['fail-fast']=='false'
    assert int(job['timeout-minutes'])<=30
    assert job['env']['MKL_CBWR']=='COMPATIBLE' and job['env']['MKL_DYNAMIC']=='FALSE'
    checkout=next(s for s in job['steps'] if 'actions/checkout@' in s.get('uses',''))
    assert checkout['with']['ref']=='${{ github.sha }}'
    commands='\n'.join(s.get('run','') for s in job['steps'])
    assert 'uv sync --locked' in commands and 'embeddinggemma_tools.audio_gpu_refinement' in commands
    assert '--fetch' in commands and '--source "$GITHUB_SHA"' in commands


def test_failed_refinement_publishes_only_safe_receipt_and_successful_models_have_matching_guard():
    steps=workflow()['jobs']['candidate-gpu-refinement']['steps']
    checks={s['id']:s for s in steps if s.get('uses')=='./.github/actions/privacy-artifact-guard'}
    uploads=[s for s in steps if 'actions/upload-artifact@' in s.get('uses','')]
    assert len(checks)==len(uploads)==2
    for upload in uploads:
        guard=next(k for k in checks if f'steps.{k}.outcome' in upload['if'])
        assert checks[guard]['with']['paths'].strip()==upload['with']['path'].strip()
        assert int(upload['with']['retention-days'])==90
        if 'segment-' in upload['with']['path']:
            assert 'success()' in checks[guard]['if'] and 'success()' in upload['if']
            assert 'partition-plan.json' in upload['with']['path']
            assert 'candidate-input' not in upload['with']['path'] and 'templates' not in upload['with']['path']
        else:
            assert 'always()' in checks[guard]['if'] and 'always()' in upload['if']
            assert upload['with']['path'].strip()=='artifacts/candidate-gpu-refinement/audio-gpu-refinement.json'
