from pathlib import Path

import yaml


PATH=Path(__file__).parents[2]/'.github/workflows/audio-native-norm-replay.yml'


def workflow():
    return yaml.load(PATH.read_text(),Loader=yaml.BaseLoader)


def test_seed_preparation_is_an_explicit_false_by_default_manual_mode():
    value=workflow()['on']['workflow_dispatch']['inputs']['prepare_gpu_candidate']
    assert value['type']=='boolean' and value['default']=='false'
    assert 'prepare_gpu_candidate' in workflow()['jobs']['probe']['if']
    assert '!inputs.prepare_gpu_candidate' in workflow()['jobs']['probe']['if']


def test_seed_job_is_read_only_and_covers_both_candidate_profiles():
    value=workflow();job=value['jobs']['candidate-gpu-seed']
    assert value['permissions']=={'contents':'read','actions':'read'}
    assert 'workflow_dispatch' in job['if'] and 'inputs.prepare_gpu_candidate' in job['if']
    assert job['strategy']['matrix']['case']==['silence','padded_noise']
    assert job['strategy']['fail-fast']=='false'
    assert int(job['timeout-minutes'])<=30
    steps=job['steps'];commands='\n'.join(x.get('run','') for x in steps)
    assert 'audio_candidate_gpu_seed' in commands and '--fetch' in commands
    assert '--source "$GITHUB_SHA"' in commands
    assert 'uv sync --locked' in commands
    assert job['env']['MKL_CBWR']=='COMPATIBLE' and job['env']['MKL_DYNAMIC']=='FALSE'


def test_only_a_successful_seed_and_exact_privacy_selection_can_publish_models():
    steps=workflow()['jobs']['candidate-gpu-seed']['steps']
    checks=[s for s in steps if s.get('uses')=='./.github/actions/privacy-artifact-guard']
    uploads=[s for s in steps if 'actions/upload-artifact@' in s.get('uses','')]
    assert len(checks)==len(uploads)==1
    check=checks[0];upload=uploads[0]
    assert "success()" in check['if']
    assert "success()" in upload['if'] and "steps.privacy_gpu_seed.outcome == 'success'" in upload['if']
    assert check['with']['paths'].strip()==upload['with']['path'].strip()
    assert 'candidate-gpu-seed/audio.onnx' in upload['with']['path']
    assert 'candidate-gpu-seed/audio-gpu-seed.json' in upload['with']['path']
    assert 'candidate-input' not in upload['with']['path']
    assert int(upload['with']['retention-days'])==90
