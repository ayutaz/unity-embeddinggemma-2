from pathlib import Path

import yaml

from workflow_privacy_assertions import assert_privacy_gated_upload


def workflow():
    return yaml.safe_load((Path(__file__).resolve().parents[2] /
        '.github/workflows/reference-context.yml').read_text(encoding='utf-8'))


def test_new_image_CPU_measurement_is_explicit_manual_and_legacy_modes_are_exclusive():
    value = workflow()
    dispatch = value.get('on', value.get(True))['workflow_dispatch']['inputs']
    assert dispatch['image_cpu_measurement']['type'] == 'boolean'
    assert dispatch['image_cpu_measurement']['default'] is False
    job = value['jobs']['image_cpu_measurement']
    assert "github.event_name == 'workflow_dispatch'" in job['if']
    assert 'inputs.image_cpu_measurement' in job['if']
    for name, legacy in value['jobs'].items():
        if name != 'image_cpu_measurement': assert '!inputs.image_cpu_measurement' in legacy['if']


def test_three_fixed_profiles_are_bounded_and_failures_remain_failures():
    job = workflow()['jobs']['image_cpu_measurement']
    assert job['runs-on'] == 'ubuntu-latest'
    assert job['strategy']['matrix'] == {'profile': ['square', 'landscape', 'portrait']}
    assert job['strategy']['fail-fast'] is False and job['timeout-minutes'] == 30
    assert job['env']['OMP_NUM_THREADS'] == job['env']['MKL_NUM_THREADS'] == '4'
    assert not job.get('continue-on-error', False)
    assert not any(step.get('continue-on-error', False) for step in job['steps'])


def test_actual_measurement_uses_original_public_data_and_separate_independent_audit():
    steps = workflow()['jobs']['image_cpu_measurement']['steps']
    commands = [(i, step.get('run', '')) for i, step in enumerate(steps)]
    fetch = next(i for i, run in commands if 'embeddinggemma_tools.image_reference_distribution' in run)
    measure = next(i for i, run in commands if 'embeddinggemma_tools.image_cpu_reference measure' in run)
    audit = next(i for i, run in commands if 'embeddinggemma_tools.image_cpu_reference audit' in run)
    assert fetch < measure < audit
    assert '--profile "$PROFILE"' in steps[measure]['run']
    assert '--source "$GITHUB_SHA"' in steps[measure]['run'] and '--source "$GITHUB_SHA"' in steps[audit]['run']
    assert not any('artifact_zip' in run or 'prepare-vision' in run or '|| true' in run for _, run in commands)
    exclusive = next(step for step in steps if step.get('name') == 'Reject incompatible legacy selections')
    assert 'original_vision' in exclusive['env']['LEGACY_SELECTED']
    assert 'original_blas_context' in exclusive['env']['LEGACY_SELECTED']


def test_partial_numerical_evidence_has_matching_privacy_guard_and_no_weights_upload():
    steps = workflow()['jobs']['image_cpu_measurement']['steps']
    index, upload = next((i, step) for i, step in enumerate(steps) if 'actions/upload-artifact@' in step.get('uses', ''))
    assert_privacy_gated_upload(upload)
    guard = steps[index - 1]
    assert guard['uses'] == './.github/actions/privacy-artifact-guard'
    assert guard['if'] == 'always()'
    assert guard['with']['paths'] == upload['with']['path']
    assert 'artifacts/image-cpu-' in upload['with']['path']
    assert 'snapshot' not in upload['with']['path'] and 'original-image-references' not in upload['with']['path']
