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
        if name not in {'image_cpu_measurement', 'image_cpu_host_comparison'}:
            assert '!inputs.image_cpu_measurement' in legacy['if']


def test_same_host_comparison_is_explicit_and_disables_independent_profile_hosts():
    value = workflow()
    inputs = value.get('on', value.get(True))['workflow_dispatch']['inputs']
    assert inputs['image_cpu_host_comparison']['default'] is False
    assert '!inputs.image_cpu_host_comparison' in value['jobs']['image_cpu_measurement']['if']
    paired = value['jobs']['image_cpu_host_comparison']
    assert "github.event_name == 'workflow_dispatch'" in paired['if']
    assert 'inputs.image_cpu_host_comparison' in paired['if']
    assert paired['strategy']['matrix'] == {'host_sample': [1, 2]}
    assert paired['strategy']['fail-fast'] is False and paired['timeout-minutes'] == 30


def test_each_host_runs_every_shape_in_both_contexts_and_propagates_any_failure():
    job = workflow()['jobs']['image_cpu_host_comparison']
    run = next(step['run'] for step in job['steps'] if step.get('name') == 'Compare all shapes under native and compatible math on this host')
    assert 'for context in native compatible' in run
    assert 'for profile in square landscape portrait' in run
    assert 'env -u MKL_CBWR' in run and 'MKL_CBWR=COMPATIBLE' in run
    assert 'embeddinggemma_tools.image_cpu_reference measure' in run
    assert 'embeddinggemma_tools.image_cpu_reference audit' in run
    assert 'status=0' in run and 'status=1' in run and 'exit "$status"' in run
    assert '|| true' not in run and not any(step.get('continue-on-error') for step in job['steps'])


def test_same_host_comparison_retains_partial_evidence_under_exact_privacy_guard():
    steps = workflow()['jobs']['image_cpu_host_comparison']['steps']
    index, upload = next((i, step) for i, step in enumerate(steps) if 'actions/upload-artifact@' in step.get('uses', ''))
    assert_privacy_gated_upload(upload)
    guard = steps[index - 1]
    assert guard['uses'] == './.github/actions/privacy-artifact-guard'
    assert guard['with']['paths'] == upload['with']['path']
    assert 'image-cpu-host-' in upload['with']['path'] and 'snapshot' not in upload['with']['path']


def test_host_only_selection_reaches_admission_and_cannot_fall_through_to_legacy_models():
    value = workflow()
    for name, job in value['jobs'].items():
        if name not in {'image_cpu_measurement', 'image_cpu_host_comparison'}:
            assert '!inputs.image_cpu_host_comparison' in job['if']
    paired = value['jobs']['image_cpu_host_comparison']
    assert '&& inputs.image_cpu_measurement &&' not in paired['if']


def test_paired_admission_checks_required_primary_mode_before_dependencies_or_models():
    steps = workflow()['jobs']['image_cpu_host_comparison']['steps']
    index, guard = next((i, step) for i, step in enumerate(steps) if step.get('name') == 'Reject incompatible legacy selections')
    assert guard['env']['MEASUREMENT_SELECTED'] == '${{ inputs.image_cpu_measurement }}'
    assert "os.environ['MEASUREMENT_SELECTED'] == 'true'" in guard['run']
    sync = next(i for i, step in enumerate(steps) if step.get('run') == 'uv sync --locked')
    assert index < sync


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
