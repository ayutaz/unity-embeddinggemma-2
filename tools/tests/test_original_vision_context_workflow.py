from pathlib import Path

import yaml
import json
import sys
import types

import pytest


def workflow():
    path = Path(__file__).resolve().parents[2] / '.github/workflows/reference-context.yml'
    return yaml.safe_load(path.read_text(encoding='utf-8'))


def test_original_vision_probe_is_separate_opt_in_and_keeps_default_matrix():
    value = workflow()
    trigger = value.get('on', value.get(True))
    inputs = trigger['workflow_dispatch'].get('inputs', {}) if trigger['workflow_dispatch'] else {}
    assert 'original_vision' in inputs, 'Frozen original vision dispatch input is missing'
    assert inputs['original_vision']['type'] == 'boolean' and inputs['original_vision']['default'] is False
    assert 'original_vision_probe' in value['jobs'], 'Frozen original vision replay job is missing'
    job = value['jobs']['original_vision_probe']
    assert 'inputs.original_vision' in job['if'] and 'workflow_dispatch' in job['if']
    assert job['strategy']['matrix']['context'] == '${{ inputs.require_avx512_native && fromJSON(\'["native"]\') || fromJSON(\'["native","avx2"]\') }}'
    assert value['jobs']['probe']['strategy']['matrix']['modality'] == ['audio', 'image']
    assert '!inputs.original_vision' in value['jobs']['probe']['if']


def test_original_vision_probe_fetches_exact_old_graph_and_never_changes_golden():
    jobs = workflow()['jobs']
    assert 'original_vision_probe' in jobs, 'Frozen original vision replay job is missing'
    steps = jobs['original_vision_probe']['steps']
    commands = '\n'.join(step.get('run', '') for step in steps)
    assert '--artifact-id 11774238857' in commands
    for member in ('vision-reference.json', 'vision-validation.json', 'square/vision.pt2', 'square/0-pixels-fp32.bin'):
        assert f'--member {member}' in commands
    for digest in ('3dfccd1a487ce04df3f8d640483ca045f7a7a56231bc1096ad0f845c923a29c6',
                   'ab1aa775b553f33a5e2892abed2cdcef7768e35fbad6fa5af9ec3286a7ba90ba',
                   'f9c5175a1d75289ce814a9651f496f9bdae5e873ac66199a5c3d03e0172c16cc',
                   'd25852af1b53b8f4ed484b760280b1859b8a3b63e3456f7cbb378f33877761b7'):
        assert digest in commands
    assert 'e2a6dc01fbca8c9be1f3bbe875325ad338de0ca0' in commands
    assert 'replay_saved_reference_context' in commands and 'reference_reproduced' in commands
    assert 'snapshot_download' not in commands and 'prepare_vision' not in commands
    assert 'lscpu --json' in commands
    upload = next(step for step in steps if 'actions/upload-artifact@' in step.get('uses', ''))
    assert upload['if'] == 'always()' and '.pt2' not in upload['with']['path']


def test_original_probe_observes_nodes_on_native_host_and_uploads_only_selected_captures():
    steps = workflow()['jobs']['original_vision_probe']['steps']
    execute = next(step['run'] for step in steps
                   if step.get('name') == 'Replay original saved graph against unchanged golden')
    assert 'capture_nodes=' in execute and "== 'native'" in execute
    assert "'add_1', 'cat', 'cat_1', 'add_15', 'mul_372', 'mul_373', 'mm_113'" in execute
    upload = next(step for step in steps if 'actions/upload-artifact@' in step.get('uses', ''))
    assert 'artifacts/original-context-results/captures/*.bin' in upload['with']['path']


def test_original_probe_checks_pool_layout_on_captured_last_encoder_output_after_failed_gate():
    steps=workflow()['jobs']['original_vision_probe']['steps']
    execute=next(step['run'] for step in steps if step.get('name')=='Replay original saved graph against unchanged golden')
    assert "'add_225'" in execute, 'Last original encoder output must be observed directly'
    step=next(step for step in steps if step.get('name')=='Compare original official and saved pooling layouts')
    assert 'always()' in step['if'] and 'native' in step['if']
    assert 'embeddinggemma_tools.vision_pooling_context' in step['run']


def test_original_probe_captures_bounded_native_first_layer_nonlinearity_operands():
    steps=workflow()['jobs']['original_vision_probe']['steps']
    execute=next(step['run'] for step in steps if step.get('name')=='Replay original saved graph against unchanged golden')
    assert 'capture_row_prefixes=' in execute
    for name in ('add_10','_softmax','view_21','gelu','mul_4'):
        assert f"'{name}'" in execute
        assert f"'{name}': 16" in execute


def test_blas_context_probe_keeps_aten_native_and_checks_original_patch_before_full_replay():
    value=workflow();trigger=value.get('on',value.get(True));inputs=trigger['workflow_dispatch']['inputs']
    assert 'original_blas_context' in inputs,'Independent original BLAS context input is missing'
    assert inputs['original_blas_context']['default']=='native' and inputs['original_blas_context']['options']==['native','avx2','avx2_isa']
    steps=value['jobs']['original_vision_probe']['steps']
    configure=next(s for s in steps if s.get('name')=='Constrain only original native BLAS context')
    assert 'original_blas_context' in configure['if'] and "matrix.context == 'native'" in configure['if']
    assert 'MKL_CBWR=AVX2' in configure['run'] and 'ATEN_CPU_CAPABILITY' not in configure['run'] and 'DNNL_MAX_CPU_ISA' not in configure['run']
    prefix=next(s for s in steps if s.get('name')=='Gate original patch before constrained BLAS full replay')
    assert 'original_blas_context' in prefix['if'] and "matrix.context == 'native'" in prefix['if']
    assert steps.index(prefix)<next(i for i,s in enumerate(steps) if s.get('name')=='Replay original saved graph against unchanged golden')
    assert 'run_saved_graph_prefix' in prefix['run'] and "'add_1'" in prefix['run']
    assert '1e99fa140cb0084e624b5019d96ae424192eff77346ef2f7ffea892c1bea0607' in prefix['run']
    assert 'full_encoder_executed=False' in prefix['run'] and 'raise SystemExit' in prefix['run']


def test_isa_only_probe_is_distinct_from_cnr_and_requires_original_first_layer_before_replay():
    value=workflow();trigger=value.get('on',value.get(True));option=trigger['workflow_dispatch']['inputs']['original_blas_context']
    assert 'avx2_isa' in option['options'],'Independent MKL ISA-only probe is missing'
    steps=value['jobs']['original_vision_probe']['steps']
    configure=next(s for s in steps if s.get('name')=='Constrain only original native BLAS instruction set')
    assert "original_blas_context == 'avx2_isa'" in configure['if'] and "matrix.context == 'native'" in configure['if']
    assert 'MKL_ENABLE_INSTRUCTIONS=AVX2' in configure['run']
    assert 'MKL_CBWR=' not in configure['run'] and 'ATEN_CPU_CAPABILITY=' not in configure['run']
    gate=next(s for s in steps if s.get('name')=='Gate original first layer before ISA-only full replay')
    assert "original_blas_context == 'avx2_isa'" in gate['if']
    assert steps.index(gate)<next(i for i,s in enumerate(steps) if s.get('name')=='Replay original saved graph against unchanged golden')
    assert 'run_saved_graph_prefix' in gate['run'] and "'add_15'" in gate['run']
    assert 'fb33092e54e997da999b87246aefe2089ce2d0f376a89586b8c329db069bb7fa' in gate['run']
    assert 'full_encoder_executed=False' in gate['run'] and 'raise SystemExit' in gate['run']


def test_rotary_only_dispatch_captures_original_position_constants_for_all_profiles_without_encoder():
    value=workflow();trigger=value.get('on',value.get(True));inputs=trigger['workflow_dispatch']['inputs']
    assert 'original_rotary_only' in inputs,'Original data-independent rotary dispatch is missing'
    assert inputs['original_rotary_only']['default'] is False
    assert '!inputs.original_rotary_only' in value['jobs']['original_vision_probe']['if']
    job=value['jobs']['original_rotary_probe'];assert 'inputs.original_rotary_only' in job['if']
    assert job['strategy']['matrix']['profile']==['square','landscape','portrait']
    commands='\n'.join(s.get('run','') for s in job['steps'])
    assert '--artifact-id 11774238857' in commands and 'run_saved_graph_prefix' in commands
    assert 'placeholder' in commands and 'full_encoder_executed=False' in commands
    assert 'replay_saved_reference_context' not in commands and 'prepare_vision' not in commands
    upload=next(s for s in job['steps'] if 'actions/upload-artifact@' in s.get('uses',''))
    assert '.pt2' not in upload['with']['path'] and 'original-rotary-' in upload['with']['name']


@pytest.mark.parametrize('capability,accepted', [('AVX512',True),('AVX2',False)])
def test_avx512_only_dispatch_rejects_wrong_actual_capability_before_model_fetch(
        tmp_path,monkeypatch,capability,accepted):
    value=workflow();trigger=value.get('on',value.get(True));inputs=trigger['workflow_dispatch']['inputs']
    assert 'require_avx512_native' in inputs, 'Explicit native AVX512 prerequisite is missing'
    assert inputs['require_avx512_native']['default'] is False
    job=value['jobs']['original_vision_probe'];matrix=job['strategy']['matrix']['context']
    assert 'inputs.require_avx512_native' in matrix and "fromJSON('[\"native\"]')" in matrix
    assert "fromJSON('[\"native\",\"avx2\"]')" in matrix
    steps=job['steps'];gate=next(s for s in steps if s.get('name')=='Require actual AVX512 before original model download')
    assert gate['if']=='inputs.require_avx512_native'
    assert steps.index(gate)<next(i for i,s in enumerate(steps) if s.get('name')=='Fetch original frozen graph and first fixture')
    fake=types.ModuleType('torch');fake.backends=types.SimpleNamespace(cpu=types.SimpleNamespace(get_cpu_capability=lambda:capability))
    monkeypatch.setitem(sys.modules,'torch',fake);work=tmp_path/'tools';work.mkdir();monkeypatch.chdir(work)
    monkeypatch.setenv('PROBE_CONTEXT','native');monkeypatch.setenv('GITHUB_SHA','b'*40)
    script=gate['run'].split("<<'PY'\n",1)[1].rsplit('\nPY',1)[0]
    if accepted:exec(compile(script,'<avx512-prerequisite>','exec'),{})
    else:
        with pytest.raises(SystemExit,match='Actual AVX512'):
            exec(compile(script,'<avx512-prerequisite>','exec'),{})
    report=json.loads((tmp_path/'artifacts/original-context-results/cpu-prerequisite.json').read_text())
    assert report['actual_cpu_capability']==capability and report['satisfied'] is accepted
    assert report['model_executed'] is False


@pytest.mark.parametrize('conditions,reproduced', [
    ([dict(executed=True, mismatched_elements=0)], True),
    ([dict(executed=True, mismatched_elements=1)], False),
    ([dict(executed=False, mismatched_elements=0)], False),
])
def test_replay_step_persists_real_report_schema_and_rejects_missing_or_failed_matches(
        tmp_path, monkeypatch, conditions, reproduced):
    step = next(step for step in workflow()['jobs']['original_vision_probe']['steps']
                if step.get('name') == 'Replay original saved graph against unchanged golden')
    script = step['run'].split("<<'PY'\n", 1)[1].rsplit('\nPY', 1)[0]
    fake = types.ModuleType('embeddinggemma_tools.reference_context_failure')
    fake.replay_saved_reference_context = lambda *args, **kwargs: dict(
        thread_contexts=dict(conditions=conditions), expected_unchanged=True)
    monkeypatch.setitem(sys.modules, fake.__name__, fake)
    work = tmp_path / 'tools'
    work.mkdir()
    monkeypatch.chdir(work)
    monkeypatch.setenv('GITHUB_SHA', 'a' * 40)
    monkeypatch.setenv('PROBE_CONTEXT', 'native')
    if reproduced:
        exec(compile(script, '<workflow-replay>', 'exec'), {})
    else:
        with pytest.raises(SystemExit, match='did not reproduce'):
            exec(compile(script, '<workflow-replay>', 'exec'), {})
    report = json.loads((tmp_path / 'artifacts/original-context-results/original-image-native.json').read_text())
    assert report['reference_reproduced'] is reproduced
    assert report['thread_contexts']['conditions'] == conditions
