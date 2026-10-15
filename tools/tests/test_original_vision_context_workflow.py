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
    assert job['strategy']['matrix']['context'] == ['native', 'avx2']
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
