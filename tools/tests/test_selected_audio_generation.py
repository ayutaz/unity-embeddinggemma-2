import copy
import hashlib
import importlib
import json
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]


def module():
    return importlib.import_module('embeddinggemma_tools.selected_audio_generation')


def documents():
    manifest = json.loads((ROOT/'tools/references/selected-audio-generation.json').read_text(encoding='utf-8-sig'))
    evidence = {key: json.loads((ROOT/entry['path']).read_text(encoding='utf-8-sig'))
                for key, entry in manifest['proofs'].items()}
    return manifest, evidence


def test_selection_requires_actual_numeric_encoder_roles_and_both_saved_player_proofs(monkeypatch):
    m = module()
    manifest, evidence = documents()
    m.validate_evidence(manifest, evidence)
    monkeypatch.setattr(m, '_runtime_tree', lambda root: manifest['sdk_runtime_tree'])
    selected = m.checked_selection(ROOT)
    assert selected['model_generation'] == 'audio-e9de158-20261018'
    assert selected['release_ready'] is False and selected['full_multimodal_acceptance'] is False


@pytest.mark.parametrize('kind', ['numeric', 'seed', 'partitions', 'encoder', 'roles', 'saved_fp32', 'saved_float16'])
def test_selection_rejects_any_missing_or_failed_required_proof(kind):
    manifest, evidence = documents()
    evidence[kind]['success'] = False
    with pytest.raises(ValueError): module().validate_evidence(manifest, evidence)
    del evidence[kind]
    with pytest.raises(ValueError): module().validate_evidence(manifest, evidence)


@pytest.mark.parametrize('problem', ['golden', 'release', 'scope', 'contract', 'encoder_values', 'encoder_error',
    'encoder_readback', 'roles', 'fp32_descriptor', 'float16_descriptor', 'float16_coefficients',
    'float16_models', 'float16_runtime', 'float16_missing_profile', 'float16_cosine', 'fp32_cosine', 'numeric_cases'])
def test_qualified_selection_cannot_be_forged_by_partial_or_mismatched_measurements(problem):
    manifest, e = documents()
    if problem == 'golden': manifest['original_golden_changed'] = True
    elif problem == 'release': manifest['release_ready'] = True
    elif problem == 'scope': manifest['scope'] = 'all_platforms'
    elif problem == 'contract': manifest['candidate_contract_sha256'] = '0'*64
    elif problem == 'encoder_values': e['encoder']['values'] -= 1
    elif problem == 'encoder_error': e['encoder']['mismatched_elements'] = 1
    elif problem == 'encoder_readback': e['encoder']['intermediate_readback_count'] = 1
    elif problem == 'roles': e['roles']['final_vector_count'] = 17
    elif problem == 'fp32_descriptor': e['saved_fp32']['player']['descriptor_sha256'] = '0'*64
    elif problem == 'float16_descriptor': e['saved_float16']['player']['descriptor_sha256'] = '0'*64
    elif problem == 'float16_coefficients': e['saved_float16']['preparation']['non_model_bytes_identical'] = False
    elif problem == 'float16_models': e['saved_float16']['preparation']['models_bound'] = 284
    elif problem == 'float16_runtime': e['saved_float16']['player']['runtime_tree_sha'] = '0'*40
    elif problem == 'float16_missing_profile': e['saved_float16']['player']['results'].pop()
    elif problem == 'float16_cosine': e['saved_float16']['player']['results'][0]['minimum_recomputed_cosine'] = .98
    elif problem == 'fp32_cosine': e['saved_fp32']['player']['results'][0]['minimum_recomputed_cosine'] = .998
    elif problem == 'numeric_cases': e['numeric']['cases'].pop()
    with pytest.raises(ValueError): module().validate_evidence(manifest, e)


def test_loader_rejects_modified_proof_bytes_and_runtime_tree(tmp_path, monkeypatch):
    m = module(); manifest, evidence = documents()
    destination = tmp_path/'tools/references/selected-audio-generation.json'
    destination.parent.mkdir(parents=True); destination.write_text(json.dumps(manifest))
    for key, entry in manifest['proofs'].items():
        path = tmp_path/entry['path']; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(evidence[key]))
    monkeypatch.setattr(m, '_runtime_tree', lambda root: '0'*40)
    with pytest.raises(ValueError): m.checked_selection(tmp_path)
    monkeypatch.setattr(m, '_runtime_tree', lambda root: manifest['sdk_runtime_tree'])
    m.checked_selection(tmp_path)
    path = tmp_path/manifest['proofs']['numeric']['path']
    data = json.loads(path.read_text()); data['source'] = '0'*40; path.write_text(json.dumps(data))
    with pytest.raises(ValueError): m.checked_selection(tmp_path)


def test_current_ci_gate_still_requires_six_actual_captures_at_current_source(monkeypatch):
    m = module(); manifest, _ = documents()
    monkeypatch.setattr(m, 'checked_selection', lambda root=None: manifest)
    result = dict(success=True, source='a'*40, model_generation=manifest['model_generation'],
                  candidate_contract_sha256=manifest['candidate_contract_sha256'], case_count=6, profile_count=2,
                  values_compared=824320, candidate_tolerance_failures=0, gpu_validated=False, generation_qualified=False)
    monkeypatch.setattr(m.candidate, 'audit_candidate_reports', lambda *args, **kwargs: copy.deepcopy(result))
    actual = m.validate_current_reports([], 'original', source='a'*40)
    assert actual['required_numeric_gate_passed'] is True and actual['gpu_validated'] is False
    assert actual['generation_qualified'] is False
    for field, value in [('source','b'*40), ('values_compared',824319), ('case_count',5),
                         ('profile_count',1), ('candidate_tolerance_failures',1), ('success',False)]:
        broken = copy.deepcopy(result); broken[field] = value
        monkeypatch.setattr(m.candidate, 'audit_candidate_reports', lambda *a, **kw: broken)
        with pytest.raises(ValueError): m.validate_current_reports([], 'original', source='a'*40)


def test_cli_checks_selection_before_network_and_never_defaults_to_another_generation(monkeypatch, tmp_path):
    m = module(); calls = []; manifest, _ = documents()
    monkeypatch.setattr(m, 'checked_selection', lambda root=None: calls.append('selection') or manifest)
    monkeypatch.setattr(m.candidate, 'fetch_candidate', lambda *a, **kw: calls.append(('fetch', kw['case_id'])))
    assert m.main(['fetch', '--output', str(tmp_path/'candidate'), '--original', str(tmp_path/'original'), '--case', 'noise']) == 0
    assert calls == ['selection', ('fetch', 'noise')]
    calls.clear()
    def rejected(root=None): raise ValueError('changed selection')
    monkeypatch.setattr(m, 'checked_selection', rejected)
    with pytest.raises(ValueError):
        m.main(['fetch', '--output', str(tmp_path/'candidate'), '--original', str(tmp_path/'original'), '--case', 'noise'])
    assert calls == []


def test_cli_writes_the_actual_audit_and_never_overwrites_it(monkeypatch, tmp_path):
    m = module(); manifest, _ = documents(); output = tmp_path/'result.json'
    monkeypatch.setattr(m, 'checked_selection', lambda root=None: manifest)
    result = {'success':True, 'required_numeric_gate_passed':True, 'gpu_validated':False}
    monkeypatch.setattr(m, 'validate_current_reports', lambda *a, **kw: copy.deepcopy(result))
    args = ['audit', '--reports', str(tmp_path), '--original', str(tmp_path/'original'), '--source', 'a'*40, '--output', str(output)]
    assert m.main(args) == 0 and json.loads(output.read_text()) == result
    with pytest.raises(ValueError): m.main(args)
    assert json.loads(output.read_text()) == result


def test_required_workflow_selects_reviewed_generation_and_still_executes_every_original_raw_input():
    workflow = yaml.load((ROOT/'.github/workflows/pinned-audio-acceptance.yml').read_text(encoding='utf-8'), Loader=yaml.BaseLoader)
    execute = workflow['jobs']['execute']; audit = workflow['jobs']['audit']
    assert execute['strategy']['matrix']['case_id'] == ['silence','impulse','sine','nyquist','noise','padded_noise']
    assert execute['strategy']['fail-fast'] == 'false' and not execute.get('continue-on-error')
    commands = '\n'.join(step.get('run','') for step in execute['steps'])
    assert 'embeddinggemma_tools.selected_audio_generation fetch-distribution' in commands
    assert '--candidate-distribution-directory ../artifacts/candidate/audio-runtime-reference' in commands
    assert '--candidate-directory ' not in commands
    assert '--raw-input --variant ordered_original_midpoint_bmm' in commands
    assert 'pinned_reference_archive resolve' not in commands
    assert not any(step.get('continue-on-error') for step in execute['steps'])
    assert audit['needs'] == ['execute'] and not audit.get('continue-on-error')
    audit_commands='\n'.join(step.get('run','') for step in audit['steps'])
    assert 'embeddinggemma_tools.selected_audio_generation audit-distribution' in audit_commands
    assert 'embeddinggemma_tools.selected_audio_generation fetch-original-distribution' in audit_commands
    for job in (execute, audit):
        for index, step in enumerate(job['steps']):
            if 'actions/upload-artifact@' in step.get('uses',''):
                previous = job['steps'][index-1]
                assert previous['uses'] == './.github/actions/privacy-artifact-guard'
                assert "outcome == 'success'" in step['if']
