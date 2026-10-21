from pathlib import Path

import yaml

from embeddinggemma_tools.ci import SCOPED_JOBS
from embeddinggemma_tools.ci_scope import SCOPE_FIELDS


ROOT = Path(__file__).resolve().parents[2]


def workflow():
    return yaml.load((ROOT / '.github/workflows/ci.yml').read_text(encoding='utf-8'), Loader=yaml.BaseLoader)


def test_workflow_selects_full_models_for_any_non_documentation_change():
    jobs = workflow()['jobs']
    for name in ('python', 'model-reference', 'package'):
        assert jobs[name]['needs'] == 'changes'
        assert jobs[name]['if'] == "${{ needs.changes.outputs.mode == 'full' }}"
        assert jobs[name]['uses'].startswith('./.github/workflows/')
        assert not jobs[name].get('continue-on-error')
    assert jobs['documentation']['needs'] == 'changes'
    assert jobs['documentation']['if'] == "${{ needs.changes.outputs.mode == 'docs' }}"
    assert jobs['workflow-lint'].get('if') is None


def test_required_gate_recomputes_scope_with_complete_git_history():
    jobs = workflow()['jobs']
    assert set(jobs['changes']['outputs']) == set(SCOPE_FIELDS)
    gate = jobs['required']
    assert gate['name'] == 'Required CI' and gate['if'] == '${{ always() }}'
    assert set(gate['needs']) == set(SCOPED_JOBS)
    for name in ('changes', 'documentation', 'required'):
        steps = jobs[name]['steps']
        checkout = next(s for s in steps if s.get('uses', '').startswith('actions/checkout@'))
        assert checkout['with']['fetch-depth'] == '0'
        assert checkout['with']['persist-credentials'] == 'false'
        assert not jobs[name].get('continue-on-error')
    commands = '\n'.join(s.get('run', '') for s in gate['steps'])
    assert 'embeddinggemma_tools.ci --scoped' in commands
    assert jobs['required']['steps'][-1]['env']['NEEDS_JSON'] == '${{ toJSON(needs) }}'


def test_documentation_job_uses_small_checker_without_model_dependencies():
    job = workflow()['jobs']['documentation']
    assert int(job['timeout-minutes']) <= 10
    commands = '\n'.join(s.get('run', '') for s in job['steps'])
    assert 'embeddinggemma_tools.documentation_ci' in commands
    assert '--no-project' in commands and 'uv sync' not in commands
    assert 'prepare-' not in commands
    assert workflow()['permissions'] == {'contents': 'read', 'actions': 'read'}
