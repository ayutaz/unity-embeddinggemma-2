from copy import deepcopy

import pytest

from embeddinggemma_tools.ci import validate_scoped_results


SCOPE = dict(mode='docs', reason='documentation', base_sha='1' * 40,
             head_sha='2' * 40, source_sha='3' * 40, change_digest='4' * 64)


def jobs_for(scope):
    full = scope['mode'] == 'full'
    return {
        'changes': {'result': 'success', 'outputs': dict(scope)},
        'documentation': {'result': 'skipped' if full else 'success'},
        'workflow-lint': {'result': 'success'},
        **{name: {'result': 'success' if full else 'skipped'}
           for name in ('python', 'model-reference', 'package')},
    }


def test_docs_only_accepts_exactly_the_declared_non_execution():
    assert validate_scoped_results(jobs_for(SCOPE), SCOPE) == []


def test_full_scope_still_requires_all_original_dependencies():
    scope = dict(SCOPE, mode='full', reason='code-or-unknown')
    jobs = jobs_for(scope)
    assert validate_scoped_results(jobs, scope) == []
    for name in ('python', 'model-reference', 'package', 'workflow-lint'):
        for result in ('failure', 'cancelled', 'skipped', '', None):
            bad = deepcopy(jobs)
            bad[name]['result'] = result
            assert name in validate_scoped_results(bad, scope)


@pytest.mark.parametrize('name', ['changes', 'documentation', 'workflow-lint'])
@pytest.mark.parametrize('result', ['failure', 'cancelled', 'skipped', '', None])
def test_lightweight_required_checks_cannot_be_missing_or_unsuccessful(name, result):
    jobs = jobs_for(SCOPE)
    jobs[name]['result'] = result
    assert name in validate_scoped_results(jobs, SCOPE)


@pytest.mark.parametrize('key', list(SCOPE))
def test_forged_scope_output_is_rejected_against_independently_recomputed_diff(key):
    jobs = jobs_for(SCOPE)
    jobs['changes']['outputs'][key] = 'forged'
    assert 'changes' in validate_scoped_results(jobs, SCOPE)


@pytest.mark.parametrize('name', ['changes', 'documentation', 'workflow-lint', 'python', 'model-reference', 'package'])
def test_missing_dependency_is_always_failure(name):
    jobs = jobs_for(SCOPE)
    del jobs[name]
    assert name in validate_scoped_results(jobs, SCOPE)


def test_unknown_scope_and_malformed_jobs_fail_closed():
    for jobs, scope in ((None, SCOPE), ([], SCOPE), (jobs_for(SCOPE), None),
                        (jobs_for(SCOPE), dict(SCOPE, mode='unknown'))):
        assert validate_scoped_results(jobs, scope)


def test_undocumented_result_does_not_count_as_model_success():
    jobs = jobs_for(SCOPE)
    jobs['model-reference']['result'] = 'success'
    assert 'model-reference' in validate_scoped_results(jobs, SCOPE)


def test_new_unknown_dependency_cannot_be_ignored_by_the_gate():
    jobs = jobs_for(SCOPE)
    jobs['unregistered-check'] = {'result': 'failure'}
    assert validate_scoped_results(jobs, SCOPE)


@pytest.mark.parametrize('key', list(SCOPE))
def test_malformed_expected_scope_cannot_approve_equal_but_invalid_outputs(key):
    scope = dict(SCOPE)
    scope[key] = None
    jobs = jobs_for(SCOPE)
    jobs['changes']['outputs'] = scope
    assert validate_scoped_results(jobs, scope)
