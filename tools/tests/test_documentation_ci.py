import json
import subprocess

import pytest

from embeddinggemma_tools.documentation_ci import audit_documentation, check_links, check_payload, check_whitespace
from embeddinggemma_tools.ci_scope import resolve_scope


def test_relative_links_headings_references_and_html_are_checked(tmp_path):
    (tmp_path / 'target.md').write_text('# 日本語 API\n# 日本語 API\n', encoding='utf-8')
    (tmp_path / 'README.md').write_text(
        '[ok](target.md#日本語-api)\n[duplicate](target.md#日本語-api-1)\n'
        '[ref][guide]\n[guide]: target.md#日本語-api\n'
        '<a href="target.md#日本語-api">guide</a>\n', encoding='utf-8')
    assert check_links(tmp_path, ['README.md', 'target.md']) == []


@pytest.mark.parametrize('content,rule', [
    ('[broken](missing.md)', 'missing-local-target'),
    ('[broken](target.md#missing)', 'missing-local-anchor'),
    ('[ref][guide]\n[guide]: missing.md', 'missing-local-target'),
    ('<img src="missing.png">', 'missing-local-target'),
    ('[escape](../outside.md)', 'outside-repository-link'),
    ('[label][missing]', 'missing-reference-definition'),
])
def test_missing_or_escaping_links_fail_without_echoing_targets(tmp_path, content, rule):
    (tmp_path / 'README.md').write_text(content, encoding='utf-8')
    (tmp_path / 'target.md').write_text('# Good\n', encoding='utf-8')
    findings = check_links(tmp_path, ['README.md'])
    assert any(f['rule'] == rule for f in findings)
    assert all(set(f) == {'path', 'line', 'rule'} for f in findings)


def test_code_examples_and_external_links_do_not_cause_network_or_false_failures(tmp_path):
    (tmp_path / 'README.md').write_text(
        '```md\n[example](missing.md)\n```\n`[inline](missing.md)`\n'
        '[external](https://example.invalid/doc#section)\n', encoding='utf-8')
    assert check_links(tmp_path, ['README.md']) == []


def test_absolute_symlink_escape_is_rejected(tmp_path):
    repo = tmp_path / 'repo'
    repo.mkdir()
    outside = tmp_path / 'outside.md'
    outside.write_text('# Secret\n', encoding='utf-8')
    try:
        (repo / 'link.md').symlink_to(outside)
    except OSError:
        pytest.skip('Symlink creation is unavailable on this host')
    (repo / 'README.md').write_text('[link](link.md)', encoding='utf-8')
    assert check_links(repo, ['README.md'])[0]['rule'] == 'outside-repository-link'


def test_actual_pr_title_and_body_are_scanned_separately():
    assert check_payload({'pull_request': {'title': 'Docs', 'body': 'Relative paths only'}}) == []
    private = 'C:' + '\\Users\\' + 'private-fixture' + '\\source'
    for field in ('title', 'body'):
        event = {'pull_request': {'title': 'Docs', 'body': 'Safe'}}
        event['pull_request'][field] = private
        findings = check_payload(event)
        assert findings and private not in json.dumps(findings)


def test_malformed_pr_payload_cannot_be_silently_ignored():
    for event in ({'pull_request': None}, {'pull_request': {}},
                  {'pull_request': {'title': 3, 'body': None}}):
        assert check_payload(event)


def test_git_whitespace_failure_is_retained_without_printing_raw_diff(tmp_path):
    def git(*args):
        return subprocess.check_output(['git', '-C', str(tmp_path), *args], text=True).strip()
    git('init', '-q')
    git('config', 'user.name', 'ci-fixture')
    git('config', 'user.email', 'ci-fixture@users.noreply.github.com')
    p = tmp_path / 'README.md'
    p.write_text('# Initial\n', encoding='utf-8')
    git('add', '.')
    git('commit', '-qm', 'Initial')
    base = git('rev-parse', 'HEAD')
    p.write_text('Trailing spaces   \n', encoding='utf-8')
    git('add', '.')
    git('commit', '-qm', 'Whitespace')
    assert check_whitespace(tmp_path, base, git('rev-parse', 'HEAD'))


def make_history(root, old_private=False):
    def git(*args):
        return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()
    git('init', '-q')
    git('config', 'user.name', 'ci-fixture')
    git('config', 'user.email', 'ci-fixture@users.noreply.github.com')
    p = root / 'README.md'
    p.write_text(('C:' + '\\Users\\' + 'private-fixture' + '\\source\n') if old_private else '# Initial\n', encoding='utf-8')
    git('add', '.')
    git('commit', '-qm', 'Initial')
    base = git('rev-parse', 'HEAD')
    p.write_text('# Clean current docs\n', encoding='utf-8')
    git('add', '.')
    git('commit', '-qm', 'Update docs')
    head = git('rev-parse', 'HEAD')
    event = {'before': base, 'after': head, 'sender': {'login': 'ci-fixture'}}
    return resolve_scope(root, 'push', event, head), event


def test_documentation_route_checks_actual_ancestry_and_payload(tmp_path):
    scope, event = make_history(tmp_path)
    profiles = []
    def public_profile(login):
        profiles.append(login)
        return {'login': login, 'name': login}
    result = audit_documentation(tmp_path, scope, event, profile_reader=public_profile)
    assert result['success'] and result['model_execution'] is False
    assert profiles == ['ci-fixture']
    event['pull_request'] = {'title': 'Docs', 'body': 'C:' + '\\Users\\' + 'private-fixture' + '\\source'}
    assert not audit_documentation(tmp_path, scope, event, profile_reader=public_profile)['success']


def test_clean_head_does_not_approve_private_ancestors(tmp_path):
    scope, event = make_history(tmp_path, old_private=True)
    result = audit_documentation(tmp_path, scope, event,
                                profile_reader=lambda login: {'login': login, 'name': login})
    assert not result['success'] and any(f['rule'] == 'home-path' for f in result['findings'])


def test_unavailable_or_unreviewed_profile_fails_closed(tmp_path):
    scope, event = make_history(tmp_path)
    for reader in (lambda login: {}, lambda login: {'login': login, 'name': 'Private Fixture Name'}):
        result = audit_documentation(tmp_path, scope, event, profile_reader=reader)
        assert not result['success']
        assert 'Private Fixture Name' not in json.dumps(result)
    assert not audit_documentation(tmp_path, dict(scope, mode='full'), event)['success']


def test_invalid_event_or_profile_container_is_a_safe_failure(tmp_path):
    scope, _ = make_history(tmp_path)
    for event in (None, [], {'repository': None}, {'sender': {'login': 'ci-fixture'}, 'repository': []}):
        result = audit_documentation(tmp_path, scope, event,
                                    profile_reader=lambda login: {'login': login, 'name': login})
        assert not result['success']
