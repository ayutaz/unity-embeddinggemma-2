import json
import subprocess

import pytest

from embeddinggemma_tools.ci_scope import classify_paths, resolve_scope, main


@pytest.mark.parametrize('paths', [
    ['README.md'], ['docs/status.md', 'docs/results/explanation.md'],
    ['docs/old.md', 'docs/new.md'],
])
def test_only_explicit_documentation_paths_use_light_checks(paths):
    assert classify_paths(paths) == 'docs'


@pytest.mark.parametrize('path', [
    'Packages/com.ayutaz.embeddinggemma/Runtime/TextEmbedder.cs',
    'Packages/com.ayutaz.embeddinggemma/Editor/Prepare.cs',
    'Packages/com.ayutaz.embeddinggemma/Samples~/TextSearch/README.md',
    'Assets/Tests/Editor/Case.cs', 'tools/references/source.json',
    'tools/uv.lock', 'tools/tests/test_ci_scope.py', '.github/workflows/ci.yml',
    '.privacy-reviewed-assets.json', 'docs/privacy.md', 'AGENTS.md',
    '.agents/skills/project-privacy/SKILL.md', 'docs/results/result.json',
    'docs/image.png', 'unknown.md', '../docs/readme.md', 'docs/../tools/readme.md',
    'docs//readme.md', 'docs\\readme.md', 'docs/line\nbreak.md', '/docs/readme.md',
])
def test_code_sensitive_unknown_or_malformed_paths_require_full_checks(path):
    assert classify_paths(['README.md', path]) == 'full'


def test_empty_or_non_list_inputs_never_select_docs():
    for paths in ([], None, 'README.md', [None], [1]):
        assert classify_paths(paths) == 'full'


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()


@pytest.fixture
def repo(tmp_path):
    git(tmp_path, 'init', '-q')
    git(tmp_path, 'config', 'user.name', 'ci-fixture')
    git(tmp_path, 'config', 'user.email', 'ci-fixture@users.noreply.github.com')
    (tmp_path / 'README.md').write_text('# Initial\n', encoding='utf-8')
    git(tmp_path, 'add', '.')
    git(tmp_path, 'commit', '-qm', 'Initial')
    return tmp_path


def commit_doc(repo):
    base = git(repo, 'rev-parse', 'HEAD')
    (repo / 'README.md').write_text('# Updated\n', encoding='utf-8')
    git(repo, 'add', '.')
    git(repo, 'commit', '-qm', 'Update docs')
    return base, git(repo, 'rev-parse', 'HEAD')


def test_real_diff_is_bound_to_event_base_head_and_source(repo):
    base, head = commit_doc(repo)
    event = {'pull_request': {'base': {'sha': base}, 'head': {'sha': head}}}
    scope = resolve_scope(repo, 'pull_request', event, head)
    assert scope['mode'] == 'docs'
    assert scope['base_sha'] == base and scope['head_sha'] == head
    assert scope['source_sha'] == head
    assert len(scope['change_digest']) == 64
    assert 'paths' not in scope  # No potentially private filenames in public output.
    assert resolve_scope(repo, 'push', {'before': base, 'after': head}, head)['mode'] == 'docs'


def test_manual_missing_or_forged_context_never_selects_docs(repo):
    base, head = commit_doc(repo)
    contexts = [
        ('workflow_dispatch', {}, head), ('pull_request', {}, head),
        ('push', {'before': '0' * 40, 'after': head}, head),
        ('push', {'before': base, 'after': base}, head),
        ('pull_request', {'pull_request': {'base': {'sha': base}, 'head': {'sha': 'f' * 40}}}, head),
        ('push', {'before': base, 'after': head}, base),
    ]
    for name, event, claimed_head in contexts:
        assert resolve_scope(repo, name, event, claimed_head)['mode'] == 'full'


def test_renames_and_deleted_code_cannot_hide_behind_doc_extension(repo):
    (repo / 'code.cs').write_text('old code', encoding='utf-8')
    git(repo, 'add', '.')
    git(repo, 'commit', '-qm', 'Add code')
    base = git(repo, 'rev-parse', 'HEAD')
    (repo / 'code.cs').rename(repo / 'docs.md')
    git(repo, 'add', '-A')
    git(repo, 'commit', '-qm', 'Rename code')
    head = git(repo, 'rev-parse', 'HEAD')
    assert resolve_scope(repo, 'push', {'before': base, 'after': head}, head)['mode'] == 'full'


def test_executable_document_is_not_lightweight(repo):
    base, _ = commit_doc(repo)
    git(repo, 'update-index', '--chmod=+x', 'README.md')
    git(repo, 'commit', '-qm', 'Mode change')
    head = git(repo, 'rev-parse', 'HEAD')
    assert resolve_scope(repo, 'push', {'before': base, 'after': head}, head)['mode'] == 'full'


def test_cli_binds_outputs_and_does_not_publish_changed_paths(repo, monkeypatch, capsys):
    base, head = commit_doc(repo)
    event = repo / 'event.json'
    event.write_text(json.dumps({'before': base, 'after': head}), encoding='utf-8')
    output = repo / 'outputs'
    for key, value in {'GITHUB_EVENT_NAME': 'push', 'GITHUB_EVENT_PATH': str(event),
                       'GITHUB_SHA': head, 'GITHUB_OUTPUT': str(output)}.items():
        monkeypatch.setenv(key, value)
    assert main(['--root', str(repo)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['mode'] == 'docs'
    assert dict(line.split('=', 1) for line in output.read_text().splitlines()) == result
    assert 'README.md' not in output.read_text()


def test_required_cli_independently_rejects_forged_documentation_scope(repo, monkeypatch, capsys):
    from embeddinggemma_tools.ci import main as required_main
    base = git(repo, 'rev-parse', 'HEAD')
    (repo / 'code.cs').write_text('Runtime change', encoding='utf-8')
    git(repo, 'add', '.')
    git(repo, 'commit', '-qm', 'Code change')
    head = git(repo, 'rev-parse', 'HEAD')
    event = repo / 'event.json'
    event.write_text(json.dumps({'before': base, 'after': head}), encoding='utf-8')
    for key, value in {'GITHUB_EVENT_NAME': 'push', 'GITHUB_EVENT_PATH': str(event),
                       'GITHUB_SHA': head}.items():
        monkeypatch.setenv(key, value)
    scope = resolve_scope(repo, 'push', {'before': base, 'after': head}, head)
    assert scope['mode'] == 'full'
    jobs = {'changes': {'result': 'success', 'outputs': dict(scope, mode='docs')},
            'documentation': {'result': 'success'}, 'workflow-lint': {'result': 'success'},
            **{name: {'result': 'skipped'} for name in ('python', 'model-reference', 'package')}}
    monkeypatch.setenv('NEEDS_JSON', json.dumps(jobs))
    assert required_main(['--scoped', '--root', str(repo)]) == 1
    assert not json.loads(capsys.readouterr().out)['success']
    jobs['changes']['outputs'] = scope
    jobs['documentation']['result'] = 'skipped'
    for name in ('python', 'model-reference', 'package'):
        jobs[name]['result'] = 'success'
    monkeypatch.setenv('NEEDS_JSON', json.dumps(jobs))
    assert required_main(['--scoped', '--root', str(repo)]) == 0


def test_actual_pr_merge_classifies_only_the_change_against_its_updated_base(repo):
    original = git(repo, 'rev-parse', 'HEAD')
    git(repo, 'checkout', '-qb', 'doc-change')
    _, source = commit_doc(repo)
    git(repo, 'checkout', '-qb', 'base-update', original)
    (repo / 'accepted-code.cs').write_text('Already accepted on base', encoding='utf-8')
    git(repo, 'add', '.')
    git(repo, 'commit', '-qm', 'Base code update')
    base = git(repo, 'rev-parse', 'HEAD')
    git(repo, 'merge', '--no-ff', '-qm', 'Synthetic PR merge', source)
    head = git(repo, 'rev-parse', 'HEAD')
    scope = resolve_scope(repo, 'pull_request',
                          {'pull_request': {'base': {'sha': base}, 'head': {'sha': source}}}, head)
    assert scope['mode'] == 'docs' and scope['source_sha'] == source
    assert scope['head_sha'] != scope['source_sha']


def test_symlink_mode_in_git_cannot_be_classified_as_docs(repo):
    base = git(repo, 'rev-parse', 'HEAD')
    oid = subprocess.check_output(['git', '-C', str(repo), 'hash-object', '-w', '--stdin'],
                                  input=b'outside.md\n').decode().strip()
    git(repo, 'update-index', '--cacheinfo', '120000', oid, 'README.md')
    git(repo, 'commit', '-qm', 'Symlink mode')
    head = git(repo, 'rev-parse', 'HEAD')
    assert resolve_scope(repo, 'push', {'before': base, 'after': head}, head)['mode'] == 'full'
