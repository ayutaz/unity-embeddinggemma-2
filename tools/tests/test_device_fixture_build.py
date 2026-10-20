"""Host staging contracts; synthetic bytes never prove original fixtures or GPU."""
import hashlib
import importlib
import json
from pathlib import Path

import pytest


def api():
    return importlib.import_module('embeddinggemma_tools.device_fixture_build')


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


@pytest.fixture
def inputs(tmp_path):
    source = tmp_path / 'inputs'
    source.mkdir()
    files = {'image/0.png': b'synthetic-image', 'audio/input.bin': b'\0\0\0\0',
             'search-reference.json': b'{"synthetic":true}'}
    rows = []
    for name, raw in files.items():
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        rows.append(dict(path=name, bytes=len(raw), sha256=digest(raw)))
    raw = json.dumps(dict(schema_version=1, files=rows)).encode()
    return source, tmp_path / 'project', raw, digest(raw)


def destination(project):
    return project / 'Assets/StreamingAssets/EmbeddingGemmaDeviceValidation'


def test_stage_preserves_exact_manifest_and_file_bytes_without_acceptance_claims(inputs):
    source, project, raw, pin = inputs
    unrelated = project / 'Assets/StreamingAssets/user-data.txt'
    unrelated.parent.mkdir(parents=True)
    unrelated.write_text('preserve')
    (source / 'unlisted.txt').write_text('not part of the inventory')
    report = api().stage_fixtures(source, project, raw, pin)
    assert report['success'] is True and report['verified_files'] == 3
    assert report['manifest_sha256'] == pin
    assert report['verified_bytes'] == sum(row['bytes'] for row in json.loads(raw)['files'])
    assert (destination(project) / 'fixture-manifest.json').read_bytes() == raw
    for row in json.loads(raw)['files']:
        assert (destination(project) / 'fixtures' / row['path']).read_bytes() == (source / row['path']).read_bytes()
    assert not (destination(project) / 'fixtures/unlisted.txt').exists()
    assert unrelated.read_text() == 'preserve'
    assert not report['original_fixture_validated'] and not report['gpu_verified']
    assert not report['build_performed'] and not report['model_execution_performed']
    assert str(project) not in json.dumps(report) and str(source) not in json.dumps(report)


@pytest.mark.parametrize('fault', ['pin', 'encoding', 'empty', 'too_large', 'duplicate_key',
    'extra_key', 'schema_bool', 'schema_wrong', 'no_files', 'too_many', 'extra_row_key',
    'size_bool', 'size_zero', 'size_limit', 'total_limit', 'sha_upper', 'sha_short',
    'traversal', 'absolute', 'backslash', 'double_slash', 'dot', 'extension', 'colon',
    'duplicate_path', 'case_collision', 'file_directory_collision', 'windows_reserved', 'trailing_dot'])
def test_invalid_inventory_is_rejected_before_any_destination_is_created(inputs, fault):
    source, project, raw, pin = inputs
    value = json.loads(raw)
    row = value['files'][0]
    mutations = {
        'extra_key': lambda: value.update(private='synthetic'),
        'schema_bool': lambda: value.update(schema_version=True),
        'schema_wrong': lambda: value.update(schema_version=2),
        'no_files': lambda: value.update(files=[]),
        'too_many': lambda: value.update(files=[row] * 513),
        'extra_row_key': lambda: row.update(private='synthetic'),
        'size_bool': lambda: row.update(bytes=True),
        'size_zero': lambda: row.update(bytes=0),
        'size_limit': lambda: row.update(bytes=64 * 2**20 + 1),
        'total_limit': lambda: value.update(files=[dict(path=f'{i}.bin', bytes=64*2**20, sha256='a'*64) for i in range(9)]),
        'sha_upper': lambda: row.update(sha256='A'*64),
        'sha_short': lambda: row.update(sha256='a'*63),
        'duplicate_path': lambda: value['files'].append(dict(row)),
        'case_collision': lambda: value['files'].append(dict(row, path=row['path'].upper())),
        'file_directory_collision': lambda: value['files'].append(dict(row, path=row['path']+'/child.bin')),
    }
    paths = dict(traversal='../x.bin', absolute='/x.bin', backslash='a\\x.bin',
                 double_slash='a//x.bin', dot='a/./x.bin', extension='x.sentis', colon='x:y.bin',
                 windows_reserved='CON.bin', trailing_dot='a./x.bin')
    if fault in mutations:
        mutations[fault]()
    elif fault in paths:
        row['path'] = paths[fault]
    raw = json.dumps(value).encode()
    if fault == 'encoding': raw = b'\xff'
    if fault == 'empty': raw = b''
    if fault == 'too_large': raw = b' ' * (256 * 1024 + 1)
    if fault == 'duplicate_key': raw = b'{"schema_version":1,"schema_version":1,"files":[]}'
    pin = '0'*64 if fault == 'pin' else digest(raw)
    with pytest.raises(ValueError):
        api().stage_fixtures(source, project, raw, pin)
    assert not destination(project).exists()


@pytest.mark.parametrize('fault', ['missing', 'size', 'hash', 'directory'])
def test_source_corruption_is_rejected_before_staging(inputs, fault):
    source, project, raw, pin = inputs
    path = source / 'image/0.png'
    if fault in {'missing', 'directory'}:
        path.unlink()
        if fault == 'directory': path.mkdir()
    else:
        data = path.read_bytes()
        path.write_bytes(data + b'x' if fault == 'size' else b'x'*len(data))
    with pytest.raises((OSError, ValueError)):
        api().stage_fixtures(source, project, raw, pin)
    assert not destination(project).exists()


def test_second_stage_preserves_existing_owned_and_user_files(inputs):
    source, project, raw, pin = inputs
    api().stage_fixtures(source, project, raw, pin)
    keep = destination(project) / 'user.txt'
    keep.write_text('preserve')
    with pytest.raises((OSError, ValueError)):
        api().stage_fixtures(source, project, raw, pin)
    assert keep.read_text() == 'preserve'


def test_parent_link_cannot_redirect_staging_outside_project(inputs, tmp_path):
    source, project, raw, pin = inputs
    outside = tmp_path / 'outside'
    outside.mkdir()
    project.mkdir()
    try:
        (project / 'Assets').symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip('OS does not allow creating test symlinks')
    with pytest.raises(ValueError):
        api().stage_fixtures(source, project, raw, pin)
    assert list(outside.iterdir()) == []


@pytest.mark.parametrize('target', ['android', 'ios'])
def test_native_build_has_verified_original_bytes_before_install_and_build(inputs, tmp_path, monkeypatch, target):
    from test_platform_ci import native_setup
    (tmp_path / 'native').mkdir()
    module, root, output, report = native_setup(tmp_path / 'native', monkeypatch, target)
    source, _, raw, pin = inputs
    manifest = tmp_path / 'manifest.json'
    manifest.write_bytes(raw)
    calls = []
    def command(args, private, stage, env, timeout):
        calls.append(stage)
        assert (destination(root) / 'fixtures/image/0.png').read_bytes() == (source / 'image/0.png').read_bytes()
        if stage == 'activation':
            raise RuntimeError('synthetic stop before actual build')
    result = module.execute(root, target, 'a'*40, output, {'UNITY_LICENSE':'synthetic'}, report,
        command=command, fixtures=(source, manifest, pin))
    assert result['fixture_staged'] is True
    assert result['fixture_staging']['verified_files'] == 3
    assert result['build_passed'] is False and result['gpu_verified'] is False
    assert calls == ['activation']


def test_fixture_integrity_failure_prevents_install_activation_and_build(inputs, tmp_path, monkeypatch):
    from test_platform_ci import native_setup
    (tmp_path / 'native').mkdir()
    module, root, output, report = native_setup(tmp_path / 'native', monkeypatch, 'android')
    source, _, raw, pin = inputs
    manifest = tmp_path / 'manifest.json'
    manifest.write_bytes(raw)
    (source / 'image/0.png').write_bytes(b'corrupt')
    result = module.execute(root, 'android', 'a'*40, output, {'UNITY_LICENSE':'synthetic'}, report,
        command=lambda *args: pytest.fail('Invalid fixtures must stop native execution'),
        fixtures=(source, manifest, pin))
    assert not result['fixture_staged'] and not result['build_passed']
    assert result['blockers'] == ['execution_failed_at_fixture_staging']


@pytest.mark.parametrize('flags', [[], ['--fixture-source', 'synthetic']])
def test_cli_fixture_options_are_an_explicit_complete_set(tmp_path, monkeypatch, capsys, flags):
    from test_platform_ci import api as platform_api, project
    root = project(tmp_path)
    module = platform_api()
    for key in ('UNITY_LICENSE', 'UNITY_SERIAL', 'UNITY_EMAIL', 'UNITY_PASSWORD'):
        monkeypatch.delenv(key, raising=False)
    arguments = ['--project', str(root), '--platform', 'android', '--source', 'a'*40,
                 '--output', str(tmp_path / 'out')]
    if flags:
        with pytest.raises(SystemExit) as exc:
            module.main(arguments + flags)
        assert exc.value.code == 2
        assert 'fixture options must be supplied together' in capsys.readouterr().err
    else:
        assert module.main(arguments) == 1


def test_cli_connects_complete_fixture_options_to_the_real_build_driver(inputs, monkeypatch):
    from test_platform_ci import api as platform_api
    source, root, raw, pin = inputs
    (root / 'ProjectSettings').mkdir(parents=True)
    (root / 'ProjectSettings/ProjectVersion.txt').write_text(
        'm_EditorVersion: 6000.3.21f1\nm_EditorVersionWithRevision: 6000.3.21f1 (c02631ffc030)\n')
    manifest = source.parent / 'manifest.json'
    manifest.write_bytes(raw)
    module, calls = platform_api(), []
    monkeypatch.setenv('UNITY_LICENSE', 'synthetic')
    monkeypatch.setattr(module.platform, 'system', lambda: 'Darwin')
    monkeypatch.setattr(module.shutil, 'disk_usage', lambda p: type('Disk', (), {'free': 100*2**30})())
    def execute(project, target, commit, output, environment, report, *, fixtures):
        calls.append(fixtures)
        report['success'] = False
    monkeypatch.setattr(module, 'execute', execute)
    assert module.main(['--project', str(root), '--platform', 'android', '--source', 'a'*40,
                        '--output', str(root / 'evidence'), '--execute', '--fixture-source', str(source),
                        '--fixture-manifest', str(manifest), '--fixture-manifest-sha256', pin]) == 1
    assert calls == [(source, manifest, pin)]


def test_changed_file_during_copy_keeps_verified_files_and_removes_only_owned_partial(inputs, monkeypatch):
    source, project, raw, pin = inputs
    module, original = api(), api()._verify
    def verify(path, row, output=None):
        if output is not None and row['path'] == 'audio/input.bin':
            path.write_bytes(b'changed')
        return original(path, row, output)
    monkeypatch.setattr(module, '_verify', verify)
    with pytest.raises(ValueError):
        module.stage_fixtures(source, project, raw, pin)
    assert (destination(project) / 'fixtures/image/0.png').read_bytes() == (source / 'image/0.png').read_bytes()
    assert not list(destination(project).rglob('*.part'))
    assert not (destination(project) / 'fixtures/audio/input.bin').exists()


def test_source_file_link_is_rejected_before_staging(inputs, tmp_path):
    source, project, raw, pin = inputs
    original = source / 'image/0.png'
    moved = tmp_path / 'outside.png'
    original.rename(moved)
    try:
        original.symlink_to(moved)
    except OSError:
        pytest.skip('OS does not allow creating test symlinks')
    with pytest.raises(ValueError):
        api().stage_fixtures(source, project, raw, pin)
    assert not destination(project).exists()


def test_native_workflow_tracks_fixture_staging_dependencies():
    import yaml
    root = Path(__file__).resolve().parents[2]
    workflow = yaml.load((root / '.github/workflows/platform-validation.yml').read_text(), Loader=yaml.BaseLoader)
    paths = workflow['on']['pull_request']['paths']
    assert 'tools/embeddinggemma_tools/device_fixture_build.py' in paths
    assert 'tools/tests/test_device_fixture_build.py' in paths
