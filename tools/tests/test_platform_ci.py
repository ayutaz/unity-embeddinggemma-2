import importlib
import json
import os
from pathlib import Path
import plistlib
import struct
import subprocess
import sys

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
SOURCE = 'a' * 40


def api():
    return importlib.import_module('embeddinggemma_tools.platform_ci')


def project(tmp_path):
    (tmp_path / 'ProjectSettings').mkdir()
    (tmp_path / 'ProjectSettings/ProjectVersion.txt').write_text(
        'm_EditorVersion: 6000.3.21f1\nm_EditorVersionWithRevision: 6000.3.21f1 (c02631ffc030)\n')
    return tmp_path


@pytest.mark.parametrize('target', ['macos', 'android', 'ios'])
def test_preflight_records_missing_license_without_build_or_gpu_claim(tmp_path, target):
    value = api().preflight(project(tmp_path), target, SOURCE, {}, system='Darwin', free_bytes=100 * 2**30)
    assert value['success'] is False
    assert value['blockers'] == ['unity_license_missing']
    assert value['source_commit'] == SOURCE
    assert not value['build_passed'] and not value['gpu_verified'] and not value['device_verified']
    assert value['module_verified'] is False


@pytest.mark.parametrize('environment,ready', [
    ({'UNITY_LICENSE': 'synthetic-license'}, True),
    ({'UNITY_SERIAL': 'synthetic-serial', 'UNITY_EMAIL': 'synthetic-account', 'UNITY_PASSWORD': 'synthetic-password'}, True),
    ({'UNITY_SERIAL': 'synthetic-serial'}, False),
    ({'UNITY_EMAIL': 'synthetic-account', 'UNITY_PASSWORD': 'synthetic-password'}, False),
])
def test_license_modes_and_receipt_never_contain_credentials(tmp_path, environment, ready):
    value = api().preflight(project(tmp_path), 'ios', SOURCE, environment, system='Darwin', free_bytes=100 * 2**30)
    assert value['success'] is ready
    assert all(secret not in json.dumps(value) for secret in environment.values())


@pytest.mark.parametrize('system,free,expected', [
    ('Windows', 100 * 2**30, 'macos_host_required'),
    ('Darwin', 1, 'insufficient_install_space'),
])
def test_platform_requirements_are_not_inferred_from_runner_label(tmp_path, system, free, expected):
    value = api().preflight(project(tmp_path), 'android', SOURCE, {'UNITY_LICENSE': 'x'}, system=system, free_bytes=free)
    assert expected in value['blockers'] and not value['success']


@pytest.mark.parametrize('source', ['main', 'A' * 40, '', 'a' * 39])
def test_unpinned_sources_are_rejected(tmp_path, source):
    with pytest.raises(ValueError):
        api().preflight(project(tmp_path), 'ios', source, {}, system='Darwin', free_bytes=100 * 2**30)


def test_editor_revision_is_fixed_not_only_the_version(tmp_path):
    root = project(tmp_path)
    (root / 'ProjectSettings/ProjectVersion.txt').write_text('m_EditorVersion: 6000.3.21f1\n')
    value = api().preflight(root, 'macos', SOURCE, {'UNITY_LICENSE': 'x'}, system='Darwin', free_bytes=100 * 2**30)
    assert 'editor_revision_mismatch' in value['blockers']


def test_unsupported_target_is_rejected_before_any_execution(tmp_path):
    with pytest.raises(ValueError):
        api().preflight(project(tmp_path), 'web', SOURCE, {})


def xml(tmp_path, *, failed=0, skipped=0, missing=False):
    names = api().TEST_CLASSES
    cases = ''.join(f'<test-case fullname="{name}.Example" result="Passed"/>' for name in names[:-1] if missing)
    if not missing:
        cases = ''.join(f'<test-case fullname="{name}.Example" result="Passed"/>' for name in names)
    path = tmp_path / 'tests.xml'
    path.write_text(f'<test-run total="{len(names)}" passed="{len(names)-failed-skipped}" failed="{failed}" skipped="{skipped}">{cases}</test-run>')
    return path


def test_editor_exit_zero_alone_cannot_pass_or_hide_skipped_classes(tmp_path):
    assert api().audit_tests(xml(tmp_path))['passed'] == len(api().TEST_CLASSES)
    for kwargs in ({'failed': 1}, {'skipped': 1}, {'missing': True}):
        with pytest.raises(ValueError):
            api().audit_tests(xml(tmp_path, **kwargs))


def build_receipt():
    return dict(success=True, injected_build=False, source_commit=SOURCE,
                build_result='Succeeded', il2cpp_compiler='Release', build_options='None',
                build_info=dict(codeCommit=SOURCE, unityVersion='6000.3.21f1', sentisVersion='2.6.1',
                                target='Android', scriptingBackend='IL2CPP', stripping='High'))


@pytest.mark.parametrize('field,bad', [('success', False), ('success', 1), ('injected_build', True),
                                     ('build_result', 'Failed'), ('source_commit', 'b'*40),
                                     ('il2cpp_compiler', 'Debug'), ('build_options', 'Development')])
def test_simulated_failed_or_wrong_source_build_receipts_are_rejected(field, bad):
    value = build_receipt()
    api().audit_build(value, 'android', SOURCE)
    value[field] = bad
    with pytest.raises(ValueError):
        api().audit_build(value, 'android', SOURCE)


@pytest.mark.parametrize('field,bad', [('unityVersion', '6000.3.16f1'), ('target', 'iOS'),
                                     ('codeCommit', 'b'*40), ('scriptingBackend', 'Mono'),
                                     ('stripping', 'Low'), ('sentisVersion', 'unknown')])
def test_build_toolchain_and_runtime_profile_are_checked(field, bad):
    value = build_receipt()
    value['build_info'][field] = bad
    with pytest.raises(ValueError):
        api().audit_build(value, 'android', SOURCE)


def test_private_command_keeps_output_and_errors_out_of_public_console(tmp_path, capsys):
    with pytest.raises(RuntimeError, match='private_command_failed'):
        api().private_command([sys.executable, '-c', 'print("synthetic-private-diagnostic");raise SystemExit(7)'],
                              tmp_path, 'probe', dict(os.environ), 10)
    assert capsys.readouterr().out == ''
    assert 'synthetic-private-diagnostic' in (tmp_path / 'probe.log').read_text()


def test_private_command_has_a_bound_and_does_not_print_timeout_diagnostics(tmp_path, capsys):
    with pytest.raises(RuntimeError, match='private_command_interrupted'):
        api().private_command([sys.executable, '-c', 'import time;time.sleep(30)'],
                              tmp_path, 'bounded', dict(os.environ), 0.05)
    assert not capsys.readouterr().out


def native_setup(tmp_path, monkeypatch, target):
    module = api()
    root = project(tmp_path)
    editor = root / 'installation/Unity.app/Contents/MacOS/Unity'
    editor.parent.mkdir(parents=True)
    editor.touch()
    hub = root / 'hub'
    hub.touch()
    monkeypatch.setattr(module, 'HUB', hub)
    monkeypatch.setattr(module, 'EDITOR', editor)
    playback = editor.parents[1] / 'PlaybackEngines'
    for name in ('iOSSupport', 'AndroidPlayer/SDK', 'AndroidPlayer/NDK', 'AndroidPlayer/OpenJDK'):
        (playback / name).mkdir(parents=True, exist_ok=True)
    (root / 'Assets/Validation/Editor').mkdir(parents=True)
    output = root / 'evidence'
    (output / 'public').mkdir(parents=True)
    monkeypatch.setattr(module.subprocess, 'check_output', lambda *a, **kw: SOURCE.encode())
    report = module.preflight(root, target, SOURCE, {'UNITY_LICENSE': 'synthetic-license'}, system='Darwin', free_bytes=100 * 2**30)
    return module, root, output, report


def make_app(directory, *, platform='iPhoneOS', cpu=0x0100000c):
    product = directory / 'Build/Products/Release-iphoneos/Validation.app'
    product.mkdir(parents=True)
    (product / 'Info.plist').write_bytes(plistlib.dumps(dict(CFBundleExecutable='Validation', CFBundleSupportedPlatforms=[platform])))
    (product / 'Validation').write_bytes(struct.pack('<IiiIIIII', 0xfeedfacf, cpu, 0, 2, 1, 8, 0, 0) + b'\0'*8)
    return product


@pytest.mark.parametrize('target', ['macos', 'android', 'ios'])
def test_actual_runner_wires_real_entry_and_requires_actual_output(tmp_path, monkeypatch, target):
    module, root, output, report = native_setup(tmp_path, monkeypatch, target)
    calls = []

    def command(args, private, stage, env, timeout):
        calls.append((args, stage))
        assert not any(key.startswith('UNITY_') for key in env)
        if stage == 'xcode-compile':
            make_app(Path(args[args.index('-derivedDataPath') + 1]))
        if stage == 'editor-tests':
            xml(private).rename(private / 'accepted.xml')
            (private / 'accepted.xml').rename(private / 'tests.xml')
        if stage == 'device-build':
            wiring = root / 'Assets/Validation/Editor/PlatformCIBuildInvocation.cs'
            text = wiring.read_text()
            assert ('BuildAndroidMixed' if target == 'android' else 'BuildIOSMixed') in text
            assert 'execute:' not in text and 'Action<' not in text
            path = Path(env['EG_CI_OUTPUT'])
            path.parent.mkdir(parents=True)
            receipt = build_receipt()
            receipt['build_info']['target'] = module.TARGETS[target]
            (path.parent / 'build.json').write_text(json.dumps(receipt))
            actual = path if target == 'android' else path / 'Unity-iPhone.xcodeproj/project.pbxproj'
            actual.parent.mkdir(parents=True, exist_ok=True)
            actual.write_bytes(b'synthetic-build-output')

    value = module.execute(root, target, SOURCE, output, {'UNITY_LICENSE': 'synthetic-license'}, report, command=command)
    assert value['success'] is True
    assert value['editor_tests_passed'] is (target == 'macos')
    assert value['build_passed'] is (target != 'macos')
    assert not value['gpu_verified'] and not value['device_verified'] and not value['signing_performed']
    assert not (output / 'private/license.ulf').exists()
    assert not (root / 'Assets/Validation/Editor/PlatformCIBuildInvocation.cs').exists()


@pytest.mark.parametrize('failure', ['missing_output', 'injected_receipt', 'process_failure', 'source_mismatch'])
def test_runner_cannot_pass_on_fake_build_or_exit_zero_without_artifact(tmp_path, monkeypatch, failure):
    module, root, output, report = native_setup(tmp_path, monkeypatch, 'android')
    if failure == 'source_mismatch':
        monkeypatch.setattr(module.subprocess, 'check_output', lambda *a, **kw: ('b'*40).encode())
    calls = []

    def command(args, private, stage, env, timeout):
        calls.append(stage)
        if stage != 'device-build':
            return
        if failure == 'process_failure':
            raise RuntimeError('synthetic-private-failure')
        path = Path(env['EG_CI_OUTPUT'])
        path.parent.mkdir(parents=True)
        receipt = build_receipt()
        receipt['injected_build'] = failure == 'injected_receipt'
        (path.parent / 'build.json').write_text(json.dumps(receipt))
        if failure != 'missing_output':
            path.write_bytes(b'synthetic')

    value = module.execute(root, 'android', SOURCE, output, {'UNITY_LICENSE': 'synthetic-license'}, report, command=command)
    assert not value['success'] and not value['build_passed']
    assert 'synthetic-private-failure' not in json.dumps(value)
    if failure == 'source_mismatch':
        assert calls == []


def test_existing_build_wiring_is_preserved_on_collision(tmp_path, monkeypatch):
    module, root, output, report = native_setup(tmp_path, monkeypatch, 'android')
    path = root / 'Assets/Validation/Editor/PlatformCIBuildInvocation.cs'
    path.write_text('user-work')
    value = module.execute(root, 'android', SOURCE, output, {'UNITY_LICENSE': 'synthetic-license'}, report,
                           command=lambda *args: None)
    assert not value['success'] and path.read_text() == 'user-work'


def test_serial_license_is_returned_even_when_activation_fails(tmp_path, monkeypatch):
    module, root, output, report = native_setup(tmp_path, monkeypatch, 'ios')
    environment = dict(UNITY_SERIAL='synthetic-serial', UNITY_EMAIL='synthetic-account', UNITY_PASSWORD='synthetic-password')
    report['license_mode'] = 'serial'
    calls = []

    def command(args, private, stage, env, timeout):
        calls.append(stage)
        if stage == 'activation':
            raise RuntimeError('synthetic-failure')

    value = module.execute(root, 'ios', SOURCE, output, environment, report, command=command)
    assert not value['success'] and value['serial_license_returned']
    assert calls == ['activation', 'license-return']


def test_public_cli_fails_closed_before_install_without_dumping_environment(tmp_path):
    root = project(tmp_path)
    output = tmp_path / 'evidence'
    environment = {k: v for k, v in os.environ.items() if not k.startswith('UNITY_')}
    result = subprocess.run([sys.executable, '-P', '-m', 'embeddinggemma_tools.platform_ci', '--project', str(root),
                             '--platform', 'ios', '--source', SOURCE, '--output', str(output)],
                            env=environment, capture_output=True, text=True)
    assert result.returncode == 1
    public = json.loads((output / 'public/result.json').read_text())
    assert public['success'] is False and public['phase'] == 'preflight'
    assert not list((output / 'private').glob('*.log'))
    assert str(root) not in result.stdout and str(root) not in json.dumps(public)


def test_broken_project_never_leaks_a_traceback_or_path_from_cli(tmp_path):
    output = tmp_path / 'evidence'
    result = subprocess.run([sys.executable, '-P', '-m', 'embeddinggemma_tools.platform_ci',
                             '--project', str(tmp_path), '--platform', 'ios', '--source', SOURCE,
                             '--output', str(output)], capture_output=True, text=True)
    assert result.returncode == 1
    assert str(tmp_path) not in result.stdout + result.stderr
    assert 'Traceback' not in result.stderr
    value = json.loads((output / 'public/result.json').read_text())
    assert value['success'] is False and value['blockers'] == ['invalid_project_configuration']


def test_workflow_has_three_independent_owned_jobs_and_only_public_receipts():
    path = ROOT / '.github/workflows/platform-validation.yml'
    value = yaml.load(path.read_text(), Loader=yaml.BaseLoader)
    assert 'pull_request_target' not in value['on']
    job = value['jobs']['platform']
    assert job['strategy']['fail-fast'] == 'false'
    assert job['strategy']['matrix']['platform'] == ['macos', 'android', 'ios']
    assert job['runs-on'] == 'macos-15-intel'
    assert value['permissions'] == {'contents': 'read'}
    assert 'head.repo.full_name == github.repository' in job['if']
    assert 'github.repository_owner' in job['if']
    steps = job['steps']
    assert all(not step.get('continue-on-error') for step in steps)
    execution = next(s for s in steps if 'embeddinggemma_tools.platform_ci' in s.get('run', ''))
    assert '--execute' in execution['run'] and '--source "$GITHUB_SHA"' in execution['run']
    for index, step in enumerate(steps):
        if 'actions/upload-artifact@' in step.get('uses', ''):
            guard = steps[index - 1]
            assert guard['uses'] == './.github/actions/privacy-artifact-guard'
            assert guard['with']['paths'] == step['with']['path']
            assert step['with']['path'] == 'artifacts/platform-ci/public/result.json'
            assert 'outcome == \'success\'' in step['if']
