import json
from pathlib import Path

import pytest

from test_platform_ci import api, native_setup, build_receipt, make_app, SOURCE


def metal_result(**changes):
    value = dict(schema=1, available=True, compute_passed=True, compared=16, mismatches=0,
                 max_buffer_bytes=2**30, recommended_working_set_bytes=2**30)
    value.update(changes)
    return value


def test_metal_result_is_a_small_probe_not_SDK_acceptance():
    value = api().audit_metal_result(json.dumps(metal_result()).encode())
    assert value['compute_passed'] is True
    assert 'gpu_verified' not in value and 'device_verified' not in value
    absent = dict(schema=1, available=False, compute_passed=False, reason='no_metal_device')
    assert api().audit_metal_result(json.dumps(absent).encode()) == absent


@pytest.mark.parametrize('changes', [dict(compared=0), dict(mismatches=1), dict(compute_passed=1),
                                     dict(available=False), dict(max_buffer_bytes=-1),
                                     dict(private_path='synthetic-private'), dict(gpu_verified=True), dict(schema=True)])
def test_metal_receipt_rejects_numeric_failures_and_unreviewed_fields(changes):
    with pytest.raises(ValueError):
        api().audit_metal_result(json.dumps(metal_result(**changes)).encode())


def test_native_metal_probe_compiles_and_executes_with_private_diagnostics(tmp_path):
    calls = []

    def command(args, private, stage, env, timeout):
        calls.append((args, stage))
        assert 'UNITY_LICENSE' not in env
        if stage == 'metal-execute':
            (private / 'metal-result.json').write_text(json.dumps(metal_result()))

    result = api().probe_metal(tmp_path, {'UNITY_LICENSE': 'synthetic'}, command=command)
    assert [row[1] for row in calls] == ['metal-compile', 'metal-execute']
    assert calls[0][0][:2] == ['xcrun', 'swiftc']
    assert result['compute_passed'] and len(result['source_sha256']) == 64


def test_metal_failure_never_returns_raw_exception_text(tmp_path):
    def command(*args):
        raise RuntimeError('synthetic-private-diagnostic')
    value = api().probe_metal(tmp_path, {}, command=command)
    assert value['compute_passed'] is False and value['reason'] == 'probe_execution_failed'
    assert 'synthetic-private-diagnostic' not in json.dumps(value)


def test_metal_probe_runs_before_missing_license_blocks_Unity(tmp_path, monkeypatch, capsys):
    module = api()
    from test_platform_ci import project
    root = project(tmp_path)
    for name in ('UNITY_LICENSE', 'UNITY_SERIAL', 'UNITY_EMAIL', 'UNITY_PASSWORD'):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(module.platform, 'system', lambda: 'Darwin')
    monkeypatch.setattr(module.shutil, 'disk_usage', lambda path: type('Disk', (), {'free': 100*2**30})())
    calls = []
    monkeypatch.setattr(module, 'probe_metal', lambda *args: calls.append('probe') or metal_result())
    monkeypatch.setattr(module, 'execute', lambda *args: pytest.fail('Unlicensed Unity must not start'))
    output = tmp_path / 'evidence'
    assert module.main(['--project', str(root), '--platform', 'macos', '--source', SOURCE,
                        '--output', str(output), '--execute', '--probe-metal']) == 1
    assert calls == ['probe']
    report = json.loads((output / 'public/result.json').read_text())
    assert report['metal_capability_probe']['compute_passed'] is True
    assert report['gpu_verified'] is False and report['build_passed'] is False
    assert report['blockers'] == ['unity_license_missing']
    assert str(root) not in capsys.readouterr().out


def test_existing_private_license_file_is_preserved_on_activation_collision(tmp_path, monkeypatch):
    module, root, output, report = native_setup(tmp_path, monkeypatch, 'ios')
    private = output / 'private'
    private.mkdir()
    license_file = private / 'license.ulf'
    license_file.write_text('unrelated-license')
    value = module.execute(root, 'ios', SOURCE, output, {'UNITY_LICENSE': 'synthetic'}, report,
                           command=lambda *args: None)
    assert not value['success'] and license_file.read_text() == 'unrelated-license'


@pytest.mark.parametrize('error_type', [OSError, RuntimeError])
def test_cleanup_supervision_error_cannot_publish_success_or_a_private_traceback(tmp_path, monkeypatch, capsys, error_type):
    module = api()
    from test_platform_ci import project
    root = project(tmp_path)
    monkeypatch.setenv('UNITY_LICENSE', 'synthetic-license')
    monkeypatch.setattr(module.platform, 'system', lambda: 'Darwin')
    monkeypatch.setattr(module.shutil, 'disk_usage', lambda path: type('Disk', (), {'free': 100*2**30})())

    def failed(*args):
        raise error_type('synthetic-private-diagnostic')

    monkeypatch.setattr(module, 'execute', failed)
    output = tmp_path / 'evidence'
    assert module.main(['--project', str(root), '--platform', 'ios', '--source', SOURCE,
                        '--output', str(output), '--execute']) == 1
    value = json.loads((output / 'public/result.json').read_text())
    assert not value['success'] and 'execution_supervision_failed' in value['blockers']
    assert 'synthetic-private-diagnostic' not in capsys.readouterr().out


@pytest.mark.parametrize('bad', ['missing', 'simulator', 'wrong_cpu', 'empty', 'path_escape', 'multiple'])
def test_ios_native_compile_requires_actual_arm64_device_product(tmp_path, bad):
    if bad != 'missing':
        product = make_app(tmp_path, platform='iPhoneSimulator' if bad == 'simulator' else 'iPhoneOS',
                           cpu=0x01000007 if bad == 'wrong_cpu' else 0x0100000c)
        if bad == 'empty':
            (product / 'Validation').write_bytes(b'')
        if bad == 'path_escape':
            import plistlib
            (product / 'Info.plist').write_bytes(plistlib.dumps(dict(CFBundleExecutable='../other', CFBundleSupportedPlatforms=['iPhoneOS'])))
        if bad == 'multiple':
            (product.parent / 'Other.app').mkdir()
    with pytest.raises(ValueError):
        api().audit_ios_product(tmp_path)


def test_ios_product_audit_exposes_hashes_and_size_without_local_paths(tmp_path):
    make_app(tmp_path)
    value = api().audit_ios_product(tmp_path)
    assert value['executable_bytes'] == 40 and len(value['executable_sha256']) == 64
    assert str(tmp_path) not in json.dumps(value)


@pytest.mark.parametrize('failure', [None, 'compile_failed', 'no_app'])
def test_ios_export_and_native_compile_are_separate_gates(tmp_path, monkeypatch, failure):
    module, root, output, report = native_setup(tmp_path, monkeypatch, 'ios')
    calls = []

    def command(args, private, stage, env, timeout):
        calls.append(stage)
        if stage == 'device-build':
            path = Path(env['EG_CI_OUTPUT'])
            actual = path / 'Unity-iPhone.xcodeproj/project.pbxproj'
            actual.parent.mkdir(parents=True)
            actual.write_text('synthetic export')
            receipt = build_receipt()
            receipt['build_info']['target'] = 'iOS'
            (path.parent / 'build.json').write_text(json.dumps(receipt))
        if stage == 'xcode-compile':
            assert args[0] == 'xcodebuild'
            assert 'CODE_SIGNING_ALLOWED=NO' in args and 'CODE_SIGNING_REQUIRED=NO' in args
            assert args[args.index('-configuration') + 1] == 'Release'
            assert args[args.index('-destination') + 1] == 'generic/platform=iOS'
            if failure == 'compile_failed':
                raise RuntimeError('synthetic-private-error')
            if failure != 'no_app':
                make_app(Path(args[args.index('-derivedDataPath') + 1]))

    value = module.execute(root, 'ios', SOURCE, output, {'UNITY_LICENSE': 'synthetic'}, report, command=command)
    assert value['build_passed'] is True
    assert value['success'] is (failure is None)
    assert value['xcode_compile_passed'] is (failure is None)
    assert value['gpu_verified'] is False and value['signing_performed'] is False
    assert 'xcode-compile' in calls
