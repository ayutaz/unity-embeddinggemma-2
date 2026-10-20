"""Native hosted-Mac contracts and device builds; never a GPU acceptance gate."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import plistlib
import re
import shutil
import signal
import subprocess
import struct
import xml.etree.ElementTree as ET

from .bounded_process import read_unity_execution_counts, write_json_atomic

VERSION = '6000.3.21f1'
REVISION = 'c02631ffc030'
TARGETS = {'macos': 'StandaloneOSX', 'android': 'Android', 'ios': 'iOS'}
TEST_CLASSES = tuple('EmbeddingGemma.Validation.Tests.' + name for name in (
    'ValidationIOSBuildTests', 'ValidationAndroidBuildTests', 'PlayerDeviceMixedEntryTests'))
HUB = Path('/Applications/Unity Hub.app/Contents/MacOS/Unity Hub')
EDITOR = Path('/Applications/Unity/Hub/Editor') / VERSION / 'Unity.app/Contents/MacOS/Unity'
METAL_SOURCE = Path(__file__).with_name('platform_metal_probe.swift')


def audit_metal_result(raw):
    """Admit only fixed fields from a sixteen-integer compute capability probe."""
    value = json.loads(raw)
    base = {'schema', 'available', 'compute_passed'}
    if (not isinstance(value, dict) or type(value.get('schema')) is not int or value.get('schema') != 1
            or type(value.get('available')) is not bool or type(value.get('compute_passed')) is not bool):
        raise ValueError('invalid_metal_probe')
    if value['compute_passed']:
        keys = base | {'compared', 'mismatches', 'max_buffer_bytes', 'recommended_working_set_bytes'}
        if (set(value) != keys or value['available'] is not True
                or any(type(value[key]) is not int for key in keys - base)
                or value['compared'] != 16 or value['mismatches'] != 0
                or value['max_buffer_bytes'] <= 0 or value['recommended_working_set_bytes'] <= 0):
            raise ValueError('invalid_metal_compute_result')
    elif set(value) != base | {'reason'} or value['reason'] not in {
            'no_metal_device', 'probe_setup_failed', 'gpu_command_failed', 'compute_mismatch'}:
        raise ValueError('invalid_metal_probe_failure')
    return value


def probe_metal(private, environment, *, command=None):
    command = private_command if command is None else command
    private = Path(private)
    private.mkdir(parents=True, exist_ok=True)
    clean = {key: value for key, value in environment.items() if not key.startswith('UNITY_')}
    result = dict(schema=1, available=False, compute_passed=False, reason='probe_execution_failed')
    try:
        command(['xcrun', 'swiftc', str(METAL_SOURCE), '-o', str(private / 'metal-probe')],
                private, 'metal-compile', clean, 120)
        command([str(private / 'metal-probe'), str(private / 'metal-result.json')],
                private, 'metal-execute', clean, 60)
        result = audit_metal_result((private / 'metal-result.json').read_bytes())
    except (OSError, ValueError, RuntimeError):
        pass
    result['source_sha256'] = hashlib.sha256(METAL_SOURCE.read_bytes()).hexdigest()
    return result


def audit_ios_product(derived):
    """Require an actual arm64 device executable independently of xcodebuild's exit."""
    products = list((Path(derived) / 'Build/Products/Release-iphoneos').glob('*.app'))
    if len(products) != 1 or products[0].is_symlink():
        raise ValueError('device_app_missing_or_ambiguous')
    app = products[0]
    try:
        info = plistlib.loads((app / 'Info.plist').read_bytes())
        name = info['CFBundleExecutable']
        if (not isinstance(name, str) or not re.fullmatch('[A-Za-z0-9_.-]+', name)
                or name in {'.', '..'} or info.get('CFBundleSupportedPlatforms') != ['iPhoneOS']):
            raise ValueError('device_app_metadata_invalid')
        executable = app / name
        if executable.is_symlink() or not executable.is_file():
            raise ValueError('device_app_executable_invalid')
        with executable.open('rb') as stream:
            header = stream.read(32)
            if len(header) != 32:
                raise ValueError('device_app_header_missing')
            magic, cpu, _, kind, commands, command_bytes, _, _ = struct.unpack('<IiiIIIII', header)
            if (magic != 0xfeedfacf or cpu != 0x0100000c or kind != 2 or commands < 1
                    or command_bytes < 8 or executable.stat().st_size < 32 + command_bytes):
                raise ValueError('arm64_device_executable_required')
            stream.seek(0)
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    except (OSError, KeyError, plistlib.InvalidFileException) as error:
        raise ValueError('device_app_product_invalid') from error
    return dict(executable_sha256=digest, executable_bytes=executable.stat().st_size,
                info_plist_sha256=hashlib.sha256((app / 'Info.plist').read_bytes()).hexdigest(),
                architecture='arm64', platform='iPhoneOS')


def preflight(project, target, source, environment, *, system=None, free_bytes=None):
    if target not in TARGETS or not re.fullmatch('[a-f0-9]{40}', source):
        raise ValueError('fixed_source_and_platform_required')
    project = Path(project)
    blockers = []
    version = (project / 'ProjectSettings/ProjectVersion.txt').read_text(encoding='utf-8')
    if f'm_EditorVersion: {VERSION}\n' not in version or f'({REVISION})' not in version:
        blockers.append('editor_revision_mismatch')
    mode = 'missing'
    if environment.get('UNITY_LICENSE', '').strip():
        mode = 'license_file'
    elif all(environment.get(key, '').strip() for key in ('UNITY_SERIAL', 'UNITY_EMAIL', 'UNITY_PASSWORD')):
        mode = 'serial'
    if mode == 'missing':
        blockers.append('unity_license_missing')
    if (system or platform.system()) != 'Darwin':
        blockers.append('macos_host_required')
    free = shutil.disk_usage(project).free if free_bytes is None else free_bytes
    if free < 24 * 2**30:
        blockers.append('insufficient_install_space')
    return dict(schema=1, success=not blockers, phase='preflight', platform=target,
                source_commit=source, unity_version=VERSION, unity_revision=REVISION,
                license_mode=mode, free_disk_bytes=free, install_admission_bytes=24 * 2**30,
                admission_is_not_SDK_minimum=True, blockers=blockers,
                module_verified=False, editor_tests_passed=False, build_passed=False,
                gpu_verified=False, device_verified=False, xcode_compile_passed=False,
                signing_performed=False, fixture_staged=False, model_execution_performed=False)


def audit_tests(path):
    counts = read_unity_execution_counts(path)
    if counts['failed']:
        raise ValueError('editor_tests_failed')
    cases = ET.parse(path).getroot().findall('.//test-case')
    for name in TEST_CLASSES:
        selected = [case for case in cases if case.get('fullname', '').startswith(name + '.')]
        if not selected or any(case.get('result') != 'Passed' for case in selected):
            raise ValueError('required_test_class_not_passed')
    return counts


def audit_build(value, target, source):
    info = value.get('build_info', {})
    expected = dict(codeCommit=source, unityVersion=VERSION, sentisVersion='2.6.1',
                    target=TARGETS[target], scriptingBackend='IL2CPP', stripping='High')
    if (value.get('success') is not True or value.get('injected_build') is not False
            or value.get('source_commit') != source or value.get('build_result') != 'Succeeded'
            or value.get('il2cpp_compiler') != 'Release' or value.get('build_options') != 'None'
            or any(info.get(key) != expected_value for key, expected_value in expected.items())):
        raise ValueError('actual_build_receipt_invalid')


def private_command(command, private, stage, environment, timeout):
    """Only this invocation's process/group is terminated at the bound; raw output stays private."""
    private = Path(private)
    private.mkdir(parents=True, exist_ok=True)
    with (private / (stage + '.log')).open('xb') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=environment,
                                   start_new_session=os.name != 'nt')
        try:
            code = process.wait(timeout=timeout)
        except BaseException:
            if process.poll() is None:
                if os.name == 'nt':
                    process.kill()
                else:
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=30)
            raise RuntimeError('private_command_interrupted') from None
    if code != 0:
        raise RuntimeError('private_command_failed')


def _build_wiring(target):
    method = {'android': 'BuildAndroidMixed', 'ios': 'BuildIOSMixed'}[target]
    return ('using System;\nnamespace EmbeddingGemma.Validation.Editor {\n'
            'public static class PlatformCIBuildInvocation { public static void Run() {\n'
            f'ValidationPlayerBuild.{method}(Environment.GetEnvironmentVariable("EG_CI_OUTPUT"), '
            'Environment.GetEnvironmentVariable("EG_CI_SOURCE"));\n} } }\n')


def execute(project, target, source, output, environment, report, *, command=private_command):
    project, output = Path(project).resolve(), Path(output).resolve()
    private = output / 'private'
    private.mkdir(exist_ok=True)
    public = output / 'public/result.json'
    # Licensing credentials are supplied only to activation/return; no command line is published.
    clean_env = {key: value for key, value in environment.items() if not key.startswith('UNITY_')}
    license_file = private / 'license.ulf'
    license_file_created = False
    serial_attempted = False
    wiring = project / 'Assets/Validation/Editor/PlatformCIBuildInvocation.cs'

    def phase(value):
        report['phase'] = value
        write_json_atomic(public, report)

    def run(args, label, timeout=900, env=None):
        command([str(arg) for arg in args], private, label, clean_env if env is None else env, timeout)

    try:
        phase('source_check')
        actual = subprocess.check_output(['git', '-C', str(project), 'rev-parse', 'HEAD'], stderr=subprocess.PIPE).decode().strip()
        if actual != source:
            raise ValueError('checkout_source_mismatch')
        phase('installation')
        if not HUB.is_file():
            run(['brew', 'install', '--cask', 'unity-hub'], 'hub-install', 600)
        if not EDITOR.is_file():
            modules = {'macos': [], 'ios': ['ios'],
                       'android': ['android', 'android-sdk-ndk-tools', 'android-open-jdk']}[target]
            args = [HUB, '--', '--headless', 'install', '--version', VERSION, '--changeset', REVISION]
            if modules:
                args += ['--module', *modules, '--childModules']
            run(args, 'editor-install', 1800)
        if not EDITOR.is_file():
            raise ValueError('editor_install_missing')
        playback = EDITOR.parents[1] / 'PlaybackEngines'
        if target != 'macos':
            module = playback / ('AndroidPlayer' if target == 'android' else 'iOSSupport')
            required = [module]
            if target == 'android':
                required += [module / name for name in ('SDK', 'NDK', 'OpenJDK')]
            if not all(item.is_dir() for item in required):
                raise ValueError('platform_module_missing')
        report['module_verified'] = True
        phase('activation')
        activation = [EDITOR, '-batchmode', '-nographics', '-quit', '-logFile', private / 'activation-editor.log']
        if report['license_mode'] == 'license_file':
            # File contents, path and raw Editor responses are excluded from the public receipt.
            with license_file.open('x', encoding='utf-8') as stream:
                license_file_created = True
                stream.write(environment['UNITY_LICENSE'])
            license_file.chmod(0o600)
            activation += ['-manualLicenseFile', license_file]
        else:
            serial_attempted = True
            activation += ['-serial', environment['UNITY_SERIAL'], '-username', environment['UNITY_EMAIL'],
                           '-password', environment['UNITY_PASSWORD']]
        run(activation, 'activation', 180)
        if target == 'macos':
            phase('editor_contract_tests')
            results = private / 'tests.xml'
            run([EDITOR, '-batchmode', '-nographics', '-projectPath', project, '-runTests',
                 '-testPlatform', 'EditMode', '-testFilter', ';'.join(TEST_CLASSES),
                 '-testResults', results, '-logFile', private / 'tests-editor.log'], 'editor-tests', 900)
            report['test_counts'] = audit_tests(results)
            report['tests_xml_sha256'] = hashlib.sha256(results.read_bytes()).hexdigest()
            report['editor_tests_passed'] = True
        else:
            phase('device_build')
            # Thin generated wiring calls the real existing build API without an injected callback.
            with wiring.open('x', encoding='utf-8') as stream:
                stream.write(_build_wiring(target))
            report['build_wiring_sha256'] = hashlib.sha256(wiring.read_bytes()).hexdigest()
            destination = private / 'build' / ('Validation.apk' if target == 'android' else 'Xcode')
            build_env = dict(clean_env, EG_CI_SOURCE=source, EG_CI_OUTPUT=str(destination))
            run([EDITOR, '-batchmode', '-nographics', '-quit', '-projectPath', project,
                 '-buildTarget', TARGETS[target], '-executeMethod',
                 'EmbeddingGemma.Validation.Editor.PlatformCIBuildInvocation.Run',
                 '-logFile', private / 'build-editor.log'], 'device-build', 1800, build_env)
            receipt = destination.parent / 'build.json'
            audit_build(json.loads(receipt.read_text()), target, source)
            required = destination if target == 'android' else destination / 'Unity-iPhone.xcodeproj/project.pbxproj'
            if not required.is_file() or required.stat().st_size == 0:
                raise ValueError('actual_build_output_missing')
            report['build_passed'] = True
            report['build_receipt_sha256'] = hashlib.sha256(receipt.read_bytes()).hexdigest()
            report['build_output_sha256'] = hashlib.sha256(required.read_bytes()).hexdigest()
            report['build_output_bytes'] = required.stat().st_size
            if target == 'ios':
                phase('xcode_native_compile')
                derived = private / 'xcode-derived'
                run(['xcodebuild', '-project', destination / 'Unity-iPhone.xcodeproj',
                     '-scheme', 'Unity-iPhone', '-configuration', 'Release', '-sdk', 'iphoneos',
                     '-destination', 'generic/platform=iOS', '-derivedDataPath', derived,
                     'CODE_SIGNING_ALLOWED=NO', 'CODE_SIGNING_REQUIRED=NO', 'build'], 'xcode-compile', 1800)
                report['ios_native_product'] = audit_ios_product(derived)
                report['xcode_compile_passed'] = True
        report['success'] = True
        phase('complete')
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError):
        # Raw errors can contain credentials or local paths. Keep only a fixed reason publicly.
        report['success'] = False
        report['blockers'].append('execution_failed_at_' + report['phase'])
    finally:
        if serial_attempted:
            try:
                run([EDITOR, '-batchmode', '-nographics', '-quit', '-returnlicense',
                     '-username', environment['UNITY_EMAIL'], '-password', environment['UNITY_PASSWORD'],
                     '-logFile', private / 'license-return-editor.log'], 'license-return', 180)
                report['serial_license_returned'] = True
            except (OSError, RuntimeError):
                report['success'] = False
                report['blockers'].append('serial_license_return_failed')
        if license_file_created and license_file.exists():
            license_file.unlink()
        # Wiring is created exclusively by this invocation; preserve unrelated files on collision.
        if 'build_wiring_sha256' in report:
            wiring.unlink(missing_ok=True)
            wiring.with_suffix('.cs.meta').unlink(missing_ok=True)
        report['completed_utc'] = datetime.now(timezone.utc).isoformat()
        write_json_atomic(public, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--platform', choices=TARGETS, required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--probe-metal', action='store_true')
    args = parser.parse_args(argv)
    try:
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / 'public').mkdir()
    except OSError:
        print(json.dumps(dict(success=False, phase='output_setup', blockers=['new_output_required'])))
        return 1
    try:
        report = preflight(args.project, args.platform, args.source, os.environ)
    except (OSError, ValueError):
        report = dict(schema=1, success=False, phase='preflight', blockers=['invalid_project_configuration'],
                      build_passed=False, gpu_verified=False, device_verified=False)
    # Preflight readiness is not an executed test/build success.
    ready = report['success']
    report['success'] = False
    report['preflight_ready'] = ready
    if args.probe_metal and args.platform == 'macos' and platform.system() == 'Darwin':
        report['metal_capability_probe'] = probe_metal(args.output / 'private', os.environ)
    write_json_atomic(args.output / 'public/result.json', report)
    if ready and args.execute:
        try:
            execute(args.project, args.platform, args.source, args.output, os.environ, report)
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError):
            report['success'] = False
            report['blockers'].append('execution_supervision_failed')
            write_json_atomic(args.output / 'public/result.json', report)
    print(json.dumps(dict(success=report['success'], phase=report['phase'], blockers=report['blockers'])))
    return 0 if report['success'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
