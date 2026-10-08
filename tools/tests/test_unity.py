import json
import subprocess

import pytest

from embeddinggemma_tools.unity import run_harness


class EditorCLI:
    def __init__(self, compile_result=None, tests=None, compile_exit=0):
        self.calls = []
        self.compile_result = compile_result or {"Success": True, "ErrorCount": 0}
        self.tests = tests or {"Success": True, "TestCount": 3, "PassedCount": 3,
                              "FailedCount": 0, "SkippedCount": 0}
        self.compile_exit = compile_exit

    def __call__(self, args, **kwargs):
        self.calls.append((args, kwargs))
        command = args[3]
        result = {"launch": {"Success": True, "Ready": True}, "compile": self.compile_result,
                  "run-tests": self.tests, "get-logs": {"Logs": []}}[command]
        stdout = result if isinstance(result, str) else json.dumps(result)
        return subprocess.CompletedProcess(args, self.compile_exit if command == "compile" else 0,
                                           stdout, "")


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "Unity project with spaces"
    (root / "ProjectSettings").mkdir(parents=True)
    (root / "ProjectSettings/ProjectVersion.txt").write_text("m_EditorVersion: 6000.3.16f1\n")
    return root


def test_complete_editor_loop_records_evidence_and_verifies_all_m1_tests(project, tmp_path):
    editor = EditorCLI()
    output = tmp_path / "evidence"
    report = run_harness(project, output, "uloop.exe", "m1", runner=editor, launch=True)
    assert report["success"] is True
    assert report["m1_reference_passed"] is True
    assert report["unity_version"] == "6000.3.16f1"
    assert [call[0][3] for call in editor.calls] == ["launch", "compile", "run-tests", "get-logs"]
    test_args, options = editor.calls[2]
    assert test_args[1:3] == ["--project-path", str(project)]
    assert test_args[4:] == ["--test-mode", "EditMode", "--filter-type", "assembly",
                             "--filter-value", "EmbeddingGemma.Editor.Tests",
                             "--unsaved-changes", "fail", "--timeout-seconds", "900"]
    assert options["cwd"] == project
    assert options["shell"] is False
    assert json.loads((output / "summary.json").read_text())["m1_reference_passed"] is True
    assert json.loads((output / "run-tests.json").read_text())["stdout"] == json.dumps(editor.tests)


@pytest.mark.parametrize("result", [
    {"Success": False, "TestCount": 3, "PassedCount": 2, "FailedCount": 1, "SkippedCount": 0},
    {"Success": True, "TestCount": 3, "PassedCount": 2, "FailedCount": 0, "SkippedCount": 1},
    {"Success": True, "TestCount": 0, "PassedCount": 0, "FailedCount": 0, "SkippedCount": 0},
    {"Success": True},
])
def test_failed_skipped_empty_or_incomplete_test_results_cannot_pass(project, tmp_path, result):
    report = run_harness(project, tmp_path / "evidence", "uloop.exe", "m1", runner=EditorCLI(tests=result))
    assert report["success"] is False
    assert report["m1_reference_passed"] is False


@pytest.mark.parametrize("compile_exit", [0, 1])
def test_compile_failure_stops_tests_and_still_collects_logs(project, tmp_path, compile_exit):
    editor = EditorCLI(compile_result={"Success": False, "ErrorCount": 1}, compile_exit=compile_exit)
    report = run_harness(project, tmp_path / "evidence", "uloop.exe", "m1", runner=editor, launch=True)
    assert report["success"] is False
    assert [call[0][3] for call in editor.calls] == ["launch", "compile", "get-logs"]


def test_compile_only_is_explicitly_not_m1_verification(project, tmp_path):
    editor = EditorCLI()
    report = run_harness(project, tmp_path / "evidence", "uloop.exe", "compile", runner=editor)
    assert report["success"] is True
    assert report["m1_reference_passed"] is False
    assert [call[0][3] for call in editor.calls] == ["compile", "get-logs"]


def test_launch_progress_and_json_are_supported(project, tmp_path):
    editor = EditorCLI()

    def progress(args, **kwargs):
        result = editor(args, **kwargs)
        if args[3] == "launch":
            result.stdout = 'Waiting for Unity...\n' + result.stdout
        return result

    report = run_harness(project, tmp_path / "evidence", "uloop.exe", "compile", runner=progress, launch=True)
    assert report["success"] is True


def test_launch_failure_does_not_attempt_compilation(project, tmp_path):
    editor = EditorCLI()

    def not_ready(args, **kwargs):
        result = editor(args, **kwargs)
        if args[3] == "launch":
            result.stdout = '{"Success": false, "Ready": false}'
        return result

    report = run_harness(project, tmp_path / "evidence", "uloop.exe", "compile", runner=not_ready, launch=True)
    assert report["success"] is False
    assert [call[0][3] for call in editor.calls] == ["launch", "get-logs"]


def test_timeout_overwrites_stale_success_and_collects_logs(project, tmp_path):
    output = tmp_path / "evidence"
    output.mkdir()
    (output / "summary.json").write_text('{"success": true, "m1_reference_passed": true}')
    editor = EditorCLI()

    def timeout(args, **kwargs):
        if args[3] == "compile":
            raise subprocess.TimeoutExpired(args, 900, output="partial compiler output")
        return editor(args, **kwargs)

    report = run_harness(project, output, "uloop.exe", "m1", runner=timeout)
    assert report["success"] is False
    assert report["m1_reference_passed"] is False
    assert "timeout" in report["error"].lower()
    assert json.loads((output / "summary.json").read_text())["success"] is False
    assert editor.calls[-1][0][3] == "get-logs"
    assert "partial compiler output" in (output / "compile.json").read_text()


def test_malformed_cli_response_is_a_failure(project, tmp_path):
    editor = EditorCLI()

    def malformed(args, **kwargs):
        result = editor(args, **kwargs)
        if args[3] == "compile":
            result.stdout = "Editor busy, no result"
        return result

    report = run_harness(project, tmp_path / "evidence", "uloop.exe", "m1", runner=malformed)
    assert report["success"] is False
    assert report["m1_reference_passed"] is False


def test_failed_cli_preserves_test_counts_and_xml_in_evidence(project, tmp_path):
    xml = project / ".uloop/outputs/TestResults/failed.xml"
    xml.parent.mkdir(parents=True)
    xml.write_text('<test-run result="Failed"/>')
    editor = EditorCLI(tests={"Success": False, "TestCount": 3, "PassedCount": 0,
                              "FailedCount": 3, "SkippedCount": 0, "XmlPath": str(xml)})

    def failed(args, **kwargs):
        result = editor(args, **kwargs)
        if args[3] == "run-tests":
            result.returncode = 1
        return result

    output = tmp_path / "evidence"
    report = run_harness(project, output, "uloop.exe", "m1", runner=failed)
    assert report["success"] is False
    assert report["tests"]["FailedCount"] == 3
    assert (output / "test-results.xml").read_text() == xml.read_text()


def test_cli_main_returns_nonzero_when_verification_fails(project, tmp_path, monkeypatch):
    from embeddinggemma_tools import unity
    monkeypatch.setattr(unity, "run_harness", lambda *args, **kwargs:
                        {"success": False, "m1_reference_passed": False, "error": "failed"})
    assert unity.main(["--project", str(project), "--output", str(tmp_path / "evidence")]) == 1
