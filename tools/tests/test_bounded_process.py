import json
import sys
from pathlib import Path
import time
import pytest

from embeddinggemma_tools import bounded_process


@pytest.mark.parametrize("error_type", [PermissionError, RuntimeError, KeyboardInterrupt])
def test_post_spawn_receipt_failure_reaps_only_the_owned_process(tmp_path, monkeypatch, error_type):
    output = tmp_path / "post-spawn-failure.json"
    original_spawn = bounded_process.subprocess.Popen
    original_save = bounded_process.write_json_atomic
    children = []
    saves = 0

    def spawn(*args, **kwargs):
        child = original_spawn(*args, **kwargs)
        children.append(child)
        return child

    def fail_after_spawn(path, report):
        nonlocal saves
        saves += 1
        if saves == 2:
            raise error_type("receipt unavailable after spawn")
        return original_save(path, report)

    monkeypatch.setattr(bounded_process.subprocess, "Popen", spawn)
    monkeypatch.setattr(bounded_process, "write_json_atomic", fail_after_spawn)
    try:
        with pytest.raises(error_type, match="receipt unavailable after spawn"):
            bounded_process.run_bounded([sys.executable, "-c", "import time; time.sleep(30)"], output, timeout=5)
        assert len(children) == 1
        assert children[0].poll() is not None, "A failed receipt must not orphan the owned process"
        report = json.loads(output.read_text())
        assert report["terminal"] and not report["success"] and not report["timed_out"]
        assert report["pid"] == children[0].pid and report["exit_code"] == children[0].returncode
        assert report["supervisor_terminated_child"] is True
        assert error_type.__name__ in report["error"]
    finally:
        # The intentional Red also must not leave this test's own subprocess behind.
        for child in children:
            if child.poll() is None:
                child.kill()
            child.wait(timeout=5)


@pytest.mark.parametrize("winerror", [5, 32, 33])
def test_transient_windows_receipt_lock_preserves_process_and_final_result(tmp_path, monkeypatch, winerror):
    output = tmp_path / "shared-receipt.json"
    replace = Path.replace
    calls = 0

    def briefly_locked(path, target):
        nonlocal calls
        calls += 1
        if Path(target) == output and calls == 2:
            error = PermissionError("transient Windows receipt sharing conflict")
            error.winerror = winerror
            raise error
        return replace(path, target)

    monkeypatch.setattr(Path, "replace", briefly_locked)
    report = bounded_process.run_bounded([sys.executable, "-c", "pass"], output, timeout=5)
    assert report["terminal"] and report["success"] and report["exit_code"] == 0
    assert json.loads(output.read_text()) == report


@pytest.mark.parametrize("winerror", [None, 5])
def test_persistent_receipt_denial_remains_an_error_with_bounded_wait(tmp_path, monkeypatch, winerror):
    output = tmp_path / "denied.json"

    def denied(path, target):
        error = PermissionError("persistent receipt permission denial")
        if winerror is not None:
            error.winerror = winerror
        raise error

    monkeypatch.setattr(Path, "replace", denied)
    started = time.monotonic()
    with pytest.raises(PermissionError, match="persistent receipt permission denial"):
        bounded_process.run_bounded([sys.executable, "-c", "pass"], output, timeout=5)
    assert time.monotonic() - started < 2


def test_timeout_stops_owned_process_and_persists_incomplete_result(tmp_path):
    report = bounded_process.run_bounded(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        tmp_path / "timeout.json", timeout=0.15)
    assert report["timed_out"] is True
    assert report["success"] is False and report["terminal"] is True
    assert report["elapsed_seconds"] < 5
    assert json.loads((tmp_path / "timeout.json").read_text())["pid"] == report["pid"]


def test_success_requires_natural_zero_exit_and_does_not_overwrite_evidence(tmp_path):
    output = tmp_path / "complete.json"
    report = bounded_process.run_bounded([sys.executable, "-c", "pass"], output, timeout=5)
    assert report["success"] and not report["timed_out"]
    assert report["exit_code"] == 0
    import pytest
    with pytest.raises(FileExistsError):
        bounded_process.run_bounded([sys.executable, "-c", "pass"], output, timeout=5)


def test_nonzero_exit_is_failed_not_timeout(tmp_path):
    report = bounded_process.run_bounded([sys.executable, "-c", "raise SystemExit(2)"],
        tmp_path / "failed.json", timeout=5)
    assert report["exit_code"] == 2 and not report["success"]
    assert not report["timed_out"]
