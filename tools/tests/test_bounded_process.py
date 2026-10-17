import json
import sys
from pathlib import Path
import time
import pytest

from embeddinggemma_tools import bounded_process


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
