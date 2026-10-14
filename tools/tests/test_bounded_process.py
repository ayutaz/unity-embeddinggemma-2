import json
import sys

from embeddinggemma_tools import bounded_process


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
