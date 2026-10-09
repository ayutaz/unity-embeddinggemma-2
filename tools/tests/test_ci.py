import json

import pytest

from embeddinggemma_tools.ci import REQUIRED_JOBS, main, validate_results


def successful_jobs():
    return {name: {"result": "success"} for name in REQUIRED_JOBS}


def test_all_required_jobs_must_succeed():
    assert validate_results(successful_jobs()) == []


@pytest.mark.parametrize("result", ["failure", "cancelled", "skipped", None, "", "unknown"])
def test_required_job_cannot_be_skipped_or_fail(result):
    jobs = successful_jobs()
    jobs[REQUIRED_JOBS[0]]["result"] = result
    assert validate_results(jobs) == [REQUIRED_JOBS[0]]


def test_missing_job_is_failure():
    jobs = successful_jobs()
    del jobs[REQUIRED_JOBS[-1]]
    assert validate_results(jobs) == [REQUIRED_JOBS[-1]]


@pytest.mark.parametrize("jobs", [None, [], "success", {name: "success" for name in REQUIRED_JOBS}])
def test_malformed_results_fail_closed(jobs):
    assert validate_results(jobs) == list(REQUIRED_JOBS)


def test_cli_success_and_failure(monkeypatch, capsys):
    monkeypatch.setenv("NEEDS_JSON", json.dumps(successful_jobs()))
    assert main() == 0
    assert json.loads(capsys.readouterr().out)["success"] is True
    monkeypatch.setenv("NEEDS_JSON", "{}")
    assert main() == 1
    assert json.loads(capsys.readouterr().out)["success"] is False


def test_cli_invalid_json_and_missing_environment(monkeypatch):
    monkeypatch.setenv("NEEDS_JSON", "not json")
    assert main() == 1
    monkeypatch.delenv("NEEDS_JSON")
    assert main() == 1
