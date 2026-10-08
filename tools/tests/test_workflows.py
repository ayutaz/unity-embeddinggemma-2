from pathlib import Path

import yaml

from embeddinggemma_tools.ci import REQUIRED_JOBS

ROOT = Path(__file__).resolve().parents[2]


def load(name):
    # BaseLoader preserves the YAML key "on" instead of treating it as a bool.
    return yaml.load((ROOT / ".github/workflows" / name).read_text(encoding="utf-8"),
                     Loader=yaml.BaseLoader)


def test_required_ci_runs_on_every_pr_including_documentation():
    workflow = load("ci.yml")
    assert "pull_request" in workflow["on"]
    assert not workflow["on"]["pull_request"]
    assert workflow["on"]["push"]["branches"] == ["main"]
    assert workflow["permissions"] == {"contents": "read"}
    assert "pull_request_target" not in workflow["on"]


def test_required_gate_covers_all_reusable_workflows_even_after_failure():
    workflow = load("ci.yml")
    gate = workflow["jobs"]["required"]
    assert gate["name"] == "Required CI"
    assert set(gate["needs"]) == set(REQUIRED_JOBS)
    assert gate["if"] == "${{ always() }}"
    assert gate["steps"][-1]["env"]["NEEDS_JSON"] == "${{ toJSON(needs) }}"
    assert "embeddinggemma_tools.ci" in gate["steps"][-1]["run"]
    assert not gate.get("continue-on-error")
    for name in REQUIRED_JOBS:
        job = workflow["jobs"][name]
        assert not job.get("if")
        assert not job.get("continue-on-error")
        assert job["uses"].startswith("./.github/workflows/")
        child = load(job["uses"].rsplit("/", 1)[-1])
        assert "workflow_call" in child["on"]
        assert not {"push", "pull_request", "pull_request_target"}.intersection(child["on"])


def test_cloud_unity_remains_manual_and_does_not_receive_fork_secrets():
    workflow = load("unity-validation.yml")
    assert set(workflow["on"]) == {"workflow_dispatch"}
    assert "secrets" not in load("ci.yml")["jobs"]["model-reference"]
