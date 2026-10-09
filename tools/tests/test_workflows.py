from pathlib import Path
import json

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


def test_package_audit_is_required_and_creates_an_isolated_consumer():
    assert "package" in REQUIRED_JOBS
    workflow = load("package-validation.yml")
    commands = "\n".join(step.get("run", "") for step in workflow["jobs"]["package"]["steps"])
    assert "embeddinggemma_tools.package" in commands
    assert "--consumer" in commands
    assert "--git-revision" in commands


def test_package_audit_failure_cannot_be_hidden_by_tee():
    workflow = load("package-validation.yml")
    audit = next(step for step in workflow["jobs"]["package"]["steps"]
                 if "embeddinggemma_tools.package" in step.get("run", ""))
    assert audit["shell"] == "bash"  # GitHub explicitly enables -eo pipefail.


def test_main_protection_requires_pr_and_actions_check_for_admins_too():
    policy = json.loads((ROOT / ".github/main-protection.json").read_text())
    assert policy["enforce_admins"] is True
    assert policy["required_status_checks"] == {
        "strict": True, "checks": [{"context": "Required CI", "app_id": 15368}]}
    assert policy["required_pull_request_reviews"] is not None
    assert policy["required_pull_request_reviews"]["required_approving_review_count"] == 0
    assert policy["required_conversation_resolution"] is True
    assert policy["required_linear_history"] is True
    assert policy["allow_force_pushes"] is False
    assert policy["allow_deletions"] is False
