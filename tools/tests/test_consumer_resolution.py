import json

import pytest

from embeddinggemma_tools.package import PACKAGE_NAME, audit_git_consumer, main

REVISION = "a" * 40
URL = f"https://github.com/ayutaz/unity-embeddinggemma-2.git?path=/Packages/{PACKAGE_NAME}#{REVISION}"


@pytest.fixture
def consumer(tmp_path):
    packages = tmp_path / "Packages"
    packages.mkdir()
    (packages / "manifest.json").write_text(json.dumps({"dependencies": {PACKAGE_NAME: URL}}))
    (packages / "packages-lock.json").write_text(json.dumps({"dependencies": {
        PACKAGE_NAME: {"version": URL, "source": "git", "depth": 0, "hash": REVISION,
                       "dependencies": {"com.unity.ai.inference": "2.6.1", "com.unity.nuget.newtonsoft-json": "3.2.2"}},
        "com.unity.ai.inference": {"version": "2.6.1", "source": "registry"},
        "com.unity.nuget.newtonsoft-json": {"version": "3.2.2", "source": "registry"},
    }}))
    return tmp_path


def test_pinned_git_resolution_is_distinct_from_unity_execution(consumer):
    result = audit_git_consumer(consumer, REVISION)
    assert result["success"] is True
    assert result["resolved_revision"] == REVISION
    assert result["unity_executed"] is False
    assert result["errors"] == []


@pytest.mark.parametrize("mutation", ["missing_lock", "bad_json", "non_object", "missing_package", "local_source",
                                    "wrong_hash", "wrong_url", "wrong_dependency", "wrong_resolved_dependency"])
def test_missing_stale_or_substituted_resolution_fails(consumer, mutation):
    path = consumer / "Packages/packages-lock.json"
    data = json.loads(path.read_text())
    if mutation == "missing_lock":
        path.unlink()
    elif mutation == "bad_json":
        path.write_text("not json")
    elif mutation == "non_object":
        path.write_text("[]")
    else:
        entry = data["dependencies"][PACKAGE_NAME]
        if mutation == "missing_package":
            del data["dependencies"][PACKAGE_NAME]
        elif mutation == "local_source":
            entry["source"] = "local"
        elif mutation == "wrong_hash":
            entry["hash"] = "b" * 40
        elif mutation == "wrong_url":
            entry["version"] = URL.replace(REVISION, "b" * 40)
        elif mutation == "wrong_dependency":
            entry["dependencies"]["com.unity.ai.inference"] = "2.7.0"
        elif mutation == "wrong_resolved_dependency":
            data["dependencies"]["com.unity.ai.inference"]["version"] = "2.7.0"
        path.write_text(json.dumps(data))
    result = audit_git_consumer(consumer, REVISION)
    assert result["success"] is False
    assert result["unity_executed"] is False
    assert result["errors"]


def test_stale_lock_does_not_override_changed_manifest(consumer):
    path = consumer / "Packages/manifest.json"
    path.write_text(json.dumps({"dependencies": {PACKAGE_NAME: "file:/local/package"}}))
    assert audit_git_consumer(consumer, REVISION)["success"] is False


@pytest.mark.parametrize("revision", [None, "main", "v0.1.0", "A" * 40, "a" * 39])
def test_audit_requires_exact_commit_not_a_floating_reference(consumer, revision):
    result = audit_git_consumer(consumer, revision)
    assert result["success"] is False
    assert result["errors"]


def test_cli_reports_resolution_without_claiming_editor_execution(consumer, capsys):
    assert main(["--verify-git-consumer", str(consumer), "--git-revision", REVISION]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["success"] is True
    assert result["unity_executed"] is False
    (consumer / "Packages/packages-lock.json").unlink()
    assert main(["--verify-git-consumer", str(consumer), "--git-revision", REVISION]) == 1


def test_cli_refuses_to_mix_verification_and_creation(consumer):
    with pytest.raises(SystemExit) as error:
        main(["--verify-git-consumer", str(consumer), "--consumer", str(consumer / "new"),
              "--git-revision", REVISION])
    assert error.value.code == 2
    assert not (consumer / "new").exists()
