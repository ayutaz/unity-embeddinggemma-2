from pathlib import Path

import yaml


def workflow():
    path = Path(__file__).resolve().parents[2] / ".github/workflows/audio-role-reference.yml"
    assert path.is_file(), "Fixed audio role reference workflow is missing"
    return yaml.safe_load(path.read_text())


def test_role_job_is_opt_in_and_pins_native_and_backbone_artifacts():
    value = workflow()
    job = value["jobs"]["reference"]
    assert "audio-role-reference" in job["if"]
    assert "github.event.pull_request.head.repo.full_name == github.repository" in job["if"]
    steps = job["steps"]
    native = next(v["run"] for v in steps if v.get("name") == "Fetch fixed native encoder files")
    backbone = next(v["run"] for v in steps if v.get("name") == "Fetch fixed document backbones")
    assert "--artifact-id 11814567883" in native
    assert "--artifact-id 11814552957" in backbone
    assert "--max-entry-bytes 2147483648" in native
    assert "--member audio-runtime-reference/frames-1119-valid-1119/audio.pt2" in native
    assert "--member audio-runtime-reference/frames-1119-valid-839/audio.pt2" in native
    assert "--member frames-1119-valid-1119/backbone.pt2" in backbone
    assert "--member frames-1119-valid-839/backbone.pt2" in backbone


def test_role_generation_uses_three_roles_reuse_and_separate_encoder_source():
    steps = workflow()["jobs"]["reference"]["steps"]
    audit_index = next(i for i, v in enumerate(steps) if v.get("name") == "Audit pinned manifests before model download")
    run_index = next(i for i, v in enumerate(steps) if v.get("name") == "Compare every role against official real-model forward")
    assert audit_index < run_index
    run = steps[run_index]["run"]
    assert "for role in document query raw" in run
    assert "--role \"$role\"" in run
    assert "--backbone-source ../artifacts/fixed-document-backbone" in run
    assert "--encoder-source-commit 4edbe17a8105233c5aaa1514faa2862b08a4f7e0" in run
    upload = next(v for v in steps if "actions/upload-artifact@" in v.get("uses", ""))
    assert "if" in upload and "always()" in upload["if"]
    assert "*.pt2" not in upload["with"]["path"]
