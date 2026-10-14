from pathlib import Path

import yaml


def workflow():
    path = Path(__file__).resolve().parents[2] / ".github/workflows/image-role-reference.yml"
    assert path.is_file(), "Fixed image role reference workflow is missing"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_image_role_job_pins_all_native_profiles_and_shared_backbones():
    job = workflow()["jobs"]["reference"]
    assert "image-role-reference" in job["if"]
    steps = job["steps"]
    native = next(v["run"] for v in steps if v.get("name") == "Fetch fixed native vision files")
    assert "--artifact-id 11830312950" in native
    for name in ("square", "landscape", "portrait"):
        assert "--member " + name + "/vision.pt2" in native
        assert "--member " + name + "/positions-int32.bin" in native
        assert "--member " + name + "/0-pixels-fp32.bin" in native
        assert "--member " + name + "/1-pixels-fp32.bin" in native
        assert "--member " + name + "/0.png" in native, "Image producer audits the complete input manifest"
        assert "--member " + name + "/1.png" in native
    assert "--member processor_config.json" in native
    backbone = next(v["run"] for v in steps if v.get("name") == "Fetch fixed document backbones")
    assert "--artifact-id 11830367896" in backbone
    assert "--member square/backbone.pt2" in backbone and "--member landscape/backbone.pt2" in backbone
    assert "portrait/backbone.pt2" not in backbone


def test_image_role_job_preserves_source_role_and_failed_context_evidence():
    steps = workflow()["jobs"]["reference"]["steps"]
    run = next(v["run"] for v in steps if v.get("name") == "Compare every role against official real-model forward")
    assert "for role in document query raw" in run
    assert "--vision-source-commit c868df5ca67f1ca8a6173a0f8e81f9f0bb7367e1" in run
    assert "--backbone-source ../artifacts/fixed-image-document-backbone" in run
    upload = next(v for v in steps if "actions/upload-artifact@" in v.get("uses", ""))
    assert upload["if"] == "always()"
    assert "*-actual-fp32.bin" in upload["with"]["path"]
    assert "*.pt2" not in upload["with"]["path"]
