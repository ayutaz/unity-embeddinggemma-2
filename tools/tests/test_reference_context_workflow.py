from pathlib import Path

import yaml
from workflow_privacy_assertions import assert_privacy_gated_upload


def job():
    path = Path(__file__).resolve().parents[2] / ".github/workflows/reference-context.yml"
    assert path.is_file(), "Standalone native context workflow is missing"
    return yaml.safe_load(path.read_text(encoding="utf-8"))["jobs"]["probe"]


def test_context_job_is_opt_in_and_compares_two_cpu_contexts_for_both_modalities():
    value = job()
    assert "reference-context-diagnostics" in value["if"]
    assert value["strategy"]["matrix"]["modality"] == ["audio", "image"]
    assert value["strategy"]["matrix"]["context"] == ["native", "avx2"]
    configure = next(step for step in value["steps"] if step.get("name") == "Constrain the common AVX2 context")
    assert configure["if"] == "matrix.context == 'avx2'"
    assert all(name in configure["run"] for name in ("ATEN_CPU_CAPABILITY=avx2", "MKL_CBWR=AVX2", "DNNL_MAX_CPU_ISA=AVX2"))


def test_context_job_uses_only_saved_encoder_and_fixture_with_pinned_manifest_audit():
    steps = job()["steps"]
    fetch = next(step["run"] for step in steps if step.get("name") == "Fetch fixed graph and first fixture")
    assert "--artifact-id 11814567883" in fetch and "--artifact-id 11830312950" in fetch
    assert "--member audio-runtime-reference/frames-1119-valid-1119/audio.pt2" in fetch
    assert "--member square/vision.pt2" in fetch and "--member square/0-pixels-fp32.bin" in fetch
    assert "backbone.pt2" not in fetch
    execute = next(step["run"] for step in steps if step.get("name") == "Replay fixed expected features under controlled threads")
    assert "replay_saved_reference_context" in execute
    assert "snapshot_download" not in execute
    upload = next(step for step in steps if "actions/upload-artifact@" in step.get("uses", ""))
    assert_privacy_gated_upload(upload)
    assert "*.pt2" not in upload["with"]["path"]
