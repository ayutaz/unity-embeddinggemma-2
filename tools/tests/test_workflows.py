from pathlib import Path
import json

import yaml

from embeddinggemma_tools.ci import REQUIRED_JOBS

ROOT = Path(__file__).resolve().parents[2]


def load(name):
    # BaseLoader preserves the YAML key "on" instead of treating it as a bool.
    return yaml.load((ROOT / ".github/workflows" / name).read_text(encoding="utf-8"),
                     Loader=yaml.BaseLoader)


def test_audio_encoder_only_dispatch_preserves_all_signals_without_extra_chain_exports():
    workflow = load("audio-runtime-reference.yml")
    option = workflow["on"]["workflow_dispatch"]["inputs"]["encoder_only"]
    assert option["type"] == "boolean" and option["default"] == "true"
    steps = workflow["jobs"]["native-reference"]["steps"]
    encoder = next(s for s in steps if "prepare-audio-encoder" in s.get("run", ""))
    assert "--native-silu --collect-onnx-failures" in encoder["run"]
    assert "if" not in encoder
    for command in ("prepare-audio-runtime-stages", "prepare-audio-embeddings"):
        step = next(s for s in steps if command in s.get("run", ""))
        assert "!inputs.encoder_only" in step["if"]
        assert "always()" in step["if"]


def test_audio_first_failure_dispatch_runs_existing_context_guarded_localization():
    workflow = load("audio-runtime-reference.yml")
    option = workflow["on"]["workflow_dispatch"]["inputs"]["diagnose_first_failure"]
    assert option["type"] == "boolean" and option["default"] == "false"
    encoder = next(s for s in workflow["jobs"]["native-reference"]["steps"]
                   if "prepare-audio-encoder" in s.get("run", ""))
    assert encoder["env"]["DIAGNOSE_FIRST_FAILURE"] == "${{ inputs.diagnose_first_failure }}"
    command = encoder["run"]
    assert 'if [ "$DIAGNOSE_FIRST_FAILURE" = true ]; then' in command
    diagnostic, collection = command.split("else", 1)
    assert "--native-silu" in diagnostic and "--collect-onnx-failures" not in diagnostic
    assert "--native-silu --collect-onnx-failures" in collection
    assert "continue-on-error" not in encoder and "|| true" not in command


def test_audio_candidate_dispatch_keeps_original_pinned_golden_and_all_six_outputs():
    workflow=load('audio-runtime-reference.yml')
    option=workflow['on']['workflow_dispatch']['inputs']['onnx_layernorm']
    assert option['type']=='choice' and option['default']=='native'
    assert option['options']==['native','torch_welford8','torch_welford8_refined_moment']
    assert workflow['permissions']['actions']=='read'
    steps=workflow['jobs']['native-reference']['steps']
    commands='\n'.join(s.get('run','') for s in steps)
    assert '11814567883' in commands and 'ac23f39904b8149dd5ccb6485afcc93d96047f0977b5faf1d640c8bd018c13c9' in commands
    validation=next(s for s in steps if 'embeddinggemma_tools.pinned_audio_onnx' in s.get('run',''))
    assert 'always()' in validation['if']
    assert '--pinned ../artifacts/pinned-audio/audio-runtime-reference' in validation['run']
    assert '--output ../artifacts/audio-pinned-onnx.json' in validation['run']
    assert not validation.get('continue-on-error')
    assert any(s.get('with',{}).get('path')=='artifacts/audio-pinned-onnx.json' and s.get('if')=='always()' for s in steps)


def test_audio_projection_candidate_is_explicit_and_preserves_default_math():
    workflow=load('audio-runtime-reference.yml')
    option=workflow['on']['workflow_dispatch']['inputs']['onnx_linear']
    assert option['type']=='choice' and option['default']=='native'
    assert option['options']==['native','compensated64']
    encoder=next(s for s in workflow['jobs']['native-reference']['steps'] if 'prepare-audio-encoder' in s.get('run',''))
    assert '--onnx-linear "$ONNX_LINEAR"' in encoder['run']


def test_required_reference_job_validates_long_multimodal_backbone_and_retains_separate_evidence():
    steps = load("model-reference.yml")["jobs"]["reference"]["steps"]
    run = [step for step in steps if "prepare-backbone" in step.get("run", "")]
    assert len(run) == 1, "required real-weight backbone validation is missing"
    assert "uv run --locked python -m embeddinggemma_tools prepare-backbone" in run[0]["run"]
    artifact = [step for step in steps if step.get("with", {}).get("name", "").startswith("multimodal-backbone-")]
    assert len(artifact) == 1
    assert artifact[0]["if"] == "always()"
    assert artifact[0]["with"]["path"] == "artifacts/multimodal-backbone/"
    assert artifact[0]["with"]["retention-days"] == "3"


def test_required_reference_job_validates_real_vision_and_preserves_partial_failure_evidence():
    steps = load("model-reference.yml")["jobs"]["reference"]["steps"]
    run = [step for step in steps if "prepare-vision" in step.get("run", "")]
    assert len(run) == 1, "required real-weight vision export gate is missing"
    assert "uv run --locked python -m embeddinggemma_tools prepare-vision" in run[0]["run"]
    artifact = [step for step in steps if step.get("with", {}).get("name", "").startswith("vision-encoder-")]
    assert len(artifact) == 1
    assert artifact[0]["if"] == "always()"
    assert artifact[0]["with"]["path"] == "artifacts/vision-encoder/"
    assert artifact[0]["with"]["retention-days"] == "3"


def test_required_reference_job_validates_fixed_audio_preprocessing_and_retains_scoped_artifact():
    steps = load("model-reference.yml")["jobs"]["reference"]["steps"]
    run = [step for step in steps if "embeddinggemma_tools.audio" in step.get("run", "")]
    assert len(run) == 1, "required audio preprocessing export gate is missing"
    assert "uv run --locked python -m embeddinggemma_tools.audio" in run[0]["run"]
    artifact = [step for step in steps if step.get("with", {}).get("name", "").startswith("audio-preprocessing-")]
    assert len(artifact) == 1
    assert artifact[0]["if"] == "always()"
    assert artifact[0]["with"]["path"] == "artifacts/audio-preprocessing/"
    assert artifact[0]["with"]["retention-days"] == "3"


def test_required_ci_runs_on_every_pr_including_documentation():
    workflow = load("ci.yml")
    assert "pull_request" in workflow["on"]
    assert not workflow["on"]["pull_request"]
    assert workflow["on"]["push"]["branches"] == ["main"]
    assert workflow["permissions"] == {"contents": "read", "actions": "read"}
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


def test_manual_unity_and_project_use_the_approved_shutdown_fix_version():
    version = '6000.3.21f1'
    project_version = (ROOT / 'ProjectSettings/ProjectVersion.txt').read_text()
    assert f'm_EditorVersion: {version}\n' in project_version
    steps = load('unity-validation.yml')['jobs']['unity']['steps']
    game_ci = next(step for step in steps if step.get('uses', '').startswith('game-ci/unity-test-runner@'))
    assert game_ci['with']['unityVersion'] == version


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


def test_search_reference_can_be_downloaded_without_the_large_model():
    workflow = load("model-reference.yml")
    steps = workflow["jobs"]["reference"]["steps"]
    artifact = next(step for step in steps if step.get("with", {}).get("name", "").startswith("search-reference-"))
    assert artifact["with"]["path"].splitlines() == ["artifacts/m1/search-reference.json", "artifacts/m1/export-validation.json"]
    assert artifact["with"]["retention-days"] == "3"
    assert artifact["with"]["if-no-files-found"] == "error"
