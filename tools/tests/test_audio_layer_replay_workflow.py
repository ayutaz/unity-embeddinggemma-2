"""The diagnostic job must bind the frozen context and retain failed receipts."""

from pathlib import Path

import yaml


def test_native_replay_job_pins_old_context_and_never_claims_full_encoder_pass():
    path = Path(__file__).resolve().parents[2] / ".github/workflows/audio-layer-replay.yml"
    assert path.is_file(), "Frozen-context native layer replay workflow is missing"
    workflow = yaml.safe_load(path.read_text(encoding="utf-8"))
    triggers = workflow.get("on", workflow.get(True))
    assert "pull_request_target" not in triggers
    assert workflow["permissions"] == {"contents": "read", "actions": "read"}
    job = workflow["jobs"]["replay"]
    assert "head.repo.full_name == github.repository" in job["if"]
    commands = "\n".join(step.get("run", "") for step in job["steps"])
    assert "--artifact-id 11814567883" in commands
    assert "6620f335563a91ccc484df7d9b89d4d889e58dab4ec3d7d0389b36abe36b454a" in commands
    assert "3edee8acdd9ebebb819ae488b753a3a94949b953aa0906995e516764932b0f9b" in commands
    assert "sha256sum --check" in commands
    assert "--reference-source 4edbe17a8105233c5aaa1514faa2862b08a4f7e0" in commands
    assert "embeddinggemma_tools.audio_layer_replay" in commands
    assert "--detail-feed-forward" in commands
    assert "--detail-attention-projections" in commands
    assert "--projection-oracle" in commands
    assert "--report ../artifacts/audio-layer-replay-report.json" in commands
    assert not any(step.get("continue-on-error") for step in job["steps"])
    uploads = [step for step in job["steps"] if step.get("uses", "").startswith("actions/upload-artifact@")]
    assert len(uploads) == 1 and uploads[0]["if"] == "always()"
    assert "audio-layer-replay-report.json" in uploads[0]["with"]["path"]


def test_replay_compares_compensated_fp32_without_relaxing_failure():
    path = Path(__file__).resolve().parents[2] / ".github/workflows/audio-layer-replay.yml"
    workflow = yaml.safe_load(path.read_text(encoding="utf-8"))
    job = workflow["jobs"]["replay"]
    commands = "\n".join(step.get("run", "") for step in job["steps"])
    assert "--compensated-projection" in commands, "Real FP32 candidate is not exercised"
    assert not any(step.get("continue-on-error") for step in job["steps"])


def test_attention_reference_scope_keeps_full_subsample_replay_as_separate_default_gate():
    path=Path(__file__).resolve().parents[2]/'.github/workflows/audio-layer-replay.yml'
    workflow=yaml.safe_load(path.read_text(encoding='utf-8'));triggers=workflow.get('on',workflow.get(True))
    dispatch=triggers['workflow_dispatch']
    assert isinstance(dispatch,dict),'Independent pinned-attention diagnostic scope is missing'
    assert dispatch['inputs']['scope']['default']=='full'
    steps=workflow['jobs']['replay']['steps']
    full=next(s for s in steps if s.get('name')=='Replay official layer 0 against unchanged native final')
    attention=next(s for s in steps if s.get('name')=='Capture attention projections against pinned layer final')
    assert "inputs.scope != 'attention'" in full['if'] and '--subsample-input' in full['run']
    assert "inputs.scope == 'attention'" in attention['if']
    assert '--detail-attention-projections' in attention['run'] and '--subsample-input' not in attention['run']
    assert '--detail-attention-ops' in attention['run']
    assert '--detail-convolution' in attention['run'],'Native convolution boundaries are missing from the pinned reference job'
    assert '--reference-source 4edbe17a8105233c5aaa1514faa2862b08a4f7e0' in attention['run']
    assert not any(s.get('continue-on-error') for s in steps)
