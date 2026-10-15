import importlib
import json
from pathlib import Path

import pytest
import yaml


def producer():
    try: module=importlib.import_module("embeddinggemma_tools.ci_audio_subsample_probe")
    except ModuleNotFoundError: pytest.fail("Bounded fixed-artifact subsample probe is missing")
    return module


def test_untrusted_prefix_files_are_rejected_before_model_loading(tmp_path,monkeypatch):
    module=producer();root=tmp_path/"input";root.mkdir();out=tmp_path/"out";out.mkdir()
    stale=out/"subsample-probe.json";stale.write_text(json.dumps(dict(success=True)))
    for name in module.PINNED_FILES:(root/name).write_bytes(b"not the pinned diagnostic")
    monkeypatch.setattr(module.onnx,"load",lambda *args,**kwargs:pytest.fail("Audit must precede graph loading"))
    with pytest.raises(ValueError,match="[Pp]inned"):module.probe(root,out,source_commit="a"*40)
    assert not stale.exists()


def test_probe_projection_covers_a_partial_final_block_with_exact_integer_oracle():
    import torch
    module=producer()
    values=(torch.arange(400,dtype=torch.float32).reshape(1,2,200)%17)-8
    weights=(torch.arange(600,dtype=torch.float32).reshape(200,3)%7)-3
    actual=module.CompensatedMatmul()(values,weights)
    assert torch.equal(actual,(values.double()@weights.double()).float())


def test_prefix_workflow_is_opt_in_small_and_retains_the_full_onnx_gate():
    root=Path(__file__).resolve().parents[2]
    path=root/".github/workflows/ci-audio-subsample-probe.yml"
    assert path.is_file(),"Prefix-only Actions probe must be reviewable before execution"
    job=yaml.safe_load(path.read_text(encoding="utf-8"))["jobs"]["probe"]
    assert job["timeout-minutes"]==15 and "ci-audio-subsample-probe" in job["if"]
    commands="\n".join(step.get("run","") for step in job["steps"])
    assert "--artifact-id 11986081282" in commands
    assert "frames-1119-valid-1119/audio-onnx-subsample-stages.onnx" in commands
    assert "embeddinggemma_tools.ci_audio_subsample_probe" in commands
    assert "prepare-audio-encoder" not in commands and "snapshot_download" not in commands
    assert not any(step.get("continue-on-error") for step in job["steps"])
    assert "ci_audio_subsample_probe" not in (root/".github/workflows/model-reference.yml").read_text(encoding="utf-8")
