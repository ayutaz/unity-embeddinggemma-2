import hashlib
import json

import numpy as np
import pytest
import torch

from embeddinggemma_tools import reference_context_failure as context


class AddOne(torch.nn.Module):
    def forward(self, values):
        return values + 1


@pytest.fixture
def saved_source(tmp_path):
    values = torch.tensor([[1., 2.]])
    torch.export.save(torch.export.export(AddOne(), (values,)), tmp_path / "encoder.pt2")
    values.numpy().astype("<f4").tofile(tmp_path / "input.bin")
    metadata = dict(source_commit="a" * 40, model_revision="b" * 40)
    reference = dict(metadata=metadata, profiles={"fixed": dict(feature_shape=[1, 2], native_graph="encoder.pt2")},
                     cases=[dict(id="silence", profile="fixed", input_features="input.bin", features=[[2., 3.]])])
    (tmp_path / "audio-encoder-reference.json").write_text(json.dumps(reference), encoding="utf-8")
    files = {name: dict(bytes=(tmp_path / name).stat().st_size, sha256=hashlib.sha256((tmp_path / name).read_bytes()).hexdigest())
             for name in ("audio-encoder-reference.json", "input.bin", "encoder.pt2")}
    audit = dict(metadata=metadata, stage="audio_encoder_projector_only", native_reference_complete=True, files=files)
    (tmp_path / "audio-encoder-validation.json").write_text(json.dumps(audit), encoding="utf-8")
    return tmp_path


def replay(*args, **kwargs):
    function = getattr(context, "replay_saved_reference_context", None)
    assert callable(function), "Audited standalone saved-reference replay is missing"
    return function(*args, **kwargs)


def test_standalone_replay_uses_fixed_program_input_and_original_expected(saved_source):
    report = replay(saved_source, "audio", source_commit="a" * 40, revision="b" * 40, case_id="silence")
    assert report["case"] == "silence" and report["reference_source"] == "a" * 40
    assert all(row["mismatched_elements"] == 0 for row in report["thread_contexts"]["conditions"])
    assert report["input_sha256"] == hashlib.sha256((saved_source / "input.bin").read_bytes()).hexdigest()
    assert report["model_weights_downloaded"] is False and report["full_encoder_validated"] is False


def test_image_replay_audits_the_selected_graph_and_pixels_without_unrelated_pngs(saved_source):
    original = json.loads((saved_source / "audio-encoder-reference.json").read_text())
    profile = original["profiles"]["fixed"]
    profile["pixel_shape"] = profile.pop("feature_shape")
    profile["graph"] = profile.pop("native_graph")
    row = original["cases"][0]
    row["id"] = "square-0"; row["pixels"] = row.pop("input_features")
    path = saved_source / "vision-reference.json"
    path.write_text(json.dumps(original), encoding="utf-8")
    audit = json.loads((saved_source / "audio-encoder-validation.json").read_text())
    audit.update(stage="vision_encoder_projector_only", success=True)
    audit["files"][path.name] = dict(bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    audit["files"]["unrelated.png"] = dict(bytes=10, sha256="c" * 64)
    (saved_source / "vision-validation.json").write_text(json.dumps(audit))
    report = replay(saved_source, "image", source_commit="a" * 40, revision="b" * 40, case_id="square-0")
    assert report["modality"] == "image" and report["case"] == "square-0"
    assert all(row["mismatched_elements"] == 0 for row in report["thread_contexts"]["conditions"])
    assert report["full_encoder_validated"] is False


@pytest.mark.parametrize("failure", ["input_hash", "producer", "graph_escape"])
def test_standalone_replay_rejects_invalid_provenance_before_program_load(saved_source, monkeypatch, failure):
    if failure == "input_hash":
        np.array([2., 3.], dtype="<f4").tofile(saved_source / "input.bin")
    elif failure == "producer":
        path = saved_source / "audio-encoder-validation.json"
        audit = json.loads(path.read_text()); audit["metadata"]["source_commit"] = "c" * 40
        path.write_text(json.dumps(audit))
    else:
        path = saved_source / "audio-encoder-reference.json"
        ref = json.loads(path.read_text()); ref["profiles"]["fixed"]["native_graph"] = "../escape.pt2"
        path.write_text(json.dumps(ref))
        audit_path = saved_source / "audio-encoder-validation.json"
        audit = json.loads(audit_path.read_text())
        audit["files"][path.name] = dict(bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        audit_path.write_text(json.dumps(audit))
    monkeypatch.setattr(torch.export, "load", lambda *_: pytest.fail("Unverified graph was loaded"))
    with pytest.raises(ValueError):
        replay(saved_source, "audio", source_commit="a" * 40, revision="b" * 40, case_id="silence")
