import hashlib
import json

import pytest

from embeddinggemma_tools.backbone_reuse import audit_backbone_source


def fixture(tmp_path, modality="audio"):
    graph = tmp_path / "small/backbone.pt2"
    graph.parent.mkdir()
    graph.write_bytes(b"independent saved program fixture")
    profile = dict(sequence_length=32, image_feature_count=0, audio_feature_count=2, feature_width=16,
                   graph="small/backbone.pt2")
    reference = dict(metadata=dict(source_commit="b" * 40, model_revision="a" * 40,
                                  input_sha256={"model.safetensors": "c" * 64}), profiles={"small": profile})
    path = tmp_path / (modality + "-embedding-reference.json")
    path.write_text(json.dumps(reference))
    receipt = dict(success=True, stage="python_" + modality + "_embeddings_only", metadata=reference["metadata"],
                   files={p.relative_to(tmp_path).as_posix(): dict(bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                          for p in (path, graph)})
    (tmp_path / (modality + "-embedding-validation.json")).write_text(json.dumps(receipt))
    expected = {"small": {k: v for k, v in profile.items() if k != "graph"}}
    return expected, reference, receipt


@pytest.mark.parametrize("modality", ["image", "audio"])
def test_backbone_reuse_returns_audited_program_and_original_producer(tmp_path, modality):
    expected, _, _ = fixture(tmp_path, modality)
    result = audit_backbone_source(tmp_path, modality, revision="a" * 40,
                                  input_sha256={"model.safetensors": "c" * 64}, profiles=expected)
    assert result["source_commit"] == "b" * 40
    assert result["programs"]["small"] == tmp_path / "small/backbone.pt2"


@pytest.mark.parametrize("problem", ["hash", "revision", "weights", "profile", "failed", "escape"])
def test_corrupt_or_incompatible_reuse_is_rejected(tmp_path, problem):
    expected, reference, receipt = fixture(tmp_path)
    revision = "a" * 40
    inputs = {"model.safetensors": "c" * 64}
    if problem == "hash":
        (tmp_path / "small/backbone.pt2").write_bytes(b"corrupt")
    elif problem == "revision":
        revision = "d" * 40
    elif problem == "weights":
        inputs["model.safetensors"] = "d" * 64
    elif problem == "profile":
        expected["small"]["audio_feature_count"] = 3
    elif problem == "failed":
        receipt["success"] = False
        (tmp_path / "audio-embedding-validation.json").write_text(json.dumps(receipt))
    else:
        reference["profiles"]["small"]["graph"] = "../outside.pt2"
        path = tmp_path / "audio-embedding-reference.json"
        path.write_text(json.dumps(reference))
        receipt["files"][path.name] = dict(bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        (tmp_path / "audio-embedding-validation.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="backbone"):
        audit_backbone_source(tmp_path, "audio", revision=revision, input_sha256=inputs, profiles=expected)
