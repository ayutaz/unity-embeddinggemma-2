import hashlib
import json
import shutil

import pytest

from embeddinggemma_tools import audio_embeddings, prepare
from test_audio_embeddings import source


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    root = tmp_path_factory.mktemp("frozen_audio_roles")
    snap, encoder = source(root)
    anchor = root / "anchor"
    audio_embeddings.prepare_audio_embeddings(snap, encoder, anchor, sequence_length=16,
        revision=prepare.MODEL_REVISION, source_commit="b" * 40)
    return snap, encoder, anchor


@pytest.mark.parametrize("role", ["query", "raw"])
def test_frozen_roles_require_all_six_document_anchors_without_executing_encoder(bundle, tmp_path, monkeypatch, role):
    snap, encoder, anchor = bundle
    original_loader = audio_embeddings.load_audio_embedding_model
    def load(path):
        core = original_loader(path)
        def forbidden(*args, **kwargs):
            pytest.fail("Explicit frozen role generation must not execute the CPU encoder")
        core.get_audio_features = forbidden
        return core
    monkeypatch.setattr(audio_embeddings, "load_audio_embedding_model", load)
    report = audio_embeddings.prepare_audio_embeddings(snap, encoder, tmp_path / "out", sequence_length=16,
        revision=prepare.MODEL_REVISION, source_commit="c" * 40, encoder_source_commit="b" * 40,
        backbone_source=anchor, role=role, frozen_encoder_features=True)
    assert report["success"] and report["stage"] == "python_audio_frozen_feature_role_reference"
    assert report["original_document_anchor_cases"] == 6 and report["case_count"] == 6
    assert report["frozen_encoder_features_validated"] is True
    assert report["python_native_saved_chain_validated"] is False
    assert report["end_to_end_validated"] is False and report["gpu_chain_validated"] is False
    reference = json.loads((tmp_path / "out/audio-embedding-reference.json").read_text())
    assert reference["metadata"]["reference_boundary"] == "original_frozen_encoder_outputs_to_official_backbone"
    assert all(row["role"] == role and row["document_anchor_cosine"] >= 0.999999 for row in reference["cases"])
    assert all(row["native_saved_chain_cosine"] is None for row in reference["cases"])


@pytest.mark.parametrize("problem", ["embedding", "missing_case", "role"])
def test_corrupt_document_anchor_cannot_authorize_new_role(bundle, tmp_path, problem):
    snap, encoder, original = bundle
    anchor = tmp_path / "anchor"
    shutil.copytree(original, anchor)
    path = anchor / "audio-embedding-reference.json"
    reference = json.loads(path.read_text())
    receipt_path = anchor / "audio-embedding-validation.json"
    receipt = json.loads(receipt_path.read_text())
    if problem == "embedding": reference["cases"][0]["embedding"] = [-x for x in reference["cases"][0]["embedding"]]
    elif problem == "missing_case": reference["cases"].pop()
    else:
        reference["metadata"]["role"] = "query"
        receipt["metadata"]["role"] = "query"
    path.write_text(json.dumps(reference))
    receipt["files"][path.name] = dict(bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="[Dd]ocument anchor"):
        audio_embeddings.prepare_audio_embeddings(snap, encoder, tmp_path / "out", sequence_length=16,
            revision=prepare.MODEL_REVISION, source_commit="c" * 40, encoder_source_commit="b" * 40,
            backbone_source=anchor, role="query", frozen_encoder_features=True)
    assert not (tmp_path / "out/audio-embedding-validation.json").exists()
