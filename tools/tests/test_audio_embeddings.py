import hashlib
import importlib
import json

import pytest
import torch
from safetensors.torch import save_file
from transformers import EmbeddingGemma2Model

from embeddinggemma_tools import model as loader, prepare
from embeddinggemma_tools.audio import prepare_audio_preprocessor
from test_audio_encoder import audio_models
from test_audio_prepare import configuration
from test_model import tiny_model
from test_prompt_reference import snapshot as tokenizer_snapshot


def required(owner, name):
    function = getattr(owner, name, None)
    assert callable(function), "Audio final embedding contract is missing: " + name
    return function


def preparation():
    try:
        module = importlib.import_module("embeddinggemma_tools.audio_embeddings")
    except ModuleNotFoundError:
        pytest.fail("Audio final reference producer is missing")
    return required(module, "prepare_audio_embeddings")


def snapshot(path, omit=None):
    tokenizer_snapshot(path)
    text = tiny_model(); audio, projection = audio_models(128)
    (path / "config.json").write_text(json.dumps(dict(text_config=text.config.to_dict(),
        audio_config=audio.config.to_dict(), image_token_id=10, audio_token_id=11)))
    configuration(path / "processor_config.json")
    state = {prefix + key: value.bfloat16().contiguous()
             for prefix, module in (("language_model.", text), ("audio_tower.", audio), ("embed_audio.", projection))
             for key, value in module.state_dict().items() if prefix + key != omit}
    state["vision_tower.must_not_load"] = torch.ones(1)
    save_file(state, path / "model.safetensors")
    return path


def source(tmp_path):
    snap = snapshot(tmp_path / "snapshot")
    preprocessing = tmp_path / "preprocessing"
    prepare_audio_preprocessor(snap / "processor_config.json", preprocessing, num_samples=1024, source_commit="b" * 40)
    encoder = tmp_path / "encoder"
    prepare.prepare_audio_encoder(snap, preprocessing, encoder, source_commit="b" * 40)
    return snap, encoder


def test_audio_official_loader_does_not_construct_the_unused_vision_tower(tmp_path):
    load = required(loader, "load_audio_embedding_model")
    core = load(snapshot(tmp_path / "snapshot"))
    assert isinstance(core, EmbeddingGemma2Model)
    assert core.vision_tower is None and core.embed_vision is None
    assert core.audio_tower.config._attn_implementation == "sdpa"
    assert not core.training and all(p.dtype == torch.float32 and p.device.type == "cpu" for p in core.parameters())
    output = core(input_ids=torch.tensor([[1,14,11,11,15,2,0,0]]),
                  attention_mask=torch.tensor([[1,1,1,1,1,1,0,0]]),
                  input_features=torch.zeros(1,6,128), input_features_mask=torch.ones(1,6,dtype=torch.bool))
    assert output.audio_hidden_states.shape == (2,16)
    assert output.last_hidden_state.shape == (1,8,24)


def test_missing_audio_projection_cannot_be_hidden_by_partial_loading(tmp_path):
    load = required(loader, "load_audio_embedding_model")
    with pytest.raises(RuntimeError, match="embedding_projection"):
        load(snapshot(tmp_path / "snapshot", omit="embed_audio.embedding_projection.weight"))


def test_native_audio_final_vectors_cover_all_six_signals_and_zero_image_profiles(tmp_path):
    run = preparation()
    snap, encoder = source(tmp_path)
    # Replay the complete-native / failed-ONNX collection status explicitly.
    audit_path = encoder / "audio-encoder-validation.json"
    audit = json.loads(audit_path.read_text())
    audit.update(success=False, onnx_case_count=6, onnx_failed_case_count=6, onnx_numerically_validated=False)
    audit_path.write_text(json.dumps(audit))
    report = run(snap, encoder, tmp_path / "out", sequence_length=16,
                 revision=prepare.MODEL_REVISION, source_commit="b" * 40)
    assert report["success"] and report["stage"] == "python_audio_embeddings_only"
    assert report["case_count"] == 6 and report["profile_count"] == 2
    assert report["python_native_saved_chain_validated"] is True
    assert report["source_encoder_validation_success"] is False
    assert report["onnx_encoder_failed_case_count"] == 6
    assert report["unity_validated"] is False and report["end_to_end_validated"] is False
    reference = json.loads((tmp_path / "out/audio-embedding-reference.json").read_text())
    assert all(spec["image_feature_count"] == 0 for spec in reference["profiles"].values())
    assert sorted(spec["audio_feature_count"] for spec in reference["profiles"].values()) == [1,2]
    for row in reference["cases"]:
        assert len(row["embedding"]) == 24 and abs(sum(value*value for value in row["embedding"])-1) < 1e-5
        assert row["input_ids"].count(11) == reference["profiles"][row["profile"]]["audio_feature_count"]
        assert row["native_saved_chain_cosine"] >= 0.999999
    for name, spec in report["files"].items():
        assert hashlib.sha256((tmp_path / "out" / name).read_bytes()).hexdigest() == spec["sha256"]


@pytest.mark.parametrize("problem", ["source", "hash", "native_incomplete"])
def test_audio_final_audit_precedes_model_loading_and_removes_stale_success(tmp_path, monkeypatch, problem):
    run = preparation()
    snap, encoder = source(tmp_path)
    if problem == "hash":
        (encoder / "audio-encoder-reference.json").write_text('{}')
    else:
        path = encoder / "audio-encoder-validation.json"
        report = json.loads(path.read_text())
        if problem == "source": report["metadata"]["source_commit"] = "c" * 40
        else: report["native_reference_complete"] = False
        path.write_text(json.dumps(report))
    output = tmp_path / "out"; output.mkdir()
    (output / "audio-embedding-validation.json").write_text('{"success":true}')
    monkeypatch.setattr(importlib.import_module(run.__module__), "load_audio_embedding_model",
                        lambda *args: pytest.fail("audit before loading weights"))
    with pytest.raises(ValueError, match="Audio reference"):
        run(snap, encoder, output, sequence_length=16, revision=prepare.MODEL_REVISION, source_commit="b" * 40)
    assert not (output / "audio-embedding-validation.json").exists()


def test_wrong_saved_audio_backbone_cannot_create_a_passing_final_reference(tmp_path, monkeypatch):
    run = preparation()
    snap, encoder = source(tmp_path)
    class Wrong(torch.nn.Module):
        def forward(self, ids, mask, image, audio):
            return torch.zeros(1,24)
    def corrupt(wrapper, ids, mask, image, audio, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.export.save(torch.export.export(Wrong(), (ids,mask,image,audio)), path)
    monkeypatch.setattr(importlib.import_module(run.__module__).export, "export_multimodal_pt2", corrupt)
    with pytest.raises(ValueError, match="Saved audio embedding"):
        run(snap, encoder, tmp_path / "out", sequence_length=16, revision=prepare.MODEL_REVISION, source_commit="b" * 40)
    assert not (tmp_path / "out/audio-embedding-validation.json").exists()
