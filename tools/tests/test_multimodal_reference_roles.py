import json

import numpy as np
import pytest
import torch
from transformers import AutoTokenizer

from embeddinggemma_tools import __main__ as cli, audio_embeddings, image_embeddings, model, prepare
from embeddinggemma_tools import export
from test_audio_embeddings import source as audio_source
from test_image_embeddings import prepare_source as image_source
from test_vision import positions


@pytest.fixture(scope="module")
def audio_fixture(tmp_path_factory):
    return audio_source(tmp_path_factory.mktemp("audio-role-source"))


@pytest.fixture(scope="module")
def image_fixture(tmp_path_factory):
    with pytest.MonkeyPatch.context() as patch:
        yield image_source(tmp_path_factory.mktemp("image-role-source"), patch)


def assert_role(reference, role, snapshot):
    assert reference["metadata"]["role"] == role
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    for row in reference["cases"]:
        assert row["role"] == role
        if role == "query":
            assert row["formatted_text"].startswith("task: search result | query: ")
        elif role == "document":
            assert row["formatted_text"].startswith("title: none | text: ")
        else:
            assert not row["formatted_text"].startswith(("title: none | text: ", "task: search result | query: "))
        independent = tokenizer(row["formatted_text"], padding="max_length", max_length=len(row["input_ids"]), truncation=False)
        assert row["input_ids"] == independent["input_ids"]
        assert row["attention_mask"] == independent["attention_mask"]


@pytest.mark.parametrize("role", ["query", "raw", "document"])
def test_audio_reference_role_and_final_vectors_match_independent_official_forward(tmp_path, audio_fixture, role):
    snapshot, encoder = audio_fixture
    output = tmp_path / role
    report = audio_embeddings.prepare_audio_embeddings(snapshot, encoder, output, sequence_length=32,
        revision=prepare.MODEL_REVISION, source_commit="b" * 40, role=role)
    reference = json.loads((output / "audio-embedding-reference.json").read_text())
    source = json.loads((encoder / "audio-encoder-reference.json").read_text())
    assert_role(reference, role, snapshot)
    core = model.load_audio_embedding_model(snapshot)
    for row in reference["cases"]:
        spec = source["profiles"][row["profile"]]
        values = torch.from_numpy(np.fromfile(encoder / row["input_features"], "<f4").copy().reshape(spec["feature_shape"]))
        frame_mask = torch.from_numpy(np.fromfile(encoder / spec["frame_mask"], "<i4").copy().reshape(spec["feature_shape"][:2])).bool()
        ids = torch.tensor([row["input_ids"]]); mask = torch.tensor([row["attention_mask"]])
        with torch.no_grad():
            native = core(input_ids=ids, attention_mask=mask, input_features=values, input_features_mask=frame_mask)
            expected = torch.nn.functional.normalize(native.last_hidden_state[:, mask[0].bool()].mean(1), dim=-1)[0]
        torch.testing.assert_close(torch.tensor(row["embedding"]), expected, rtol=1e-5, atol=1e-6)
    assert report["python_native_saved_chain_validated"]
    assert not report["unity_validated"]


@pytest.mark.parametrize("role", ["query", "raw", "document"])
def test_image_reference_role_and_final_vectors_match_independent_official_forward(tmp_path, image_fixture, role):
    snapshot, vision = image_fixture
    output = tmp_path / role
    report = image_embeddings.prepare_image_embeddings(snapshot, vision, output, sequence_length=32,
        revision="a" * 40, source_commit="b" * 40, role=role)
    reference = json.loads((output / "image-embedding-reference.json").read_text())
    assert_role(reference, role, snapshot)
    core = model.load_image_embedding_model(snapshot)
    for row in reference["cases"]:
        pixels = torch.from_numpy(np.fromfile(vision / row["pixels"], "<f4").copy().reshape(1, 18, 12))
        ids = torch.tensor([row["input_ids"]]); mask = torch.tensor([row["attention_mask"]])
        with torch.no_grad():
            native = core(input_ids=ids, attention_mask=mask, pixel_values=pixels, image_position_ids=positions(width=3))
            expected = torch.nn.functional.normalize(native.last_hidden_state[:, mask[0].bool()].mean(1), dim=-1)[0]
        torch.testing.assert_close(torch.tensor(row["embedding"]), expected, rtol=1e-5, atol=1e-6)
    assert not report["unity_validated"]


@pytest.mark.parametrize("modality", ["audio", "image"])
def test_invalid_role_fails_before_model_loading(tmp_path, monkeypatch, modality):
    module = audio_embeddings if modality == "audio" else image_embeddings
    monkeypatch.setattr(module, "load_" + modality + "_embedding_model", lambda *_: pytest.fail("Model loaded before role validation"))
    run = getattr(module, "prepare_" + modality + "_embeddings")
    with pytest.raises(ValueError, match="role"):
        run(tmp_path, tmp_path, tmp_path / "out", revision="a" * 40, source_commit="b" * 40, role="unknown")


@pytest.mark.parametrize("modality,source_flag", [("audio", "--encoder-source"), ("image", "--vision-source")])
def test_cli_forwards_explicit_role(tmp_path, monkeypatch, modality, source_flag):
    seen = {}
    def run(*args, **kwargs):
        seen.update(kwargs)
        return dict(success=True, stage="python_" + modality + "_embeddings_only", case_count=6, profile_count=2,
                    unity_validated=False, gpu_chain_validated=False, onnx_encoder_numerically_validated=False)
    monkeypatch.setattr(cli, "prepare_" + modality + "_embeddings", run)
    cli.main(["prepare-" + modality + "-embeddings", "--snapshot", str(tmp_path), source_flag, str(tmp_path),
              "--output", str(tmp_path / "out"), "--role", "query", source_flag + "-commit", "b" * 40,
              "--backbone-source", str(tmp_path / "backbones")])
    assert seen["role"] == "query"
    assert seen["encoder_source_commit" if modality == "audio" else "vision_source_commit"] == "b" * 40
    assert seen["backbone_source"] == tmp_path / "backbones"


def test_audio_reference_producer_and_reused_encoder_producer_are_separate(tmp_path, audio_fixture):
    snapshot, encoder = audio_fixture
    report = audio_embeddings.prepare_audio_embeddings(snapshot, encoder, tmp_path / "out", sequence_length=32,
        revision=prepare.MODEL_REVISION, source_commit="c" * 40, encoder_source_commit="b" * 40, role="raw")
    assert report["metadata"]["source_commit"] == "c" * 40
    assert report["metadata"]["encoder_source_commit"] == "b" * 40


def test_image_reference_producer_and_reused_vision_producer_are_separate(tmp_path, image_fixture):
    snapshot, vision = image_fixture
    report = image_embeddings.prepare_image_embeddings(snapshot, vision, tmp_path / "out", sequence_length=32,
        revision="a" * 40, source_commit="c" * 40, vision_source_commit="b" * 40, role="raw")
    assert report["metadata"]["source_commit"] == "c" * 40
    assert report["metadata"]["vision_source_commit"] == "b" * 40


def test_audio_raw_role_reuses_audited_document_backbones_without_export(tmp_path, audio_fixture, monkeypatch):
    snapshot, encoder = audio_fixture
    source = tmp_path / "document"
    original = audio_embeddings.prepare_audio_embeddings(snapshot, encoder, source, sequence_length=32,
        revision=prepare.MODEL_REVISION, source_commit="b" * 40)
    monkeypatch.setattr(export, "export_multimodal_pt2", lambda *_: pytest.fail("Unchanged backbone was exported again"))
    report = audio_embeddings.prepare_audio_embeddings(snapshot, encoder, tmp_path / "raw", sequence_length=32,
        revision=prepare.MODEL_REVISION, source_commit="c" * 40, encoder_source_commit="b" * 40,
        role="raw", backbone_source=source)
    assert report["metadata"]["backbone_source_commit"] == "b" * 40
    assert report["python_native_saved_chain_validated"]
    for name, value in original["files"].items():
        if name.endswith("backbone.pt2"):
            assert report["files"][name] == value


def test_image_query_role_reuses_audited_document_backbone_without_export(tmp_path, image_fixture, monkeypatch):
    snapshot, vision = image_fixture
    source = tmp_path / "document"
    original = image_embeddings.prepare_image_embeddings(snapshot, vision, source, sequence_length=32,
        revision="a" * 40, source_commit="b" * 40)
    monkeypatch.setattr(export, "export_multimodal_pt2", lambda *_: pytest.fail("Unchanged backbone was exported again"))
    report = image_embeddings.prepare_image_embeddings(snapshot, vision, tmp_path / "query", sequence_length=32,
        revision="a" * 40, source_commit="c" * 40, vision_source_commit="b" * 40,
        role="query", backbone_source=source)
    assert report["metadata"]["backbone_source_commit"] == "b" * 40
    assert report["success"]
    for name, value in original["files"].items():
        if name.endswith("backbone.pt2"):
            assert report["files"][name] == value


@pytest.mark.parametrize("modality", ["audio", "image"])
def test_official_feature_drift_records_failed_context_without_replacing_reference(
        tmp_path, audio_fixture, image_fixture, monkeypatch, modality):
    module = audio_embeddings if modality == "audio" else image_embeddings
    snapshot, encoder = audio_fixture if modality == "audio" else image_fixture
    load = getattr(module, "load_" + modality + "_embedding_model")
    core = load(snapshot)
    forward = core.forward
    def changed(*args, **kwargs):
        result = forward(*args, **kwargs)
        name = modality + "_hidden_states"
        setattr(result, name, getattr(result, name) + .25)
        return result
    monkeypatch.setattr(core, "forward", changed)
    monkeypatch.setattr(module, "load_" + modality + "_embedding_model", lambda *_: core)
    output = tmp_path / modality
    run = getattr(module, "prepare_" + modality + "_embeddings")
    with pytest.raises(ValueError, match="encoder features"):
        run(snapshot, encoder, output, sequence_length=32,
            revision=prepare.MODEL_REVISION if modality == "audio" else "a" * 40,
            source_commit="b" * 40, role="raw")
    failure_path = output / (modality + "-role-context-failure.json")
    assert failure_path.is_file(), "Numeric context failure diagnostics were discarded"
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    assert failure["success"] is False and failure["role"] == "raw"
    assert failure["mismatched_elements"] > 0
    assert .24 < failure["maximum_absolute_error"] < .26
    assert failure["expected_unchanged"] and failure["runtime"]["torch_threads"] > 0
    saved = failure.get("saved_encoder_vs_expected")
    assert saved is not None, "Saved encoder context was not compared after official feature drift"
    assert saved["mismatched_elements"] == 0 and saved["same_shape"] and saved["actual_finite"]
    contexts = failure["saved_encoder_thread_contexts"]
    assert [row["actual_threads"] for row in contexts["conditions"]] == [1, 2, 4]
    assert torch.get_num_threads() == contexts["original_threads"]
    assert not (output / (modality + "-embedding-validation.json")).exists()
