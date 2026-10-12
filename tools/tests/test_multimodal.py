import pytest
import torch
from torch.nn import functional as F

from embeddinggemma_tools import export
from test_model import tiny_model


IMAGE, AUDIO = 70, 71  # Deliberately outside the tiny vocabulary.


def wrapper(model=None):
    factory = getattr(export, "MultimodalEmbeddingExport", None)
    assert factory is not None, "multimodal feature input export has not been implemented"
    return factory(model or tiny_model(), 6, image_token_id=IMAGE, audio_token_id=AUDIO,
                   image_feature_count=2, audio_feature_count=1)


def inputs():
    return (torch.tensor([[2, IMAGE, AUDIO, 3, IMAGE, 0]]),
            torch.tensor([[1, 1, 1, 1, 1, 0]]),
            torch.arange(32, dtype=torch.float32).reshape(2, 16) / 32,
            torch.linspace(-0.4, 0.4, 16).reshape(1, 16))


def independent(model, ids, mask, image, audio):
    safe_ids = ids.clone()
    safe_ids[(ids == IMAGE) | (ids == AUDIO)] = model.config.pad_token_id
    hidden = model.get_input_embeddings()(safe_ids)
    # Follow the official multimodal model, independently of the export path.
    hidden = hidden.masked_scatter((ids == IMAGE).unsqueeze(-1).expand_as(hidden), image)
    hidden = hidden.masked_scatter((ids == AUDIO).unsqueeze(-1).expand_as(hidden), audio)
    hidden = model(inputs_embeds=hidden, attention_mask=mask).last_hidden_state
    return F.normalize(hidden[:, mask[0].bool()].mean(dim=1), dim=-1)


def test_noncontiguous_oov_features_match_official_injection_and_pooling():
    model = tiny_model()
    actual = wrapper(model)(*inputs())
    torch.testing.assert_close(actual, independent(model, *inputs()), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("problem", ["missing_image", "extra_audio", "padded_slot", "wrong_width",
                                     "nonfinite", "invalid_id", "nonbinary_mask", "wrong_dtype"])
def test_invalid_layout_or_features_are_rejected_before_inference(problem):
    ids, mask, image, audio = inputs()
    if problem == "missing_image":
        image = image[:1]
    elif problem == "extra_audio":
        audio = audio.repeat(2, 1)
    elif problem == "padded_slot":
        mask[0, 1] = 0
    elif problem == "wrong_width":
        image = image[:, :15]
    elif problem == "nonfinite":
        audio[0, 0] = float("nan")
    elif problem == "invalid_id":
        ids[0, 3] = 100
    elif problem == "nonbinary_mask":
        mask[0, 3] = 2
    elif problem == "wrong_dtype":
        image = image.double()
    with pytest.raises(ValueError):
        wrapper()(ids, mask, image, audio)


@pytest.mark.parametrize("modality", ["image", "audio", "text"])
def test_single_modality_and_no_features_preserve_the_text_contract(modality):
    model = tiny_model()
    factory = getattr(export, "MultimodalEmbeddingExport", None)
    assert factory is not None, "multimodal export missing"
    ids = torch.tensor([[2, 3, 4, 1, 0, 0]])
    mask = torch.tensor([[1, 1, 1, 1, 0, 0]])
    image, audio = torch.empty(0, 16), torch.empty(0, 16)
    if modality == "image":
        ids[0, 1] = IMAGE
        image = torch.ones(1, 16) * 0.25
    elif modality == "audio":
        ids[0, 2] = AUDIO
        audio = torch.ones(1, 16) * -0.25
    actual = factory(model, 6, image_token_id=IMAGE, audio_token_id=AUDIO,
                     image_feature_count=len(image), audio_feature_count=len(audio))(ids, mask, image, audio)
    torch.testing.assert_close(actual, independent(model, ids, mask, image, audio), rtol=1e-5, atol=1e-6)


def test_saved_export_accepts_changed_positions_features_and_padding_without_masked_scatter(tmp_path):
    model = tiny_model()
    instance = wrapper(model)
    save = getattr(export, "export_multimodal_pt2", None)
    assert save is not None, "multimodal saved export missing"
    destination = tmp_path / "multimodal.pt2"
    save(instance, *inputs(), destination)
    ids = torch.tensor([[IMAGE, 5, IMAGE, AUDIO, 0, 0]])
    mask = torch.tensor([[1, 1, 1, 1, 0, 0]])
    image, audio = inputs()[2:]
    image, audio = image.flip(0) * 0.5, audio * 0.75
    program = torch.export.load(destination)
    actual = program.module()(ids, mask, image, audio)
    torch.testing.assert_close(actual, independent(model, ids, mask, image, audio), rtol=1e-5, atol=1e-6)
    targets = {str(node.target) for node in program.graph.nodes if node.op == "call_function"}
    assert not any("masked_scatter" in target or "_assert_tensor_metadata" in target for target in targets)


def test_longer_profile_crosses_local_attention_window_with_both_modalities():
    model = tiny_model()
    factory = getattr(export, "MultimodalEmbeddingExport", None)
    assert factory is not None, "long multimodal profile missing"
    ids = torch.tensor([[2, IMAGE, 3, 4, 5, 6, AUDIO, 1, 0, 0]])
    mask = (ids != 0).long()
    image, audio = torch.ones(1, 16) / 4, torch.ones(1, 16) / 8
    instance = factory(model, 10, image_token_id=IMAGE, audio_token_id=AUDIO,
                       image_feature_count=1, audio_feature_count=1)
    actual = instance(ids, mask, image, audio)
    torch.testing.assert_close(actual, independent(model, ids, mask, image, audio), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("change", [dict(sequence_length=0), dict(image_feature_count=-1),
                                    dict(image_feature_count=True), dict(audio_token_id=IMAGE),
                                    dict(image_token_id=0), dict(audio_feature_count=5)])
def test_invalid_profile_is_rejected_when_constructed(change):
    arguments = dict(sequence_length=6, image_token_id=IMAGE, audio_token_id=AUDIO,
                     image_feature_count=2, audio_feature_count=1)
    arguments.update(change)
    with pytest.raises(ValueError):
        export.MultimodalEmbeddingExport(tiny_model(), **arguments)
