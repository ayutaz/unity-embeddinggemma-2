import json

import pytest
import torch
from safetensors.torch import save_file
from transformers import Gemma4AudioConfig, Gemma4AudioModel
from transformers.models.embedding_gemma2.modeling_embedding_gemma2 import EmbeddingGemma2MultimodalEmbedder

from embeddinggemma_tools import export, model as loader
from test_model import tiny_model


def audio_models(mel_width=8):
    torch.manual_seed(29)
    config = Gemma4AudioConfig(hidden_size=16, num_hidden_layers=2, num_attention_heads=2,
                              subsampling_conv_channels=[mel_width, 4], output_proj_dims=12,
                              attention_chunk_size=4, attention_context_left=5, attention_context_right=2,
                              conv_kernel_size=3)
    audio = Gemma4AudioModel(config).float().eval()
    assert audio.config._attn_implementation == "sdpa", "Keep the pinned official audio default mask semantics"
    return audio, EmbeddingGemma2MultimodalEmbedder(config, tiny_model().config).eval()


def mask(frames=25, valid=25):
    return (torch.arange(frames)[None] < valid).to(torch.int64)


def factory():
    function = getattr(export, "AudioEncoderExport", None)
    assert function is not None, "fixed-profile audio encoder export is missing"
    return function


@pytest.mark.parametrize("frames,valid", [(25, 25), (25, 17), (17, 9), (5, 1)])
def test_fixed_audio_projection_matches_official_chunked_encoder_and_padding(frames, valid):
    audio, projection = audio_models()
    frame_mask = mask(frames, valid)
    features = torch.randn(1, frames, 8) * frame_mask[:, :, None]
    output = audio(input_features=features, attention_mask=frame_mask.bool())
    expected = projection(output.last_hidden_state)[output.attention_mask]
    actual = factory()(audio, projection, frame_mask)(features)
    assert actual.shape == ((valid + 3) // 4, 16)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_saved_audio_export_matches_changed_features_without_unfold_or_nonzero(tmp_path):
    audio, projection = audio_models()
    frame_mask = mask(25, 17)
    wrapper = factory()(audio, projection, frame_mask)
    save = getattr(export, "export_audio_pt2", None)
    assert save is not None, "saved audio program is missing"
    features = torch.randn(1, 25, 8) * frame_mask[:, :, None]
    save(wrapper, features, tmp_path / "audio.pt2")
    changed = features * 0.6
    output = audio(input_features=changed, attention_mask=frame_mask.bool())
    expected = projection(output.last_hidden_state)[output.attention_mask]
    program = torch.export.load(tmp_path / "audio.pt2")
    torch.testing.assert_close(program.module()(changed), expected, rtol=1e-5, atol=1e-6)
    targets = {str(node.target) for node in program.graph.nodes if node.op == "call_function"}
    assert not any(term in target for term in ("unfold", "nonzero", "masked_scatter", "_assert_tensor_metadata") for target in targets)


def test_loads_only_audio_and_projector_as_float32(tmp_path):
    audio, projection = audio_models()
    (tmp_path / "config.json").write_text(json.dumps({"audio_config": audio.config.to_dict(),
                                                       "text_config": tiny_model().config.to_dict()}))
    weights = {"audio_tower." + key: value.bfloat16().contiguous() for key, value in audio.state_dict().items()}
    weights.update({"embed_audio." + key: value.bfloat16().contiguous() for key, value in projection.state_dict().items()})
    weights.update({"language_model.unused": torch.ones(1), "vision_tower.unused": torch.ones(1)})
    save_file(weights, tmp_path / "model.safetensors")
    function = getattr(loader, "load_audio_models", None)
    assert function is not None, "audio-only strict weight loading is missing"
    actual_audio, actual_projection = function(tmp_path)
    assert not actual_audio.training and not actual_projection.training
    assert actual_audio.config._attn_implementation == "sdpa"
    for prefix, actual in (("audio_tower.", actual_audio), ("embed_audio.", actual_projection)):
        for name, tensor in actual.state_dict().items():
            assert tensor.dtype == torch.float32
            torch.testing.assert_close(tensor, weights[prefix + name].float(), rtol=0, atol=0)


def test_missing_audio_projection_weight_is_rejected(tmp_path):
    audio, _ = audio_models()
    (tmp_path / "config.json").write_text(json.dumps({"audio_config": audio.config.to_dict(),
                                                       "text_config": tiny_model().config.to_dict()}))
    save_file({"audio_tower." + key: value.contiguous() for key, value in audio.state_dict().items()},
              tmp_path / "model.safetensors")
    function = getattr(loader, "load_audio_models", None)
    assert function is not None, "strict audio projector loading is missing"
    with pytest.raises(RuntimeError, match="embedding_projection"):
        function(tmp_path)


@pytest.mark.parametrize("invalid", [
    torch.ones(2, 25, dtype=torch.int64), torch.ones(1, 25), mask(25, 0),
    mask(25, 17).flip(1), torch.full((1, 25), 2, dtype=torch.int64), torch.ones(1, 0, dtype=torch.int64),
])
def test_rejects_non_prefix_empty_or_non_integer_frame_profiles(invalid):
    audio, projection = audio_models()
    with pytest.raises(ValueError, match="profile"):
        factory()(audio, projection, invalid)


@pytest.mark.parametrize("invalid", [
    torch.zeros(2, 25, 8), torch.zeros(1, 24, 8), torch.zeros(1, 25, 7),
    torch.zeros(1, 25, 8, dtype=torch.float64), torch.full((1, 25, 8), float("nan")),
    torch.full((1, 25, 8), float("inf")), torch.ones(1, 25, 8),
])
def test_rejects_invalid_features_or_nonzero_padding(invalid):
    audio, projection = audio_models()
    wrapper = factory()(audio, projection, mask(25, 17))
    with pytest.raises(ValueError, match="features"):
        wrapper(invalid)


def test_rejects_eager_audio_reference_with_incompatible_float_mask_semantics():
    audio, projection = audio_models()
    audio.config._attn_implementation = "eager"
    with pytest.raises(ValueError, match="sdpa"):
        factory()(audio, projection, mask())
