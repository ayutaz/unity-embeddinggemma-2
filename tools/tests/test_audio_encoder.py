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


def test_saved_attention_preserves_native_learned_query_scale_without_softplus_decomposition(tmp_path):
    audio, projection = audio_models()
    attention = audio.layers[0].self_attn
    with torch.no_grad():
        attention.per_dim_scale.copy_(torch.tensor([-8, -2, -0.3, 0, 0.1, 1, 5, 8.49]))
    native_scale = torch.nn.functional.softplus(attention.per_dim_scale).detach().clone()
    wrapper = export._AudioAttentionExport(attention, 7).eval()
    hidden = torch.randn(1, 7, audio.config.hidden_size)
    positions = audio.rel_pos_enc(hidden)
    blocked = factory()(audio, projection, mask()).blocked_mask
    destination = tmp_path / "attention.pt2"
    export._save_program(wrapper, (hidden, positions, blocked), destination)
    saved = torch.export.load(destination)
    # Per-dimension scale is a fixed learned inference coefficient. Replacing
    # native Softplus with log1p(exp(x)) changes its FP32 rounding; compute it
    # once with the official operation and retain that tensor in the graph.
    targets = {str(node.target) for node in saved.graph.nodes if node.op == "call_function"}
    assert "aten.exp.default" not in targets and "aten.log1p.default" not in targets, (
        "Fixed learned query scale must not be recomputed with decomposed Softplus"
    )
    coefficients = [*saved.state_dict.values(), *saved.constants.values()]
    assert any(value.shape == native_scale.shape and torch.equal(value, native_scale)
               for value in coefficients if isinstance(value, torch.Tensor)), (
        "Saved attention must retain the exact native learned scale"
    )
    changed = hidden * 0.6
    expected = attention(changed, positions, blocked)[0]
    torch.testing.assert_close(saved.module()(changed, positions, blocked), expected, rtol=1e-5, atol=1e-6)


def test_saved_audio_silu_preserves_native_division_rounding_at_counterexamples(tmp_path):
    class ActivationProbe(torch.nn.Module):
        def validate_inputs(self, features):
            assert features.dtype == torch.float32 and torch.isfinite(features).all()

        def forward(self, features):
            return torch.nn.functional.silu(features)

    # Fixed counterexamples to Core ATen's x * sigmoid(x), plus signed values.
    # Native CPU SiLU computes x / (1 + exp(-x)); do not add a rounded reciprocal
    # multiplication. These are FP32 arithmetic tests, not real encoder proof.
    inputs = torch.tensor([8.01, 5.0, 1.0, 0.1, -0.3, -2.0, -8.0])
    expected = torch.nn.functional.silu(inputs)
    path = tmp_path / "audio-silu.pt2"
    export.export_audio_pt2(ActivationProbe(), inputs, path)
    saved = torch.export.load(path)
    actual = saved.module()(inputs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    targets = {str(node.target) for node in saved.graph.nodes if node.op == "call_function"}
    assert "aten.silu.default" not in targets and "aten.sigmoid.default" not in targets
    assert "aten.exp.default" in targets and "aten.div.Tensor" in targets
    default_path = tmp_path / "default-silu.pt2"
    export._save_program(ActivationProbe(), (inputs,), default_path)
    default = torch.export.load(default_path)
    assert "aten.sigmoid.default" in {str(node.target) for node in default.graph.nodes if node.op == "call_function"}


@pytest.mark.parametrize("shape", [(17,), (1, 2, 4096)])
def test_saved_audio_silu_preserves_native_vectorized_rounding(tmp_path, shape):
    class ActivationProbe(torch.nn.Module):
        def validate_inputs(self, features):
            assert features.dtype == torch.float32 and torch.isfinite(features).all()

        def forward(self, features):
            return torch.nn.functional.silu(features)

    torch.manual_seed(412)
    inputs = torch.randn(shape) * 2
    expected = torch.nn.functional.silu(inputs)
    path = tmp_path / "vector-silu.pt2"
    export.export_audio_pt2(ActivationProbe(), inputs, path)
    with torch.no_grad():
        actual = torch.export.load(path).module()(inputs)
    indices = torch.nonzero(actual.flatten() != expected.flatten()).flatten()
    examples = [(inputs.flatten()[i].item(), expected.flatten()[i].item(), actual.flatten()[i].item())
                for i in indices[:4]]
    assert indices.numel() == 0, f"Native vectorized SiLU mismatch: {indices.numel()} elements; {examples}"


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
