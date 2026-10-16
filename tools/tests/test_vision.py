import json

import pytest
import torch
from safetensors.torch import save_file
from transformers import Gemma4VisionConfig, Gemma4VisionModel
from transformers.models.embedding_gemma2.modeling_embedding_gemma2 import EmbeddingGemma2MultimodalEmbedder

from embeddinggemma_tools import export, model as loader
from test_model import tiny_model


def vision_models(**kwargs):
    torch.manual_seed(17)
    config = Gemma4VisionConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                              num_attention_heads=2, num_key_value_heads=2, head_dim=8,
                              patch_size=2, pooling_kernel_size=3, position_embedding_size=24, **kwargs)
    config._attn_implementation = "eager"
    return Gemma4VisionModel(config).eval(), EmbeddingGemma2MultimodalEmbedder(config, tiny_model().config).eval()


def positions(width=6, height=3, maximum=18):
    values = [[x, y] for y in range(height) for x in range(width)]
    return torch.tensor([values + [[-1, -1]] * (maximum - len(values))])


def require_export():
    factory = getattr(export, "VisionEncoderExport", None)
    assert factory is not None, "fixed-profile vision encoder export is missing"
    return factory


@pytest.mark.parametrize("width", [3, 6])
def test_fixed_spatial_pooling_and_projector_match_official_vision_with_padding(width):
    vision, projection = vision_models()
    pixels = torch.rand(1, 18, 12)
    coords = positions(width=width)
    expected = projection(vision(pixel_values=pixels, pixel_position_ids=coords).last_hidden_state)
    actual = require_export()(vision, projection, coords)(pixels)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    assert actual.shape == (width // 3, 16)


def test_saved_vision_export_handles_changed_pixels_without_dynamic_boolean_selection(tmp_path):
    vision, projection = vision_models()
    instance = require_export()(vision, projection, positions(width=3))
    save = getattr(export, "export_vision_pt2", None)
    assert save is not None, "saved vision program is missing"
    output = tmp_path / "vision.pt2"
    save(instance, torch.rand(1, 18, 12), output)
    changed = torch.rand(1, 18, 12) * 0.7
    expected = projection(vision(pixel_values=changed, pixel_position_ids=positions(width=3)).last_hidden_state)
    program = torch.export.load(output)
    torch.testing.assert_close(program.module()(changed), expected, rtol=1e-5, atol=1e-6)
    targets = {str(node.target) for node in program.graph.nodes if node.op == "call_function"}
    assert not any("nonzero" in target or "masked_scatter" in target for target in targets)


def test_loads_only_vision_and_its_projection_as_float32(tmp_path):
    vision, projection = vision_models()
    (tmp_path / "config.json").write_text(json.dumps({"vision_config": vision.config.to_dict(),
                                                       "text_config": tiny_model().config.to_dict()}))
    weights = {"vision_tower." + key: value.bfloat16().contiguous() for key, value in vision.state_dict().items()}
    weights.update({"embed_vision." + key: value.bfloat16().contiguous() for key, value in projection.state_dict().items()})
    weights.update({"language_model.unused": torch.ones(1), "audio_tower.unused": torch.ones(1)})
    save_file(weights, tmp_path / "model.safetensors")
    function = getattr(loader, "load_vision_models", None)
    assert function is not None, "vision-only strict weight loading is missing"
    actual_vision, actual_projection = function(tmp_path)
    assert not actual_vision.training and not actual_projection.training
    for prefix, actual in (("vision_tower.", actual_vision), ("embed_vision.", actual_projection)):
        for name, tensor in actual.state_dict().items():
            assert tensor.dtype == torch.float32
            torch.testing.assert_close(tensor, weights[prefix + name].float(), rtol=0, atol=0)


def test_missing_vision_projection_weight_is_rejected(tmp_path):
    vision, _ = vision_models()
    (tmp_path / "config.json").write_text(json.dumps({"vision_config": vision.config.to_dict(),
                                                       "text_config": tiny_model().config.to_dict()}))
    save_file({"vision_tower." + key: value.contiguous() for key, value in vision.state_dict().items()},
              tmp_path / "model.safetensors")
    function = getattr(loader, "load_vision_models", None)
    assert function is not None, "strict projector loading is missing"
    with pytest.raises(RuntimeError, match="embedding_projection"):
        function(tmp_path)


@pytest.mark.parametrize("invalid", [
    torch.zeros(2, 18, 2, dtype=torch.int64),
    torch.zeros(1, 18, 2),
    torch.full((1, 18, 2), -1),
    positions(maximum=19),
    positions(width=5, height=3),
    positions(width=6).flip(1),
    positions(width=6) + 24,
])
def test_rejects_invalid_or_non_rectangular_position_profiles(invalid):
    vision, projection = vision_models()
    with pytest.raises(ValueError, match="profile"):
        require_export()(vision, projection, invalid)


@pytest.mark.parametrize("invalid", [
    torch.rand(2, 18, 12), torch.rand(1, 17, 12), torch.rand(1, 18, 11),
    torch.rand(1, 18, 12).half(), torch.full((1, 18, 12), float("nan")),
    torch.full((1, 18, 12), float("inf")), torch.full((1, 18, 12), -0.01),
    torch.full((1, 18, 12), 1.01),
])
def test_rejects_pixels_outside_fixed_rgb_float32_contract(invalid):
    vision, projection = vision_models()
    instance = require_export()(vision, projection, positions())
    with pytest.raises(ValueError, match="pixel"):
        instance(invalid)


def test_standardized_vision_matches_official():
    vision, projection = vision_models(standardize=True)
    vision.std_bias.copy_(torch.randn(16))
    vision.std_scale.copy_(torch.rand(16) + 0.5)
    pixels = torch.rand(1, 18, 12)
    expected = projection(vision(pixel_values=pixels, pixel_position_ids=positions(width=3)).last_hidden_state)
    instance = require_export()(vision, projection, positions(width=3))
    torch.testing.assert_close(instance(pixels), expected, rtol=1e-5, atol=1e-6)
