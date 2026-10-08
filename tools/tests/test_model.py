import json

import pytest
import torch
from safetensors.torch import save_file
from transformers import EmbeddingGemma2TextConfig, EmbeddingGemma2TextModel

from embeddinggemma_tools.model import load_text_model


def tiny_model():
    torch.manual_seed(42)
    config = EmbeddingGemma2TextConfig(
        vocab_size=64, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        head_dim=8, global_head_dim=16, num_global_key_value_heads=1,
        hidden_size_per_layer_input=8, embedding_dim=24, sliding_window=2,
    )
    config._attn_implementation = "eager"
    return EmbeddingGemma2TextModel(config).eval()


def save_snapshot(path, model, omit=None):
    (path / "config.json").write_text(json.dumps({"text_config": model.config.to_dict()}))
    state = {"language_model." + key: value.to(torch.bfloat16).contiguous()
             for key, value in model.state_dict().items() if key != omit}
    state["vision_tower.unused"] = torch.ones(2)
    save_file(state, path / "model.safetensors")


def test_loads_only_text_weights_as_float32_without_losing_projection(tmp_path):
    original = tiny_model()
    save_snapshot(tmp_path, original)
    loaded = load_text_model(tmp_path)
    assert not loaded.training
    assert loaded.config._attn_implementation == "eager"
    assert all(p.dtype == torch.float32 for p in loaded.parameters())
    assert set(loaded.state_dict()) == set(original.state_dict())
    for key, value in original.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key], value.bfloat16().float(), rtol=0, atol=0)


def test_missing_projection_weight_is_rejected(tmp_path):
    save_snapshot(tmp_path, tiny_model(), omit="embedding_projection.weight")
    with pytest.raises(RuntimeError, match="embedding_projection"):
        load_text_model(tmp_path)
