"""Load the text backbone without constructing image or audio encoders."""

import json
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import EmbeddingGemma2TextConfig, EmbeddingGemma2TextModel, Gemma4VisionConfig, Gemma4VisionModel
from transformers.models.embedding_gemma2.modeling_embedding_gemma2 import EmbeddingGemma2MultimodalEmbedder


def load_text_model(snapshot: Path) -> EmbeddingGemma2TextModel:
    configuration = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    config = EmbeddingGemma2TextConfig.from_dict(configuration["text_config"])
    config._attn_implementation = "eager"
    model = EmbeddingGemma2TextModel(config).float()
    prefix = "language_model."
    with safe_open(snapshot / "model.safetensors", framework="pt", device="cpu") as weights:
        state = {key.removeprefix(prefix): weights.get_tensor(key).to(torch.float32)
                 for key in weights.keys() if key.startswith(prefix)}
    model.load_state_dict(state, strict=True)
    return model.eval()


def load_vision_models(snapshot: Path):
    """Load only the official vision tower and its projection to text hidden width."""
    configuration = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    config = Gemma4VisionConfig.from_dict(configuration["vision_config"])
    config._attn_implementation = "eager"
    vision = Gemma4VisionModel(config).float()
    projection = EmbeddingGemma2MultimodalEmbedder(
        config, EmbeddingGemma2TextConfig.from_dict(configuration["text_config"])
    ).float()
    with safe_open(snapshot / "model.safetensors", framework="pt", device="cpu") as weights:
        for prefix, module in (("vision_tower.", vision), ("embed_vision.", projection)):
            state = {key.removeprefix(prefix): weights.get_tensor(key).to(torch.float32)
                     for key in weights.keys() if key.startswith(prefix)}
            module.load_state_dict(state, strict=True)
    return vision.eval(), projection.eval()
