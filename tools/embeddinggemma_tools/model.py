"""Load the text backbone without constructing image or audio encoders."""

import json
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import EmbeddingGemma2TextConfig, EmbeddingGemma2TextModel


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
