"""Fixed-shape fp32 export of the official text backbone and sentence pooling."""

from pathlib import Path

import torch
from torch import nn
from torch.fx.node import map_aggregate

from .text import pool_and_normalize


class TextEmbeddingExport(nn.Module):
    def __init__(self, model: nn.Module, sequence_length: int):
        super().__init__()
        self.model = model
        positions = torch.arange(sequence_length)
        self.register_buffer("positions", positions.unsqueeze(0))
        self.register_buffer(
            "local_band",
            (positions.unsqueeze(0) - positions.unsqueeze(1)).abs() <= model.config.sliding_window,
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        masks = self.attention_masks(input_ids, attention_mask)
        output = self.model(
            input_ids=input_ids, attention_mask=masks, position_ids=self.positions, return_dict=True
        )
        return pool_and_normalize(output.last_hidden_state, attention_mask)

    def attention_masks(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict:
        valid_keys = attention_mask[:, None, None, :].bool()
        zero = torch.zeros((), dtype=torch.float32, device=input_ids.device)
        blocked = torch.full((), torch.finfo(torch.float32).min, device=input_ids.device)
        return {
            "full_attention": torch.where(valid_keys, zero, blocked),
            "sliding_attention": torch.where(valid_keys & self.local_band, zero, blocked),
        }


class MultimodalEmbeddingExport(TextEmbeddingExport):
    """Fixed batch-one backbone taking already projected image/audio soft tokens.

    Counts and width are fixed by the export profile. Eager input validation is
    intentionally outside the numerical graph; Unity callers must enforce the
    same ID, mask, slot-count and finite-feature contract before scheduling it.
    No encoder or multimodal projection is constructed by this wrapper.
    """

    def __init__(self, model, sequence_length, *, image_token_id, audio_token_id,
                 image_feature_count, audio_feature_count):
        if type(sequence_length) is not int or not 2 <= sequence_length <= 8192:
            raise ValueError("sequence length must be between 2 and 8192")
        if any(type(count) is not int or count < 0 for count in (image_feature_count, audio_feature_count)):
            raise ValueError("feature counts must be nonnegative integers")
        if image_feature_count + audio_feature_count > sequence_length:
            raise ValueError("features exceed the token budget")
        if (any(type(token) is not int or token < 0 or token == model.config.pad_token_id
                for token in (image_token_id, audio_token_id)) or image_token_id == audio_token_id):
            raise ValueError("modality IDs must be distinct nonnegative non-padding IDs")
        super().__init__(model, sequence_length)
        self.sequence_length = sequence_length
        self.image_token_id = image_token_id
        self.audio_token_id = audio_token_id
        self.image_feature_count = image_feature_count
        self.audio_feature_count = audio_feature_count

    def validate_inputs(self, input_ids, attention_mask, image_features, audio_features):
        if input_ids.dtype != torch.int64 or attention_mask.dtype != torch.int64:
            raise ValueError("IDs and mask must be int64")
        if input_ids.shape != (1, self.sequence_length) or attention_mask.shape != input_ids.shape:
            raise ValueError("input shape does not match the fixed batch-one profile")
        if not torch.all((attention_mask == 0) | (attention_mask == 1)) or not attention_mask.any():
            raise ValueError("attention mask must be binary and contain valid tokens")
        multimodal = (input_ids == self.image_token_id) | (input_ids == self.audio_token_id)
        text_ids = input_ids[~multimodal]
        if torch.any(text_ids < 0) or torch.any(text_ids >= self.model.config.vocab_size):
            raise ValueError("text token ID is outside the vocabulary")
        for token_id, count, features in (
            (self.image_token_id, self.image_feature_count, image_features),
            (self.audio_token_id, self.audio_feature_count, audio_features),
        ):
            slots = input_ids == token_id
            if features.shape != (count, self.model.config.hidden_size) or int(slots.sum()) != count:
                raise ValueError("feature count/width does not match the profile and placeholders")
            if torch.any(slots & (attention_mask == 0)):
                raise ValueError("multimodal placeholders must not be padding")
            if features.dtype != torch.float32 or not torch.isfinite(features).all():
                raise ValueError("projected features must be finite float32")
            if features.device != input_ids.device or attention_mask.device != input_ids.device:
                raise ValueError("inputs must be on the same device")

    def forward(self, input_ids, attention_mask, image_features, audio_features):
        if not torch.compiler.is_compiling():
            self.validate_inputs(input_ids, attention_mask, image_features, audio_features)
        image_slots = input_ids == self.image_token_id
        audio_slots = input_ids == self.audio_token_id
        safe_ids = torch.where(image_slots | audio_slots, self.model.config.pad_token_id, input_ids)
        embeddings = self.model.get_input_embeddings()(safe_ids)
        # Ordered gather + Where replaces dynamic masked_scatter. A sentinel
        # handles positions preceding the first slot and zero-feature profiles.
        for slots, features in ((image_slots, image_features), (audio_slots, audio_features)):
            padded = torch.cat((torch.zeros_like(embeddings[0, :1]), features), dim=0)
            indices = slots.to(torch.int64).cumsum(dim=1)
            gathered = padded[indices]
            embeddings = torch.where(slots.unsqueeze(-1), gathered, embeddings)
        output = self.model(inputs_embeds=embeddings, attention_mask=self.attention_masks(input_ids, attention_mask),
                            position_ids=self.positions, return_dict=True)
        return pool_and_normalize(output.last_hidden_state, attention_mask)


class VisionEncoderExport(nn.Module):
    """Official vision tower + projector, with a fixed rectangular patch profile.

    Static pooling and padding selection avoid dynamic boolean indexing. Pixels
    are RGB patches in processor order, float32 in [0, 1]. This produces features
    for the shared backbone, not final sentence embeddings. Unity must enforce
    the same input contract outside the exported numerical graph.
    """

    def __init__(self, vision, projection, pixel_position_ids):
        super().__init__()
        kernel = vision.config.pooling_kernel_size
        coords = pixel_position_ids
        if (coords.dtype != torch.int64 or coords.ndim != 3 or coords.shape[0] != 1
                or coords.shape[2] != 2 or coords.shape[1] == 0 or coords.shape[1] % (kernel * kernel)):
            raise ValueError("vision profile requires batch-one int64 positions and whole pooling cells")
        pairs = coords[0].tolist()
        valid = [pair for pair in pairs if pair != [-1, -1]]
        if not valid or any(min(pair) < 0 or max(pair) >= vision.config.position_embedding_size for pair in valid):
            raise ValueError("vision profile positions must be nonempty and inside the position table")
        width, height = max(pair[0] for pair in valid) + 1, max(pair[1] for pair in valid) + 1
        expected = [[x, y] for y in range(height) for x in range(width)]
        if (width % kernel or height % kernel or valid != expected
                or pairs != expected + [[-1, -1]] * (len(pairs) - len(expected))):
            raise ValueError("vision profile must be a complete row-major rectangle with trailing padding")
        self.vision = vision
        self.projection = projection
        self.pixel_shape = (1, len(pairs), 3 * vision.config.patch_size**2)
        self.feature_count = width * height // (kernel * kernel)
        self.register_buffer("positions", coords.clone())
        padding = (coords == -1).all(dim=-1)
        self.register_buffer("padding", padding)
        self.register_buffer("attention", torch.where(
            (~padding)[:, None, None, :], 0.0, torch.finfo(torch.float32).min))
        weights = torch.zeros(1, self.feature_count, len(pairs), dtype=torch.float32, device=coords.device)
        for index, (x, y) in enumerate(valid):
            weights[0, (x // kernel) + (width // kernel) * (y // kernel), index] = 1 / (kernel * kernel)
        self.register_buffer("pooling_weights", weights)
        self.eval()

    def validate_inputs(self, pixels):
        if pixels.dtype != torch.float32 or pixels.shape != self.pixel_shape:
            raise ValueError("pixel shape and dtype must match the fixed RGB float32 profile")
        if not torch.isfinite(pixels).all() or torch.any(pixels < 0) or torch.any(pixels > 1):
            raise ValueError("pixels must be finite and in [0, 1]")
        if pixels.device != self.positions.device:
            raise ValueError("pixels and profile must be on the same device")

    def forward(self, pixels):
        if not torch.compiler.is_compiling():
            self.validate_inputs(pixels)
        hidden = self.vision.patch_embedder(pixels, self.positions, self.padding)
        position_embeddings = self.vision.encoder.rotary_emb(hidden, self.positions)
        for layer in self.vision.encoder.layers[:self.vision.config.num_hidden_layers]:
            hidden = layer(hidden, attention_mask=self.attention, position_embeddings=position_embeddings,
                           position_ids=self.positions)
        # The official pooler zeroes padding, averages each spatial cell, then
        # scales in fp32. The fixed weights select only valid output cells.
        hidden = hidden.masked_fill(self.padding.unsqueeze(-1), 0.0)
        hidden = (self.pooling_weights @ hidden.float())[0] * self.vision.config.hidden_size**0.5
        if self.vision.config.standardize:
            hidden = (hidden - self.vision.std_bias.float()) * self.vision.std_scale.float()
        return self.projection(hidden)


def export_vision_pt2(wrapper, pixels, destination):
    wrapper.validate_inputs(pixels)
    _save_program(wrapper.eval(), (pixels,), destination)


def export_pt2(
    model: nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor, destination: Path
) -> None:
    wrapper = TextEmbeddingExport(model, input_ids.shape[1]).eval()
    _save_program(wrapper, (input_ids, attention_mask), destination)


def export_multimodal_pt2(wrapper, input_ids, attention_mask, image_features, audio_features, destination):
    wrapper.validate_inputs(input_ids, attention_mask, image_features, audio_features)
    _save_program(wrapper.eval(), (input_ids, attention_mask, image_features, audio_features), destination)


def _save_program(wrapper, inputs, destination):
    with torch.no_grad():
        program = torch.export.export(wrapper, inputs)
        program = program.run_decompositions()
    # Sentis' Torch JSON reader stores graph scalar integers as System.Int32.
    # Core ATen uses INT64_MAX for open-ended slices, even on fixed-size inputs.
    def clamp_scalar(value):
        return max(-(2**31), min(2**31 - 1, value)) if type(value) is int else value

    for node in program.graph.nodes:
        node.args = map_aggregate(node.args, clamp_scalar)
        node.kwargs = map_aggregate(node.kwargs, clamp_scalar)
    # Torch 2.14 emits metadata-only assertions that Sentis 2.6 cannot import.
    # erase_node refuses removal if a node has users; numerical operations stay intact.
    for node in list(program.graph.nodes):
        if node.op == "call_function" and node.target == torch.ops.aten._assert_tensor_metadata.default:
            program.graph.erase_node(node)
    program.graph.lint()
    program.graph_module.recompile()
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.export.save(program, destination)
