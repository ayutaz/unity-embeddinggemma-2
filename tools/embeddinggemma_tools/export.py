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
            # Sentis imports multidimensional advanced indexing through
            # GatherND / Transpose. Keep the feature placement's rank explicit
            # with a flat Gather followed by the fixed batch-one shape.
            gathered = torch.index_select(padded, 0, indices.reshape(-1)).reshape(
                1, self.sequence_length, self.model.config.hidden_size)
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


class _AudioAttentionExport(nn.Module):
    """Official relative chunk attention with static Gather replacing Unfold."""

    def __init__(self, attention, sequence_length):
        super().__init__()
        self.attention = attention
        blocks = (sequence_length + attention.chunk_size - 1) // attention.chunk_size
        self.register_buffer("context_indices", torch.arange(blocks)[:, None] * attention.chunk_size
                             + torch.arange(attention.context_size)[None, :])
        # This learned coefficient is constant for an inference export. Native
        # Softplus and its Core ATen log1p(exp(x)) decomposition differ in FP32
        # rounding; preserve the official value before graph decomposition.
        # Keep the two query multiplications in their original order.
        self.register_buffer("per_dimension_query_scale",
                             torch.nn.functional.softplus(attention.per_dim_scale.detach()).clone())

    def context(self, hidden):
        attention = self.attention
        padded = torch.nn.functional.pad(hidden, (0, 0, 0, 0, attention.max_past_horizon,
                                                   attention.max_future_horizon + attention.chunk_size - 1))
        return padded[:, self.context_indices].contiguous()

    def forward(self, hidden, positions, mask):
        attention = self.attention
        batch, length, _ = hidden.shape
        shape = (batch, length, attention.num_heads, attention.head_dim)
        query = attention.q_proj(hidden).float().view(shape)
        query = query * attention.q_scale * self.per_dimension_query_scale
        query = attention._convert_to_block(query)
        key = self.context(attention.k_proj(hidden).float().view(shape) * attention.k_scale)
        value = self.context(attention.v_proj(hidden).float().view(shape))
        relative = attention.relative_k_proj(positions).view(-1, attention.num_heads, attention.head_dim)
        queries = query.permute(0, 3, 1, 2, 4)
        content = queries @ key.permute(0, 3, 1, 4, 2)
        bias = queries.reshape(batch, attention.num_heads, -1, attention.head_dim) @ relative.permute(1, 2, 0)
        bias = attention._rel_shift(bias.reshape(batch, attention.num_heads, query.shape[1], attention.chunk_size, -1))
        logits = torch.tanh((content + bias) / attention.softcap) * attention.softcap
        logits = logits.masked_fill(~mask, attention.config.attention_invalid_logits_value)
        probabilities = torch.nn.functional.softmax(logits, dim=-1, dtype=torch.float32).to(value.dtype)
        output = (probabilities @ value.permute(0, 3, 1, 2, 4)).permute(0, 2, 3, 1, 4)
        output = output.reshape(batch, query.shape[1] * attention.chunk_size, -1)[:, :length].contiguous()
        return attention.post(output.to(hidden.dtype))


class AudioEncoderExport(nn.Module):
    """Fixed prefix-mask audio encoder + projector, not final embeddings.

    Host validation is outside the saved graph. Unity must enforce the same
    finite FP32 features / zero right-padding and fixed frame-mask profile.
    """

    def __init__(self, audio, projection, frame_mask):
        super().__init__()
        if audio.config._attn_implementation != "sdpa":
            raise ValueError("audio export reference requires the official sdpa boolean-mask semantics")
        if (frame_mask.dtype != torch.int64 or frame_mask.ndim != 2 or frame_mask.shape[0] != 1
                or frame_mask.shape[1] == 0 or not torch.all((frame_mask == 0) | (frame_mask == 1))
                or not frame_mask.any() or torch.any(frame_mask[:, 1:] > frame_mask[:, :-1])):
            raise ValueError("audio profile requires a nonempty batch-one int64 binary prefix mask")
        self.audio = audio
        self.projection = projection
        self.feature_shape = (1, frame_mask.shape[1], audio.config.subsampling_conv_channels[0])
        self.register_buffer("frame_mask", frame_mask.clone())
        output_mask = frame_mask[:, ::4].bool()
        length = output_mask.shape[1]
        self.feature_count = int(output_mask.sum())
        chunk = audio.config.attention_chunk_size
        past = audio.config.attention_context_left - 1
        future = audio.config.attention_context_right
        blocks = (length + chunk - 1) // chunk
        queries = torch.arange(blocks)[:, None, None] * chunk + torch.arange(chunk)[None, :, None]
        keys = torch.arange(blocks)[:, None, None] * chunk + torch.arange(chunk + past + future)[None, None, :] - past
        distance = queries - keys
        valid_keys = (keys >= 0) & (keys < length) & output_mask[0, keys.clamp(0, length - 1)]
        allowed = ((distance >= 0) & (distance < past)) | ((distance < 0) & (-distance < future))
        self.register_buffer("blocked_mask", (valid_keys & (queries < length) & allowed)[None, None])
        self.attentions = nn.ModuleList(_AudioAttentionExport(layer.self_attn, length) for layer in audio.layers)
        self.eval()

    def validate_inputs(self, features):
        if features.dtype != torch.float32 or features.shape != self.feature_shape or not torch.isfinite(features).all():
            raise ValueError("audio features must match the finite float32 profile")
        if features.device != self.frame_mask.device or torch.any(features[self.frame_mask == 0] != 0):
            raise ValueError("audio features require the same device and zero right padding")

    def forward(self, features):
        return self._forward(features, collect_stages=False)

    def _forward(self, features, *, collect_stages, detail_layer=None):
        if not torch.compiler.is_compiling():
            self.validate_inputs(features)
        hidden, _ = self.audio.subsample_conv_projection(features, self.frame_mask.bool())
        stages = [hidden] if collect_stages else []
        positions = self.audio.rel_pos_enc(hidden)
        for index, (layer, attention) in enumerate(zip(self.audio.layers[:self.audio.config.num_hidden_layers], self.attentions)):
            detailed = collect_stages and index == detail_layer
            cap = min(layer.gradient_clipping, torch.finfo(layer.norm_pre_attn.weight.dtype).max)
            hidden = layer.feed_forward1(hidden)
            if detailed:
                stages.append(hidden)
            residual = hidden
            hidden = layer.norm_pre_attn(hidden.clamp(-cap, cap))
            if detailed:
                stages.append(hidden)
            hidden = attention(hidden, positions, self.blocked_mask)
            if detailed:
                stages.append(hidden)
            hidden = layer.norm_post_attn(hidden.clamp(-cap, cap))
            if detailed:
                stages.append(hidden)
            hidden = layer.lconv1d(hidden + residual)
            if detailed:
                stages.append(hidden)
            hidden = layer.feed_forward2(hidden)
            if detailed:
                stages.append(hidden)
            hidden = layer.norm_out(hidden.clamp(-cap, cap))
            if collect_stages:
                stages.append(hidden)
        hidden = self.audio.output_proj(hidden)
        projected = self.projection(hidden)[0, :self.feature_count]
        if collect_stages:
            return tuple([*stages, hidden, projected])
        return projected


class AudioSubsampleStagesExport(nn.Module):
    """Failure-only fixed-profile trace of the official subsample module."""

    stage_names = tuple(f"layer{index}.{part}" for index in (0, 1)
                        for part in ("masked_input", "conv", "norm", "relu")) + ("projection_input", "projection")

    def __init__(self, module, input_shape, frame_mask):
        super().__init__()
        self.module = module
        self.feature_shape = tuple(input_shape)
        self.register_buffer("frame_mask", frame_mask.clone())
        self.eval()

    def validate_inputs(self, features):
        if (features.dtype != torch.float32 or tuple(features.shape) != self.feature_shape
                or not torch.isfinite(features).all() or features.device != self.frame_mask.device
                or torch.any(features[self.frame_mask == 0] != 0)):
            raise ValueError("subsample features require the fixed finite float32 profile and zero padding")

    def forward(self, features):
        if not torch.compiler.is_compiling():
            self.validate_inputs(features)
        hidden, frame_mask, stages = features.unsqueeze(1), self.frame_mask.bool(), []
        for layer in (self.module.layer0, self.module.layer1):
            hidden = hidden * frame_mask[:, None, :, None]
            stages.append(hidden)
            hidden = layer.conv(hidden.to(layer.conv.weight.dtype))
            stages.append(hidden)
            normalized = layer.norm(hidden.permute(0, 2, 3, 1))
            stages.append(normalized)
            hidden = layer.act(normalized.permute(0, 3, 1, 2).contiguous())
            stages.append(hidden)
            frame_mask = frame_mask[:, ::2]
        batch, _, length, _ = hidden.shape
        hidden = hidden.permute(0, 2, 3, 1).contiguous().reshape(batch, length, -1)
        return (*stages, hidden, self.module.input_proj_linear(hidden))


class AudioEncoderStagesExport(AudioEncoderExport):
    """Failure-only trace using the same numerical path as the normal export."""

    detail_parts = ("feed_forward1", "norm_pre_attn", "self_attn", "norm_post_attn", "lconv1d", "feed_forward2")

    def __init__(self, audio, projection, frame_mask, *, detail_layer=None):
        if detail_layer is not None and (type(detail_layer) is not int or not 0 <= detail_layer < audio.config.num_hidden_layers):
            raise ValueError("audio detail layer must be an existing integer layer index")
        super().__init__(audio, projection, frame_mask)
        self.detail_layer = detail_layer

    @property
    def stage_names(self):
        names = ["subsample"]
        for index in range(self.audio.config.num_hidden_layers):
            if index == self.detail_layer:
                names.extend(f"layer_{index}.{part}" for part in self.detail_parts)
            names.append(f"layer_{index}")
        return (*names, "output_projection", "projected_features")

    def forward(self, features):
        return self._forward(features, collect_stages=True, detail_layer=self.detail_layer)


class AudioFeedForwardStagesExport(nn.Module):
    """Small failure-only trace; preserve values before native in-place updates."""

    stage_names = ("clamp_input", "pre_layer_norm", "ffw_layer_1", "activation", "ffw_layer_2",
                   "clamp_output", "post_layer_norm", "post_layer_scale", "residual_output")

    def __init__(self, module, input_shape):
        super().__init__()
        self.module = module
        self.input_shape = tuple(input_shape)
        self.eval()

    def validate_inputs(self, hidden):
        if (hidden.dtype != torch.float32 or tuple(hidden.shape) != self.input_shape
                or hidden.device != self.module.pre_layer_norm.weight.device or not torch.isfinite(hidden).all()):
            raise ValueError("audio feed-forward input must match the finite float32 native profile")

    def forward(self, hidden):
        if not torch.compiler.is_compiling():
            self.validate_inputs(hidden)
        residual = hidden
        cap = min(self.module.gradient_clipping, torch.finfo(hidden.dtype).max)
        hidden = hidden.clamp(-cap, cap)
        values = [hidden.clone()]
        hidden = self.module.pre_layer_norm(hidden)
        values.append(hidden.clone())
        hidden = self.module.ffw_layer_1(hidden)
        values.append(hidden.clone())
        hidden = self.module.act_fn(hidden)
        values.append(hidden.clone())
        hidden = self.module.ffw_layer_2(hidden)
        values.append(hidden.clone())
        hidden = hidden.clamp(-cap, cap)
        values.append(hidden.clone())
        hidden = self.module.post_layer_norm(hidden)
        values.append(hidden.clone())
        hidden *= self.module.post_layer_scale
        values.append(hidden.clone())
        hidden += residual
        return tuple([*values, hidden])


def export_audio_pt2(wrapper, features, destination):
    # Keep the native kernel for Python accuracy. Sentis imports the standard
    # Swish ONNX companion, not this native-reference Torch program.
    export_audio_native_pt2(wrapper,features,destination)


def export_audio_native_pt2(wrapper, features, destination):
    """Retain the actual CPU SiLU operation for the independent Python gate."""
    wrapper.validate_inputs(features)
    from torch._decomp import core_aten_decompositions
    decompositions=core_aten_decompositions()
    decompositions.pop(torch.ops.aten.silu.default,None)
    _save_program(wrapper.eval(),(features,),destination,decompositions=decompositions)


def fp32_fma(a, b, c):
    """Single-rounding candidate for bounded normal FP32 inputs, without FP64 ops.

    Dekker product and TwoSum retain the residual before the final rounding.
    This is a candidate for the bounded LayerNorm statistics, not general IEEE
    FMA with overflow, underflow, NaN, or infinity handling.
    """
    product = a * b
    a_split, b_split = a * 4097.0, b * 4097.0
    a_high, b_high = a_split - (a_split - a), b_split - (b_split - b)
    a_low, b_low = a - a_high, b - b_high
    product_error = ((a_high * b_high - product) + a_high * b_low + a_low * b_high) + a_low * b_low
    total = product + c
    recovered = total - product
    sum_error = (product - (total - recovered)) + (c - recovered)
    return total + (product_error + sum_error)


def fp32_block192_matmul(values, weights):
    """Experimental fixed FP32 static projection with block192 FMA/store order.

    Inferred from a pinned full-280-row SGEMM replay, not a guarantee about all
    MKL shapes or instruction sets. Borrows the existing KxN coefficient bytes.
    Product inference stays FP32; the independent test oracle may use FP64.
    """
    if not torch.compiler.is_compiling() and not torch.jit.is_tracing():
        if (values.ndim not in (2, 3) or weights.ndim != 2 or values.shape[-1] != weights.shape[0]
                or values.shape[-1] < 1 or weights.shape[1] < 1
                or values.dtype != torch.float32 or weights.dtype != torch.float32
                or values.device != weights.device or not torch.isfinite(values).all()
                or not torch.isfinite(weights).all() or torch.any(values.abs() > 1e6)
                or torch.any(weights.abs() > 100)):
            raise ValueError('Block192 projection requires bounded finite matching FP32 matrices')
    width = int(weights.shape[0])
    blocks = (width + 191) // 192
    padding = blocks * 192 - width
    rank = values.ndim
    chunks = torch.nn.functional.pad(values, (0, padding)).reshape(*values.shape[:-1], blocks, 192)
    chunks = chunks.permute(rank-1, *range(rank-1), rank)
    coefficients = torch.nn.functional.pad(weights, (0, 0, 0, padding)).reshape(blocks, 192, weights.shape[1])
    partial = torch.zeros_like(chunks[..., :1])
    for index in range(192):
        right = coefficients[:, index].reshape(blocks, *([1] * (values.ndim-1)), weights.shape[1])
        partial = fp32_fma(chunks[..., index:index+1], right, partial)
    total = partial[0]
    for index in range(1, blocks):
        total = total + partial[index]
    return total


def fp32_cascade_mean1024(values):
    """Pinned AVX2 eight-lane FP32 sum/divide order for contiguous 1024 rows.

    Experimental standard-op candidate; this is not a general reduction or a
    guarantee about other CPU instruction sets or tensor layouts. Mirrors the
    four-accumulator, 16-step cascade and final lane order in pinned SumKernel:
    https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/native/cpu/SumKernel.cpp
    """
    return _fp32_cascade_mean(values, 1024)


def fp32_cascade_mean1536(values):
    """Same pinned eight-lane order for the audio projection's 1536-wide rows."""
    return _fp32_cascade_mean(values, 1536)


def fp32_cascade_mean768(values):
    """Opt-in contiguous vision RMS reduction, including its half-block tail.

    Independent CPU/ONNX checks qualify this arithmetic only; actual GPU and
    original full-image numerical gates remain separate requirements.
    """
    return _fp32_cascade_sum768(values) / 768


def _fp32_cascade_sum768(values):
    if not torch.compiler.is_compiling() and not torch.jit.is_tracing():
        if (values.dtype != torch.float32 or values.ndim < 2 or values.shape[-1] != 768
                or not values.numel() or not values.is_contiguous() or not torch.isfinite(values).all()):
            raise ValueError("vision cascade requires finite contiguous 768-wide FP32 rows")
    grouped = torch.nn.functional.pad(values, (0, 256)).reshape(-1, 2, 16, 4, 8)
    partial = _fp32_ordered_sum(grouped, -3)
    accumulated = _fp32_ordered_sum(partial, -4)
    combined = _fp32_ordered_sum(accumulated, -2)
    total = _fp32_ordered_sum(combined, -1)
    return total.reshape_as(values[..., :1])


def _fp32_cascade_mean(values, width):
    if not torch.compiler.is_compiling() and not torch.jit.is_tracing():
        if (values.dtype != torch.float32 or values.ndim < 2 or values.shape[-1] != width
                or not values.numel() or not values.is_contiguous() or not torch.isfinite(values).all()):
            raise ValueError("cascade mean requires finite contiguous fixed-width FP32 rows")
    # Independent chunks / ILP accumulators / SIMD lanes are batched together.
    # Each dependent addition still occurs in the original order.
    grouped = values.reshape(-1, width // 512, 16, 4, 8)
    partial = _fp32_ordered_sum(grouped, -3)
    accumulated = _fp32_ordered_sum(partial, -4)
    combined = _fp32_ordered_sum(accumulated, -2)
    total = _fp32_ordered_sum(combined, -1)
    return total.reshape_as(values[..., :1]) / width


def _fp32_ordered_sum(values, axis):
    parts = torch.unbind(values, dim=axis)
    total = torch.zeros_like(parts[0])
    for part in parts:
        total = total + part
    return total.unsqueeze(axis)


def fp32_outer_cascade_mean1024(values):
    """Pinned AVX2 outer reduction for one transposed 280x1024 convolution.

    The first 256 time rows use multi_row_sum (four cascade levels); the last
    24 use row_sum's four ILP accumulators. The fixed shape and CPU strides are
    an explicit reference contract, not inferred from an ONNX tensor shape.
    """
    if not torch.compiler.is_compiling() and not torch.jit.is_tracing():
        if (values.dtype != torch.float32 or tuple(values.shape) != (1, 280, 1024)
                or values.stride() != (286720, 1, 280) or not torch.isfinite(values).all()):
            raise ValueError("outer cascade requires finite transposed 1x280x1024 FP32 values")
    head = values[:, :256, :].reshape(1, 256, 4, 16, 16)
    head = _fp32_ordered_sum(head, -1)
    head = _fp32_ordered_sum(head, -2)
    head = _fp32_ordered_sum(head, -3).reshape(1, 256, 1)
    tail = values[:, 256:, :].reshape(1, 24, 16, 16, 4)
    tail = _fp32_ordered_sum(tail, -2)
    tail = _fp32_ordered_sum(tail, -3)
    tail = _fp32_ordered_sum(tail, -1).reshape(1, 24, 1)
    return torch.cat((head, tail), dim=1) / 1024


def fp32_refined_rstd(values, *, maximum_statistic=1e6):
    """Bounded FP32 sqrt then reciprocal, correcting GPU approximation residuals.

    Keep native's two rounding steps instead of a contracted rsqrt. This is an
    experimental standard-op candidate for positive statistics in [1e-6, 1e6],
    with an explicit 2e6 upper bound for the final vision RMS candidate,
    not an IEEE implementation for arbitrary FP32 inputs.
    """
    if maximum_statistic not in (1e6, 2e6):
        raise ValueError("refinement requires a qualified bounded FP32 statistic domain")
    if not torch.compiler.is_compiling() and not torch.jit.is_tracing():
        if (values.dtype != torch.float32 or not values.numel() or not torch.isfinite(values).all()
                or torch.any(values < 1e-6) or torch.any(values > maximum_statistic)):
            raise ValueError("refinement requires bounded positive FP32 statistics")
    root = torch.sqrt(values)
    residual = fp32_fma(-root, root, values)
    rounded_root = root + (residual * 0.5) / root
    inverse = 1.0 / rounded_root
    return _fp32_correct_reciprocal(rounded_root, inverse)


def _fp32_correct_reciprocal(rounded_root, inverse):
    """Select the closest reciprocal around a bounded normal FP32 GPU seed.

    A Newton step can round to a midpoint and leave a faithful seed one ULP
    away, so comparing only seed and refined values is insufficient. Examine
    both adjacent representable values using a relative step strictly above
    half an ULP, including powers of two. No integer bitcast or FP64 is exported.
    """
    ones = torch.ones_like(rounded_root)

    def error_pair(quotient):
        # The product is near one. TwoProd followed by a compensated
        # subtraction retains the low residual that a single Newton/FMA
        # rounds away at hard reciprocal midpoints. All operations stay FP32.
        product = rounded_root * quotient
        product_low = fp32_fma(rounded_root, quotient, -product)
        difference = ones - product
        high = difference - product_low
        low = (difference - high) - product_low
        return high.abs(), torch.where(high < 0, -low, low)

    # The next FP32 number above 2**-24 makes the increment strictly greater
    # than half an ULP at an exact binade boundary; away from it, still <1 ULP.
    step = inverse * (2.0**-24 + 2.0**-47)
    best = inverse
    best_high, best_low = error_pair(best)
    for candidate in (inverse + step, inverse - step):
        high, low = error_pair(candidate)
        closer = (high < best_high) | ((high == best_high) & (low < best_low))
        best = torch.where(closer, candidate, best)
        best_high = torch.where(closer, high, best_high)
        best_low = torch.where(closer, low, best_low)
    return best


def fp32_cascade_mean64(values):
    """Opt-in contiguous vision head reduction with two four-vector steps."""
    if not torch.compiler.is_compiling() and not torch.jit.is_tracing():
        if (values.dtype != torch.float32 or values.ndim < 2 or values.shape[-1] != 64
                or not values.numel() or not values.is_contiguous() or not torch.isfinite(values).all()):
            raise ValueError("vision head cascade requires finite contiguous 64-wide FP32 rows")
    grouped = values.reshape(-1, 2, 4, 8)
    partial = _fp32_ordered_sum(grouped, -3)
    combined = _fp32_ordered_sum(partial, -2)
    total = _fp32_ordered_sum(combined, -1)
    return total.reshape_as(values[..., :1]) / 64


class VisionHeadRmsExport(nn.Module):
    """Opt-in scaled/unscaled 64-wide RMS candidate; borrows original scale.

    Local equation parity does not prove original native-context or GPU parity.
    """
    def __init__(self, scale, epsilon):
        super().__init__()
        if (scale is not None and (not isinstance(scale, torch.Tensor) or scale.dtype != torch.float32
                or tuple(scale.shape) != (64,) or not scale.is_contiguous() or not torch.isfinite(scale).all())):
            raise ValueError("vision head RMS requires the original finite FP32 width-64 scale")
        if float(torch.tensor(epsilon, dtype=torch.float32)) != float(torch.tensor(1e-6, dtype=torch.float32)):
            raise ValueError("vision head RMS requires epsilon 1e-6")
        self.register_buffer("scale", scale)

    def forward(self, values):
        mean = fp32_cascade_mean64(values.square())
        inverse = fp32_refined_rstd(mean + 1e-6)
        normalized = values * inverse
        return normalized if self.scale is None else normalized * self.scale


class VisionRmsExport(nn.Module):
    """Opt-in fixed 768-wide FP32 RMS candidate; borrows the original scale.

    CPU/ONNX parity is not original-context or GPU encoder acceptance. Keep
    scalar division and refined inverse operations explicit in exported graphs.
    """
    def __init__(self, scale, epsilon):
        super().__init__()
        if (not isinstance(scale, torch.Tensor) or scale.dtype != torch.float32
                or tuple(scale.shape) != (768,) or not scale.is_contiguous()
                or not torch.isfinite(scale).all()
                or float(torch.tensor(epsilon, dtype=torch.float32)) != float(torch.tensor(1e-6, dtype=torch.float32))):
            raise ValueError("vision RMS requires the original finite FP32 width-768 scale and epsilon 1e-6")
        self.register_buffer("scale", scale)
        self.epsilon = 1e-6

    def forward(self, values):
        mean = fp32_cascade_mean768(values.square())
        inverse = fp32_refined_rstd(mean + self.epsilon)
        return (values * inverse) * self.scale


class VisionInlineRmsExport(VisionRmsExport):
    """Opt-in single-model RMS candidate with FP32 quotient residual correction.

    Keeps importer reciprocal multiplication from determining the mean's last
    bit. Original-context, GPU and full encoder gates remain independent.
    """
    def forward(self, values):
        total = _fp32_cascade_sum768(values.square())
        quotient = total / 768
        residual = fp32_fma(-quotient, torch.full_like(total, 768), total)
        mean = fp32_fma(residual, torch.full_like(total, 1 / 768), quotient)
        inverse = fp32_refined_rstd(mean + self.epsilon)
        return (values * inverse) * self.scale


class VisionTailRmsExport(nn.Module):
    """Opt-in unscaled final width-768 vision RMS; fixed FP32 statistics <=2e6.

    This preserves the existing RMS exporters and their default domain. CPU
    equation/export parity does not prove GPU or original-golden acceptance.
    """
    def forward(self, values):
        total = _fp32_cascade_sum768(values.square())
        quotient = total / 768
        residual = fp32_fma(-quotient, torch.full_like(total, 768), total)
        mean = fp32_fma(residual, torch.full_like(total, 1 / 768), quotient)
        inverse = fp32_refined_rstd(mean + 1e-6, maximum_statistic=2e6)
        return values * inverse


class TorchWelfordLayerNorm(nn.Module):
    """Editor export candidate for the pinned eight-lane FP32 moment order.

    Mirrors the bounded 32/128-wide cases in pinned PyTorch moments_utils.h:
    https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/native/cpu/moments_utils.h
    Native source and weights are borrowed; never change the golden reference.
    """
    def __init__(self, source, *, refine_inverse=False, fuse_scalar_mean=False, fuse_scalar_moment=False):
        super().__init__()
        if source.normalized_shape not in ((32,), (128,)):
            raise ValueError("candidate supports only width 32 or 128")
        if source.weight is None or source.weight.dtype != torch.float32 or source.bias is not None:
            raise ValueError("candidate requires unbiased FP32 LayerNorm scale")
        self.weight = source.weight
        self.eps = source.eps
        self.width = source.normalized_shape[0]
        self.refine_inverse = refine_inverse
        # Linux reference compiler contracts the scalar AddMoments mean update.
        # Opt in explicitly; existing candidate graphs retain their old order.
        self.fuse_scalar_mean = fuse_scalar_mean
        self.fuse_scalar_moment = fuse_scalar_moment

    def forward(self, values):
        if not torch.compiler.is_compiling():
            if values.dtype != torch.float32 or values.shape[-1] != self.width:
                raise ValueError("candidate requires fixed-width FP32 values")
        mean = torch.zeros_like(values[..., :8])
        moment = torch.zeros_like(mean)
        count = self.width // 8
        for index in range(count):
            item = values[..., index*8:(index+1)*8]
            delta = item - mean
            # Store the reciprocal as FP32, as the pinned vector constants do.
            coefficient = values.new_tensor(1.0 / (index + 1))
            mean = fp32_fma(coefficient, delta, mean)
            moment = fp32_fma(delta, item - mean, moment)
        total_mean = torch.zeros_like(mean[..., :1])
        total_moment = torch.zeros_like(total_mean)
        for index in range(8):
            previous = index * count
            delta = mean[..., index:index+1] - total_mean
            coefficient = values.new_tensor(count / (previous + count))
            total_mean = (fp32_fma(coefficient, delta, total_mean) if self.fuse_scalar_mean
                          else total_mean + coefficient * delta)
            increment=(fp32_fma(delta*delta*coefficient,values.new_tensor(previous),moment[...,index:index+1])
                       if self.fuse_scalar_moment else moment[...,index:index+1]+delta*delta*coefficient*previous)
            total_moment = total_moment + increment
        variance = total_moment / self.width + self.eps
        inverse = fp32_refined_rstd(variance) if self.refine_inverse else 1.0 / torch.sqrt(variance)
        return (values - total_mean) * inverse * self.weight


def export_audio_native_onnx(wrapper, features, destination, *, layernorm_algorithm="native", linear_algorithm="native"):
    """Bridge native SiLU to standard ops while preserving FP32 division order."""
    import tempfile
    import onnx
    destination=Path(destination)
    destination.parent.mkdir(parents=True,exist_ok=True)
    if layernorm_algorithm not in ("native", "torch_welford8", "torch_welford8_refined_moment"):
        raise ValueError("Unknown ONNX LayerNorm algorithm")
    if linear_algorithm not in ("native", "compensated64"):
        raise ValueError("Unknown ONNX linear algorithm")
    restored = []
    domain="com.ayutaz.export"
    def silu_symbolic(graph, value):
        return graph.op(domain+"::NativeSiLU",value).setType(value.type())
    torch.onnx.register_custom_op_symbolic("aten::silu",silu_symbolic,20)
    try:
        if layernorm_algorithm in ("torch_welford8", "torch_welford8_refined_moment"):
            for parent in list(wrapper.modules()):
                for name, child in list(parent.named_children()):
                    if isinstance(child, nn.LayerNorm):
                        refined = layernorm_algorithm == "torch_welford8_refined_moment"
                        replacement = TorchWelfordLayerNorm(child,refine_inverse=refined,fuse_scalar_moment=refined)
                        restored.append((parent, name, child))
                        parent.add_module(name, replacement)
            if not restored:
                raise ValueError("Welford export candidate requires source LayerNorm modules")
        if linear_algorithm == "compensated64":
            from types import SimpleNamespace
            from .audio_layer_replay import CompensatedClippedLinear
            linear_count = 0
            for parent in list(wrapper.modules()):
                for name, child in list(parent.named_children()):
                    if isinstance(child, nn.Linear):
                        replacement = CompensatedClippedLinear(SimpleNamespace(linear=child, use_clipped_linears=False),
                                                               chunk_size=64, include_bias=True)
                        restored.append((parent, name, child))
                        parent.add_module(name, replacement)
                        linear_count += 1
            if not linear_count:
                raise ValueError("Linear export candidate requires source Linear modules")
        with tempfile.TemporaryDirectory(prefix="audio-onnx-",dir=destination.parent) as directory:
            if not Path(directory).resolve().is_relative_to(destination.parent.resolve()):
                raise ValueError("ONNX temporary path escaped the output directory")
            intermediate=Path(directory)/"intermediate.onnx"
            with torch.no_grad():
                torch.onnx.export(wrapper.eval(),(features,),intermediate,dynamo=False,opset_version=20,
                    input_names=["input"],output_names=["output"],custom_opsets={domain:1})
            model=onnx.load(intermediate)
            model=onnx.version_converter.convert_version(model,24)
            reserved = {name for node in model.graph.node for name in (*node.input, *node.output)}
            reserved.update(value.name for value in model.graph.initializer)
            def fresh(base):
                name = base
                while name in reserved:
                    name += "_"
                reserved.add(name)
                return name
            one = fresh("__embeddinggemma_silu_one")
            model.graph.initializer.append(onnx.helper.make_tensor(one, onnx.TensorProto.FLOAT, [], [1.0]))
            expanded = []
            for index, node in enumerate(model.graph.node):
                if node.domain==domain and node.op_type=="NativeSiLU":
                    if len(node.input) != 1 or len(node.output) != 1:
                        raise ValueError("Native SiLU requires one FP32 input and output")
                    negative = fresh(f"__embeddinggemma_silu_{index}_negative")
                    exponential = fresh(f"__embeddinggemma_silu_{index}_exp")
                    denominator = fresh(f"__embeddinggemma_silu_{index}_denominator")
                    # PyTorch CPU SiLU divides x by 1+exp(-x). The ONNX Swish
                    # evaluator multiplies x by the reciprocal and rounds again.
                    expanded.extend((onnx.helper.make_node("Neg", [node.input[0]], [negative]),
                        onnx.helper.make_node("Exp", [negative], [exponential]),
                        onnx.helper.make_node("Add", [one, exponential], [denominator]),
                        onnx.helper.make_node("Div", [node.input[0], denominator], list(node.output), name=node.name)))
                elif node.domain not in ("","ai.onnx"):
                    raise ValueError("Audio ONNX export retained a nonstandard runtime domain")
                else:
                    expanded.append(node)
            del model.graph.node[:]
            model.graph.node.extend(expanded)
            imports=[item for item in model.opset_import if item.domain!=domain]
            del model.opset_import[:]
            model.opset_import.extend(imports)
            onnx.checker.check_model(model)
            onnx.save_model(model,destination)
    finally:
        for parent, name, child in restored:
            parent.add_module(name, child)
        torch.onnx.unregister_custom_op_symbolic("aten::silu",20)


def export_pt2(
    model: nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor, destination: Path
) -> None:
    wrapper = TextEmbeddingExport(model, input_ids.shape[1]).eval()
    _save_program(wrapper, (input_ids, attention_mask), destination)


def export_multimodal_pt2(wrapper, input_ids, attention_mask, image_features, audio_features, destination):
    wrapper.validate_inputs(input_ids, attention_mask, image_features, audio_features)
    _save_program(wrapper.eval(), (input_ids, attention_mask, image_features, audio_features), destination)


def _save_program(wrapper, inputs, destination, *, decompositions=None):
    with torch.no_grad():
        program = torch.export.export(wrapper, inputs)
        program = program.run_decompositions(decompositions)
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
