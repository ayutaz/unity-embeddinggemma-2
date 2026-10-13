import json

import pytest
import torch
from safetensors.torch import save_file

from embeddinggemma_tools import audio_layer_replay
from test_audio_encoder import audio_models, mask


def capture():
    value = getattr(audio_layer_replay, "capture_parts", None)
    assert callable(value), "Pinned-input native layer replay is missing"
    return value


def fixture():
    audio, projection = audio_models()
    features = torch.randn(1, 25, 8, generator=torch.Generator().manual_seed(17))
    frame_mask = mask()
    with torch.no_grad():
        hidden, output_mask = audio.subsample_conv_projection(features, frame_mask.bool())
        positions = audio.rel_pos_enc(hidden)
    from embeddinggemma_tools.export import AudioEncoderExport
    blocked = AudioEncoderExport(audio, projection, frame_mask).blocked_mask
    with torch.no_grad():
        expected = audio.layers[0](hidden.clone(), attention_mask=blocked, position_embeddings=positions)
    return audio.layers[0], hidden, blocked, positions, expected


def test_replay_keeps_pinned_input_and_matches_independent_native_final():
    run = capture()
    layer, hidden, blocked, positions, expected = fixture()
    original = hidden.clone()
    values = run(layer, hidden, blocked, positions, expected)
    assert list(values) == ["feed_forward1", "norm_pre_attn", "self_attn", "norm_post_attn", "lconv1d", "feed_forward2", "norm_out"]
    assert torch.equal(hidden, original), "Replay must never mutate its pinned input"
    assert torch.equal(values["norm_out"], expected)
    assert all(torch.isfinite(value).all() for value in values.values())


@pytest.mark.parametrize("fail", [False, True])
def test_replay_preserves_external_hooks_and_removes_own_hooks(fail, monkeypatch):
    run = capture()
    layer, hidden, blocked, positions, expected = fixture()
    modules = [getattr(layer, name) for name in ("feed_forward1", "norm_pre_attn", "self_attn", "norm_post_attn", "lconv1d", "feed_forward2", "norm_out")]
    external = layer.norm_out.register_forward_hook(lambda *args: None)
    before = [set(value._forward_hooks) for value in modules]
    if fail:
        monkeypatch.setattr(layer.self_attn, "forward", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("fixture failure")))
    try:
        if fail:
            with pytest.raises(RuntimeError, match="fixture failure"):
                run(layer, hidden, blocked, positions, expected)
        else:
            run(layer, hidden, blocked, positions, expected)
        assert [set(value._forward_hooks) for value in modules] == before
    finally:
        external.remove()


def test_changed_native_final_is_rejected_before_any_result_is_accepted():
    run = capture()
    layer, hidden, blocked, positions, expected = fixture()
    with pytest.raises(ValueError, match="pinned.*final"):
        run(layer, hidden, blocked, positions, expected + 0.1)


def test_loader_allocates_only_selected_layer_and_loads_exact_pinned_weights(tmp_path):
    load = getattr(audio_layer_replay, "load_layer", None)
    assert callable(load), "Selected audio layer loader is missing"
    audio, _ = audio_models()
    config = dict(audio_config=audio.config.to_dict())
    (tmp_path / "config.json").write_text(json.dumps(config))
    save_file({"audio_tower.layers.0." + key: value.contiguous() for key, value in audio.layers[0].state_dict().items()}, tmp_path / "model.safetensors")
    layer, loaded_config = load(tmp_path, index=0)
    assert loaded_config.hidden_size == audio.config.hidden_size
    assert layer.training is False
    assert all(torch.equal(value, audio.layers[0].state_dict()[key]) for key, value in layer.state_dict().items())
    assert len(layer.state_dict()) == len(audio.layers[0].state_dict())
