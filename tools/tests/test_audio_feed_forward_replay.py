import pytest
import torch

from embeddinggemma_tools import audio_layer_replay
from test_audio_layer_replay import fixture, bundle


def capture():
    function = getattr(audio_layer_replay, "capture_feed_forward_parts", None)
    assert callable(function), "Pinned feed-forward detail capture is missing"
    return function


def test_feed_forward_details_match_independent_unmutated_module_outputs():
    run = capture()
    layer, hidden, blocked, positions, expected = fixture()
    ff = layer.feed_forward1
    with torch.no_grad():
        normalized = ff.pre_layer_norm(hidden.clamp(-ff.gradient_clipping, ff.gradient_clipping))
        first = ff.ffw_layer_1(normalized)
        activated = ff.act_fn(first)
        second = ff.ffw_layer_2(activated)
        post = ff.post_layer_norm(second.clamp(-ff.gradient_clipping, ff.gradient_clipping)).clone()
    values = run(layer, hidden, blocked, positions, expected)
    assert list(values) == ["pre_layer_norm", "ffw_layer_1", "act_fn", "ffw_layer_2", "post_layer_norm"]
    for name, value in zip(values, (normalized, first, activated, second, post)):
        assert torch.equal(values[name], value), f"{name} output changed or was captured after in-place residual updates"


def test_feed_forward_capture_failure_removes_only_owned_nested_hooks(monkeypatch):
    run = capture()
    layer, hidden, blocked, positions, expected = fixture()
    modules = [getattr(layer.feed_forward1, name) for name in ("pre_layer_norm", "ffw_layer_1", "act_fn", "ffw_layer_2", "post_layer_norm")]
    external = modules[0].register_forward_hook(lambda *args: None)
    before = [set(module._forward_hooks) for module in modules]
    monkeypatch.setattr(layer.feed_forward1.ffw_layer_2, "forward", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("detail fixture failure")))
    try:
        with pytest.raises(RuntimeError, match="detail fixture failure"):
            run(layer, hidden, blocked, positions, expected)
        assert [set(module._forward_hooks) for module in modules] == before
    finally:
        external.remove()


def test_feed_forward_details_reject_changed_pinned_final():
    run = capture()
    layer, hidden, blocked, positions, expected = fixture()
    with pytest.raises(ValueError, match="pinned.*final"):
        run(layer, hidden, blocked, positions, expected + 0.1)


def test_detail_producer_adds_five_outputs_without_changing_original_seven(tmp_path):
    snapshot, context = bundle(tmp_path)
    base = audio_layer_replay.prepare(snapshot, context, tmp_path / "base", reference_source="a" * 40, diagnostic_source="c" * 40)
    detail = audio_layer_replay.prepare(snapshot, context, tmp_path / "detail", reference_source="a" * 40, diagnostic_source="c" * 40, detail_feed_forward=True)
    assert len(detail["parts"]) == 12
    for name in base["parts"]:
        assert detail["parts"][name] == base["parts"][name]
    assert detail["native_final_exact_match"] is True
    assert detail["full_encoder_validated"] is False


def test_cli_passes_explicit_detail_request_to_the_audited_producer(tmp_path, monkeypatch):
    seen = {}
    def prepare(*args, **kwargs):
        seen.update(kwargs)
        return dict(context_validated=True, native_final_exact_match=True,
                    full_encoder_validated=False, unity_validated=False)
    monkeypatch.setattr(audio_layer_replay, "prepare", prepare)
    code = audio_layer_replay.main(["--snapshot", str(tmp_path), "--context", str(tmp_path),
        "--output", str(tmp_path / "out"), "--reference-source", "a" * 40,
        "--diagnostic-source", "c" * 40, "--report", str(tmp_path / "report.json"),
        "--detail-feed-forward"])
    assert code == 0 and seen["detail_feed_forward"] is True
