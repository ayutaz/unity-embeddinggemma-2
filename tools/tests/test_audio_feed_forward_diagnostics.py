"""Inspect the first divergent feed-forward using independently captured native input."""

import json

import pytest
import torch

from embeddinggemma_tools import export, prepare
from test_audio_encoder import audio_models, mask


NAMES = ("clamp_input", "pre_layer_norm", "ffw_layer_1", "activation", "ffw_layer_2",
         "clamp_output", "post_layer_norm", "post_layer_scale", "residual_output")


def contract():
    wrapper = getattr(export, "AudioFeedForwardStagesExport", None)
    capture = getattr(prepare, "capture_official_audio_feed_forward", None)
    diagnose = getattr(prepare, "diagnose_audio_feed_forward_stages", None)
    assert wrapper is not None, "Standalone feed-forward stage export is missing"
    assert capture is not None, "Official feed-forward input/stage capture is missing"
    assert diagnose is not None, "Failure-only feed-forward diagnostics are missing"
    return wrapper, capture, diagnose


def independent_native(module, hidden):
    """Hooks observe the native modules; clones precede native in-place updates."""
    values, handles = {}, []
    def observe(name, mod):
        def hook(mod, args, output):
            values[name] = output.detach().clone()
            if name == "pre_layer_norm":
                values["clamp_input"] = args[0].detach().clone()
            if name == "post_layer_norm":
                values["clamp_output"] = args[0].detach().clone()
        handles.append(mod.register_forward_hook(hook))
    try:
        for name, mod in [("pre_layer_norm", module.pre_layer_norm), ("ffw_layer_1", module.ffw_layer_1),
                          ("activation", module.act_fn), ("ffw_layer_2", module.ffw_layer_2),
                          ("post_layer_norm", module.post_layer_norm)]:
            observe(name, mod)
        with torch.no_grad():
            values["residual_output"] = module(hidden).clone()
        values["post_layer_scale"] = values["post_layer_norm"] * module.post_layer_scale
        return tuple(values[name] for name in NAMES)
    finally:
        for handle in handles:
            handle.remove()


def test_saved_standalone_trace_preserves_native_stages_changed_input_and_inplace_snapshots(tmp_path):
    cls, _, _ = contract()
    audio, _ = audio_models()
    module = audio.layers[0].feed_forward1
    hidden = torch.randn(1, 7, 16)
    wrapper = cls(module, hidden.shape)
    assert wrapper.stage_names == NAMES
    expected = independent_native(module, hidden)
    with torch.no_grad():
        for actual, native in zip(wrapper(hidden), expected, strict=True):
            assert torch.equal(actual, native)
    graph = tmp_path / "ff.pt2"
    export.export_audio_pt2(wrapper, hidden, graph)
    changed = hidden * 0.63
    with torch.no_grad():
        saved = torch.export.load(graph).module()(changed)
    for actual, native in zip(saved, independent_native(module, changed), strict=True):
        torch.testing.assert_close(actual, native, rtol=1e-5, atol=1e-6)
    assert not torch.equal(saved[6], saved[-1]), "post-norm snapshot must survive residual updates"


@pytest.mark.parametrize("part", ["feed_forward1", "feed_forward2"])
def test_official_capture_uses_actual_selected_module_input_and_removes_only_own_hooks(part):
    _, capture, _ = contract()
    audio, _ = audio_models()
    features, frame_mask = torch.randn(1, 25, 8), mask(25, 17)
    features *= frame_mask[:, :, None]
    module = getattr(audio.layers[1], part)
    captured = []
    external = module.register_forward_pre_hook(lambda mod, args: captured.append(args[0].clone()))
    before = [(set(m._forward_hooks), set(m._forward_pre_hooks)) for m in audio.modules()]
    try:
        hidden, actual = capture(audio, features, frame_mask, detail_layer=1, part=part)
        assert torch.equal(hidden, captured[0])
        for value, expected in zip(actual, independent_native(module, hidden), strict=True):
            assert torch.equal(value, expected)
        assert [(set(m._forward_hooks), set(m._forward_pre_hooks)) for m in audio.modules()] == before
    finally:
        external.remove()


def test_native_capture_removes_hooks_when_selected_module_fails(monkeypatch):
    _, capture, _ = contract()
    audio, _ = audio_models()
    module = audio.layers[0].feed_forward1
    external = module.post_layer_norm.register_forward_hook(lambda *args: None)
    before = [(set(m._forward_hooks), set(m._forward_pre_hooks)) for m in audio.modules()]
    def fail(*args, **kwargs):
        raise RuntimeError("native linear failed")
    monkeypatch.setattr(module.ffw_layer_2, "forward", fail)
    try:
        with pytest.raises(RuntimeError, match="native linear failed"):
            capture(audio, torch.randn(1, 25, 8), mask(), detail_layer=0, part="feed_forward1")
        assert [(set(m._forward_hooks), set(m._forward_pre_hooks)) for m in audio.modules()] == before
    finally:
        external.remove()


@pytest.mark.parametrize("layer,part", [(-1, "feed_forward1"), (2, "feed_forward1"),
                                       (True, "feed_forward1"), (0, "self_attn")])
def test_capture_rejects_invalid_selector(layer, part):
    _, capture, _ = contract()
    audio, _ = audio_models()
    with pytest.raises(ValueError, match="feed-forward selector"):
        capture(audio, torch.randn(1, 25, 8), mask(), detail_layer=layer, part=part)


@pytest.mark.parametrize("kind", ["shape", "dtype", "nan"])
def test_standalone_export_rejects_invalid_input_before_saving(tmp_path, kind):
    cls, _, _ = contract()
    audio, _ = audio_models()
    wrapper = cls(audio.layers[0].feed_forward1, (1, 7, 16))
    hidden = torch.randn(1, 7, 16)
    if kind == "shape":
        hidden = hidden[:, :6]
    elif kind == "dtype":
        hidden = hidden.double()
    else:
        hidden[0, 0, 0] = float("nan")
    path = tmp_path / "invalid.pt2"
    with pytest.raises(ValueError, match="feed-forward input"):
        export.export_audio_pt2(wrapper, hidden, path)
    assert not path.exists()


@pytest.mark.parametrize("same_stage", [True, False])
def test_diagnosis_gates_localization_on_exact_standalone_vs_full_trace(tmp_path, same_stage):
    _, capture, diagnose = contract()
    audio, projection = audio_models()
    features, frame_mask = torch.randn(1, 25, 8), mask()
    path = tmp_path / "full.pt2"
    full = export.AudioEncoderStagesExport(audio, projection, frame_mask, detail_layer=0)
    export.export_audio_pt2(full, features, path)
    with torch.no_grad():
        output = torch.export.load(path).module()(features)
    failed_stage = output[full.stage_names.index("layer_0.feed_forward1")]
    if not same_stage:
        failed_stage = failed_stage + 0.1
    result = diagnose(audio, features, frame_mask, tmp_path, failed_stage,
                      detail_layer=0, part="feed_forward1")
    assert result["stage"] == "audio_feed_forward_diagnostics_only"
    assert result["standalone_matches_full_stage"] is same_stage
    assert result["localization_validated"] is same_stage
    assert result["feature_rtol"] == 1e-4 and result["feature_atol"] == 1e-5
    assert [row["name"] for row in result["stages"]] == list(NAMES)
    assert result["first_divergent_stage"] is None
    assert result["encoders_validated"] is False
    assert result["end_to_end_validated"] is False
    assert len(result["native_input_sha256"]) == 64
    assert (tmp_path / "audio-layer-0-feed_forward1-stages.pt2").exists()


@pytest.mark.parametrize("same_final,child_fails,has_numeric_failure", [(True, False, True), (True, True, True),
                                                                    (False, False, True), (True, False, False)])
def test_layer_receipt_persists_before_child_diagnostic_and_keeps_failed_final_gate(tmp_path, monkeypatch, same_final, child_fails, has_numeric_failure):
    contract()
    audio, projection = audio_models()
    features, frame_mask = torch.randn(1, 25, 8), mask()
    path = tmp_path / "normal.pt2"
    export.export_audio_pt2(export.AudioEncoderExport(audio, projection, frame_mask), features, path)
    with torch.no_grad():
        failed_saved = torch.export.load(path).module()(features)
    if not same_final:
        failed_saved = failed_saved + 0.1
    compare = prepare.compare_audio_stage_outputs
    def inject(names, expected, actual):
        result = compare(names, expected, actual)
        result["stages"][0]["maximum_absolute_error"] = 0
        result["stages"][1]["maximum_absolute_error"] = 1e-6
        result["first_divergent_stage"] = "layer_1" if has_numeric_failure else None
        return result  # Artificial localization decision only, not a numeric acceptance.
    calls = []
    def child(*args, **kwargs):
        calls.append(kwargs["part"])
        persisted = json.loads((tmp_path / "audio-layer-0-diagnostics.json").read_text())
        assert persisted["trace_matches_failed_graph"] is True
        if child_fails:
            raise RuntimeError("standalone export interrupted")
        return {"localization_validated": False}
    monkeypatch.setattr(prepare, "compare_audio_stage_outputs", inject)
    monkeypatch.setattr(prepare, "diagnose_audio_feed_forward_stages", child)
    result = prepare.diagnose_audio_layer_stages(audio, projection, features, frame_mask, tmp_path,
                                               failed_saved, detail_layer=0)
    expect_child = same_final and has_numeric_failure
    assert calls == (["feed_forward1"] if expect_child else [])
    assert result["trace_matches_failed_graph"] is same_final
    persisted = json.loads((tmp_path / "audio-layer-0-diagnostics.json").read_text())
    if expect_child and child_fails:
        assert "standalone export interrupted" in persisted["feed_forward_diagnostics_error"]
    elif expect_child:
        assert persisted["feed_forward_diagnostics"]["localization_validated"] is False
    else:
        assert "feed_forward_diagnostics" not in persisted
