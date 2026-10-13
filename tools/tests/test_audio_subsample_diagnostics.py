import pytest
import torch
import onnx
from onnx.reference import ReferenceEvaluator

from embeddinggemma_tools import export, prepare
from test_audio_encoder import audio_models, mask


def required(owner, name):
    value = getattr(owner, name, None)
    assert callable(value), f"Subsample diagnostic contract is missing: {name}"
    return value


def native_stages(audio, features, frame_mask):
    values, handles = {}, []
    module = audio.subsample_conv_projection
    def before(name):
        def capture(layer, args): values[name] = args[0].detach().clone()
        return capture
    def after(name):
        def capture(layer, args, output): values[name] = output.detach().clone()
        return capture
    for index, layer in enumerate((module.layer0, module.layer1)):
        prefix = f"layer{index}."
        handles.append(layer.conv.register_forward_pre_hook(before(prefix + "masked_input")))
        for name, child in (("conv", layer.conv), ("norm", layer.norm), ("relu", layer.act)):
            handles.append(child.register_forward_hook(after(prefix + name)))
    handles.append(module.input_proj_linear.register_forward_pre_hook(before("projection_input")))
    handles.append(module.input_proj_linear.register_forward_hook(after("projection")))
    try:
        with torch.no_grad(): module(features, frame_mask.bool())
    finally:
        for handle in handles: handle.remove()
    names = tuple(f"layer{index}.{part}" for index in (0, 1)
                  for part in ("masked_input", "conv", "norm", "relu")) + ("projection_input", "projection")
    return names, tuple(values[name] for name in names)


@pytest.mark.parametrize("valid", [25, 17])
def test_subsample_trace_keeps_masks_layout_and_official_stages(tmp_path, valid):
    factory = required(export, "AudioSubsampleStagesExport")
    capture = required(prepare, "capture_official_audio_subsample_stages")
    audio, _ = audio_models()
    frame_mask = mask(25, valid)
    features = torch.randn(1, 25, 8, generator=torch.Generator().manual_seed(41)) * frame_mask[:, :, None]
    names, expected = native_stages(audio, features, frame_mask)
    wrapper = factory(audio.subsample_conv_projection, features.shape, frame_mask)
    assert wrapper.stage_names == names
    with torch.no_grad(): actual = wrapper(features)
    captured = capture(audio, features, frame_mask)
    for computed, hooked, native in zip(actual, captured, expected, strict=True):
        torch.testing.assert_close(computed, native, rtol=0, atol=0)
        torch.testing.assert_close(hooked, native, rtol=0, atol=0)
    path = tmp_path / "subsample.onnx"
    export.export_audio_native_onnx(wrapper, features, path)
    model = onnx.load(path)
    assert all(node.domain in ("", "ai.onnx") for node in model.graph.node)
    changed = features * 0.6
    values = ReferenceEvaluator(model).run(None, {"input": changed.numpy()})
    _, expected = native_stages(audio, changed, frame_mask)
    for saved, native in zip(values, expected, strict=True):
        __import__("numpy").testing.assert_allclose(saved, native.numpy(), rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("problem", ["trace", "reference"])
def test_subsample_diagnostics_cannot_validate_a_changed_context(tmp_path, problem):
    run = required(prepare, "diagnose_audio_onnx_subsample")
    audio, _ = audio_models()
    frame_mask = mask(25, 17)
    features = torch.zeros(1, 25, 8)
    with torch.no_grad(): original, _ = audio.subsample_conv_projection(features, frame_mask.bool())
    report = run(audio, features, frame_mask, tmp_path,
                 original + 0.1 if problem == "trace" else original,
                 original + 0.2 if problem == "reference" else original)
    assert report["localization_validated"] is False
    assert report["encoders_validated"] is False
    assert report["end_to_end_validated"] is False


@pytest.mark.parametrize("fail", [False, True])
def test_subsample_capture_removes_only_its_hooks_even_on_failure(monkeypatch, fail):
    capture = required(prepare, "capture_official_audio_subsample_stages")
    audio, _ = audio_models()
    module = audio.subsample_conv_projection
    external = module.layer0.conv.register_forward_hook(lambda *args: None)
    modules = [module.layer0.conv, module.layer0.norm, module.layer0.act,
               module.layer1.conv, module.layer1.norm, module.layer1.act, module.input_proj_linear]
    before = [(set(child._forward_hooks), set(child._forward_pre_hooks)) for child in modules]
    if fail:
        def broken(*args): raise RuntimeError("subsample capture failed")
        monkeypatch.setattr(module.layer1.conv, "forward", broken)
        with pytest.raises(RuntimeError, match="subsample capture failed"):
            capture(audio, torch.zeros(1, 25, 8), mask(25, 17))
    else:
        capture(audio, torch.zeros(1, 25, 8), mask(25, 17))
    assert [(set(child._forward_hooks), set(child._forward_pre_hooks)) for child in modules] == before
    external.remove()


@pytest.mark.parametrize("problem", ["shape", "dtype", "nonfinite", "padding"])
def test_subsample_trace_refuses_invalid_fixed_profile_features(problem):
    factory = required(export, "AudioSubsampleStagesExport")
    audio, _ = audio_models()
    wrapper = factory(audio.subsample_conv_projection, (1, 25, 8), mask(25, 17))
    features = torch.zeros(1, 25, 8)
    if problem == "shape": features = features[:, :24]
    elif problem == "dtype": features = features.double()
    elif problem == "nonfinite": features[0, 0, 0] = float("nan")
    else: features[0, -1, 0] = 1
    with pytest.raises(ValueError, match="subsample"):
        wrapper(features)


@pytest.mark.parametrize("secondary_fails", [False, True])
def test_coarse_failure_routes_original_subsample_values_to_secondary_diagnostic(tmp_path, monkeypatch, secondary_fails):
    import onnx.reference
    audio, projection = audio_models()
    features, frame_mask = torch.zeros(1, 25, 8), mask(25, 17)
    native = prepare.capture_official_audio_stages(audio, projection, features, frame_mask)
    actual = tuple(value.clone() for value in native)
    actual[0].add_(0.01)
    actual[-1].add_(0.03)
    class Evaluator:
        def __init__(self, path): pass
        def run(self, *args): return [value.numpy() for value in actual]
    def save(wrapper, features, path): path.write_bytes(b"coarse-diagnostic-test-only")
    calls = []
    def probe(*args):
        calls.append(args)
        if secondary_fails: raise RuntimeError("secondary subsample failed")
        return dict(localization_validated=False, encoders_validated=False)
    monkeypatch.setattr(onnx.reference, "ReferenceEvaluator", Evaluator)
    monkeypatch.setattr(prepare, "export_audio_native_onnx", save)
    monkeypatch.setattr(prepare, "diagnose_audio_onnx_subsample", probe)
    report = prepare.diagnose_audio_onnx_stages(
        audio, projection, features, frame_mask, tmp_path, actual[-1], official_reference=native[-1])
    assert len(calls) == 1, "The first coarse subsample failure must trigger the selected operator trace"
    assert torch.equal(calls[0][-2], actual[0]) and torch.equal(calls[0][-1], native[0])
    assert report["encoders_validated"] is False and report["end_to_end_validated"] is False
    if secondary_fails:
        assert "secondary subsample failed" in report["subsample_diagnostics_error"]
    else:
        assert report["subsample_diagnostics"]["localization_validated"] is False
