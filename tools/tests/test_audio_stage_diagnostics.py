"""Stage diagnostics must retain independent references and the original failure."""

import json

import pytest
import torch

from embeddinggemma_tools import export, prepare
from test_audio_encoder import audio_models, mask
from test_audio_encoder_prepare import fixture


def required(owner, name):
    value = getattr(owner, name, None)
    assert value is not None, f"audio stage diagnostic contract is missing: {name}"
    return value


def official_stages(audio, projection, features, frame_mask):
    # Independent test oracle, never the export wrapper's forward computation.
    values, handles = [], []
    modules = [audio.subsample_conv_projection, *audio.layers, audio.output_proj]
    for index, module in enumerate(modules):
        def capture(module, args, output, index=index):
            values.append((output[0] if index == 0 else output).detach().clone())
        handles.append(module.register_forward_hook(capture))
    try:
        with torch.no_grad():
            result = audio(input_features=features, attention_mask=frame_mask.bool())
            values.append(projection(result.last_hidden_state)[result.attention_mask].detach().clone())
    finally:
        for handle in handles:
            handle.remove()
    return tuple(values)


def test_trace_export_has_ordered_official_stages_and_saved_changed_input(tmp_path):
    trace = required(export, "AudioEncoderStagesExport")
    audio, projection = audio_models()
    frame_mask = mask(25, 17)
    features = torch.randn(1, 25, 8) * frame_mask[:, :, None]
    wrapper = trace(audio, projection, frame_mask)
    assert wrapper.stage_names == ("subsample", "layer_0", "layer_1", "output_projection", "projected_features")
    expected = official_stages(audio, projection, features, frame_mask)
    actual = wrapper(features)
    assert isinstance(actual, tuple) and len(actual) == len(expected)
    for saved, native in zip(actual, expected, strict=True):
        torch.testing.assert_close(saved, native, rtol=1e-5, atol=1e-6)
    path = tmp_path / "trace.pt2"
    export.export_audio_pt2(wrapper, features, path)
    changed = features * 0.6
    actual = torch.export.load(path).module()(changed)
    expected = official_stages(audio, projection, changed, frame_mask)
    for saved, native in zip(actual, expected, strict=True):
        torch.testing.assert_close(saved, native, rtol=1e-5, atol=1e-6)
    normal = export.AudioEncoderExport(audio, projection, frame_mask)(changed)
    assert isinstance(normal, torch.Tensor)
    torch.testing.assert_close(normal, actual[-1], rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("fail", [False, True])
def test_official_capture_removes_only_its_hooks_even_on_exception(monkeypatch, fail):
    capture = required(prepare, "capture_official_audio_stages")
    audio, projection = audio_models()
    frame_mask, features = mask(), torch.randn(1, 25, 8)
    external = audio.layers[0].register_forward_hook(lambda *args: None)
    modules = [audio.subsample_conv_projection, *audio.layers, audio.output_proj]
    before = [set(module._forward_hooks) for module in modules]
    expected = official_stages(audio, projection, features, frame_mask)
    if fail:
        def broken(*args, **kwargs):
            raise RuntimeError("official probe failure")
        monkeypatch.setattr(audio.layers[1], "forward", broken)
        with pytest.raises(RuntimeError, match="official probe failure"):
            capture(audio, projection, features, frame_mask)
    else:
        actual = capture(audio, projection, features, frame_mask)
        for value, native in zip(actual, expected, strict=True):
            torch.testing.assert_close(value, native, rtol=0, atol=0)
    assert [set(module._forward_hooks) for module in modules] == before
    external.remove()


def test_stage_comparison_identifies_first_numeric_divergence():
    compare = required(prepare, "compare_audio_stage_outputs")
    names = ("subsample", "layer_0", "projected_features")
    expected = (torch.ones(1, 2, 4), torch.ones(1, 2, 4), torch.ones(2, 4))
    actual = (expected[0].clone(), expected[1] + 0.01, expected[2] + 0.02)
    result = compare(names, expected, actual)
    assert result["first_divergent_stage"] == "layer_0"
    assert [row["success"] for row in result["stages"]] == [True, False, False]
    assert result["stages"][1]["mismatched_elements"] == 8
    assert result["stages"][1]["maximum_tolerance_ratio"] > 1


@pytest.mark.parametrize("problem", ["count", "shape", "nonfinite", "dtype", "names"])
def test_stage_comparison_rejects_incomplete_or_invalid_trace(problem):
    compare = required(prepare, "compare_audio_stage_outputs")
    names, expected, actual = ("subsample",), (torch.ones(1, 2, 4),), (torch.ones(1, 2, 4),)
    if problem == "count":
        actual = ()
    elif problem == "shape":
        actual = (torch.ones(1, 1, 4),)
    elif problem == "nonfinite":
        actual = (torch.full((1, 2, 4), float("nan")),)
    elif problem == "dtype":
        actual = (actual[0].double(),)
    else:
        names = ("subsample", "subsample")
    with pytest.raises(ValueError, match="stage"):
        compare(names, expected, actual)


@pytest.mark.parametrize("diagnostic_fails", [False, True])
def test_prepare_keeps_original_numeric_failure_when_stage_diagnostics_run(tmp_path, monkeypatch, diagnostic_fails):
    snapshot, inputs = fixture(tmp_path, monkeypatch)
    class Wrong(torch.nn.Module):
        def forward(self, features):
            return torch.zeros(2, 16)
    def corrupt(wrapper, features, path):
        torch.export.save(torch.export.export(Wrong(), (features,)), path)
    calls = []
    def diagnose(*args, **kwargs):
        calls.append((args, kwargs))
        if diagnostic_fails:
            raise RuntimeError("stage probe failed")
        return {"first_divergent_stage": "layer_0", "stages": [{"name": "layer_0", "success": False}],
                "trace_matches_failed_graph": True}
    monkeypatch.setattr(prepare, "export_audio_pt2", corrupt)
    monkeypatch.setattr(prepare, "diagnose_audio_stages", diagnose, raising=False)
    output = tmp_path / "out"
    with pytest.raises(ValueError, match="audio encoder export mismatch: silence"):
        prepare.prepare_audio_encoder(snapshot, inputs, output, source_commit="b" * 40)
    assert len(calls) == 1, "a numeric failure must invoke stage diagnostics exactly once"
    failure = json.loads((output / "audio-encoder-failure.json").read_text())
    assert failure["success"] is False and failure["case_id"] == "silence"
    assert failure["saved_vs_eager_maximum_absolute_error"] > 0
    assert not (output / "audio-encoder-validation.json").exists()
    if diagnostic_fails:
        assert "stage probe failed" in failure["stage_diagnostics_error"]
    else:
        assert failure["stage_diagnostics"]["first_divergent_stage"] == "layer_0"


@pytest.mark.parametrize("same_final", [True, False])
def test_real_small_stage_diagnostic_records_trace_and_original_graph_agreement(tmp_path, same_final):
    diagnose = required(prepare, "diagnose_audio_stages")
    audio, projection = audio_models()
    features, frame_mask = torch.randn(1, 25, 8), mask()
    graph = tmp_path / "normal.pt2"
    export.export_audio_pt2(export.AudioEncoderExport(audio, projection, frame_mask), features, graph)
    original = torch.export.load(graph).module()(features)
    if not same_final:
        original = original + 0.1
    report = diagnose(audio, projection, features, frame_mask, tmp_path, original)
    assert report["first_divergent_stage"] is None
    assert len(report["stages"]) == 5 and all(row["success"] for row in report["stages"])
    assert report["trace_matches_failed_graph"] is same_final
    assert (report["trace_vs_failed_graph_maximum_absolute_error"] == 0) is same_final
    assert report["graph_sha256"] == __import__("hashlib").sha256((tmp_path / "audio-stages.pt2").read_bytes()).hexdigest()
    assert report["stage"] == "audio_encoder_stage_diagnostics_only"


def test_original_failure_is_persisted_before_starting_expensive_diagnostics(tmp_path, monkeypatch):
    snapshot, inputs = fixture(tmp_path, monkeypatch)
    class Wrong(torch.nn.Module):
        def forward(self, features):
            return torch.zeros(2, 16)
    def corrupt(wrapper, features, path):
        torch.export.save(torch.export.export(Wrong(), (features,)), path)
    persisted, output = [], tmp_path / "out"
    def diagnose(*args):
        path = output / "audio-encoder-failure.json"
        persisted.append(path.exists())
        if path.exists():
            failure = json.loads(path.read_text())
            assert failure["success"] is False and failure["case_id"] == "silence"
            assert (output / "failed-official-features-fp32.bin").exists()
            assert (output / "failed-saved-features-fp32.bin").exists()
        raise RuntimeError("diagnostic interrupted after original receipt")
    monkeypatch.setattr(prepare, "export_audio_pt2", corrupt)
    monkeypatch.setattr(prepare, "diagnose_audio_stages", diagnose, raising=False)
    with pytest.raises(ValueError, match="audio encoder export mismatch: silence"):
        prepare.prepare_audio_encoder(snapshot, inputs, output, source_commit="b" * 40)
    assert persisted == [True], "Original numeric receipt must survive a killed diagnostic process"
