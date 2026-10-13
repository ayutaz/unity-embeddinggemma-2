import json
import inspect

import pytest
import torch

from embeddinggemma_tools import export, vision_diagnostics as diagnostics
from test_vision import positions, vision_models


def required(name):
    value = getattr(diagnostics, name, None)
    assert value is not None, f"vision diagnostic contract is missing: {name}"
    return value


def native_trace(vision, projection, pixels, coords):
    # Oracle uses the official full model, not the diagnostic wrapper's arithmetic.
    modules = [vision.patch_embedder, vision.encoder.rotary_emb, *vision.encoder.layers, vision,
               projection.embedding_pre_projection_norm, projection]
    values, handles = [], []
    for index, module in enumerate(modules):
        def capture(module, args, output, index=index):
            if index == 1:
                values.extend(value.detach().clone() for value in output)
                return
            value = output.last_hidden_state if index == len(modules) - 3 else output
            values.append(value.detach().clone())
        handles.append(module.register_forward_hook(capture))
    try:
        with torch.no_grad():
            projection(vision(pixel_values=pixels, pixel_position_ids=coords).last_hidden_state)
    finally:
        for handle in handles:
            handle.remove()
    return tuple(values)


@pytest.mark.parametrize("width,standardize", [(3, False), (6, True)])
def test_saved_stages_match_independent_hooks_and_keep_original_final(tmp_path, width, standardize):
    factory = required("VisionEncoderStagesExport")
    vision, projection = vision_models(standardize=standardize)
    if standardize:
        vision.std_bias.copy_(torch.randn(16)); vision.std_scale.copy_(torch.rand(16) + 0.5)
    coords, pixels = positions(width=width), torch.rand(1, 18, 12)
    instance = factory(vision, projection, coords)
    assert instance.stage_names == ("patch_embeddings", "rotary_cos", "rotary_sin", "layer_0", "vision_features",
                                    "pre_projection_norm", "projected_features")
    traced, ordinary = tmp_path / "trace.pt2", tmp_path / "ordinary.pt2"
    export.export_vision_pt2(instance, pixels, traced)
    export.export_vision_pt2(export.VisionEncoderExport(vision, projection, coords), pixels, ordinary)
    changed = pixels * 0.6
    actual = torch.export.load(traced).module()(changed)
    assert len(actual) == len(instance.stage_names)
    for stage, expected in zip(actual, native_trace(vision, projection, changed, coords), strict=True):
        torch.testing.assert_close(stage, expected, rtol=1e-5, atol=1e-6)
    assert torch.equal(actual[-1], torch.export.load(ordinary).module()(changed))


@pytest.mark.parametrize("fail", [False, True])
def test_capture_removes_only_its_hooks_on_success_and_failure(monkeypatch, fail):
    capture = required("capture_official_vision_stages")
    vision, projection = vision_models()
    coords, pixels = positions(), torch.rand(1, 18, 12)
    modules = [vision.patch_embedder, *vision.encoder.layers, vision,
               projection.embedding_pre_projection_norm, projection]
    external = vision.encoder.layers[0].register_forward_hook(lambda *args: None)
    before = [set(module._forward_hooks) for module in modules]
    try:
        if fail:
            def broken(*args, **kwargs):
                raise RuntimeError("native vision capture failed")
            monkeypatch.setattr(vision.encoder.layers[0], "forward", broken)
            with pytest.raises(RuntimeError, match="native vision capture failed"):
                capture(vision, projection, pixels, coords)
        else:
            expected = native_trace(vision, projection, pixels, coords)
            for actual, expected_value in zip(capture(vision, projection, pixels, coords), expected, strict=True):
                assert torch.equal(actual, expected_value)
        assert [set(module._forward_hooks) for module in modules] == before
    finally:
        external.remove()


def test_fixed_tolerance_records_first_micro_difference_and_first_failure():
    compare = required("compare_vision_stages")
    expected = (torch.ones(2), torch.ones(2), torch.ones(2))
    actual = (expected[0].clone(), expected[1] + 1e-6, expected[2] + 0.01)
    result = compare(("patch", "layer_0", "projector"), expected, actual)
    assert result["success"] is False
    assert result["first_nonzero_stage"] == "layer_0"
    assert result["first_divergent_stage"] == "projector"
    assert result["stages"][1]["success"] is True
    assert result["stages"][2]["maximum_tolerance_ratio"] > 1
    assert result["feature_atol"] == 1e-5 and result["feature_rtol"] == 1e-4


@pytest.mark.parametrize("actual", [(torch.ones(3),), (torch.tensor([float("nan")]),), ()])
def test_invalid_stage_count_shape_or_nonfinite_output_cannot_pass(actual):
    compare = required("compare_vision_stages")
    with pytest.raises(ValueError, match="stage"):
        compare(("patch",), (torch.ones(1),), actual)


def test_reference_manifest_has_native_binaries_hashes_and_original_context(tmp_path):
    prepare = required("prepare_vision_diagnostics")
    vision, projection = vision_models()
    coords, pixels = positions(width=3), torch.rand(1, 18, 12)
    ordinary = tmp_path / "ordinary.pt2"
    export.export_vision_pt2(export.VisionEncoderExport(vision, projection, coords), pixels, ordinary)
    result = prepare(vision, projection, pixels, coords, tmp_path / "out", ordinary,
                     source_commit="a" * 40, revision="b" * 40)
    assert result["success"] is True
    assert result["stage"] == "vision_encoder_diagnostics_only"
    assert result["unity_validated"] is False and result["end_to_end_validated"] is False
    assert result["trace_matches_original_saved"] is True
    assert result["localization_validated"] is False  # Requires Sentis trace vs original as well.
    assert "cpu_math_environment" in result["metadata"], "Diagnostic native math provenance cannot be omitted."
    assert result["metadata"]["cpu_math_environment"]["intraop_threads"] > 0
    assert len(result["stages"]) == 7
    for stage in result["stages"]:
        assert (tmp_path / "out" / stage["reference_file"]).stat().st_size == 4 * torch.tensor(stage["shape"]).prod()
    assert "vision-stages.pt2" in result["files"] and "pixels-fp32.bin" in result["files"]
    assert json.loads((tmp_path / "out/vision-diagnostics.json").read_text()) == result


def test_changed_original_output_prevents_localization_and_success(tmp_path):
    prepare = required("prepare_vision_diagnostics")
    vision, projection = vision_models()
    coords, pixels = positions(width=3), torch.rand(1, 18, 12)
    class Wrong(torch.nn.Module):
        def forward(self, pixels):
            return torch.ones(1, 16) * 999
    ordinary = tmp_path / "wrong.pt2"
    torch.export.save(torch.export.export(Wrong(), (pixels,)), ordinary)
    result = prepare(vision, projection, pixels, coords, tmp_path / "out", ordinary,
                     source_commit="a" * 40, revision="b" * 40)
    assert result["success"] is False
    assert result["trace_matches_original_saved"] is False
    assert result["localization_validated"] is False
    assert result["trace_vs_original_maximum_absolute_error"] > 1


def test_both_profile_inputs_get_independent_native_references_and_context_checks(tmp_path):
    prepare = required("prepare_vision_diagnostics")
    assert "additional_pixels" in inspect.signature(prepare).parameters, "The second real profile input cannot be omitted."
    vision, projection = vision_models()
    coords, pixels = positions(width=3), torch.rand(1, 18, 12)
    ordinary = tmp_path / "ordinary.pt2"
    export.export_vision_pt2(export.VisionEncoderExport(vision, projection, coords), pixels, ordinary)
    report = prepare(vision, projection, pixels, coords, tmp_path / "out", ordinary,
                     additional_pixels=(pixels * 0.6,), source_commit="a" * 40, revision="b" * 40)
    assert report["metadata"]["case_count"] == 2
    assert len(report["cases"]) == 2 and report["success"] is True
    assert all(case["trace_matches_original_saved"] for case in report["cases"])
    assert report["cases"][0]["pixels_file"] != report["cases"][1]["pixels_file"]
    for case, value in zip(report["cases"], (pixels, pixels * 0.6), strict=True):
        expected = native_trace(vision, projection, value, coords)
        for stage, native in zip(case["stages"], expected, strict=True):
            import numpy as np
            actual = np.fromfile(tmp_path / "out" / stage["reference_file"], dtype="<f4").reshape(stage["shape"])
            assert torch.equal(torch.from_numpy(actual), native)


@pytest.mark.parametrize("field,value", [("source_commit", "local"), ("revision", "main")])
def test_mutable_source_or_model_cannot_leave_stale_success(tmp_path, field, value):
    prepare = required("prepare_vision_diagnostics")
    output = tmp_path / "out"; output.mkdir()
    stale = output / "vision-diagnostics.json"; stale.write_text('{"success":true}')
    arguments = dict(source_commit="a" * 40, revision="b" * 40); arguments[field] = value
    with pytest.raises(ValueError, match="immutable"):
        prepare(None, None, None, None, output, tmp_path / "ordinary.pt2", **arguments)
    assert not stale.exists()
