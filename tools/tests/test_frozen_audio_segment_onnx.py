import json
from pathlib import Path

import numpy as np
import onnx
import pytest
import torch

from embeddinggemma_tools import frozen_audio_segment_onnx as probe
from embeddinggemma_tools.frozen_audio_prefix_onnx import FrozenStageReference, exact_difference


def program(*, bypass=False):
    root = torch.nn.Module()
    root.register_buffer("weight", torch.arange(12, dtype=torch.float32).reshape(4, 3) / 13)
    graph = torch.fx.Graph()
    value = graph.placeholder("features")
    cut = graph.call_function(torch.ops.aten.mul.Tensor, (value, 2.0))
    weight = graph.get_attr("weight")
    output = graph.call_function(torch.ops.aten.mm.default, (cut, weight))
    if bypass:
        residual = graph.call_function(torch.ops.aten.sum.dim_IntList, (value, [-1], True))
        output = graph.call_function(torch.ops.aten.add.Tensor, (output, residual))
    graph.output(output)
    return torch.fx.GraphModule(root, graph), cut.name, output.name


def frozen(value, stage):
    return FrozenStageReference(value, dict(stage=stage, expected_sha256="synthetic-test-only"))


def test_segment_replaces_only_cut_and_borrows_original_weights():
    source, cut, end = program()
    before = str(source.graph)
    segment = probe.extract_segment(source, cut, end)
    x = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    assert segment.weight is source.weight
    assert exact_difference(segment(x * 2), source(x)) == 0
    assert len([n for n in segment.graph.nodes if n.op == "placeholder"]) == 1
    assert not any(n.target == torch.ops.aten.mul.Tensor for n in segment.graph.nodes)
    assert str(source.graph) == before


@pytest.mark.parametrize("cut,end", [("missing", "mm_default"), ("mul_tensor", "missing"),
                                      ("features", "mm_default"), ("mm_default", "mul_tensor"),
                                      ("mul_tensor", "mul_tensor")])
def test_segment_rejects_invalid_or_reversed_boundaries(cut, end):
    source, _, _ = program()
    with pytest.raises(ValueError, match="boundary"):
        probe.extract_segment(source, cut, end)


def test_segment_rejects_uncut_original_input_dependencies():
    source, cut, end = program(bypass=True)
    with pytest.raises(ValueError, match="input|dependency"):
        probe.extract_segment(source, cut, end)


def test_segment_exports_standard_fp32_and_records_frozen_input_scope(tmp_path):
    source, cut, end = program()
    x = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    expected = source(x)
    report = probe.probe_segment(source, frozen(x * 2, "subsample"),
                                 frozen(expected, "projected_features"), tmp_path,
                                 from_boundary=cut, to_boundary=end)
    assert report["complete"] and report["native_wrapper_control_bits_equal"]
    assert report["native_segment_vs_reference_metrics"]["exact_difference_count"] == 0
    assert report["scope"] == "audited_frozen_input_audio_segment_onnx_diagnosis_only"
    assert report["input_provenance"]["stage"] == "subsample"
    assert report["target_provenance"]["stage"] == "projected_features"
    assert not report["full_encoder_validated"] and not report["gpu_validated"]
    assert not report["end_to_end_validated"]
    assert report["cases"][0]["metrics"]["mismatched_elements"] == 0
    assert report["cases"][0]["coefficients_unchanged"]
    model = onnx.load(tmp_path / report["cases"][0]["graph"])
    onnx.checker.check_model(model)
    assert all(n.domain in ("", "ai.onnx") for n in model.graph.node)
    assert all(t.data_type != onnx.TensorProto.DOUBLE for t in model.graph.initializer)
    saved = np.fromfile(tmp_path / "reference-output-fp32.bin", dtype="<f4")
    np.testing.assert_array_equal(saved, expected.numpy().ravel())


def test_segment_keeps_native_and_onnx_failures_visible(tmp_path):
    source, cut, end = program()
    x = torch.zeros(2, 4)
    report = probe.probe_segment(source, frozen(x, "subsample"), frozen(torch.ones(2, 3), "projected_features"),
                                 tmp_path, from_boundary=cut, to_boundary=end)
    assert report["complete"]
    assert report["native_segment_vs_reference_metrics"]["mismatched_elements"] == 6
    assert report["cases"][0]["metrics"]["mismatched_elements"] == 6
    assert not report["full_encoder_validated"]


def test_segment_refuses_overwriting_and_unaudited_reference(tmp_path):
    source, cut, end = program()
    x = torch.zeros(2, 4)
    with pytest.raises(ValueError, match="audited"):
        probe.probe_segment(source, x, frozen(torch.zeros(2, 3), "projected_features"), tmp_path,
                            from_boundary=cut, to_boundary=end)
    receipt = tmp_path / "frozen-audio-segment-onnx.json"
    receipt.write_text("original evidence")
    with pytest.raises(ValueError, match="overwrite"):
        probe.probe_segment(source, frozen(x, "subsample"), frozen(torch.zeros(2, 3), "projected_features"),
                            tmp_path, from_boundary=cut, to_boundary=end)
    assert receipt.read_text() == "original evidence"


def test_cli_loads_both_audited_stages_and_passes_original_program(tmp_path, monkeypatch):
    source, cut, end = program()
    x = torch.zeros(2, 4)
    monkeypatch.setenv("GITHUB_SHA", "a" * 40)
    monkeypatch.setattr(probe, "load_frozen_input", lambda p: (source, x, torch.ones(2, 3), {}))
    monkeypatch.setattr(probe, "boundary_for_stage", lambda p, stage: cut if stage == "subsample" else end)
    loaded, called = [], []
    def load(directory, stage):
        loaded.append(stage)
        return frozen(x if stage == "subsample" else torch.zeros(2, 3), stage)
    monkeypatch.setattr(probe, "load_frozen_stage_reference", load)
    def run(*args, **kwargs):
        called.append((args, kwargs))
        return dict(complete=True, cases=[])
    monkeypatch.setattr(probe, "probe_segment", run)
    assert probe.main(["--input", str(tmp_path), "--output", str(tmp_path / "out"),
                       "--frozen-stage-source", str(tmp_path), "--from-stage", "subsample",
                       "--stage", "projected_features"]) == 0
    assert loaded == ["subsample", "projected_features"]
    assert called[0][0][0] is source
    assert called[0][1]["variants"] == ("native",)


def test_workflow_segment_is_explicit_and_downloads_both_audited_boundaries():
    path = Path(__file__).resolve().parents[2] / ".github/workflows/frozen-audio-prefix-onnx.yml"
    text = path.read_text()
    assert "from_stage:" in text and "default: none" in text
    assert 'PROBE_FROM_STAGE' in text and 'frozen_audio_segment_onnx' in text
    assert '--from-stage' in text and 'source_index' in text
    assert 'timeout-minutes: 20' in text


def test_segment_rejects_native_layer_norm_instead_of_silently_skipping_audit():
    root = torch.nn.Module()
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    norm = graph.call_function(torch.ops.aten.native_layer_norm.default, (x, [32], None, None, 1e-6))
    graph.output(norm)
    segment = torch.fx.GraphModule(root, graph)
    with pytest.raises(ValueError, match="normalization"):
        probe.SegmentExport(segment, "native")


def test_segment_rms_variant_audits_real_nodes_and_exports_the_original_equation(tmp_path):
    class RMS(torch.nn.Module):
        def forward(self, x):
            return x * (x.pow(2).mean(-1, keepdim=True) + 1e-6).pow(-0.5)
    x = torch.randn(2, 1024, generator=torch.Generator().manual_seed(27))
    exported = torch.export.export(RMS(), (x,)).run_decompositions({}).module()
    segment = torch.fx.GraphModule(exported, exported.graph)
    # square is represented as pow(2) in the pinned saved program.
    square = next(n for n in segment.graph.nodes if n.target == torch.ops.aten.pow.Tensor_Scalar)
    assert square.args[1] == 2
    wrapper = probe.SegmentExport(segment, "refined_mean_moment_rms")
    assert len(wrapper.rms_means) == len(wrapper.rms_inverses) == 1
    y = wrapper(x)
    torch.testing.assert_close(y, RMS()(x), atol=1e-5, rtol=1e-4)
    path = tmp_path / "rms.onnx"
    probe.export_audio_native_onnx(wrapper, x, path)
    actual = probe.ReferenceEvaluator(str(path)).run(None, {"input": x.numpy()})[0]
    np.testing.assert_allclose(actual, y.detach().numpy(), atol=1e-5, rtol=1e-4)


def test_segment_silu_variant_is_explicit_and_tracks_original_silu_nodes(tmp_path):
    root = torch.nn.Module()
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    cut = graph.call_function(torch.ops.aten.mul.Tensor, (x, 2.0))
    activated = graph.call_function(torch.ops.aten.silu.default, (cut,))
    graph.output(activated)
    source = torch.fx.GraphModule(root, graph)
    values = torch.linspace(-10, 10, 128).reshape(2, 64)
    before = str(source.graph)
    report = probe.probe_segment(source, frozen(values, "subsample"),
        frozen(torch.nn.functional.silu(values), "layer_0"), tmp_path,
        from_boundary=cut.name, to_boundary=activated.name, variants=("native_silu",))
    assert probe.VARIANTS[0] == "native"
    assert report["cases"][0]["silu_replacements"] == 1
    assert report["cases"][0]["metrics"]["exact_difference_count"] == 0
    assert report["native_wrapper_control_bits_equal"]
    assert str(source.graph) == before


def test_segment_silu_candidate_rejects_a_segment_without_silu():
    source, cut, end = program()
    with pytest.raises(ValueError, match="SiLU"):
        probe.SegmentExport(probe.extract_segment(source, cut, end), "native_silu")
