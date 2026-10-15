import operator
from pathlib import Path

import numpy as np
import onnx
import pytest
import torch

from embeddinggemma_tools import frozen_audio_prefix_onnx as probe


def rms_program(width=1024, epsilon=1e-6, squared=True, transposed=False):
    root = torch.nn.Module()
    root.register_buffer("norm_scale", torch.linspace(0.75, 1.25, 128))
    root.register_buffer("rms_scale", torch.linspace(0.9, 1.1, width))
    graph = torch.fx.Graph()
    values = graph.placeholder("values")
    scale = graph.get_attr("norm_scale")
    normal = graph.call_function(torch.ops.aten.native_layer_norm.default, (values, [128], scale, None, 1e-6))
    hidden = graph.call_function(operator.getitem, (normal, 0))
    hidden = (graph.call_function(torch.ops.aten.repeat.default, (hidden, [1, 1, width // 128])) if width >= 128
              else graph.call_function(torch.ops.aten.slice.Tensor, (hidden, -1, 0, width)))
    if transposed:
        hidden = graph.call_function(torch.ops.aten.permute.default, (hidden, [0, 2, 1]))
        hidden = graph.call_function(torch.ops.aten.clone.default, (hidden,), {"memory_format": torch.contiguous_format})
        hidden = graph.call_function(torch.ops.aten.permute.default, (hidden, [0, 2, 1]))
    square = graph.call_function(torch.ops.aten.pow.Tensor_Scalar, (hidden, 2)) if squared else hidden
    mean = graph.call_function(torch.ops.aten.mean.dim, (square, [-1], True))
    statistic = graph.call_function(torch.ops.aten.add.Tensor, (mean, epsilon))
    inverse = graph.call_function(torch.ops.aten.pow.Tensor_Scalar, (statistic, -0.5))
    normalized = graph.call_function(torch.ops.aten.mul.Tensor, (hidden, inverse))
    output = graph.call_function(torch.ops.aten.mul.Tensor, (normalized, graph.get_attr("rms_scale")))
    graph.output(output)
    return torch.fx.GraphModule(root, graph), output.name


@pytest.mark.parametrize("width", [1024, 1536])
def test_explicit_rms_candidate_keeps_equation_and_source_and_exports_fp32(tmp_path, width):
    source, boundary = rms_program(width)
    values = torch.randn(1, 3, 128, generator=torch.Generator().manual_seed(71)) * 5
    before_graph, before_weights = str(source.graph), probe.weight_digests(source)
    expected = source(values).detach()
    report = probe.probe_prefix(source, values, expected, tmp_path, boundary=boundary, variants=("refined_mean_moment_rms",))
    assert report["complete"] and report["native_wrapper_control_bits_equal"]
    assert str(source.graph) == before_graph and probe.weight_digests(source) == before_weights
    candidate = probe.PrefixExport(source, "refined_mean_moment_rms")(values).detach().numpy()
    actual = np.fromfile(tmp_path / report["cases"][0]["capture"], dtype="<f4").reshape(candidate.shape)
    np.testing.assert_allclose(actual, candidate, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(actual, expected.numpy(), rtol=1e-5, atol=1e-5)
    assert not report["full_encoder_validated"] and not report["gpu_validated"]
    model = onnx.load(tmp_path / report["cases"][0]["graph"])
    assert all(node.domain in ("", "ai.onnx") for node in model.graph.node)
    assert all(value.data_type != onnx.TensorProto.DOUBLE for value in model.graph.initializer)


def test_fixed_convolution_rms_layout_has_export_parity(tmp_path):
    source, boundary = rms_program(transposed=True)
    values = torch.randn(1, 280, 128, generator=torch.Generator().manual_seed(72))
    expected = source(values).detach()
    report = probe.probe_prefix(source, values, expected, tmp_path, boundary=boundary, variants=("refined_mean_moment_rms",))
    assert report["complete"] and report["cases"][0]["rms_mean_replacements"] == 1
    candidate = probe.PrefixExport(source, "refined_mean_moment_rms")(values).detach().numpy()
    actual = np.fromfile(tmp_path / report["cases"][0]["capture"], dtype="<f4").reshape(candidate.shape)
    np.testing.assert_allclose(actual, candidate, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(actual, expected.numpy(), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("epsilon,squared", [(1e-5, True), (1e-6, False)])
def test_rms_rewrite_rejects_unaudited_mean_or_epsilon(epsilon, squared):
    source, _ = rms_program(epsilon=epsilon, squared=squared)
    with pytest.raises(ValueError, match="RMS"):
        probe.PrefixExport(source, "refined_mean_moment_rms")


def test_rms_candidate_rejects_unqualified_width():
    source, _ = rms_program(width=64)
    with pytest.raises(ValueError, match="RMS"):
        probe.PrefixExport(source, "refined_mean_moment_rms")(torch.zeros(1, 3, 128))


def test_rms_variant_is_explicit_and_not_part_of_default_prefix_variants():
    assert "refined_mean_moment_rms" not in probe.VARIANTS
    assert "refined_mean_moment_rms" in probe.EXPERIMENTAL_VARIANTS
    path = Path(__file__).resolve().parents[2] / ".github/workflows/frozen-audio-prefix-onnx.yml"
    assert "refined_mean_moment_rms" in path.read_text()


@pytest.mark.parametrize("stage,count", [("subsample", 0), ("layer_5", 54), ("layer_11", 108), ("projected_features", 109)])
def test_pinned_stage_requires_complete_rms_coverage(stage, count):
    assert probe.expected_rms_count(stage) == count
