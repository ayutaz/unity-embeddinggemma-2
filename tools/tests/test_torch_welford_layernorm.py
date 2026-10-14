import pytest
import torch
import onnx
from onnx.reference import ReferenceEvaluator

from embeddinggemma_tools import export


def required(name):
    value = getattr(export, name, None)
    assert callable(value), f"Faithful FP32 LayerNorm arithmetic missing: {name}"
    return value


def test_fma_emulation_preserves_the_single_rounding_counterexample():
    fma = required("fp32_fma")
    a = torch.tensor([1.0000001192092896, 4097., -4097.])
    b = torch.tensor([0.9999998807907104, 4097., 4097.])
    c = torch.tensor([-1., -16785408., 16785408.])
    expected = (a.double() * b.double() + c.double()).float()
    assert not torch.equal(a * b + c, expected), "Expose the separate-rounding failure"
    assert torch.equal(fma(a, b, c), expected)


@pytest.mark.parametrize("width", [32, 128])
def test_welford8_layernorm_matches_independent_native_statistics_and_keeps_source(width):
    build = required("TorchWelfordLayerNorm")
    source = torch.nn.LayerNorm(width, eps=1e-6, bias=False).float().eval()
    values = torch.randn(3, width, generator=torch.Generator().manual_seed(14)) * 10
    source.weight.data.copy_(torch.linspace(0.8, 1.2, width))
    before = source.weight.detach().clone()
    candidate = build(source)
    actual = candidate(values)
    with torch.no_grad(): expected = source(values)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
    assert candidate.weight is source.weight and torch.equal(source.weight, before)
    assert actual.dtype == torch.float32


@pytest.mark.parametrize("width", [7, 64, 256])
def test_welford_candidate_rejects_unproven_normalized_widths(width):
    with pytest.raises(ValueError, match="32 or 128"):
        required("TorchWelfordLayerNorm")(torch.nn.LayerNorm(width))


def test_fma_counterexamples_cover_cancellation_and_finite_product_range():
    fma = required("fp32_fma")
    generator = torch.Generator().manual_seed(18)
    a = torch.randn(2048, generator=generator) * 4096
    b = torch.randn(2048, generator=generator) / 16
    c = -(a * b)
    expected = (a.double() * b.double() + c.double()).float()
    assert torch.equal(fma(a, b, c), expected)


@pytest.mark.parametrize("width", [32, 128])
def test_welford_export_uses_only_standard_fp32_ops_and_matches_native(tmp_path, width):
    build = required("TorchWelfordLayerNorm")
    source = torch.nn.LayerNorm(width, eps=1e-6, bias=False).float().eval()
    values = torch.randn(3, width, generator=torch.Generator().manual_seed(14)) * 10
    path = tmp_path / f"welford{width}.onnx"
    export.export_audio_native_onnx(build(source), values, path)
    graph = onnx.load(path)
    onnx.checker.check_model(graph)
    assert all(n.domain in ("", "ai.onnx") for n in graph.graph.node)
    assert all(w.data_type != onnx.TensorProto.DOUBLE for w in graph.graph.initializer)
    actual = ReferenceEvaluator(graph).run(None, {"input": values.numpy()})[0]
    with torch.no_grad(): expected = source(values).numpy()
    __import__('numpy').testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
