import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator
import pytest
import torch
from embeddinggemma_tools import export


def fused(a, b, c):
    return (a.double() * b.double() + c.double()).float()


def oracle(values, weight, *, fuse_mean):
    width = values.shape[-1]
    count = width // 8
    mean = torch.zeros_like(values[..., :8])
    moment = torch.zeros_like(mean)
    for index in range(count):
        item = values[..., index*8:(index+1)*8]
        delta = item - mean
        mean = fused(values.new_tensor(1 / (index+1)), delta, mean)
        moment = fused(delta, item-mean, moment)
    total_mean = torch.zeros_like(mean[..., :1])
    total_moment = torch.zeros_like(total_mean)
    for lane in range(8):
        previous = lane * count
        delta = mean[..., lane:lane+1] - total_mean
        coefficient = values.new_tensor(count / (previous+count))
        total_mean = fused(coefficient, delta, total_mean) if fuse_mean else total_mean + coefficient * delta
        total_moment = total_moment + (moment[..., lane:lane+1] + delta * delta * coefficient * previous)
    inverse = export.fp32_refined_rstd(total_moment / width + 1e-6)
    return (values-total_mean) * inverse * weight


@pytest.mark.parametrize('width', [32, 128])
def test_opt_in_scalar_mean_fusion_matches_independent_rounding(width):
    values = torch.randn(3, width, generator=torch.Generator().manual_seed(120))
    source = torch.nn.LayerNorm(width, eps=1e-6, bias=False)
    before = source.weight.detach().clone()
    model = export.TorchWelfordLayerNorm(source, refine_inverse=True, fuse_scalar_mean=True)
    expected = oracle(values, source.weight, fuse_mean=True)
    assert not torch.equal(expected, oracle(values, source.weight, fuse_mean=False))
    assert torch.equal(model(values), expected)
    assert model.weight is source.weight
    assert torch.equal(source.weight, before)


def test_scalar_fusion_default_preserves_previous_eight_lane_order():
    values = torch.randn(3, 32, generator=torch.Generator().manual_seed(120))
    source = torch.nn.LayerNorm(32, eps=1e-6, bias=False)
    implicit = export.TorchWelfordLayerNorm(source, refine_inverse=True)
    explicit = export.TorchWelfordLayerNorm(source, refine_inverse=True, fuse_scalar_mean=False)
    assert torch.equal(implicit(values), explicit(values))
    assert torch.equal(implicit(values), oracle(values, source.weight, fuse_mean=False))


def test_scalar_fused_export_is_standard_fp32_and_preserves_rounding(tmp_path):
    values = torch.randn(2, 32, generator=torch.Generator().manual_seed(120))
    source = torch.nn.LayerNorm(32, eps=1e-6, bias=False)
    model = export.TorchWelfordLayerNorm(source, refine_inverse=True, fuse_scalar_mean=True)
    path = tmp_path / 'scalar-fused.onnx'
    export.export_audio_native_onnx(model, values, path)
    graph = onnx.load(path)
    onnx.checker.check_model(graph)
    assert all(n.domain in ('', 'ai.onnx') for n in graph.graph.node)
    assert all(i.data_type != onnx.TensorProto.DOUBLE for i in graph.graph.initializer)
    actual = ReferenceEvaluator(graph).run(None, {'input': values.numpy()})[0]
    np.testing.assert_array_equal(actual, oracle(values, source.weight, fuse_mean=True).detach().numpy())
