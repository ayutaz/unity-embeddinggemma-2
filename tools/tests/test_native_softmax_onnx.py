import numpy as np
import onnx
import pytest
import torch
from onnx.reference import ReferenceEvaluator

from embeddinggemma_tools.export import export_audio_native_onnx
from embeddinggemma_tools.streaming_reference import run_flat_reference


def candidate():
    from embeddinggemma_tools import native_softmax_onnx
    return native_softmax_onnx.fp32_native_softmax24


def oracle(values):
    x = values.numpy().astype('f8')
    exponential = np.exp(x - x.max(axis=-1, keepdims=True)).astype('f4')
    lanes = (exponential[..., :8] + exponential[..., 8:16]) + exponential[..., 16:]
    left = (lanes[..., 0] + lanes[..., 4]) + (lanes[..., 2] + lanes[..., 6])
    right = (lanes[..., 1] + lanes[..., 5]) + (lanes[..., 3] + lanes[..., 7])
    inverse = np.float32(1) / (left + right)
    return exponential * inverse[..., None]


@pytest.mark.parametrize('shape', [(8, 24), (1, 2, 3, 4, 24)])
def test_fixed_softmax_preserves_masks_normalization_and_eight_lane_reduction(shape):
    values = torch.randn(shape, generator=torch.Generator().manual_seed(351)) * 8
    values[..., 3] = -float('inf')
    values[..., 9] = -1e9
    actual = candidate()(values).numpy()
    np.testing.assert_allclose(actual, oracle(values), atol=1e-7, rtol=5e-7)
    np.testing.assert_allclose(actual.sum(axis=-1, dtype=np.float64), 1, atol=2e-7, rtol=0)
    assert np.isfinite(actual).all() and np.all(actual >= 0)
    assert np.count_nonzero(actual[..., [3, 9]]) == 0


def test_fixed_softmax_actual_standard_fp32_onnx_matches_eager_and_masked_tails(tmp_path):
    function = candidate()
    class Module(torch.nn.Module):
        def forward(self, values):
            return function(values)
    values = torch.linspace(-130, 8, 192).reshape(8, 24)
    values[0, 2] = -float('inf')
    values[1, 4] = -torch.finfo(torch.float32).max
    path = tmp_path / 'softmax.onnx'
    export_audio_native_onnx(Module(), values, path)
    model = onnx.load(path)
    assert all(n.domain in ('', 'ai.onnx') for n in model.graph.node)
    assert not any(n.op_type == 'Softmax' for n in model.graph.node)
    assert all(t.data_type != onnx.TensorProto.DOUBLE for t in model.graph.initializer)
    for node in model.graph.node:
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.TENSOR:
                assert attribute.t.data_type != onnx.TensorProto.DOUBLE
    actual = run_flat_reference(ReferenceEvaluator(model), {'input': values.numpy()})[0]
    np.testing.assert_array_equal(actual.view('u4'), function(values).numpy().view('u4'))
    np.testing.assert_allclose(actual, oracle(values), atol=1e-7, rtol=5e-7)


@pytest.mark.parametrize('problem', ['dtype', 'width', 'empty', 'nan', 'positive_infinity', 'all_masked'])
def test_fixed_softmax_rejects_unqualified_contracts(problem):
    values = torch.zeros(2, 24)
    if problem == 'dtype': values = values.double()
    if problem == 'width': values = values[:, :16]
    if problem == 'empty': values = values[:0]
    if problem == 'nan': values[0, 0] = float('nan')
    if problem == 'positive_infinity': values[0, 0] = float('inf')
    if problem == 'all_masked': values[0] = -float('inf')
    with pytest.raises(ValueError):
        candidate()(values)
