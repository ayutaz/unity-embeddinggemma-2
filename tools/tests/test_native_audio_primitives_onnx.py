import numpy as np
import onnx
import pytest
import torch
from onnx.reference import ReferenceEvaluator

from embeddinggemma_tools.export import export_audio_native_onnx
from embeddinggemma_tools.streaming_reference import run_flat_reference


def candidate(name):
    from embeddinggemma_tools import native_audio_primitives_onnx
    return getattr(native_audio_primitives_onnx, name)


def test_sigmoid_companion_keeps_decomposed_glu_gate_semantics():
    function = candidate('fp32_native_sigmoid')
    values = torch.linspace(-80, 80, 256).reshape(2, 128)
    actual = function(values)
    torch.testing.assert_close(actual, torch.sigmoid(values), atol=1e-7, rtol=5e-7)
    assert torch.isfinite(actual).all() and torch.all((actual >= 0) & (actual <= 1))


def test_glu_keeps_reciprocal_rounding_and_handles_original_mathematical_tails(tmp_path):
    function = candidate('fp32_native_glu')
    values = torch.randn(3, 64, generator=torch.Generator().manual_seed(361)) * 6
    values[0, :2] = torch.tensor([-0., 0.])
    class Module(torch.nn.Module):
        def forward(self, values): return function(values)
    graph = tmp_path / 'glu.onnx'
    export_audio_native_onnx(Module(), values, graph)
    model = onnx.load(graph)
    assert all(n.domain in ('', 'ai.onnx') for n in model.graph.node)
    actual = run_flat_reference(ReferenceEvaluator(model), {'input': values.numpy()})[0]
    np.testing.assert_array_equal(actual.view('u4'), function(values).numpy().view('u4'))
    torch.testing.assert_close(torch.from_numpy(actual), torch.nn.functional.glu(values), atol=1e-6, rtol=1e-6)
    assert actual.view('u4')[0, 0] == np.uint32(0x80000000)
    tails = values.clone()
    tails[0, 32:36] = torch.tensor([-1e10, -100., 100., 1e10])
    with np.errstate(over='ignore'):
        actual = run_flat_reference(ReferenceEvaluator(model), {'input': tails.numpy()})[0]
    np.testing.assert_allclose(actual, torch.nn.functional.glu(tails).numpy(), atol=1e-6, rtol=1e-6)


def test_depthwise_actual_standard_fp32_onnx_matches_independent_forward_fma_without_channel_mixing(tmp_path):
    function = candidate('fp32_native_depthwise5')
    values = torch.randn(2, 4, 9, generator=torch.Generator().manual_seed(362))
    weights = torch.randn(4, 1, 5, generator=torch.Generator().manual_seed(363)) * 0.1
    expected = torch.zeros(2, 4, 5)
    for k in range(5):
        expected = (values[:, :, k:k+5].double() * weights[:, 0, k].double()[None, :, None]
                    + expected.double()).float()
    class Module(torch.nn.Module):
        def __init__(self): super().__init__(); self.register_buffer('weight', weights)
        def forward(self, values): return function(values, self.weight)
    graph = tmp_path / 'conv.onnx'
    export_audio_native_onnx(Module(), values, graph)
    model = onnx.load(graph)
    assert all(n.domain in ('', 'ai.onnx') for n in model.graph.node)
    assert not any(n.op_type == 'Conv' for n in model.graph.node)
    assert all(t.data_type != onnx.TensorProto.DOUBLE for t in model.graph.initializer)
    actual = run_flat_reference(ReferenceEvaluator(model), {'input': values.numpy()})[0]
    np.testing.assert_array_equal(actual.view('u4'), expected.numpy().view('u4'))


@pytest.mark.parametrize('name,values', [('fp32_native_glu', torch.ones(2, 5)),
    ('fp32_native_glu', torch.ones(2, 4, dtype=torch.float64)),
    ('fp32_native_glu', torch.full((2, 4), float('nan'))),
    ('fp32_native_glu', torch.empty(0, 4))])
def test_glu_rejects_unaudited_contract(name, values):
    with pytest.raises(ValueError): candidate(name)(values)


@pytest.mark.parametrize('problem', ['dtype', 'channels', 'kernel', 'length', 'nonfinite', 'bound'])
def test_depthwise_rejects_unaudited_contract(problem):
    values, weights = torch.ones(1, 4, 9), torch.ones(4, 1, 5)
    if problem == 'dtype': values = values.double()
    if problem == 'channels': weights = weights[:2]
    if problem == 'kernel': weights = weights[:, :, :3]
    if problem == 'length': values = values[:, :, :3]
    if problem == 'nonfinite': values[0, 0, 0] = float('inf')
    if problem == 'bound': values[0, 0, 0] = 2e6
    with pytest.raises(ValueError): candidate('fp32_native_depthwise5')(values, weights)
