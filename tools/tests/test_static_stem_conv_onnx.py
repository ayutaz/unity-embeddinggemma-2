import numpy as np
import onnx
import pytest
import torch
import torch.nn.functional as F
from onnx.reference import ReferenceEvaluator

from embeddinggemma_tools import compensated_conv
from embeddinggemma_tools.export import export_audio_native_onnx
from embeddinggemma_tools.streaming_reference import run_flat_reference
from test_native_stem_conv import oracle


def candidate():
    from embeddinggemma_tools import static_stem_conv_onnx
    return static_stem_conv_onnx.static_stem_conv_export


def test_static_stem_forward_is_original_convolution_passthrough_not_expensive_fixed_order(monkeypatch):
    function = candidate()
    def forbidden(*args): pytest.fail('Native export control must not execute the explicit FMA graph')
    monkeypatch.setattr(compensated_conv, 'fp32_native_stem_conv2d', forbidden)
    x = torch.randn(1, 16, 7, 6, generator=torch.Generator().manual_seed(381))
    w = torch.randn(4, 16, 3, 3, generator=torch.Generator().manual_seed(382)) * 0.1
    assert torch.equal(function(x, w).view(torch.int32), F.conv2d(x, w, stride=2, padding=1).view(torch.int32))


@pytest.mark.parametrize('channels', [1, 16, 32])
def test_actual_static_stem_onnx_matches_independent_kernel_channel_block_fma_order(channels, tmp_path):
    function = candidate()
    x = torch.randn(2, channels, 7, 6, generator=torch.Generator().manual_seed(383))
    w = torch.randn(4, channels, 3, 3, generator=torch.Generator().manual_seed(384)) * 0.1
    class Module(torch.nn.Module):
        def __init__(self): super().__init__(); self.register_buffer('weight', w)
        def forward(self, x): return function(x, self.weight)
    path = tmp_path / 'static-stem.onnx'
    export_audio_native_onnx(Module(), x, path)
    model = onnx.load(path)
    assert all(n.domain in ('', 'ai.onnx') for n in model.graph.node)
    assert not any(n.op_type == 'Conv' for n in model.graph.node)
    assert all(w.data_type != onnx.TensorProto.DOUBLE for w in model.graph.initializer)
    for node in model.graph.node:
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.TENSOR:
                assert attribute.t.data_type != onnx.TensorProto.DOUBLE
    actual = run_flat_reference(ReferenceEvaluator(model), {'input': x.numpy()})[0]
    np.testing.assert_array_equal(actual.view('u4'), oracle(x, w).numpy().view('u4'))


@pytest.mark.parametrize('problem', ['dtype', 'channels', 'nonfinite', 'bound'])
def test_static_stem_refuses_unqualified_native_contract(problem):
    x, w = torch.ones(1, 1, 5, 4), torch.ones(2, 1, 3, 3)
    if problem == 'dtype': x = x.double()
    if problem == 'channels': x, w = x.repeat(1, 8, 1, 1), w.repeat(1, 8, 1, 1)
    if problem == 'nonfinite': x[0, 0, 0, 0] = float('inf')
    if problem == 'bound': w[0, 0, 0, 0] = 101
    with pytest.raises(ValueError): candidate()(x, w)
