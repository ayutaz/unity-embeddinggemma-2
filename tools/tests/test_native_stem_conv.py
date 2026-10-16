import numpy as np
import onnx
import pytest
import torch
import torch.nn.functional as F
from onnx.reference import ReferenceEvaluator

from embeddinggemma_tools import compensated_conv
from embeddinggemma_tools.export import export_audio_native_onnx
from embeddinggemma_tools.streaming_reference import run_flat_reference


def candidate():
    function = getattr(compensated_conv, 'fp32_native_stem_conv2d', None)
    assert callable(function), 'Single-channel original stem FMA order is missing'
    return function


def oracle(values, weight):
    batch, channels, height, width = values.shape
    patches = F.unfold(values, 3, padding=1, stride=2).transpose(1, 2)
    flat = weight.reshape(weight.shape[0], -1)
    block = 1 if channels == 1 else 16
    total = None
    for start in range(0, channels, block):
        part = torch.zeros(batch, patches.shape[1], weight.shape[0])
        for kernel in range(9):
            for channel in range(start, start + block):
                index = channel * 9 + kernel
                part = (patches[..., index:index+1].double() * flat[:, index].double() + part.double()).float()
        total = part if total is None else total + part
    return total.transpose(1, 2).reshape(batch, weight.shape[0], (height+1)//2, (width+1)//2)


@pytest.mark.parametrize('channels', [1, 16, 32])
def test_stem_conv_preserves_single_channel_and_original_block16_fma_store_order(channels):
    values = torch.randn(2, channels, 5, 4, generator=torch.Generator().manual_seed(371))
    weight = torch.randn(3, channels, 3, 3, generator=torch.Generator().manual_seed(372)) * 0.1
    before = weight.clone()
    actual = candidate()(values, weight)
    assert torch.equal(actual.view(torch.int32), oracle(values, weight).view(torch.int32))
    assert torch.equal(weight, before)


def test_single_channel_actual_onnx_keeps_padding_stride_and_fp32_rounding(tmp_path):
    function = candidate()
    values = torch.randn(2, 1, 5, 4, generator=torch.Generator().manual_seed(373))
    weight = torch.randn(3, 1, 3, 3, generator=torch.Generator().manual_seed(374)) * 0.1
    class Module(torch.nn.Module):
        def __init__(self): super().__init__(); self.register_buffer('weight', weight)
        def forward(self, x): return function(x, self.weight)
    graph = tmp_path / 'stem.onnx'
    export_audio_native_onnx(Module(), values, graph)
    model = onnx.load(graph)
    assert all(n.domain in ('', 'ai.onnx') for n in model.graph.node)
    assert all(w.data_type != onnx.TensorProto.DOUBLE for w in model.graph.initializer)
    actual = run_flat_reference(ReferenceEvaluator(model), {'input': values.numpy()})[0]
    np.testing.assert_array_equal(actual.view('u4'), oracle(values, weight).numpy().view('u4'))


@pytest.mark.parametrize('problem', ['channels', 'dtype', 'kernel', 'nonfinite', 'bound'])
def test_stem_candidate_rejects_unaudited_contract(problem):
    values, weight = torch.ones(1, 1, 5, 4), torch.ones(3, 1, 3, 3)
    if problem == 'channels': values, weight = values.repeat(1, 8, 1, 1), weight.repeat(1, 8, 1, 1)
    if problem == 'dtype': values = values.double()
    if problem == 'kernel': weight = weight[:, :, :2]
    if problem == 'nonfinite': values[0, 0, 0, 0] = float('nan')
    if problem == 'bound': values[0, 0, 0, 0] = 2e6
    with pytest.raises(ValueError): candidate()(values, weight)
