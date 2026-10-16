import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator
import pytest
import torch
import torch.nn.functional as F

from embeddinggemma_tools import export


def required():
    from embeddinggemma_tools import compensated_conv
    return compensated_conv.CompensatedConv2d


def source():
    torch.manual_seed(81)
    return torch.nn.Conv2d(8,4,3,stride=2,padding=1,bias=False).float()


def test_compensated_conv_preserves_real_convolution_shape_and_source():
    original=source();before=original.weight.detach().clone()
    candidate=required()(original,chunk_size=16)
    values=torch.randn(2,8,7,6,generator=torch.Generator().manual_seed(82))
    expected=F.conv2d(values.double(),original.weight.double(),stride=2,padding=1).float()
    actual=candidate(values)
    torch.testing.assert_close(actual,expected,atol=1e-5,rtol=1e-4)
    assert actual.shape==(2,4,4,3) and actual.dtype==torch.float32
    assert original.training and candidate.weight is original.weight and torch.equal(original.weight,before)


def test_compensated_conv_retains_small_term_between_opposite_large_channels():
    original=torch.nn.Conv2d(3,1,3,stride=2,padding=1,bias=False)
    with torch.no_grad():
        original.weight.zero_();original.weight[0,:,1,1]=torch.tensor([16777216.,1.,-16777216.])
    values=torch.ones(1,3,3,3)
    expected=F.conv2d(values.double(),original.weight.double(),stride=2,padding=1).float()
    assert torch.equal(expected,torch.ones(1,1,2,2))
    assert not torch.equal(original(values),expected)
    assert torch.equal(required()(original,chunk_size=9)(values),expected)


@pytest.mark.parametrize('problem',['bias','stride','kernel','groups','dilation','half','nonfinite','chunk'])
def test_compensated_conv_rejects_unproven_weight_and_geometry_contract(problem):
    original=source();chunk=16
    if problem=='bias':original=torch.nn.Conv2d(8,4,3,stride=2,padding=1)
    elif problem=='stride':original.stride=(1,1)
    elif problem=='kernel':original=torch.nn.Conv2d(8,4,5,stride=2,padding=1,bias=False)
    elif problem=='groups':original=torch.nn.Conv2d(8,4,3,stride=2,padding=1,bias=False,groups=2)
    elif problem=='dilation':original.dilation=(2,2)
    elif problem=='half':original.half()
    elif problem=='nonfinite':
        with torch.no_grad():original.weight[0,0,0,0]=float('nan')
    elif problem=='chunk':chunk=0
    with pytest.raises(ValueError):required()(original,chunk_size=chunk)


@pytest.mark.parametrize('values',[
    torch.ones(1,7,5,5),torch.ones(1,8,5,5,dtype=torch.float64),
    torch.ones(0,8,5,5),torch.full((1,8,5,5),float('nan')),
])
def test_compensated_conv_rejects_wrong_or_nonfinite_input(values):
    with pytest.raises(ValueError):required()(source(),chunk_size=16)(values)


def test_compensated_conv_onnx_is_standard_fp32_and_matches_independent_oracle(tmp_path):
    original=source();candidate=required()(original,chunk_size=16)
    values=torch.randn(1,8,7,6,generator=torch.Generator().manual_seed(83))
    path=tmp_path/'conv.onnx';export.export_audio_native_onnx(candidate,values,path)
    model=onnx.load(path);onnx.checker.check_model(model)
    assert all(n.domain in ('','ai.onnx') for n in model.graph.node)
    assert all(w.data_type!=onnx.TensorProto.DOUBLE for w in model.graph.initializer)
    actual=ReferenceEvaluator(model).run(None,{'input':values.numpy()})[0]
    expected=F.conv2d(values.double(),original.weight.double(),stride=2,padding=1).float().detach().numpy()
    np.testing.assert_allclose(actual,expected,atol=1e-5,rtol=1e-4)
