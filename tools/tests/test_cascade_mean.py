import pytest
import torch
import onnx
from onnx.reference import ReferenceEvaluator

from embeddinggemma_tools import export


def required():
    function=getattr(export,'fp32_cascade_mean1024',None)
    assert callable(function),'Pinned eight-lane cascade mean is missing'
    return function


@pytest.mark.parametrize('positive',[False,True])
def test_cascade_mean_matches_independent_contiguous_native_fp32_order(positive):
    values=torch.randn(2,5,1024,generator=torch.Generator().manual_seed(51))
    if positive:values=values.square()
    before=values.clone()
    actual=required()(values)
    assert torch.equal(actual,values.mean(-1,keepdim=True))
    assert actual.dtype==torch.float32 and torch.equal(values,before)


@pytest.mark.parametrize('values',[
    torch.ones(1,1024,dtype=torch.float16),torch.ones(1,128),torch.ones(0,1024),
    torch.full((1,1024),float('nan')),
])
def test_cascade_mean_rejects_unproven_shapes_and_arithmetic(values):
    with pytest.raises(ValueError):required()(values)


def test_cascade_export_is_standard_fp32_and_keeps_native_mean(tmp_path):
    function=required()
    class Mean(torch.nn.Module):
        def forward(self,values):return function(values)
    values=torch.randn(2,1024,generator=torch.Generator().manual_seed(51)).square()
    path=tmp_path/'mean.onnx';export.export_audio_native_onnx(Mean(),values,path)
    model=onnx.load(path);onnx.checker.check_model(model)
    assert all(i.data_type!=onnx.TensorProto.DOUBLE for i in model.graph.initializer)
    assert all(n.domain in ('','ai.onnx') for n in model.graph.node)
    actual=ReferenceEvaluator(model).run(None,{'input':values.numpy()})[0]
    __import__('numpy').testing.assert_array_equal(actual,values.mean(-1,keepdim=True).numpy())
