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


@pytest.mark.parametrize('positive',[False,True])
def test_projection_norm1536_keeps_the_same_independent_native_order(positive):
    function=getattr(export,'fp32_cascade_mean1536',None)
    assert callable(function),'1536-wide projection mean is missing'
    values=torch.randn(2,5,1536,generator=torch.Generator().manual_seed(52))
    if positive:values=values.square()
    before=values.clone();actual=function(values)
    assert torch.equal(actual,values.mean(-1,keepdim=True)) and torch.equal(values,before)


@pytest.mark.parametrize('width',[1024,1536])
def test_cascade_export_has_bounded_graph_size_without_changing_order(tmp_path,width):
    function=getattr(export,f'fp32_cascade_mean{width}')
    class Mean(torch.nn.Module):
        def forward(self,values):return function(values)
    values=torch.randn(2,5,width,generator=torch.Generator().manual_seed(72)).square()
    path=tmp_path/'compact.onnx';export.export_audio_native_onnx(Mean(),values,path)
    model=onnx.load(path)
    assert len(model.graph.node)<400,'Vectorize independent accumulators, preserve their addition order'
    actual=ReferenceEvaluator(model).run(None,{'input':values.numpy()})[0]
    __import__('numpy').testing.assert_array_equal(actual,values.mean(-1,keepdim=True).numpy())


@pytest.mark.parametrize('positive',[False,True])
def test_outer_layout_mean_matches_native_for_all_280_time_rows(positive):
    function=getattr(export,'fp32_outer_cascade_mean1024',None)
    assert callable(function),'Pinned transposed convolution mean is missing'
    values=torch.randn(1,1024,280,generator=torch.Generator().manual_seed(74)).transpose(1,2)
    if positive:values=values.square()
    assert values.stride()==(286720,1,280)
    before=values.clone();actual=function(values)
    assert torch.equal(actual,values.mean(-1,keepdim=True))
    assert torch.equal(values,before) and actual.shape==(1,280,1)


@pytest.mark.parametrize('values',[
    torch.ones(1,280,1024),torch.ones(1,1024,279).transpose(1,2),
    torch.ones(1,1024,280,dtype=torch.float16).transpose(1,2),
    torch.full((1,1024,280),float('nan')).transpose(1,2),
])
def test_outer_layout_rejects_unproven_reference_context(values):
    with pytest.raises(ValueError):export.fp32_outer_cascade_mean1024(values)


def test_outer_layout_onnx_preserves_native_head_and_tail_bits(tmp_path):
    class Mean(torch.nn.Module):
        def forward(self,x):return export.fp32_outer_cascade_mean1024(x)
    values=torch.randn(1,1024,280,generator=torch.Generator().manual_seed(77)).transpose(1,2).square()
    path=tmp_path/'outer.onnx';export.export_audio_native_onnx(Mean(),values,path)
    model=onnx.load(path);onnx.checker.check_model(model)
    assert len(model.graph.node)<400
    assert all(n.domain in ('','ai.onnx') for n in model.graph.node)
    actual=ReferenceEvaluator(model).run(None,{'input':values.numpy()})[0]
    __import__('numpy').testing.assert_array_equal(actual,values.mean(-1,keepdim=True).numpy())
