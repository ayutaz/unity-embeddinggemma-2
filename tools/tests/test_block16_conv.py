import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator
import pytest
import torch
import torch.nn.functional as F
from embeddinggemma_tools import compensated_conv,export


def required():
    function=getattr(compensated_conv,'fp32_block16_conv2d',None)
    assert callable(function),'Pinned block16 zero/FMA/store convolution is missing'
    return function


def oracle(values,weight):
    batch,channels,height,width=values.shape;outputs=weight.shape[0]
    patches=F.unfold(values,3,padding=1,stride=2).transpose(1,2)
    flat=weight.reshape(outputs,-1);total=None
    for start in range(0,channels,16):
        partial=torch.zeros(batch,patches.shape[1],outputs)
        for kernel in range(9):
            for channel in range(start,start+16):
                index=channel*9+kernel
                partial=(patches[...,index:index+1].double()*flat[:,index].double()+partial.double()).float()
        total=partial if total is None else partial+total
    return total.transpose(1,2).reshape(batch,outputs,(height+1)//2,(width+1)//2)


@pytest.mark.parametrize('channels',[16,32])
def test_block16_conv_matches_independent_fused_rounding_and_store_order(channels):
    values=torch.randn(1,channels,5,4,generator=torch.Generator().manual_seed(101))
    weight=torch.randn(4,channels,3,3,generator=torch.Generator().manual_seed(102))*.1
    before=weight.clone();actual=required()(values,weight)
    assert torch.equal(actual,oracle(values,weight))
    assert torch.equal(weight,before) and actual.dtype==torch.float32


@pytest.mark.parametrize('problem',['channels','dtype','nonfinite','shape'])
def test_block16_conv_refuses_unproven_input_contract(problem):
    values=torch.ones(1,16,5,4);weight=torch.ones(4,16,3,3)
    if problem=='channels':values=values[:,:8];weight=weight[:,:8]
    elif problem=='dtype':values=values.double()
    elif problem=='nonfinite':values[0,0,0,0]=float('nan')
    elif problem=='shape':weight=weight[:,:,:2,:]
    with pytest.raises(ValueError):required()(values,weight)


def test_block16_export_uses_standard_fp32_without_changing_fused_order(tmp_path):
    run=required()
    class Conv(torch.nn.Module):
        def __init__(self,w):super().__init__();self.register_buffer('weight',w)
        def forward(self,x):return run(x,self.weight)
    values=torch.randn(1,16,3,3,generator=torch.Generator().manual_seed(103))
    weight=torch.randn(2,16,3,3,generator=torch.Generator().manual_seed(104))*.1
    path=tmp_path/'block16.onnx';export.export_audio_native_onnx(Conv(weight),values,path)
    model=onnx.load(path);onnx.checker.check_model(model)
    assert all(n.domain in ('','ai.onnx') for n in model.graph.node)
    assert all(w.data_type!=onnx.TensorProto.DOUBLE for w in model.graph.initializer)
    actual=ReferenceEvaluator(model).run(None,{'input':values.numpy()})[0]
    np.testing.assert_array_equal(actual,oracle(values,weight).numpy())
