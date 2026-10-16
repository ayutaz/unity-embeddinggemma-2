import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator
import pytest
import torch
from embeddinggemma_tools import export


def required():
    function=getattr(export,'fp32_block192_matmul',None)
    assert callable(function),'FP32 block192 reset/FMA/store-add matmul is missing'
    return function


def oracle(values,weights):
    total=None
    for start in range(0,weights.shape[0],192):
        partial=torch.zeros(*values.shape[:-1],weights.shape[1])
        for index in range(start,min(start+192,weights.shape[0])):
            partial=(values[...,index:index+1].double()*weights[index].double()+partial.double()).float()
        total=partial if total is None else total+partial
    return total


@pytest.mark.parametrize('shape',[(3,192),(1,3,200),(1,3,1024)])
def test_block192_matmul_matches_independent_fused_rounding_and_tail_order(shape):
    values=torch.randn(shape,generator=torch.Generator().manual_seed(140))
    weights=torch.randn(shape[-1],5,generator=torch.Generator().manual_seed(141))*.1
    before=weights.clone();actual=required()(values,weights)
    assert torch.equal(actual,oracle(values,weights))
    assert torch.equal(weights,before) and actual.dtype==torch.float32


@pytest.mark.parametrize('problem',['dtype','nonfinite','shape','bound'])
def test_block192_matmul_refuses_unproven_input_contract(problem):
    values=torch.ones(2,200);weights=torch.ones(200,3)
    if problem=='dtype':values=values.double()
    elif problem=='nonfinite':weights[0,0]=float('nan')
    elif problem=='shape':weights=weights[:-1]
    elif problem=='bound':values[0,0]=2e6
    with pytest.raises(ValueError):required()(values,weights)


def test_block192_export_is_standard_fp32_and_preserves_rounding(tmp_path):
    function=required()
    class Matmul(torch.nn.Module):
        def __init__(self,w):super().__init__();self.register_buffer('weight',w)
        def forward(self,x):return function(x,self.weight)
    values=torch.randn(1,2,200,generator=torch.Generator().manual_seed(142))
    weight=torch.randn(200,3,generator=torch.Generator().manual_seed(143))*.1
    path=tmp_path/'blocked.onnx';export.export_audio_native_onnx(Matmul(weight),values,path)
    graph=onnx.load(path);onnx.checker.check_model(graph)
    assert all(n.domain in ('','ai.onnx') for n in graph.graph.node)
    assert all(i.data_type!=onnx.TensorProto.DOUBLE for i in graph.graph.initializer)
    actual=ReferenceEvaluator(graph).run(None,{'input':values.numpy()})[0]
    np.testing.assert_array_equal(actual,oracle(values,weight).numpy())
