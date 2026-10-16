import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator
import pytest
import torch

from embeddinggemma_tools import export


def required():
    function=getattr(export,'fp32_cascade_mean768',None)
    assert callable(function),'Pinned 768-wide FP32 reduction is missing'
    return function


@pytest.mark.parametrize('positive',[False,True])
def test_vision_width768_matches_independent_native_reduction_without_mutating_input(positive):
    values=torch.randn(64,768,generator=torch.Generator().manual_seed(718))
    if positive:values=values.square()
    before=values.clone();actual=required()(values)
    np.testing.assert_array_equal(actual.numpy(),values.mean(-1,keepdim=True).numpy())
    assert torch.equal(values,before) and actual.dtype==torch.float32


@pytest.mark.parametrize('values',[torch.ones(2,767),torch.ones(2,768,dtype=torch.float16),
                                  torch.full((2,768),float('nan')),torch.ones(768,2).t()])
def test_vision_reduction_rejects_unproved_shape_type_finiteness_or_layout(values):
    with pytest.raises(ValueError):required()(values)


def test_vision_reduction_onnx_keeps_standard_fp32_native_bits(tmp_path):
    function=required()
    class Mean(torch.nn.Module):
        def forward(self,values):return function(values)
    values=torch.randn(3,768,generator=torch.Generator().manual_seed(719)).square()
    path=tmp_path/'vision-mean.onnx';export.export_audio_native_onnx(Mean(),values,path)
    model=onnx.load(path);onnx.checker.check_model(model)
    assert len(model.graph.node)<450 and all(n.domain in ('','ai.onnx') for n in model.graph.node)
    assert all(v.data_type!=onnx.TensorProto.DOUBLE for v in model.graph.initializer)
    actual=ReferenceEvaluator(model).run(None,{'input':values.numpy()})[0]
    np.testing.assert_array_equal(actual,values.mean(-1,keepdim=True).numpy())
