import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator
import pytest
import torch

from embeddinggemma_tools import export


def independent_mean(values):
    """Scalar FP32 interpretation of pinned SumKernel's 16-lane/four-ILP row."""
    data=values.numpy().reshape(-1,values.shape[-1])
    result=[]
    for row in data:
        accum=np.zeros((4,16),dtype=np.float32)
        for start in range(0,len(row),64):
            for item in range(4):
                for lane in range(16):
                    accum[item,lane]=np.float32(accum[item,lane]+row[start+item*16+lane])
        combined=accum[0].copy()
        for item in range(1,4):combined=np.add(combined,accum[item],dtype=np.float32)
        total=np.float32(0)
        for lane in combined:total=np.float32(total+lane)
        result.append(np.float32(total/np.float32(len(row))))
    return torch.from_numpy(np.array(result,dtype=np.float32)).reshape(*values.shape[:-1],1)


@pytest.mark.parametrize('width',[64,768])
def test_vector16_mean_preserves_independent_fp32_accumulation_order(width):
    data=torch.rand(7,width,generator=torch.Generator().manual_seed(364))*torch.logspace(-3,3,width)
    method=export.fp32_cascade_mean64 if width==64 else export.fp32_cascade_mean768
    actual=method(data,vector_width=16)
    assert torch.equal(actual,independent_mean(data))
    assert torch.equal(method(data),method(data,vector_width=8))
    assert not torch.equal(actual,method(data,vector_width=8)), 'Fixture must distinguish reduction orders'


@pytest.mark.parametrize('width',[64,768])
def test_vector16_rms_uses_original_scale_and_independent_mean(width):
    data=torch.randn(7,width,generator=torch.Generator().manual_seed(365))*3
    scale=torch.linspace(.5,1.5,width)
    cls=export.VisionHeadRmsExport if width==64 else export.VisionInlineRmsExport
    model=cls(scale,1e-6,vector_width=16)
    target=(data*export.fp32_refined_rstd(independent_mean(data.square())+1e-6))*scale
    np.testing.assert_array_equal(model(data).numpy().view(np.int32),target.numpy().view(np.int32))
    assert model.scale.data_ptr()==scale.data_ptr()


def test_vector16_unscaled_tail_exports_standard_fp32_equation_without_changing_default(tmp_path):
    data=torch.arange(3*768,dtype=torch.float32).reshape(3,768)*.4
    model=export.VisionTailRmsExport(vector_width=16)
    target=data*export.fp32_refined_rstd(independent_mean(data.square())+1e-6,maximum_statistic=2e6)
    path=tmp_path/'vector16.onnx'
    export.export_audio_native_onnx(model,data,path)
    graph=onnx.load(path)
    onnx.checker.check_model(graph)
    assert all(node.domain in ('','ai.onnx') for node in graph.graph.node)
    assert all(value.data_type!=onnx.TensorProto.DOUBLE for value in graph.graph.initializer)
    actual=ReferenceEvaluator(graph).run(None,{'input':data.numpy()})[0]
    np.testing.assert_array_equal(actual.view(np.int32),target.numpy().view(np.int32))


def test_unqualified_vector_width_is_rejected():
    with pytest.raises(ValueError,match='vector'):
        export.VisionTailRmsExport(vector_width=4)
