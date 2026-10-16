import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator
import pytest
import torch

from embeddinggemma_tools import export


def required():
    result=getattr(export,'VisionRmsExport',None)
    assert callable(result),'Fixed-width vision RMS export candidate is missing'
    return result


@pytest.mark.parametrize('zero',[False,True])
def test_vision_rms_keeps_borrowed_scale_input_and_native_fp32_equation(zero):
    generator=torch.Generator().manual_seed(754)
    values=torch.randn(64,768,generator=generator)
    if zero:values.zero_()
    scale=torch.rand(768,generator=generator)+.5
    before=values.clone();scale_before=scale.clone();module=required()(scale,1e-6)
    actual=module(values)
    expected=(values*torch.pow(values.square().mean(-1,keepdim=True)+1e-6,-.5))*scale
    np.testing.assert_array_equal(actual.numpy(),expected.numpy())
    assert torch.equal(values,before) and torch.equal(scale,scale_before) and module.scale.data_ptr()==scale.data_ptr()


@pytest.mark.parametrize('problem',['width','dtype','nonfinite','epsilon'])
def test_vision_rms_requires_original_fixed_fp32_scale_and_epsilon(problem):
    scale=torch.ones(768);epsilon=1e-6
    if problem=='width':scale=torch.ones(767)
    elif problem=='dtype':scale=scale.half()
    elif problem=='nonfinite':scale[0]=float('nan')
    elif problem=='epsilon':epsilon=2e-6
    with pytest.raises(ValueError):required()(scale,epsilon)


@pytest.mark.parametrize('values',[torch.ones(2,767),torch.ones(2,768,dtype=torch.float16),torch.full((2,768),float('nan'))])
def test_vision_rms_rejects_wrong_values_before_export(values):
    module=required()(torch.ones(768),1e-6)
    with pytest.raises(ValueError):module(values)


def test_vision_rms_onnx_uses_only_standard_fp32_ops_and_matches_independent_native_equation(tmp_path):
    generator=torch.Generator().manual_seed(755)
    values=torch.randn(3,768,generator=generator);scale=torch.rand(768,generator=generator)+.5
    module=required()(scale,1e-6);path=tmp_path/'rms.onnx'
    export.export_audio_native_onnx(module,values,path)
    model=onnx.load(path);onnx.checker.check_model(model)
    assert all(n.domain in ('','ai.onnx') for n in model.graph.node)
    assert all(v.data_type!=onnx.TensorProto.DOUBLE for v in model.graph.initializer)
    actual=ReferenceEvaluator(model).run(None,{'input':values.numpy()})[0]
    expected=(values*torch.pow(values.square().mean(-1,keepdim=True)+1e-6,-.5))*scale
    np.testing.assert_array_equal(actual,expected.numpy())
