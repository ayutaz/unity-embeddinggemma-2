import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator
import pytest
import torch

from embeddinggemma_tools import export


def required():
    result=getattr(export,'VisionHeadRmsExport',None)
    assert callable(result),'Scaled and unscaled fixed-width vision head RMS export is missing'
    return result


@pytest.mark.parametrize('scaled',[False,True])
@pytest.mark.parametrize('shape',[(64,64),(1,5,12,64)])
def test_head_rms_preserves_native_fp32_equation_and_borrowed_input_scale(scaled,shape):
    generator=torch.Generator().manual_seed(757)
    values=torch.randn(shape,generator=generator)
    values.reshape(-1,64)[0].zero_()
    scale=torch.rand(64,generator=generator)+.5 if scaled else None
    before=values.clone();scale_before=scale.clone() if scaled else None
    module=required()(scale,1e-6);actual=module(values)
    expected=values*torch.pow(values.square().mean(-1,keepdim=True)+1e-6,-.5)
    if scaled:expected=expected*scale
    np.testing.assert_array_equal(actual.numpy(),expected.numpy())
    assert torch.equal(values,before)
    if scaled:
        assert torch.equal(scale,scale_before) and module.scale.data_ptr()==scale.data_ptr()
    else:
        assert module.scale is None


@pytest.mark.parametrize('problem',['width','dtype','nonfinite','epsilon'])
def test_head_rms_requires_original_finite_fp32_scale_and_epsilon(problem):
    scale=torch.ones(64);epsilon=1e-6
    if problem=='width':scale=torch.ones(63)
    elif problem=='dtype':scale=scale.half()
    elif problem=='nonfinite':scale[0]=float('nan')
    elif problem=='epsilon':epsilon=2e-6
    with pytest.raises(ValueError):required()(scale,epsilon)


@pytest.mark.parametrize('values',[torch.ones(2,63),torch.ones(2,64,dtype=torch.float16),torch.full((2,64),float('nan'))])
def test_head_rms_rejects_invalid_input_before_export(values):
    module=required()(None,1e-6)
    with pytest.raises(ValueError):module(values)


@pytest.mark.parametrize('scaled',[False,True])
def test_head_rms_standard_fp32_onnx_keeps_native_equation_and_scale(tmp_path,scaled):
    generator=torch.Generator().manual_seed(758);values=torch.randn(3,64,generator=generator)
    scale=torch.rand(64,generator=generator)+.5 if scaled else None
    module=required()(scale,1e-6);path=tmp_path/'head-rms.onnx'
    export.export_audio_native_onnx(module,values,path)
    model=onnx.load(path);onnx.checker.check_model(model)
    assert all(node.domain in ('','ai.onnx') for node in model.graph.node)
    assert all(value.data_type!=onnx.TensorProto.DOUBLE for value in model.graph.initializer)
    if scaled:
        assert any(onnx.numpy_helper.to_array(value).tobytes()==scale.numpy().tobytes() for value in model.graph.initializer)
    actual=ReferenceEvaluator(model).run(None,{'input':values.numpy()})[0]
    expected=values*torch.pow(values.square().mean(-1,keepdim=True)+1e-6,-.5)
    if scaled:expected=expected*scale
    np.testing.assert_array_equal(actual,expected.numpy())
