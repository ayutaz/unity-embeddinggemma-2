import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator
import pytest
import torch

from embeddinggemma_tools import export


def required():
    result=getattr(export,'VisionInlineRmsExport',None)
    assert callable(result),'Single-model vision RMS with refined mean division is missing'
    return result


@pytest.mark.parametrize('shape',[(128,768),(1,17,768)])
def test_inline_rms_matches_independent_fp32_equation_and_borrows_original_scale(shape):
    generator=torch.Generator().manual_seed(761)
    values=torch.randn(shape,generator=generator);values.reshape(-1,768)[0].zero_()
    scale=torch.rand(768,generator=generator)+.5;before=values.clone();scale_before=scale.clone()
    module=required()(scale,1e-6);actual=module(values)
    expected=(values*torch.pow(values.square().mean(-1,keepdim=True)+1e-6,-.5))*scale
    np.testing.assert_array_equal(actual.numpy().view(np.int32),expected.numpy().view(np.int32))
    assert module.scale.data_ptr()==scale.data_ptr() and torch.equal(scale,scale_before) and torch.equal(values,before)


def test_inline_rms_export_uses_standard_fp32_ops_and_preserves_native_equation_and_scale(tmp_path):
    generator=torch.Generator().manual_seed(762);values=torch.randn(5,768,generator=generator)
    scale=torch.rand(768,generator=generator)+.5;path=tmp_path/'inline-rms.onnx'
    export.export_audio_native_onnx(required()(scale,1e-6),values,path)
    model=onnx.load(path);onnx.checker.check_model(model)
    assert len(model.graph.node)<400
    assert all(node.domain in ('','ai.onnx') for node in model.graph.node)
    assert all(value.data_type!=onnx.TensorProto.DOUBLE for value in model.graph.initializer)
    assert any(onnx.numpy_helper.to_array(value).tobytes()==scale.numpy().tobytes() for value in model.graph.initializer)
    actual=ReferenceEvaluator(model).run(None,{'input':values.numpy()})[0]
    expected=(values*torch.pow(values.square().mean(-1,keepdim=True)+1e-6,-.5))*scale
    np.testing.assert_array_equal(actual.view(np.int32),expected.numpy().view(np.int32))
