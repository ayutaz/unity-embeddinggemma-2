import numpy as np
import onnx
from onnx import helper,numpy_helper,TensorProto
from onnx.reference import ReferenceEvaluator
import pytest
import torch

from embeddinggemma_tools import export


def required():
    from embeddinggemma_tools import refine_rms_onnx
    return refine_rms_onnx.refine_rms_means


def model(width=1024):
    initializers=[numpy_helper.from_array(np.array(2.,np.float32),'two'),
        numpy_helper.from_array(np.array(-.5,np.float32),'negative-half'),
        numpy_helper.from_array(np.array(1e-6,np.float32),'epsilon'),
        numpy_helper.from_array(np.array([-1],np.int64),'axis')]
    nodes=[helper.make_node('Pow',['input','two'],['squared'],name='norm/Pow'),
        helper.make_node('ReduceMean',['squared','axis'],['mean'],keepdims=1,name='norm/ReduceMean'),
        helper.make_node('Add',['mean','epsilon'],['statistics'],name='norm/Add'),
        helper.make_node('Pow',['statistics','negative-half'],['inverse'],name='norm/Pow_1'),
        helper.make_node('Mul',['input','inverse'],['output'],name='norm/Mul')]
    graph=helper.make_graph(nodes,'rms',[helper.make_tensor_value_info('input',TensorProto.FLOAT,[1,2,width])],
        [helper.make_tensor_value_info('output',TensorProto.FLOAT,[1,2,width])],initializers,
        value_info=[helper.make_tensor_value_info('squared',TensorProto.FLOAT,[1,2,width])])
    return helper.make_model(graph,opset_imports=[helper.make_opsetid('',24)])


def templates(tmp_path):
    class Mean(torch.nn.Module):
        def forward(self,x):return export.fp32_cascade_mean1024(x)
    class Inverse(torch.nn.Module):
        def forward(self,x):return export.fp32_refined_rstd(x)
    result=[]
    for name,module,x in [('mean',Mean(),torch.ones(1,1,1024)),('inverse',Inverse(),torch.ones(1,1,1))]:
        path=tmp_path/(name+'.onnx');export.export_audio_native_onnx(module,x,path);result.append(onnx.load(path))
    return result


def test_rms_rewrite_preserves_coefficients_and_matches_native_order(tmp_path):
    run=required();source=model();before=[w.SerializeToString() for w in source.graph.initializer]
    mean,inverse=templates(tmp_path);report=run(source,{1024:mean},inverse,expected_count=1)
    onnx.checker.check_model(source)
    assert report['replaced_rms_count']==1 and report['existing_initializer_bytes_preserved']
    assert [w.SerializeToString() for w in source.graph.initializer[:len(before)]]==before
    values=torch.randn(1,2,1024,generator=torch.Generator().manual_seed(56))
    expected=values*torch.pow(values.square().mean(-1,keepdim=True)+1e-6,-.5)
    actual=ReferenceEvaluator(source).run(None,{'input':values.numpy()})[0]
    np.testing.assert_allclose(actual,expected.numpy(),rtol=1e-4,atol=1e-5)


@pytest.mark.parametrize('problem',['width','axis','square','inverse','count','double_template'])
def test_bad_rms_context_is_rejected_before_mutation(tmp_path,problem):
    run=required();source=model(512 if problem=='width' else 1024);mean,inverse=templates(tmp_path)
    count=2 if problem=='count' else 1
    if problem=='axis':source.graph.initializer[3].CopyFrom(numpy_helper.from_array(np.array([1],np.int64),'axis'))
    elif problem=='square':source.graph.initializer[0].CopyFrom(numpy_helper.from_array(np.array(3.,np.float32),'two'))
    elif problem=='inverse':source.graph.initializer[1].CopyFrom(numpy_helper.from_array(np.array(-1.,np.float32),'negative-half'))
    elif problem=='double_template':mean.graph.input[0].type.tensor_type.elem_type=TensorProto.DOUBLE
    before=source.SerializeToString()
    with pytest.raises(ValueError):run(source,{1024:mean},inverse,expected_count=count)
    assert source.SerializeToString()==before


def test_explicit_layout_exclusion_preserves_unproven_mean_node(tmp_path):
    run=required();source=model();mean,inverse=templates(tmp_path)
    excluded=helper.make_node('ReduceMean',['input','axis'],['unproven'],keepdims=1,name='conv_norm/ReduceMean')
    before=excluded.SerializeToString();source.graph.node.append(excluded)
    report=run(source,{1024:mean},inverse,expected_count=1,excluded_names=('conv_norm/ReduceMean',))
    assert next(n for n in source.graph.node if n.name=='conv_norm/ReduceMean').SerializeToString()==before
    assert report['excluded_mean_names']==['conv_norm/ReduceMean']


def test_missing_exclusion_cannot_hide_a_context_error(tmp_path):
    run=required();source=model();mean,inverse=templates(tmp_path);before=source.SerializeToString()
    with pytest.raises(ValueError):run(source,{1024:mean},inverse,expected_count=1,excluded_names=('missing',))
    assert source.SerializeToString()==before
