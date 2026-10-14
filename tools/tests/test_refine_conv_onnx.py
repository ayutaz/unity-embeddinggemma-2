import hashlib
import numpy as np
import onnx
from onnx import helper,numpy_helper,TensorProto
from onnx.reference import ReferenceEvaluator
import pytest
import torch
import torch.nn.functional as F

from embeddinggemma_tools.compensated_conv import fp32_compensated_conv2d


def required():
    from embeddinggemma_tools import refine_conv_onnx
    return refine_conv_onnx.refine_conv2d


def original():
    weight=np.random.default_rng(85).normal(size=(4,8,3,3)).astype(np.float32)
    init=numpy_helper.from_array(weight,'weight')
    node=helper.make_node('Conv',['input','weight'],['output'],name='conv',
        strides=[2,2],pads=[1,1,1,1],dilations=[1,1],group=1,kernel_shape=[3,3])
    graph=helper.make_graph([node],'source',[helper.make_tensor_value_info('input',TensorProto.FLOAT,[1,8,7,6])],
        [helper.make_tensor_value_info('output',TensorProto.FLOAT,[1,4,4,3])],[init])
    return helper.make_model(graph,opset_imports=[helper.make_opsetid('',24)]),weight


def template(tmp_path):
    class Conv(torch.nn.Module):
        def forward(self,values,weight):return fp32_compensated_conv2d(values,weight,chunk_size=16)
    path=tmp_path/'template.onnx'
    torch.onnx.export(Conv(),(torch.ones(1,8,7,6),torch.ones(4,8,3,3)),path,dynamo=False,
        opset_version=20,input_names=['values','weight'],output_names=['output'])
    return onnx.version_converter.convert_version(onnx.load(path),24)


def test_conv_rewrite_keeps_existing_weight_bytes_and_native_tolerance(tmp_path):
    run=required();model,weight=original();compiled=template(tmp_path)
    before=model.graph.initializer[0].SerializeToString()
    report=run(model,compiled,node_name='conv',expected_weight_sha256=hashlib.sha256(weight.tobytes()).hexdigest())
    onnx.checker.check_model(model)
    assert model.graph.initializer[0].SerializeToString()==before and report['existing_initializer_bytes_preserved']
    values=torch.randn(1,8,7,6,generator=torch.Generator().manual_seed(86))
    expected=F.conv2d(values.double(),torch.from_numpy(weight).double(),stride=2,padding=1).float()
    actual=ReferenceEvaluator(model).run(None,{'input':values.numpy()})[0]
    np.testing.assert_allclose(actual,expected.numpy(),atol=1e-5,rtol=1e-4)


@pytest.mark.parametrize('problem',['missing','stride','bias','hash','shape','double','half_cast'])
def test_conv_context_is_rejected_before_any_graph_mutation(tmp_path,problem):
    run=required();model,weight=original();compiled=template(tmp_path);name='conv';digest=hashlib.sha256(weight.tobytes()).hexdigest()
    if problem=='missing':name='missing'
    elif problem=='stride':next(a for a in model.graph.node[0].attribute if a.name=='strides').ints[:]=[1,1]
    elif problem=='bias':model.graph.node[0].input.append('weight')
    elif problem=='hash':digest='0'*64
    elif problem=='shape':compiled.graph.input[0].type.tensor_type.shape.dim[2].dim_value=9
    elif problem=='double':compiled.graph.input[1].type.tensor_type.elem_type=TensorProto.DOUBLE
    elif problem=='half_cast':compiled.graph.node.append(helper.make_node('Cast',['values'],['unsupported_half'],to=TensorProto.FLOAT16))
    before=model.SerializeToString()
    with pytest.raises(ValueError):run(model,compiled,node_name=name,expected_weight_sha256=digest)
    assert model.SerializeToString()==before
