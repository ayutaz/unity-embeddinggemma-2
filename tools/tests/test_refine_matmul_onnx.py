import hashlib
import numpy as np
import onnx
from onnx import helper,numpy_helper,TensorProto
from onnx.reference import ReferenceEvaluator
import pytest
import torch
from embeddinggemma_tools import export


def required():
    import importlib
    try: module=importlib.import_module('embeddinggemma_tools.refine_matmul_onnx')
    except ModuleNotFoundError: pytest.fail('Audited fixed FP32 matmul rewrite is missing')
    return module.refine_static_matmul


@pytest.fixture
def context(tmp_path):
    x=torch.randn(1,2,200,generator=torch.Generator().manual_seed(150))
    w=torch.randn(200,3,generator=torch.Generator().manual_seed(151))*.1
    graph=helper.make_graph([helper.make_node('MatMul',['input','weight'],['output'],name='projection')],
        'projection',[helper.make_tensor_value_info('input',TensorProto.FLOAT,list(x.shape))],
        [helper.make_tensor_value_info('output',TensorProto.FLOAT,[1,2,3])],
        initializer=[numpy_helper.from_array(w.numpy(),name='weight')])
    model=helper.make_model(graph,opset_imports=[helper.make_opsetid('',24)],ir_version=9)
    class Template(torch.nn.Module):
        def forward(self,a,b):return export.fp32_block192_matmul(a,b)
    path=tmp_path/'template.onnx'
    torch.onnx.export(Template(),(x,w),path,dynamo=False,opset_version=20,input_names=['a','b'],output_names=['out'])
    template=onnx.version_converter.convert_version(onnx.load(path),24)
    return model,template,x,w,hashlib.sha256(w.numpy().astype('<f4').tobytes()).hexdigest()


def test_fixed_matmul_rewrite_keeps_existing_weight_bytes_and_rounding(context):
    model,template,x,w,sha=context
    before=[i.SerializeToString() for i in model.graph.initializer]
    report=required()(model,template,node_name='projection',expected_weight_sha256=sha)
    onnx.checker.check_model(model)
    assert report['replaced_matmul_count']==1 and report['existing_initializer_bytes_preserved']
    assert [i.SerializeToString() for i in model.graph.initializer[:len(before)]]==before
    actual=ReferenceEvaluator(model).run(None,{'input':x.numpy()})[0]
    np.testing.assert_array_equal(actual,export.fp32_block192_matmul(x,w).numpy())


@pytest.mark.parametrize('problem',['missing','hash','shape','fp16','duplicate'])
def test_fixed_matmul_rewrite_refuses_unproven_context_before_mutation(context,problem):
    model,template,_,_,sha=context
    name='missing' if problem=='missing' else 'projection'
    if problem=='hash':sha='0'*64
    elif problem=='shape':template.graph.input[0].type.tensor_type.shape.dim[1].dim_value=3
    elif problem=='fp16':template.graph.input[0].type.tensor_type.elem_type=TensorProto.FLOAT16
    elif problem=='duplicate':model.graph.node.append(model.graph.node[0])
    before=model.SerializeToString()
    with pytest.raises(ValueError):required()(model,template,node_name=name,expected_weight_sha256=sha)
    assert model.SerializeToString()==before
