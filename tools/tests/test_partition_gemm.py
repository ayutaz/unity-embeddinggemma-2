import hashlib

import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh, TensorProto as T
from onnx.reference import ReferenceEvaluator
import pytest

from embeddinggemma_tools import partition_projections as planner
from test_projection_coefficients import projection


@pytest.mark.parametrize('kind,expanded',[('Gemm',False),('MatMul',True)])
def test_gemm_and_broadcast_constant_views_reconnect_after_gpu_projection_cut(kind,expanded):
    model,weight=projection(kind,expanded=expanded);model=onnx.shape_inference.infer_shapes(model)
    before=model.SerializeToString();digest=hashlib.sha256(weight.T.astype('<f4').tobytes()).hexdigest()
    parts,plan=planner.partition_static_projections(model,node_names=['project'],expected_weight_sha256=[digest])
    assert model.SerializeToString()==before and len(parts)==2
    operation=plan['operations'][0]
    matrix=planner.projection_weight_array(model,operation)
    np.testing.assert_array_equal(matrix,weight.T)
    values={'input':np.arange(8 if expanded else 2,dtype=np.float32).reshape(4,1,2) if expanded else np.array([[1.,2.]],np.float32)}
    for index,part in enumerate(parts):
        onnx.checker.check_model(part)
        result=ReferenceEvaluator(part).run(None,{v.name:values[v.name] for v in part.graph.input})
        aliases=plan['segments'][index]['output_aliases']
        values.update(zip([aliases.get(v.name,v.name) for v in part.graph.output],result))
        if index==0:values[operation['output']]=values[operation['input']]@matrix
    expected=ReferenceEvaluator(model).run(None,{'input':values['input']})[0]
    np.testing.assert_array_equal(values['out'],expected)
    assert plan['source_initializer_bytes_preserved'] is True
    assert operation['source_weight_sha256']==hashlib.sha256(weight.tobytes()).hexdigest()


def test_changed_coefficient_plan_cannot_write_a_different_weight():
    model,weight=projection();model=onnx.shape_inference.infer_shapes(model)
    digest=hashlib.sha256(weight.T.astype('<f4').tobytes()).hexdigest()
    _,plan=planner.partition_static_projections(model,node_names=['project'],expected_weight_sha256=[digest])
    plan['operations'][0]['weight_sha256']='0'*64
    with pytest.raises(ValueError):planner.projection_weight_array(model,plan['operations'][0])


def biased():
    model,weight=projection();model.graph.node[0].input.append('bias')
    next(a for a in model.graph.node[0].attribute if a.name=='beta').f=1.
    model.graph.initializer.append(nh.from_array(np.array([1.,-2.,3.],np.float32),name='bias'))
    return model,weight


def test_one_biased_gemm_keeps_original_bias_in_the_following_gpu_segment():
    from embeddinggemma_tools import projection_coefficients as coefficients
    model,weight=biased();before=[x.SerializeToString() for x in model.graph.initializer]
    expected=ReferenceEvaluator(model).run(None,{'input':np.array([[1.,2.]],np.float32)})[0]
    report=coefficients.split_gemm_bias(model)
    assert report['split_biased_gemm_count']==1 and report['source_initializer_bytes_preserved'] is True
    assert before==[x.SerializeToString() for x in model.graph.initializer]
    onnx.checker.check_model(model)
    digest=hashlib.sha256(weight.T.astype('<f4').tobytes()).hexdigest()
    parts,plan=planner.partition_static_projections(model,node_names=['project'],expected_weight_sha256=[digest])
    values={'input':np.array([[1.,2.]],np.float32)}
    for index,part in enumerate(parts):
        outputs=ReferenceEvaluator(part).run(None,{v.name:values[v.name] for v in part.graph.input})
        aliases=plan['segments'][index]['output_aliases'];values.update(zip([aliases.get(v.name,v.name) for v in part.graph.output],outputs))
        if index==0:
            op=plan['operations'][0];values[op['output']]=values[op['input']]@planner.projection_weight_array(model,op)
    np.testing.assert_array_equal(values['out'],expected)
    assert any(node.op_type=='Add' for node in parts[-1].graph.node)


@pytest.mark.parametrize('problem',['beta','alpha','shape','datatype'])
def test_unqualified_bias_split_fails_before_mutation(problem):
    from embeddinggemma_tools import projection_coefficients as coefficients
    model,_=biased()
    if problem in ('alpha','beta'):next(a for a in model.graph.node[0].attribute if a.name==problem).f=.5
    if problem=='shape':model.graph.initializer[-1].dims[:]=[1,3]
    if problem=='datatype':model.graph.initializer[-1].data_type=T.DOUBLE
    before=model.SerializeToString()
    with pytest.raises(ValueError):coefficients.split_gemm_bias(model)
    assert model.SerializeToString()==before
