import hashlib
import pytest
import numpy as np
from onnx import helper,numpy_helper,TensorProto
from embeddinggemma_tools import partition_projections
from test_partition_projections import model


def required():
    function=getattr(partition_projections,'apply_projection_shape_audit',None)
    assert callable(function),'Hash-bound fixed Sentis shape audit is missing'
    return function


def context():
    source,w=model();source.graph.value_info[1].type.tensor_type.shape.dim[1].ClearField('dim_value')
    graph_sha=hashlib.sha256(source.SerializeToString()).hexdigest();weight_sha=hashlib.sha256(w.tobytes()).hexdigest()
    audit=dict(success=True,scope='fixed_model_metadata_only_no_inference',source_graph_sha256=graph_sha,
        projections=[dict(weight_sha256=weight_sha,weight_shape=[4,4],input_shape=[1,4],output_shape=[1,4])])
    return source,audit,graph_sha


def test_verified_shapes_fill_only_unknown_metadata_and_preserve_node_and_weight_bytes():
    source,audit,sha=context()
    before_nodes=[n.SerializeToString() for n in source.graph.node];before_weights=[w.SerializeToString() for w in source.graph.initializer]
    result=required()(source,audit,expected_graph_sha256=sha)
    assert result['projection_count']==2
    assert [n.SerializeToString() for n in source.graph.node]==before_nodes
    assert [w.SerializeToString() for w in source.graph.initializer]==before_weights
    assert source.graph.value_info[1].type.tensor_type.shape.dim[1].dim_value==4


@pytest.mark.parametrize('problem',['source','failed','weight','shape','conflict'])
def test_shape_audit_rejects_unproven_context_before_mutation(problem):
    source,audit,sha=context()
    if problem=='source':audit['source_graph_sha256']='0'*64
    elif problem=='failed':audit['success']=False
    elif problem=='weight':audit['projections'][0]['weight_sha256']='0'*64
    elif problem=='shape':audit['projections'][0]['input_shape']=[0,4]
    elif problem=='conflict':audit['projections'][0]['input_shape']=[2,4];audit['projections'][0]['output_shape']=[2,4]
    before=source.SerializeToString()
    with pytest.raises(ValueError):required()(source,audit,expected_graph_sha256=sha)
    assert source.SerializeToString()==before


def constant_context():
    source,audit,_=context()
    x=np.arange(8,dtype=np.float32).reshape(2,4)
    w=np.eye(4,dtype=np.float32)*2
    source.graph.initializer.append(numpy_helper.from_array(w,name='position_weight'))
    source.graph.node.extend([
        helper.make_node('Constant',[],['position'],value=numpy_helper.from_array(x)),
        helper.make_node('MatMul',['position','position_weight'],['position_out'],name='position_projection')])
    source.graph.value_info.extend([helper.make_tensor_value_info('position',TensorProto.FLOAT,[2,4]),
                                   helper.make_tensor_value_info('position_out',TensorProto.FLOAT,[2,4])])
    sha=hashlib.sha256(source.SerializeToString()).hexdigest()
    audit.update(success=False,source_graph_sha256=sha,projection_count=1,error='Coverage assertion: missing folded constants')
    return source,audit,sha


def test_complete_shape_audit_accounts_for_constant_folded_projections_without_computing_them():
    source,audit,sha=constant_context();before=source.SerializeToString()
    function=getattr(partition_projections,'complete_projection_shape_audit',None)
    assert callable(function),'Constant-folded projection shape coverage is missing'
    completed=function(source,audit,expected_graph_sha256=sha)
    assert completed['success'] and completed['projection_count']==3
    assert completed['runtime_projection_count']==2 and completed['constant_projection_count']==1
    assert completed['original_runtime_audit_success'] is False
    assert completed['projections'][-1]['input_shape']==[2,4]
    assert completed['gpu_validated'] is False
    assert source.SerializeToString()==before
    required()(source,completed,expected_graph_sha256=sha)


@pytest.mark.parametrize('problem',['missing_runtime','source','constant_conflict','wrong_scope'])
def test_completion_rejects_unproven_runtime_or_constant_shapes(problem):
    source,audit,sha=constant_context()
    if problem=='missing_runtime':audit['projections']=[]
    elif problem=='source':audit['source_graph_sha256']='0'*64
    elif problem=='wrong_scope':audit['scope']='gpu_numeric_pass'
    elif problem=='constant_conflict':
        source.graph.value_info[-2].type.tensor_type.shape.dim[0].dim_value=3
        sha=hashlib.sha256(source.SerializeToString()).hexdigest();audit['source_graph_sha256']=sha
    before=source.SerializeToString()
    function=getattr(partition_projections,'complete_projection_shape_audit',None)
    assert callable(function),'Constant-folded projection shape coverage is missing'
    with pytest.raises(ValueError):function(source,audit,expected_graph_sha256=sha)
    assert source.SerializeToString()==before


def boundary_context():
    source,_,sha=context()
    audit=dict(success=True,scope='fixed_boundary_metadata_only_no_inference',source_graph_sha256=sha,
               boundaries={v.name:[1,4] for v in (*source.graph.input,*source.graph.output,*source.graph.value_info)})
    return source,audit,sha


def test_boundary_audit_fills_fixed_shapes_without_changing_operations_or_coefficients():
    source,audit,sha=boundary_context()
    function=getattr(partition_projections,'apply_boundary_shape_audit',None)
    assert callable(function),'GPU boundary shape overlay is missing'
    nodes=[n.SerializeToString() for n in source.graph.node];weights=[w.SerializeToString() for w in source.graph.initializer]
    result=function(source,audit,expected_graph_sha256=sha)
    assert result['boundary_count']==5 and result['full_encoder_validated'] is False
    assert source.graph.value_info[1].type.tensor_type.shape.dim[1].dim_value==4
    assert nodes==[n.SerializeToString() for n in source.graph.node]
    assert weights==[w.SerializeToString() for w in source.graph.initializer]


@pytest.mark.parametrize('problem',['source','failed','zero','missing_name','known_conflict','rank'])
def test_boundary_overlay_rejects_bad_audit_without_partial_mutation(problem):
    source,audit,sha=boundary_context()
    if problem=='source':audit['source_graph_sha256']='0'*64
    elif problem=='failed':audit['success']=False
    elif problem=='zero':audit['boundaries']['b']=[1,0]
    elif problem=='missing_name':audit['boundaries']['missing']=[1,4]
    elif problem=='known_conflict':audit['boundaries']['c']=[2,4]
    elif problem=='rank':audit['boundaries']['c']=[1,1,4]
    before=source.SerializeToString()
    function=getattr(partition_projections,'apply_boundary_shape_audit',None)
    assert callable(function),'GPU boundary shape overlay is missing'
    with pytest.raises(ValueError):function(source,audit,expected_graph_sha256=sha)
    assert source.SerializeToString()==before


def test_boundary_shape_audit_accepts_original_integer_shape_metadata_and_scalar_rank():
    source,audit,_=boundary_context()
    source.graph.value_info.append(helper.make_tensor_value_info('axis',TensorProto.INT64,[]))
    sha=hashlib.sha256(source.SerializeToString()).hexdigest()
    audit['source_graph_sha256']=sha;audit['boundaries']['axis']=[]
    result=partition_projections.apply_boundary_shape_audit(source,audit,expected_graph_sha256=sha)
    assert result['boundary_count']==6
    assert source.graph.value_info[-1].type.tensor_type.elem_type==TensorProto.INT64
