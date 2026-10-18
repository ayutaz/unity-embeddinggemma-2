import hashlib
import importlib

import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh, TensorProto as T
import pytest


def module(): return importlib.import_module('embeddinggemma_tools.projection_coefficients')


def projection(kind='Gemm', *, expanded=False):
    weight=np.arange(6,dtype=np.float32).reshape(3,2)
    tensor=nh.from_array(weight,name='weight')
    if kind=='Gemm': nodes=[h.make_node('Gemm',['input','weight'],['out'],name='project',alpha=1.,beta=0.,transB=1)]
    else:
        nodes=[h.make_node('Transpose',['weight'],['transposed'],name='transpose',perm=[1,0])]
        if expanded:
            nodes += [h.make_node('Constant',[],['axes'],name='axes',value=nh.from_array(np.array([0],np.int64))),
                      h.make_node('Unsqueeze',['transposed','axes'],['batched'],name='batch'),
                      h.make_node('Constant',[],['target'],name='target',value=nh.from_array(np.array([4,2,3],np.int64))),
                      h.make_node('Expand',['batched','target'],['right'],name='expand')]
        nodes += [h.make_node('MatMul',['input','right' if expanded else 'transposed'],['out'],name='project')]
    x=[4,1,2] if expanded else [1,2];y=[4,1,3] if expanded else [1,3]
    graph=h.make_graph(nodes,'projection',[h.make_tensor_value_info('input',T.FLOAT,x)],[h.make_tensor_value_info('out',T.FLOAT,y)],initializer=[tensor])
    return h.make_model(graph,opset_imports=[h.make_opsetid('',24)],ir_version=9),weight


@pytest.mark.parametrize('kind,expanded',[('Gemm',False),('MatMul',False),('MatMul',True)])
def test_static_projection_keeps_original_coefficient_bytes_and_resolves_exact_matrix(kind,expanded):
    model,weight=projection(kind,expanded=expanded);before=model.SerializeToString()
    matrix,report=module().resolve_projection(model,'project')
    np.testing.assert_array_equal(matrix,weight.T)
    assert matrix.dtype==np.float32 and matrix.shape==(2,3)
    assert report['weight_sha256']==hashlib.sha256(weight.T.astype('<f4').tobytes()).hexdigest()
    assert report['source_weight_sha256']==hashlib.sha256(weight.tobytes()).hexdigest()
    assert report['weight']=='weight' and report['weight_shape']==[2,3]
    assert report['coefficient_only_metadata_views'] is True
    assert model.SerializeToString()==before


def test_batched_weights_with_distinct_matrices_are_not_a_single_static_projection():
    model,_=projection('MatMul',expanded=True)
    model.graph.initializer.append(nh.from_array(np.arange(24,dtype=np.float32).reshape(4,2,3),name='different'))
    model.graph.node[-1].input[1]='different'
    with pytest.raises(ValueError):module().resolve_projection(model,'project')


@pytest.mark.parametrize('problem',['alpha','beta','transA','bias','external','datatype','arithmetic','dynamic','cycle'])
def test_unverified_projection_semantics_fail_without_mutating_source(problem):
    model,_=projection();node=model.graph.node[0]
    if problem in ('alpha','beta','transA'):
        value={'alpha':.5,'beta':1.,'transA':1}[problem]
        for attr in node.attribute:
            if attr.name==problem:attr.CopyFrom(h.make_attribute(problem,value));break
        else:node.attribute.append(h.make_attribute(problem,value))
    if problem=='bias':node.input.append('bias')
    if problem=='external':model.graph.initializer[0].data_location=T.EXTERNAL
    if problem=='datatype':model.graph.initializer[0].data_type=T.DOUBLE
    if problem=='arithmetic':
        node.input[1]='altered';model.graph.node.insert(0,h.make_node('Neg',['weight'],['altered'],name='neg'))
    if problem=='dynamic':node.input[1]='input'
    if problem=='cycle':
        node.input[1]='loop';model.graph.node.insert(0,h.make_node('Identity',['loop'],['loop'],name='loop'))
    before=model.SerializeToString()
    with pytest.raises(ValueError):module().resolve_projection(model,'project')
    assert model.SerializeToString()==before


def test_shape_only_integer_graph_can_resolve_expand_without_tensor_values():
    model,weight=projection('MatMul',expanded=True)
    nodes=list(model.graph.node)
    nodes[3:4]=[
        h.make_node('Shape',['input'],['input_shape'],name='shape'),
        h.make_node('Constant',[],['zero'],name='zero',value=nh.from_array(np.array([0],np.int64))),
        h.make_node('Gather',['input_shape','zero'],['batch_count'],name='batch_count'),
        h.make_node('Constant',[],['tail'],name='tail',value=nh.from_array(np.array([2,3],np.int64))),
        h.make_node('Concat',['batch_count','tail'],['target'],name='shape_concat',axis=0)]
    del model.graph.node[:];model.graph.node.extend(nodes)
    matrix,_=module().resolve_projection(model,'project')
    np.testing.assert_array_equal(matrix,weight.T)


def test_torch_expand_integer_shape_program_keeps_only_broadcast_weight_copies():
    model,weight=projection('MatMul',expanded=True);nodes=list(model.graph.node)
    nodes[3:4]=[
        h.make_node('Constant',[],['rank'],name='rank',value=nh.from_array(np.array([3],np.int64))),
        h.make_node('ConstantOfShape',['rank'],['ones'],name='ones',value=nh.from_array(np.array([1],np.int64))),
        h.make_node('Constant',[],['minus_one'],name='minus_one',value=nh.from_array(np.array(-1,np.int64))),
        h.make_node('Mul',['ones','minus_one'],['minus_ones'],name='minus_ones'),
        h.make_node('Constant',[],['requested'],name='requested',value=nh.from_array(np.array([4,-1,-1],np.int64))),
        h.make_node('Equal',['requested','minus_ones'],['keep'],name='keep'),
        h.make_node('Where',['keep','ones','requested'],['target'],name='target')]
    del model.graph.node[:];model.graph.node.extend(nodes)
    matrix,_=module().resolve_projection(model,'project')
    np.testing.assert_array_equal(matrix,weight.T)


def test_large_or_float_integer_metadata_is_rejected_before_expansion():
    model,_=projection('MatMul',expanded=True)
    for shape in [np.array([1000000000,2,3],np.int64),np.array([4.,2.,3.],np.float32)]:
        model.graph.node[3].attribute[0].t.CopyFrom(nh.from_array(shape))
        with pytest.raises(ValueError):module().resolve_projection(model,'project')
