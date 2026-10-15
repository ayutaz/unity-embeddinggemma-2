import hashlib
import importlib
import numpy as np
import onnx
from onnx import helper,numpy_helper,TensorProto
from onnx.reference import ReferenceEvaluator
import pytest


def required():
    try: module=importlib.import_module('embeddinggemma_tools.partition_projections')
    except ModuleNotFoundError: pytest.fail('GPU projection partition planner is missing')
    return module.partition_static_projections


def model():
    weight=np.arange(16,dtype=np.float32).reshape(4,4)/16
    nodes=[helper.make_node('MatMul',['input','weight'],['a'],name='first'),
           helper.make_node('Relu',['a'],['b'],name='activation'),
           helper.make_node('MatMul',['b','weight'],['c'],name='second'),
           helper.make_node('Add',['c','input'],['out'],name='residual')]
    def info(name):return helper.make_tensor_value_info(name,TensorProto.FLOAT,[1,4])
    graph=helper.make_graph(nodes,'two-projections',[info('input')],[info('out')],
        initializer=[numpy_helper.from_array(weight,name='weight')],value_info=[info(n) for n in ['a','b','c']])
    return helper.make_model(graph,opset_imports=[helper.make_opsetid('',24)],ir_version=9),weight


def silu_model():
    source,w=model();nodes=list(source.graph.node)
    nodes[1:2]=[helper.make_node('Neg',['a'],['neg'],name='silu_neg'),
                helper.make_node('Exp',['neg'],['exp'],name='silu_exp'),
                helper.make_node('Add',['exp','one'],['den'],name='silu_add'),
                helper.make_node('Div',['a','den'],['b'],name='silu')]
    del source.graph.node[:];source.graph.node.extend(nodes)
    source.graph.initializer.append(numpy_helper.from_array(np.array(1,dtype=np.float32),name='one'))
    source.graph.value_info.extend([helper.make_tensor_value_info(n,TensorProto.FLOAT,[1,4]) for n in ['neg','exp','den']])
    return source,w


def test_silu_and_matrix_boundaries_reconnect_without_retaining_silu_internal_nodes():
    source,w=silu_model();before=source.SerializeToString();sha=hashlib.sha256(w.tobytes()).hexdigest()
    parts,plan=required()(source,node_names=['first','second'],expected_weight_sha256=[sha,sha],silu_node_names=['silu'])
    assert source.SerializeToString()==before and len(parts)==4
    assert [op['kind'] for op in plan['operations']]==['matmul','silu','matmul']
    assert len(plan['projections'])==2
    values={'input':np.array([[1,-2,3,-4]],dtype=np.float32)}
    for index,part in enumerate(parts):
        onnx.checker.check_model(part)
        assert not any(n.name.startswith('silu') for n in part.graph.node)
        result=ReferenceEvaluator(part).run(None,{v.name:values[v.name] for v in part.graph.input})
        aliases=plan['segments'][index]['output_aliases']
        values.update(dict(zip([aliases.get(v.name,v.name) for v in part.graph.output],result)))
        if index<len(plan['operations']):
            op=plan['operations'][index];x=values[op['input']]
            values[op['output']]=x@w if op['kind']=='matmul' else x/(np.float32(1)+np.exp(-x))
    np.testing.assert_array_equal(values['out'],ReferenceEvaluator(source).run(None,{'input':values['input']})[0])


@pytest.mark.parametrize('problem',['missing','duplicate','constant','operator','shared','shape'])
def test_silu_partition_refuses_unverified_chain_without_mutating_original(problem):
    source,w=silu_model();names=['silu'];sha=hashlib.sha256(w.tobytes()).hexdigest()
    if problem=='missing':names=['missing']
    elif problem=='duplicate':names=['silu','silu']
    elif problem=='constant':source.graph.initializer[-1].CopyFrom(numpy_helper.from_array(np.array(2,dtype=np.float32),name='one'))
    elif problem=='operator':source.graph.node[2].op_type='Relu'
    elif problem=='shared':source.graph.node[-1].input[1]='exp'
    elif problem=='shape':source.graph.value_info[1].type.tensor_type.shape.dim[1].ClearField('dim_value')
    before=source.SerializeToString()
    with pytest.raises(ValueError):required()(source,node_names=['first','second'],expected_weight_sha256=[sha,sha],silu_node_names=names)
    assert source.SerializeToString()==before


def test_silu_partition_preserves_unrelated_unnamed_onnx_nodes():
    source,w=silu_model()
    for node in source.graph.node[1:4]:node.name=''
    # The exporter does not name internal Neg/Exp/Add nodes.
    source.graph.node[-1].name=''
    sha=hashlib.sha256(w.tobytes()).hexdigest()
    parts,plan=required()(source,node_names=['first','second'],expected_weight_sha256=[sha,sha],silu_node_names=['silu'])
    assert len(parts)==4 and [n.op_type for n in parts[-1].graph.node]==['Add']


def dynamic_model():
    def info(name,shape):return helper.make_tensor_value_info(name,TensorProto.FLOAT,shape)
    nodes=[helper.make_node('Transpose',['input'],['right'],name='transpose',perm=[0,2,1]),
           helper.make_node('MatMul',['input','right'],['product'],name='dynamic'),
           helper.make_node('Relu',['product'],['out'],name='finish')]
    graph=helper.make_graph(nodes,'dynamic',[info('input',[2,2,3])],[info('out',[2,2,2])],
        value_info=[info('right',[2,3,2]),info('product',[2,2,2])])
    return helper.make_model(graph,opset_imports=[helper.make_opsetid('',24)],ir_version=9)


def query_scale_model():
    def info(name):return helper.make_tensor_value_info(name,TensorProto.FLOAT,[2,3])
    nodes=[helper.make_node('Mul',['input','scalar'],['scaled'],name='first_scale'),
           helper.make_node('Mul',['scaled','vector'],['out'],name='query_scale')]
    graph=helper.make_graph(nodes,'scaling',[info('input')],[info('out')],value_info=[info('scaled')],
        initializer=[numpy_helper.from_array(np.array(.127517431974411,dtype=np.float32),name='scalar'),
                     numpy_helper.from_array(np.array([.004766542,.023,.1278625],dtype=np.float32),name='vector')])
    return helper.make_model(graph,opset_imports=[helper.make_opsetid('',24)],ir_version=9)


def test_query_scale_partition_preserves_two_original_roundings_and_coefficient_bytes():
    source=query_scale_model();before=source.SerializeToString()
    parts,plan=required()(source,node_names=[],expected_weight_sha256=[],query_scale_node_names=['query_scale'])
    assert source.SerializeToString()==before and len(parts)==2 and plan['projections']==[]
    operation=plan['operations'][0];assert operation['kind']=='sequential_scale'
    assert operation['inputs']==['input'] and operation['scale_weight']=='vector'
    vector=numpy_helper.to_array(source.graph.initializer[1]);scalar=numpy_helper.to_array(source.graph.initializer[0])
    assert operation['first_scale']==float(scalar)
    assert operation['weight_sha256']==hashlib.sha256(vector.astype('<f4').tobytes()).hexdigest()
    values={'input':np.arange(6,dtype=np.float32).reshape(2,3)}
    for index,part in enumerate(parts):
        onnx.checker.check_model(part)
        assert not any(n.name in ('first_scale','query_scale') for n in part.graph.node)
        result=ReferenceEvaluator(part).run(None,{v.name:values[v.name] for v in part.graph.input})
        aliases=plan['segments'][index]['output_aliases'];values.update(zip([aliases.get(v.name,v.name) for v in part.graph.output],result))
        if index==0:values['out']=(values['input']*scalar)*vector
    np.testing.assert_array_equal(values['out'],ReferenceEvaluator(source).run(None,{'input':values['input']})[0])


@pytest.mark.parametrize('problem',['missing','duplicate','scalar','vector','operator','shared','shape'])
def test_query_scale_partition_refuses_unverified_chain_without_mutation(problem):
    source=query_scale_model();names=['query_scale']
    if problem=='missing':names=['missing']
    elif problem=='duplicate':names*=2
    elif problem=='scalar':source.graph.initializer[0].CopyFrom(numpy_helper.from_array(np.array([1,2],dtype=np.float32),name='scalar'))
    elif problem=='vector':source.graph.initializer[1].CopyFrom(numpy_helper.from_array(np.array([1,2],dtype=np.float32),name='vector'))
    elif problem=='operator':source.graph.node[0].op_type='Add'
    elif problem=='shared':source.graph.output.append(helper.make_tensor_value_info('scaled',TensorProto.FLOAT,[2,3]))
    elif problem=='shape':source.graph.input[0].type.tensor_type.shape.dim[1].ClearField('dim_value')
    before=source.SerializeToString()
    with pytest.raises(ValueError):required()(source,node_names=[],expected_weight_sha256=[],query_scale_node_names=names)
    assert source.SerializeToString()==before


def test_dynamic_matmul_partition_binds_both_live_gpu_operands():
    source=dynamic_model();before=source.SerializeToString()
    parts,plan=required()(source,node_names=[],expected_weight_sha256=[],dynamic_node_names=['dynamic'])
    assert source.SerializeToString()==before and len(parts)==2 and plan['projections']==[]
    op=plan['operations'][0];assert op['kind']=='batched_matmul' and op['inputs']==['input','right']
    values={'input':np.arange(12,dtype=np.float32).reshape(2,2,3)}
    for index,part in enumerate(parts):
        onnx.checker.check_model(part);outputs=ReferenceEvaluator(part).run(None,{v.name:values[v.name] for v in part.graph.input})
        aliases=plan['segments'][index]['output_aliases'];values.update(dict(zip([aliases.get(v.name,v.name) for v in part.graph.output],outputs)))
        if index==0:values[op['output']]=values[op['inputs'][0]]@values[op['inputs'][1]]
    np.testing.assert_array_equal(values['out'],ReferenceEvaluator(source).run(None,{'input':values['input']})[0])


def test_later_shape_dependency_keeps_original_gpu_boundary_live_without_recomputing_it():
    source=dynamic_model();source.graph.node[-1].CopyFrom(helper.make_node('Identity',['product'],['middle'],name='carry'))
    source.graph.node.extend([
        helper.make_node('Shape',['product'],['dimensions'],name='shape'),
        helper.make_node('MatMul',['middle','middle'],['second_product'],name='second'),
        helper.make_node('Reshape',['second_product','dimensions'],['out'],name='restore')])
    source.graph.value_info.extend([
        helper.make_tensor_value_info('middle',TensorProto.FLOAT,[2,2,2]),
        helper.make_tensor_value_info('second_product',TensorProto.FLOAT,[2,2,2]),
        helper.make_tensor_value_info('dimensions',TensorProto.INT64,[3])])
    before=source.SerializeToString()
    parts,plan=required()(source,node_names=[],expected_weight_sha256=[],dynamic_node_names=['dynamic','second'])
    assert source.SerializeToString()==before
    assert 'product' in plan['segments'][1]['outputs']
    values={'input':np.arange(12,dtype=np.float32).reshape(2,2,3)}
    for index,part in enumerate(parts):
        onnx.checker.check_model(part)
        assert not any(n.name in ('dynamic','second') for n in part.graph.node)
        result=ReferenceEvaluator(part).run(None,{v.name:values[v.name] for v in part.graph.input})
        aliases=plan['segments'][index]['output_aliases']
        values.update(zip([aliases.get(v.name,v.name) for v in part.graph.output],result))
        if index<len(plan['operations']):
            op=plan['operations'][index];values[op['output']]=values[op['inputs'][0]]@values[op['inputs'][1]]
    np.testing.assert_array_equal(values['out'],ReferenceEvaluator(source).run(None,{'input':values['input']})[0])


@pytest.mark.parametrize('problem',['missing','duplicate','shape','inner','batch'])
def test_dynamic_matmul_boundary_rejects_invalid_metadata_before_mutation(problem):
    source=dynamic_model();names=['dynamic']
    if problem=='missing':names=['missing']
    elif problem=='duplicate':names=['dynamic','dynamic']
    elif problem=='shape':source.graph.value_info[0].type.tensor_type.shape.dim[0].ClearField('dim_value')
    elif problem=='inner':source.graph.value_info[0].type.tensor_type.shape.dim[1].dim_value=4
    elif problem=='batch':source.graph.value_info[0].type.tensor_type.shape.dim[0].dim_value=3
    before=source.SerializeToString()
    with pytest.raises(ValueError):required()(source,node_names=[],expected_weight_sha256=[],dynamic_node_names=names)
    assert source.SerializeToString()==before


def test_partition_reconnects_exact_residual_and_preserves_source_bytes():
    source,w=model();before=source.SerializeToString();sha=hashlib.sha256(w.tobytes()).hexdigest()
    parts,plan=required()(source,node_names=['first','second'],expected_weight_sha256=[sha,sha])
    assert source.SerializeToString()==before and len(parts)==3
    values={'input':np.array([[1,-2,3,-4]],dtype=np.float32)}
    for index,part in enumerate(parts):
        onnx.checker.check_model(part)
        outputs=ReferenceEvaluator(part).run(None,{v.name:values[v.name] for v in part.graph.input})
        aliases=plan['segments'][index]['output_aliases']
        values.update(dict(zip([aliases.get(v.name,v.name) for v in part.graph.output],outputs)))
        if index<2:
            projection=plan['projections'][index]
            values[projection['output']]=values[projection['input']]@w
    expected=ReferenceEvaluator(source).run(None,{'input':values['input']})[0]
    np.testing.assert_array_equal(values['out'],expected)
    assert plan['source_initializer_bytes_preserved']
    assert 'input' in plan['segments'][1]['outputs']
    assert [p['weight_sha256'] for p in plan['projections']]==[sha,sha]


def test_each_segment_declares_only_used_inputs_including_residual_passthrough():
    source,w=model();sha=hashlib.sha256(w.tobytes()).hexdigest()
    parts,_=required()(source,node_names=['first','second'],expected_weight_sha256=[sha,sha])
    for part in parts:
        used={v for n in part.graph.node for v in n.input}|{v.name for v in part.graph.output}
        assert all(v.name in used for v in part.graph.input)


def test_passthrough_outputs_have_explicit_identity_for_sentis_import():
    source,w=model();sha=hashlib.sha256(w.tobytes()).hexdigest()
    parts,_=required()(source,node_names=['first','second'],expected_weight_sha256=[sha,sha])
    for part in parts:
        inputs={v.name for v in part.graph.input}
        assert not inputs.intersection(v.name for v in part.graph.output)


def test_constant_input_projection_is_cut_and_reconnected_without_cpu_folding():
    source,w=model();constant=np.ones((1,4),dtype=np.float32)
    nodes=list(source.graph.node)
    nodes[1:1]=[helper.make_node('Constant',[],['position'],value=numpy_helper.from_array(constant)),
                helper.make_node('MatMul',['position','weight'],['relative'],name='position_projection')]
    nodes[-1].output[0]='sum'
    nodes.append(helper.make_node('Add',['sum','relative'],['out'],name='with_position'))
    del source.graph.node[:];source.graph.node.extend(nodes)
    source.graph.value_info.extend([helper.make_tensor_value_info(n,TensorProto.FLOAT,[1,4]) for n in ['position','relative','sum']])
    sha=hashlib.sha256(w.tobytes()).hexdigest()
    parts,plan=required()(source,node_names=['first','position_projection','second'],expected_weight_sha256=[sha]*3)
    values={'input':np.array([[1,-2,3,-4]],dtype=np.float32)}
    for index,part in enumerate(parts):
        onnx.checker.check_model(part)
        assert not any(n.name in ['first','position_projection','second'] for n in part.graph.node)
        outputs=ReferenceEvaluator(part).run(None,{v.name:values[v.name] for v in part.graph.input})
        aliases=plan['segments'][index]['output_aliases']
        values.update(dict(zip([aliases.get(v.name,v.name) for v in part.graph.output],outputs)))
        if index<3:
            projection=plan['projections'][index]
            values[projection['output']]=values[projection['input']]@w
    expected=ReferenceEvaluator(source).run(None,{'input':values['input']})[0]
    np.testing.assert_array_equal(values['out'],expected)


@pytest.mark.parametrize('problem',['missing','duplicate','order','hash','shape'])
def test_partition_refuses_unproven_boundaries_before_mutation(problem):
    source,w=model();sha=hashlib.sha256(w.tobytes()).hexdigest();names=['first','second'];hashes=[sha,sha]
    if problem=='missing':names=['missing','second']
    elif problem=='duplicate':names=['first','first']
    elif problem=='order':names=['second','first']
    elif problem=='hash':hashes[0]='0'*64
    elif problem=='shape':source.graph.value_info[1].type.tensor_type.shape.dim[1].ClearField('dim_value')
    before=source.SerializeToString()
    with pytest.raises(ValueError):required()(source,node_names=names,expected_weight_sha256=hashes)
    assert source.SerializeToString()==before
