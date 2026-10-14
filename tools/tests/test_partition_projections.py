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
