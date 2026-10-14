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
