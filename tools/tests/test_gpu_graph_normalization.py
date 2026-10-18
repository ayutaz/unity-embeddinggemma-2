import importlib

import numpy as np
import onnx
from onnx import helper as h,numpy_helper as nh
from onnx.reference import ReferenceEvaluator
import pytest

from test_partition_projections import glu_model
from embeddinggemma_tools.partition_projections import partition_static_projections


def module():return importlib.import_module('embeddinggemma_tools.gpu_graph_normalization')


def sliced_glu():
    model=glu_model();nodes=list(model.graph.node)
    constants=[]
    for name,value in [('start0',[0]),('end0',[3]),('start1',[3]),('end1',[6]),('axis',[-1]),('step',[1])]:
        constants.append(nh.from_array(np.array(value,np.int64),name=name))
    model.graph.initializer.extend(constants)
    nodes[:1]=[h.make_node('Slice',['input','start0','end0','axis','step'],['a'],name='first_half'),
        h.make_node('Slice',['input','start1','end1','axis','step'],['b'],name='second_half')]
    del model.graph.node[:];model.graph.node.extend(nodes)
    return model


def test_closed_slice_glu_becomes_a_split_with_identical_values_and_can_be_cut():
    model=sliced_glu();values=np.arange(12,dtype=np.float32).reshape(2,6)/3-2
    expected=ReferenceEvaluator(model).run(None,{'input':values})[0]
    weights=[w.SerializeToString() for w in model.graph.initializer]
    report=module().normalize_glu_slices(model)
    assert report['normalized_glu_count']==1 and report['source_initializer_bytes_preserved'] is True
    assert [w.SerializeToString() for w in model.graph.initializer]==weights
    onnx.checker.check_model(model)
    np.testing.assert_array_equal(ReferenceEvaluator(model).run(None,{'input':values})[0],expected)
    parts,plan=partition_static_projections(model,node_names=[],expected_weight_sha256=[],glu_node_names=['glu'])
    assert len(parts)==2 and plan['operations'][0]['input_shape']==[2,6]


@pytest.mark.parametrize('problem',['axis','start','end','step','shared','different_input'])
def test_unproved_glu_slices_are_rejected_before_mutation(problem):
    model=sliced_glu()
    name={'axis':'axis','start':'start1','end':'end0','step':'step'}.get(problem)
    if name:
        value={'axis':[0],'start':[2],'end':[2],'step':[2]}[problem]
        next(w for w in model.graph.initializer if w.name==name).CopyFrom(nh.from_array(np.array(value,np.int64),name=name))
    if problem=='shared':model.graph.output.append(h.make_tensor_value_info('a',onnx.TensorProto.FLOAT,[2,3]))
    if problem=='different_input':model.graph.node[1].input[0]='a'
    before=model.SerializeToString()
    with pytest.raises(ValueError):module().normalize_glu_slices(model)
    assert model.SerializeToString()==before
