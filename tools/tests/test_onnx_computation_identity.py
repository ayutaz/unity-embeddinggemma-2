import hashlib
import importlib

import onnx
from onnx import TensorProto,helper
import pytest


def module():
    return importlib.import_module('embeddinggemma_tools.onnx_computation_identity')


def model():
    graph=helper.make_graph([helper.make_node('Add',['x','weight'],['y'],name='add')],'fixture',
        [helper.make_tensor_value_info('x',TensorProto.FLOAT,[2])],
        [helper.make_tensor_value_info('y',TensorProto.FLOAT,[2])],
        [helper.make_tensor('weight',TensorProto.FLOAT,[2],[0.25,0.75])])
    return helper.make_model(graph,opset_imports=[helper.make_opsetid('',13)])


def write(path,value):
    onnx.save_model(value,path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_only_doc_strings_and_metadata_props_are_ignored_without_editing_source(tmp_path):
    original=model();left=tmp_path/'original.onnx';left_sha=write(left,original)
    changed=model();changed.doc_string='/'+'home'+'/'+'synthetic'+'/source.py'
    changed.metadata_props.add(key='source',value='debug')
    changed.graph.doc_string='graph debug'
    changed.graph.node[0].doc_string='node debug'
    changed.graph.node[0].metadata_props.add(key='trace',value='debug')
    changed.graph.initializer[0].doc_string='tensor debug'
    changed.graph.input[0].doc_string='input debug'
    right=tmp_path/'changed.onnx';right_sha=write(right,changed)
    before=right.read_bytes()
    a=module().fingerprint(left,expected_sha256=left_sha)
    b=module().fingerprint(right,expected_sha256=right_sha)
    assert a['computation_sha256']==b['computation_sha256'] and left_sha!=right_sha
    assert right.read_bytes()==before
    assert b['ignored_metadata_fields']==['doc_string','metadata_props']
    assert b['metadata_fields_removed']==7
    assert b['node_count']==1 and b['tensor_count']==1
    assert b['model_executed'] is False and b['gpu_validated'] is False


@pytest.mark.parametrize('change',['weight','operator','edge','shape','opset','ir_version'])
def test_computational_changes_always_change_identity(tmp_path,change):
    original=model();left=tmp_path/'original.onnx';a=module().fingerprint(left,expected_sha256=write(left,original))
    changed=model()
    if change=='weight':changed.graph.initializer[0].float_data[0]=0.5
    if change=='operator':changed.graph.node[0].op_type='Sub'
    if change=='edge':changed.graph.node[0].input[0]='weight'
    if change=='shape':changed.graph.input[0].type.tensor_type.shape.dim[0].dim_value=3
    if change=='opset':changed.opset_import[0].version+=1
    if change=='ir_version':changed.ir_version+=1
    right=tmp_path/'changed.onnx';b=module().fingerprint(right,expected_sha256=write(right,changed))
    assert a['computation_sha256']!=b['computation_sha256']


def test_external_tensor_references_are_rejected_without_reading_them(tmp_path):
    value=model();tensor=value.graph.initializer[0]
    tensor.data_location=TensorProto.EXTERNAL;tensor.external_data.add(key='location',value='missing-weights.bin')
    path=tmp_path/'external.onnx';sha=write(path,value)
    with pytest.raises(ValueError,match='embedded'):module().fingerprint(path,expected_sha256=sha)


def test_changed_source_is_rejected_before_protobuf_loading(tmp_path,monkeypatch):
    path=tmp_path/'model.onnx';sha=write(path,model())
    with path.open('ab') as stream:stream.write(b'changed')
    monkeypatch.setattr(onnx,'load_model',lambda *a,**k:pytest.fail('Changed source reached parser'))
    with pytest.raises(ValueError,match='hash'):module().fingerprint(path,expected_sha256=sha)


def test_nested_graph_metadata_is_removed_but_nested_operations_remain_bound(tmp_path):
    value=model();body=model().graph
    value.graph.node[0].attribute.append(helper.make_attribute('body',body))
    left=tmp_path/'left.onnx';a=module().fingerprint(left,expected_sha256=write(left,value))
    value.graph.node[0].attribute[0].g.node[0].doc_string='nested debug'
    right=tmp_path/'right.onnx';b=module().fingerprint(right,expected_sha256=write(right,value))
    assert a['computation_sha256']==b['computation_sha256']
    value.graph.node[0].attribute[0].g.node[0].op_type='Sub'
    changed=tmp_path/'nested-change.onnx';c=module().fingerprint(changed,expected_sha256=write(changed,value))
    assert a['computation_sha256']!=c['computation_sha256']


def test_components_distinguish_weight_names_from_weight_values_and_routing(tmp_path):
    value=model();left=tmp_path/'left.onnx';a=module().fingerprint(left,expected_sha256=write(left,value))
    value.graph.initializer[0].name='renamed'
    value.graph.node[0].input[1]='renamed'
    right=tmp_path/'renamed.onnx';b=module().fingerprint(right,expected_sha256=write(right,value))
    assert a['components']['initializer_values_sha256']==b['components']['initializer_values_sha256']
    assert a['components']['named_initializer_values_sha256']!=b['components']['named_initializer_values_sha256']
    assert a['components']['graph_without_initializers_sha256']!=b['components']['graph_without_initializers_sha256']
    value.graph.initializer[0].float_data[0]=0.5
    changed=tmp_path/'changed.onnx';c=module().fingerprint(changed,expected_sha256=write(changed,value))
    assert b['components']['initializer_values_sha256']!=c['components']['initializer_values_sha256']


def test_component_diagnostics_never_silently_accept_reordered_initializer_bytes(tmp_path):
    value=model();value.graph.initializer.append(helper.make_tensor('extra',TensorProto.FLOAT,[1],[0.5]))
    left=tmp_path/'left.onnx';a=module().fingerprint(left,expected_sha256=write(left,value))
    entries=list(value.graph.initializer);del value.graph.initializer[:];value.graph.initializer.extend(reversed(entries))
    right=tmp_path/'right.onnx';b=module().fingerprint(right,expected_sha256=write(right,value))
    for key in ('initializer_values_sha256','named_initializer_values_sha256','graph_without_initializers_sha256'):
        assert a['components'][key]==b['components'][key]
    assert a['components']['initializer_order_sha256']!=b['components']['initializer_order_sha256']
    assert a['computation_sha256']!=b['computation_sha256']
