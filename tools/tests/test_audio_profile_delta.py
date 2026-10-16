import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import helper,numpy_helper
from onnx.reference import ReferenceEvaluator
import pytest

from embeddinggemma_tools import audio_profile_delta as module


def models():
    def model(valid):
        mask=np.array([1]*valid+[0]*(8-valid),dtype=np.float32)
        nodes=[helper.make_node('Mul',['input','mask'],['masked'],name='mask_input'),
               helper.make_node('Constant',[],['end'],name='end',value=numpy_helper.from_array(np.array([valid],dtype=np.int64))),
               helper.make_node('Slice',['masked','start','end','axes'],['output'],name='slice')]
        graph=helper.make_graph(nodes,'profile',[helper.make_tensor_value_info('input',onnx.TensorProto.FLOAT,[8])],
                                [helper.make_tensor_value_info('output',onnx.TensorProto.FLOAT,[valid])],
                                [numpy_helper.from_array(mask,'mask'),numpy_helper.from_array(np.array([0],dtype=np.int64),'start'),
                                 numpy_helper.from_array(np.array([0],dtype=np.int64),'axes')])
        return helper.make_model(graph,opset_imports=[helper.make_opsetid('',20)])
    return model(8),model(6)


def sha(model):return hashlib.sha256(model.SerializeToString()).hexdigest()


def compare(left,right):
    return module.compare_profiles(left,right,source_sha256=sha(left),target_sha256=sha(right))


def test_profile_difference_reports_original_bytes_without_mutating_models():
    left,right=models();before=(left.SerializeToString(),right.SerializeToString())
    result=compare(left,right)
    assert result['success'] and result['scope']=='original_onnx_profile_structure_comparison_only'
    assert not result['gpu_validated'] and not result['numeric_validated']
    assert result['source_sha256']==sha(left) and result['target_sha256']==sha(right)
    assert [r['name'] for r in result['changed_initializers']]==['mask']
    assert result['changed_initializers'][0]['consumers']==[dict(node='mask_input',input_index=1)]
    assert result['changed_initializers'][0]['before_proto_base64']
    assert result['changed_initializers'][0]['before']['shape']==[8]
    assert result['changed_initializers'][0]['after']['values']==[1,1,1,1,1,1,0,0]
    assert [r['name'] for r in result['changed_nodes']]==['end']
    assert result['changed_nodes'][0]['consumers']==[dict(node='slice',input_index=2)]
    assert result['changed_nodes'][0]['after']['attributes'][0]['tensor']['values']==[6]
    assert result['outputs'][0]['before']['shape']==[8] and result['outputs'][0]['after']['shape']==[6]
    assert result['unchanged_initializer_count']==2
    assert before==(left.SerializeToString(),right.SerializeToString())


@pytest.mark.parametrize('problem',['source_hash','target_hash','node_order','node_input','op_type','weight_shape','input_shape','large_change'])
def test_profile_comparison_rejects_unproved_structure_or_unbounded_payload(problem):
    left,right=models();kwargs=dict(source_sha256=sha(left),target_sha256=sha(right),max_changed_tensor_bytes=1024)
    if problem=='source_hash':kwargs['source_sha256']='0'*64
    elif problem=='target_hash':kwargs['target_sha256']='0'*64
    elif problem=='node_order':right.graph.node.reverse()
    elif problem=='node_input':right.graph.node[0].input[0]='other'
    elif problem=='op_type':right.graph.node[0].op_type='Add'
    elif problem=='weight_shape':right.graph.initializer[0].dims[0]=7
    elif problem=='input_shape':right.graph.input[0].type.tensor_type.shape.dim[0].dim_value=7
    elif problem=='large_change':kwargs['max_changed_tensor_bytes']=8
    if problem not in ('source_hash','target_hash'):
        kwargs['target_sha256']=sha(right)
    before=copy.deepcopy((left.SerializeToString(),right.SerializeToString()))
    with pytest.raises(ValueError):module.compare_profiles(left,right,**kwargs)
    assert before==(left.SerializeToString(),right.SerializeToString())


def test_cli_reports_original_delta_without_writing_models_and_refuses_overwrite(tmp_path):
    left,right=models();source=tmp_path/'source.onnx';target=tmp_path/'target.onnx';output=tmp_path/'report.json'
    onnx.save(left,source);onnx.save(right,target)
    args=['--source',str(source),'--target',str(target),'--source-sha256',sha(left),'--target-sha256',sha(right),'--output',str(output)]
    module.main(args)
    assert json.loads(output.read_text())['changed_initializers'][0]['name']=='mask'
    assert sorted(p.name for p in tmp_path.iterdir())==['report.json','source.onnx','target.onnx']
    before=output.read_bytes()
    with pytest.raises(ValueError,match='already exists'):module.main(args)
    assert output.read_bytes()==before


def test_profile_consumers_follow_identity_aliases_to_stable_semantic_nodes():
    left,right=models()
    for model in (left,right):
        model.graph.node.insert(0,helper.make_node('Identity',['mask'],['aliased'],name='unstable_export_id'))
        model.graph.node[1].input[1]='aliased'
    assert compare(left,right)['changed_initializers'][0]['consumers']==[dict(node='mask_input',input_index=1)]


def delta_sha(delta):return hashlib.sha256(json.dumps(delta,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def apply(parts,delta,**overrides):
    kwargs=dict(expected_delta_sha256=delta_sha(delta),expected_source_sha256=delta['source_sha256'],expected_target_sha256=delta['target_sha256'])
    kwargs.update(overrides)
    return module.apply_profile_delta(parts,delta,**kwargs)


def test_audited_profile_delta_applies_to_renamed_derived_masks_and_preserves_other_values():
    left,right=models();delta=compare(left,right);derived=copy.deepcopy(left)
    derived.graph.initializer[0].name='derived_mask'
    derived.graph.node.insert(0,helper.make_node('Identity',['derived_mask'],['alias'],name='different_alias'))
    derived.graph.node[1].input[1]='alias';derived.graph.node[2].name='different_constant_name'
    unchanged=[v.SerializeToString() for v in derived.graph.initializer[1:]]
    report=apply([derived],delta)
    assert report['profile_items_applied']==2 and not report['gpu_validated']
    assert [v.SerializeToString() for v in derived.graph.initializer[1:]]==unchanged
    assert derived.graph.initializer[0].name=='derived_mask'
    onnx.checker.check_model(derived)
    values=np.arange(8,dtype=np.float32)
    np.testing.assert_array_equal(ReferenceEvaluator(derived).run(None,{'input':values})[0],ReferenceEvaluator(right).run(None,{'input':values})[0])
    assert list(derived.graph.output[0].type.tensor_type.shape.dim)[0].dim_value==6


@pytest.mark.parametrize('problem',['source_sha','target_sha','delta_sha','mask_value','consumer','crop_value','output_shape'])
def test_profile_application_rejects_wrong_scope_before_any_partition_is_changed(problem):
    left,right=models();delta=compare(left,right);derived=copy.deepcopy(left);kwargs={}
    if problem=='source_sha':kwargs['expected_source_sha256']='0'*64
    elif problem=='target_sha':kwargs['expected_target_sha256']='0'*64
    elif problem=='delta_sha':kwargs['expected_delta_sha256']='0'*64
    elif problem=='mask_value':derived.graph.initializer[0].CopyFrom(numpy_helper.from_array(np.zeros(8,dtype=np.float32),'mask'))
    elif problem=='consumer':derived.graph.node[0].name='missing_consumer'
    elif problem=='crop_value':derived.graph.node[1].attribute[0].t.CopyFrom(numpy_helper.from_array(np.array([7],dtype=np.int64)))
    elif problem=='output_shape':derived.graph.output[0].type.tensor_type.shape.dim[0].dim_value=7
    before=derived.SerializeToString()
    with pytest.raises(ValueError):apply([derived],delta,**kwargs)
    assert derived.SerializeToString()==before


def test_all_partitions_are_audited_before_mutating_an_earlier_valid_partition():
    left,right=models();first=copy.deepcopy(left);last=copy.deepcopy(left);delta=compare(left,right)
    last.graph.initializer[0].CopyFrom(numpy_helper.from_array(np.zeros(8,dtype=np.float32),'mask'))
    before=(first.SerializeToString(),last.SerializeToString())
    with pytest.raises(ValueError):apply([first,last],delta)
    assert before==(first.SerializeToString(),last.SerializeToString())


def test_actions_comparison_uses_both_immutable_original_graphs_without_model_download_from_hf():
    text=(Path(__file__).parents[2]/'.github/workflows/audio-profile-delta.yml').read_text()
    for required in ('11814567883','frames-1119-valid-1119/audio.onnx','frames-1119-valid-839/audio.onnx',
                     '7770a8af258f2de32c5587497b820596ea4e1ed7ec5382c3ee90eada2bec7e4a',
                     'af2e97b777349fa2e5f74790d24bb4b2d899c6a8438f372253737bc5940944c7',
                     'embeddinggemma_tools.audio_profile_delta','max-entry-bytes','if: always()'):
        assert required in text
    assert 'hf_hub_download' not in text and 'continue-on-error' not in text
