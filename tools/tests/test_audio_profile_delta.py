import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import helper,numpy_helper
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
    assert result['changed_initializers'][0]['before']['shape']==[8]
    assert result['changed_initializers'][0]['after']['values']==[1,1,1,1,1,1,0,0]
    assert [r['name'] for r in result['changed_nodes']]==['end']
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


def test_actions_comparison_uses_both_immutable_original_graphs_without_model_download_from_hf():
    text=(Path(__file__).parents[2]/'.github/workflows/audio-profile-delta.yml').read_text()
    for required in ('11814567883','frames-1119-valid-1119/audio.onnx','frames-1119-valid-839/audio.onnx',
                     '7770a8af258f2de32c5587497b820596ea4e1ed7ec5382c3ee90eada2bec7e4a',
                     'af2e97b777349fa2e5f74790d24bb4b2d899c6a8438f372253737bc5940944c7',
                     'embeddinggemma_tools.audio_profile_delta','max-entry-bytes','if: always()'):
        assert required in text
    assert 'hf_hub_download' not in text and 'continue-on-error' not in text
