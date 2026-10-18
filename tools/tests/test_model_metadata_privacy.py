import hashlib,json,zipfile
from pathlib import Path

import onnx
from onnx import helper,TensorProto
import pytest
import tomllib,yaml

from embeddinggemma_tools.privacy_artifacts import audit


def private_home():
    return 'C:'+chr(92)+'Users'+chr(92)+'fixture-person'+chr(92)+'work'


def pt2(path,*,model=None,other=None,comment=b''):
    with zipfile.ZipFile(path,'w') as archive:
        archive.comment=comment
        archive.writestr('archive/models/model.json',json.dumps(model or {'graph_module':{'graph':{'nodes':[]}}}))
        archive.writestr('archive/version','1')
        if other:archive.writestr('archive/data/weights/weight.json',json.dumps(other))


def model(path,*,where=None,value=None):
    node=helper.make_node('Identity',['x'],['y'])
    graph=helper.make_graph([node],'safe',[helper.make_tensor_value_info('x',TensorProto.FLOAT,[1])],[helper.make_tensor_value_info('y',TensorProto.FLOAT,[1])])
    proto=helper.make_model(graph)
    if where=='model':proto.doc_string=value
    elif where=='graph':proto.graph.doc_string=value
    elif where=='node':proto.graph.node[0].doc_string=value
    elif where=='value':proto.graph.input[0].doc_string=value
    elif where=='property':proto.metadata_props.add(key='source',value=value)
    elif where=='tensor':proto.graph.initializer.add(name='constant',data_type=TensorProto.FLOAT,dims=[1],doc_string=value,raw_data=b'\0'*4)
    path.write_bytes(proto.SerializeToString())


@pytest.mark.parametrize('suffix',['.pt2','.onnx'])
def test_unparseable_model_cannot_pass_as_an_unchecked_numeric_buffer(tmp_path,suffix):
    path=tmp_path/('model'+suffix);path.write_bytes(b'\0\1\2')
    assert audit(tmp_path,[path.name])


@pytest.mark.parametrize('where',['stack_trace','other_model_field','tensor_metadata','archive_comment'])
def test_pt2_metadata_private_paths_fail_without_echo_or_mutation(tmp_path,where):
    path=tmp_path/'model.pt2';value=private_home();doc={'graph_module':{'graph':{'nodes':[]}}};other=None;comment=b''
    if where=='stack_trace':doc['graph_module']['graph']['nodes']=[{'metadata':{'stack_trace':value}}]
    elif where=='other_model_field':doc['source']=value
    elif where=='tensor_metadata':other={'source':value}
    else:comment=value.encode()
    pt2(path,model=doc,other=other,comment=comment);before=path.read_bytes();stamp=path.stat().st_mtime_ns
    findings=audit(tmp_path,[path.name]);assert findings and value not in str(findings) and 'fixture-person' not in str(findings)
    assert path.read_bytes()==before and path.stat().st_mtime_ns==stamp


@pytest.mark.parametrize('where',['model','graph','node','value','property','tensor'])
def test_onnx_textual_metadata_is_checked_in_every_message(tmp_path,where):
    path=tmp_path/'model.onnx';value=private_home();model(path,where=where,value=value)
    before=hashlib.sha256(path.read_bytes()).hexdigest();findings=audit(tmp_path,[path.name])
    assert findings and value not in str(findings) and 'fixture-person' not in str(findings)
    assert hashlib.sha256(path.read_bytes()).hexdigest()==before


def test_safe_model_metadata_and_opaque_numerical_data_remain_separate(tmp_path):
    pt2(tmp_path/'safe.pt2');model(tmp_path/'safe.onnx')
    (tmp_path/'capture.bin').write_bytes(b'\0'+private_home().encode())
    assert audit(tmp_path,['.'])==[]


def test_artifact_guard_installs_the_pinned_onnx_metadata_parser():
    root=Path(__file__).resolve().parents[2]
    lock=tomllib.loads((root/'tools/uv.lock').read_text())
    version=next(row['version'] for row in lock['package'] if row['name']=='onnx')
    action=yaml.load((root/'.github/actions/privacy-artifact-guard/action.yml').read_text(),Loader=yaml.BaseLoader)
    command='\n'.join(step.get('run','') for step in action['runs']['steps'])
    assert '--with onnx=='+version in command


def test_pt2_posix_home_and_encoded_private_metadata_fail(tmp_path):
    value='/'+'home'+'/'+'fixture-person'+'/work'
    pt2(tmp_path/'safe.pt2',model={'source':value.replace('/','%2F')})
    assert audit(tmp_path,['safe.pt2'])


@pytest.mark.parametrize('repeated',[False,True])
def test_onnx_string_attributes_are_text_not_opaque_float_buffers(tmp_path,repeated):
    path=tmp_path/'model.onnx';model(path);proto=onnx.load_model(path)
    value=private_home();proto.graph.node[0].attribute.append(helper.make_attribute('source',[value] if repeated else value))
    path.write_bytes(proto.SerializeToString());findings=audit(tmp_path,[path.name])
    assert findings and value not in str(findings)


def test_onnx_tensor_raw_data_is_not_decoded_as_metadata(tmp_path):
    path=tmp_path/'model.onnx';model(path);proto=onnx.load_model(path)
    payload=b'\0'+private_home().encode()
    proto.graph.initializer.add(name='opaque',data_type=TensorProto.UINT8,dims=[len(payload)],raw_data=payload)
    path.write_bytes(proto.SerializeToString());before=path.read_bytes()
    assert audit(tmp_path,[path.name])==[] and path.read_bytes()==before


def test_onnx_external_weights_are_not_loaded_by_the_metadata_guard(tmp_path):
    path=tmp_path/'model.onnx';model(path);proto=onnx.load_model(path)
    tensor=proto.graph.initializer.add(name='external',data_type=TensorProto.FLOAT,dims=[1],data_location=TensorProto.EXTERNAL)
    tensor.external_data.add(key='location',value='missing-weights.data')
    path.write_bytes(proto.SerializeToString());assert not (tmp_path/'missing-weights.data').exists()
    assert audit(tmp_path,[path.name])==[]
    tensor.external_data[0].value='../outside.data';path.write_bytes(proto.SerializeToString())
    assert audit(tmp_path,[path.name])


def test_missing_onnx_parser_fails_closed(tmp_path,monkeypatch):
    import builtins
    path=tmp_path/'model.onnx';model(path);original=builtins.__import__
    def unavailable(name,*args,**kwargs):
        if name=='onnx':raise ImportError('Parser unavailable')
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',unavailable)
    assert audit(tmp_path,[path.name])


def test_native_silu_producer_uses_metadata_safe_serialization_without_changing_kernel(tmp_path):
    from embeddinggemma_tools.native_silu_probe import prepare_silu_probe
    report=prepare_silu_probe(tmp_path,source_commit='a'*40)
    assert report['success'] and report['case_count']==2
    assert audit(tmp_path,['*.pt2','*.onnx'])==[]


def test_export_producer_omits_only_optional_node_traces_and_keeps_values(tmp_path,monkeypatch):
    import torch
    from embeddinggemma_tools.export import _save_program
    class Tiny(torch.nn.Module):
        def forward(self,x):return torch.relu(x+1)
    original=torch.export.ExportedProgram.run_decompositions
    def decomposed(self,*args,**kwargs):
        result=original(self,*args,**kwargs)
        for node in result.graph.nodes:node.meta['stack_trace']=private_home()
        return result
    monkeypatch.setattr(torch.export.ExportedProgram,'run_decompositions',decomposed)
    source=torch.tensor([-2.,0.,2.]);unchanged=source.clone();path=tmp_path/'tiny.pt2'
    _save_program(Tiny(),(source,),path)
    with zipfile.ZipFile(path) as archive:
        names=[name for name in archive.namelist() if name.endswith('/models/model.json')];assert len(names)==1
        metadata=json.loads(archive.read(names[0]))
        assert all('stack_trace' not in node.get('metadata',{}) for node in metadata['graph_module']['graph']['nodes'])
    torch.testing.assert_close(torch.export.load(path).module()(source),torch.tensor([0.,1.,3.]),rtol=0,atol=0)
    torch.testing.assert_close(source,unchanged,rtol=0,atol=0)
    assert audit(tmp_path,['tiny.pt2'])==[]
