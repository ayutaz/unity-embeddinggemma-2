import hashlib,json
from pathlib import Path
import numpy as np
import pytest
import torch
from embeddinggemma_tools import reference_context_failure as context


def forbidden_encoder(value):
    raise RuntimeError('Encoder inference must not execute')


def program():
    root=torch.nn.Module();root.register_parameter('weight',torch.nn.Parameter(torch.arange(12,dtype=torch.float32).reshape(4,3)/64))
    graph=torch.fx.Graph();x=graph.placeholder('pixels');w=graph.get_attr('weight')
    transpose=graph.call_function(torch.ops.aten.permute.default,(w,[1,0]));mm=graph.call_function(torch.ops.aten.mm.default,(x,transpose))
    graph.output(graph.call_function(forbidden_encoder,(mm,)))
    return torch.fx.GraphModule(root,graph),mm.name


def run(*args,**kwargs):
    function=getattr(context,'probe_saved_static_matmuls',None)
    assert callable(function),'Saved-shape static matmul probe is missing'
    return function(*args,**kwargs)


def test_probe_preserves_full_native_shape_and_transposed_weight_layout_but_saves_only_prefix(tmp_path):
    model,name=program();before=model.weight.detach().clone();code=model.code;destination=tmp_path/'capture'
    report=run(model,destination,(name,),rows=9,prefix_rows=3,seed=71)
    assert report['full_encoder_executed'] is False and report['learned_weights_changed'] is False
    assert torch.equal(model.weight,before) and model.code==code
    row,=report['matrices'];assert row['native_input_shape']==[9,3] and row['native_output_shape']==[9,4]
    assert row['saved_input_shape']==[3,3] and row['weight_stride']==[1,3]
    values=torch.randn((9,3),generator=torch.Generator().manual_seed(row['seed']))
    expected=(values@before.t())[:3].numpy()
    for key in ('input','weights','expected'):
        item=row[key];raw=(destination/item['file']).read_bytes();assert hashlib.sha256(raw).hexdigest()==item['sha256']
    actual=np.fromfile(destination/row['expected']['file'],'<f4').reshape(3,4)
    np.testing.assert_array_equal(actual.view('u4'),expected.view('u4'))
    weights=np.fromfile(destination/row['weights']['file'],'<f4').reshape(3,4)
    np.testing.assert_array_equal(weights,before.t().numpy())


@pytest.mark.parametrize('problem',['missing','duplicate','bad_rows','bad_prefix'])
def test_probe_rejects_invalid_requests_before_writing_files(tmp_path,problem):
    model,name=program();names=(name,);kwargs=dict(rows=9,prefix_rows=3,seed=71)
    if problem=='missing':names=('absent',)
    elif problem=='duplicate':names=(name,name)
    elif problem=='bad_rows':kwargs['rows']=0
    else:kwargs['prefix_rows']=10
    destination=tmp_path/'capture'
    with pytest.raises(ValueError):run(model,destination,names,**kwargs)
    assert not destination.exists()


def test_probe_refuses_to_overwrite_existing_evidence(tmp_path):
    model,name=program();destination=tmp_path/'capture';destination.mkdir();(destination/'keep').write_text('keep')
    with pytest.raises(FileExistsError):run(model,destination,(name,),rows=9,prefix_rows=3,seed=71)
    assert (destination/'keep').read_text()=='keep'
