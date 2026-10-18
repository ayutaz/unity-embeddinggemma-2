import hashlib
import importlib
import json

import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh, TensorProto as T
import pytest


def module():return importlib.import_module('embeddinggemma_tools.audio_gpu_refinement')


@pytest.fixture
def targets():
    def info(name,shape):return h.make_tensor_value_info(name,T.FLOAT,shape)
    inputs=[];nodes=[];weights=[];outputs=[];infos=[]
    for prefix,width,shape in [('fresh_a',128,[1,560,64,128]),('fresh_b',32,[1,280,32,32])]:
        stat=prefix+'_stat';data=prefix+'_data';root=prefix+'_root';inverse=prefix+'_inverse';normalized=prefix+'_normalized';out=prefix+'_out';weight=prefix+'_weight'
        inputs.extend([info(stat,[*shape[:-1],1]),info(data,shape)])
        weights.append(nh.from_array(np.ones(width,np.float32),name=weight))
        nodes.extend([h.make_node('Sqrt',[stat],[root],name=prefix+'/Sqrt'),h.make_node('Reciprocal',[root],[inverse],name=prefix+'/Reciprocal'),
            h.make_node('Mul',[data,inverse],[normalized],name=prefix+'/normalize'),h.make_node('Mul',[normalized,weight],[out],name=prefix+'/scale')])
        infos.extend([info(root,[*shape[:-1],1]),info(inverse,[*shape[:-1],1]),info(normalized,shape),info(out,shape)])
        outputs.append(info(out,shape))
    stem=np.arange(32*128*3*3,dtype=np.float32).reshape(32,128,3,3)/40000
    weights.extend([nh.from_array(stem,name='stem_weight'),nh.from_array(np.ones((1024,1,5),np.float32),name='outer_weight'),
        nh.from_array(np.array(2,np.float32),name='two'),nh.from_array(np.array(1e-6,np.float32),name='epsilon'),
        nh.from_array(np.array(-.5,np.float32),name='negative_half'),nh.from_array(np.array([-1],np.int64),name='axes')])
    inputs.extend([info('stem_data',[1,128,560,64]),info('outer_data',[1,1024,284]),info('inner_data',[1,280,1536])])
    nodes.extend([h.make_node('Conv',['stem_data','stem_weight'],['stem_out'],name='arbitrary_stem',kernel_shape=[3,3],strides=[2,2],pads=[1,1,1,1],dilations=[1,1],group=1),
        h.make_node('Conv',['outer_data','outer_weight'],['outer_conv'],name='arbitrary_depthwise',kernel_shape=[5],strides=[1],pads=[0,0],dilations=[1],group=1024),
        h.make_node('Transpose',['outer_conv'],['outer_transpose'],name='outer_transpose',perm=[0,2,1]),
        h.make_node('Clip',['outer_transpose'],['outer_clip'],name='outer_clip')])
    infos.extend([info('stem_out',[1,32,280,32]),info('outer_conv',[1,1024,280]),info('outer_transpose',[1,280,1024]),info('outer_clip',[1,280,1024])])
    outputs.append(info('stem_out',[1,32,280,32]))
    for name,data,width in [('outer','outer_clip',1024),('inner','inner_data',1536)]:
        nodes.extend([h.make_node('Pow',[data,'two'],[name+'_square'],name=name+'_square'),
            h.make_node('ReduceMean',[name+'_square','axes'],[name+'_mean'],name=name+'_mean',keepdims=1),
            h.make_node('Add',[name+'_mean','epsilon'],[name+'_stat'],name=name+'_epsilon'),
            h.make_node('Pow',[name+'_stat','negative_half'],[name+'_inverse'],name=name+'_inverse')])
        infos.extend([info(name+'_square',[1,280,width]),info(name+'_mean',[1,280,1]),info(name+'_stat',[1,280,1])]);outputs.append(info(name+'_inverse',[1,280,1]))
    model=h.make_model(h.make_graph(nodes,'targets',inputs,outputs,initializer=weights,value_info=infos),opset_imports=[h.make_opsetid('',24)],ir_version=9)
    onnx.checker.check_model(model)
    return model,hashlib.sha256(stem.tobytes()).hexdigest()


def test_target_audit_uses_geometry_and_scale_dimensions_without_old_node_names(targets):
    model,sha=targets;before=model.SerializeToString()
    report=module().audit_refinement_targets(model,expected_rms=2,expected_outer=1,stem_weight_sha256=sha)
    assert sorted(x['width'] for x in report['normalizations'])==[32,128]
    assert report['outer_rms_mean_names']==['outer_mean']
    assert report['stem_conv_name']=='arbitrary_stem'
    assert report['rms_count']==2 and model.SerializeToString()==before


@pytest.mark.parametrize('problem',['rms_count','outer_group','norm_weight','stem_weight','unknown_shape','transpose'])
def test_unproved_refinement_targets_fail_before_mutation(targets,problem):
    model,sha=targets;expected=2
    if problem=='rms_count':expected=3
    if problem=='outer_group':next(a for n in model.graph.node if n.name=='arbitrary_depthwise' for a in n.attribute if a.name=='group').i=1
    if problem=='norm_weight':model.graph.initializer[0].data_type=T.FLOAT16
    if problem=='stem_weight':sha='0'*64
    if problem=='unknown_shape':next(v for v in model.graph.value_info if v.name=='outer_square').type.tensor_type.shape.dim[1].dim_param='time'
    if problem=='transpose':next(a for n in model.graph.node if n.name=='outer_transpose' for a in n.attribute if a.name=='perm').ints[:]=[0,1,2]
    before=model.SerializeToString()
    with pytest.raises(ValueError):module().audit_refinement_targets(model,expected_rms=expected,expected_outer=1,stem_weight_sha256=sha)
    assert model.SerializeToString()==before


def test_untrusted_seed_never_reaches_onnx_deserialization(tmp_path,monkeypatch):
    root=tmp_path/'input';root.mkdir();(root/'audio.onnx').write_bytes(b'untrusted');(root/'audio-gpu-seed.json').write_text('{}')
    monkeypatch.setattr(onnx,'load_model',lambda *a,**k:pytest.fail('Untrusted model was deserialized'))
    with pytest.raises(ValueError):module().checked_seed(root,case_id='silence')


def test_refinement_templates_remain_standard_fp32_with_the_fixed_stem_geometry(tmp_path):
    templates=module().make_templates(tmp_path/'templates')
    for model in [templates['inverse'],templates['fma'],templates['outer'],templates['conv'],*templates['means'].values()]:
        onnx.checker.check_model(model)
        assert all(n.domain in ('','ai.onnx') for n in model.graph.node)
        assert all(w.data_type not in (T.DOUBLE,T.FLOAT16,T.BFLOAT16) for w in model.graph.initializer)
    conv=templates['conv']
    assert [d.dim_value for d in conv.graph.input[0].type.tensor_type.shape.dim]==[1,128,560,64]
    assert [d.dim_value for d in conv.graph.output[0].type.tensor_type.shape.dim]==[1,32,280,32]


def test_existing_output_is_rejected_before_seed_read_or_conversion(tmp_path,monkeypatch):
    m=module();output=tmp_path/'output';output.mkdir();marker=output/'evidence.txt';marker.write_text('keep')
    monkeypatch.setattr(m,'checked_seed',lambda *a,**k:pytest.fail('seed read before output check'))
    with pytest.raises(ValueError):m.prepare(tmp_path/'input',output,case_id='silence',source='a'*40)
    assert marker.read_text()=='keep'


@pytest.fixture
def seed_archive(tmp_path,monkeypatch):
    import copy,io,zipfile
    from embeddinggemma_tools import artifact_zip
    m=module();data=b'synthetic-not-deserialized-onnx'
    report={'success':True,'source':m.SOURCE,'onnx_bytes':len(data),'onnx_sha256':hashlib.sha256(data).hexdigest(),
        'source_tensor_hashes_preserved_after_export':True,'numeric_acceptance':False,'gpu_validated':False,'generation_qualified':False,'original_golden_changed':False}
    raw=json.dumps(report).encode();stream=io.BytesIO()
    with zipfile.ZipFile(stream,'w') as archive:
        archive.writestr('audio.onnx',data);archive.writestr('audio-gpu-seed.json',raw)
    encoded=stream.getvalue();spec={'artifact_id':77,'archive_sha256':hashlib.sha256(encoded).hexdigest(),'report_sha256':hashlib.sha256(raw).hexdigest(),
        'onnx_sha256':report['onnx_sha256'],'onnx_bytes':len(data)}
    monkeypatch.setitem(m.SEEDS,'silence',spec)
    metadata={'id':77,'digest':'sha256:'+spec['archive_sha256'],'workflow_run':{'id':m.RUN,'head_sha':m.SOURCE,'repository_id':1408387878,'head_repository_id':1408387878}}
    def reader(*a):return artifact_zip.RangeReader(len(encoded),lambda start,stop:encoded[start:stop]),copy.deepcopy(metadata)
    monkeypatch.setattr(artifact_zip,'github_artifact_reader',reader)
    return m,tmp_path/'download',metadata,data


@pytest.mark.parametrize('problem',[None,'digest','source','fork'])
def test_seed_fetch_checks_provenance_and_pinned_member_bytes_before_loading(seed_archive,problem,monkeypatch):
    m,output,metadata,data=seed_archive
    if problem=='digest':metadata['digest']='sha256:'+'0'*64
    if problem=='source':metadata['workflow_run']['head_sha']='b'*40
    if problem=='fork':metadata['workflow_run']['head_repository_id']=1
    monkeypatch.setattr(onnx,'load_model',lambda *a,**k:pytest.fail('fetch deserialized a model'))
    if problem:
        with pytest.raises(ValueError):m.fetch_seed(output,case_id='silence')
    else:
        m.fetch_seed(output,case_id='silence')
        path,report=m.checked_seed(output,case_id='silence')
        assert path.read_bytes()==data and report['gpu_validated'] is False


def test_partition_publication_writes_canonical_gemm_weights_and_reconnects_original_bias(tmp_path):
    from onnx.reference import ReferenceEvaluator
    from test_partition_gemm import biased
    from embeddinggemma_tools.projection_coefficients import split_gemm_bias
    from embeddinggemma_tools.partition_projections import partition_static_projections
    m=module();model,original=biased();split_gemm_bias(model)
    matrix=original.T.copy();sha=hashlib.sha256(matrix.astype('<f4').tobytes()).hexdigest()
    parts,plan=partition_static_projections(model,node_names=['project'],expected_weight_sha256=[sha])
    m.write_partition_files(model,parts,plan,tmp_path)
    op=plan['operations'][0]
    stored=np.fromfile(tmp_path/op['file'],dtype='<f4').reshape(op['weight_shape'])
    np.testing.assert_array_equal(stored,matrix)
    values={'input':np.array([[1.,2.]],np.float32)}
    for index,segment in enumerate(plan['segments']):
        path=tmp_path/segment['file'];assert plan['files'][path.name]['sha256']==hashlib.sha256(path.read_bytes()).hexdigest()
        derived=onnx.load(path);outputs=ReferenceEvaluator(derived).run(None,{v.name:values[v.name] for v in derived.graph.input})
        aliases=segment['output_aliases'];values.update(zip([aliases.get(v.name,v.name) for v in derived.graph.output],outputs))
        if index==0:values[op['output']]=values[op['input']]@stored
    expected=ReferenceEvaluator(model).run(None,{'input':values['input']})[0]
    np.testing.assert_array_equal(values['out'],expected)
    assert plan['metadata_audits']['all_segments_textual_metadata_checked'] is True


def test_failed_preparation_keeps_safe_phase_receipt_and_source_bytes(tmp_path,monkeypatch):
    m=module();root=tmp_path/'input';root.mkdir();path=root/'audio.onnx';path.write_bytes(b'original')
    monkeypatch.setattr(m,'checked_seed',lambda *a,**k:(path,{}))
    monkeypatch.setattr(onnx,'load_model',lambda *a,**k:object())
    def fail(model,output,*,phase):
        phase('audit_coefficients');raise ValueError('private diagnostic must not be in public receipt')
    monkeypatch.setattr(m,'refine_and_partition',fail)
    with pytest.raises(ValueError):m.prepare(root,tmp_path/'out',case_id='silence',source='a'*40)
    receipt=json.loads((tmp_path/'out/audio-gpu-refinement.json').read_text())
    assert receipt['success'] is False and receipt['phase']=='audit_coefficients'
    assert receipt['error_type']=='ValueError' and 'private diagnostic' not in json.dumps(receipt)
    assert receipt['gpu_validated'] is False and receipt['generation_qualified'] is False
    assert path.read_bytes()==b'original'


def test_successful_preparation_binds_exact_source_and_stays_unqualified(tmp_path,monkeypatch):
    m=module();root=tmp_path/'input';root.mkdir();path=root/'audio.onnx';path.write_bytes(b'original')
    monkeypatch.setattr(m,'checked_seed',lambda *a,**k:(path,{'provenance':{'model_generation':'synthetic-unqualified'}}))
    monkeypatch.setattr(onnx,'load_model',lambda *a,**k:object())
    plan={'operations':[{}],'segments':[{},{}],'original_initializer_sha256':'b'*64}
    monkeypatch.setattr(m,'refine_and_partition',lambda *a,**k:plan)
    out=tmp_path/'out';report=m.prepare(root,out,case_id='padded_noise',source='a'*40)
    saved=json.loads((out/'partition-plan.json').read_text())
    assert saved['producer']=='a'*40 and saved['source_graph_sha256']==m.SEEDS['padded_noise']['onnx_sha256']
    assert saved['seed_run']==m.RUN and saved['seed_artifact_id']==m.SEEDS['padded_noise']['artifact_id']
    assert saved['profile']=='frames-1119-valid-839'
    assert report['plan_sha256']==hashlib.sha256((out/'partition-plan.json').read_bytes()).hexdigest()
    assert report['success'] is True and report['numeric_acceptance'] is False
    assert report['gpu_validated'] is False and report['generation_qualified'] is False


def test_cli_rejects_bad_source_without_fetch_and_never_prints_raw_error(tmp_path,monkeypatch,capsys):
    m=module()
    monkeypatch.setattr(m,'fetch_seed',lambda *a,**k:pytest.fail('bad source reached network'))
    result=m.main(['--fetch','--input',str(tmp_path/'in'),'--output',str(tmp_path/'out'),
        '--case','silence','--source','invalid'])
    report=json.loads(capsys.readouterr().out)
    assert result==1 and report=={'success':False,'error_type':'ValueError','gpu_validated':False,'generation_qualified':False}
