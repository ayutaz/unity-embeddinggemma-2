import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
import torch.nn.functional as F


def required():
    from embeddinggemma_tools import conv_reference_context
    return conv_reference_context.replay_frozen_conv_context


def fixture(tmp_path,monkeypatch):
    encoder=tmp_path/'encoder';subsample=tmp_path/'subsample';encoder.mkdir();subsample.mkdir()
    generator=torch.Generator().manual_seed(92)
    values=torch.randn(1,128,5,4,generator=generator)
    weight=torch.randn(32,128,3,3,generator=generator)*.01
    expected=F.conv2d(values,weight,stride=2,padding=1)
    def identity(path):return dict(bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    graph=encoder/'native.pt2';graph.write_bytes(b'audited fake graph for unit loader')
    metadata=dict(source_commit='4edbe17a8105233c5aaa1514faa2862b08a4f7e0',model_revision='914f7f89142e33e77833254d9c9b90c3cef7303b')
    reference=dict(metadata=metadata,profiles={'frames-1119-valid-1119':dict(native_graph='native.pt2')})
    audit=dict(metadata=metadata,native_reference_complete=True,files={'native.pt2':identity(graph)})
    (encoder/'audio-encoder-reference.json').write_text(json.dumps(reference))
    (encoder/'audio-encoder-validation.json').write_text(json.dumps(audit))
    parts={}
    binary=subsample/'audio-layer-replay';binary.mkdir()
    for name,value in [('layer0.act',values),('layer1.conv',expected)]:
        path=binary/(name+'.bin');value.numpy().astype('<f4').tofile(path)
        parts['subsample.'+name]=dict(file=path.name,shape=list(value.shape),**identity(path))
    replay=dict(success=True,subsample_native_final_exact_match=True,reference_source=metadata['source_commit'],
        profile='frames-1119-valid-1119',parts=parts)
    (subsample/'audio-layer-replay-report.json').write_text(json.dumps(replay))
    calls=[]
    def load(path):calls.append(path);return SimpleNamespace(state_dict={'audio.subsample_conv_projection.layer1.conv.weight':weight})
    monkeypatch.setattr(torch.export,'load',load)
    args=dict(encoder=encoder,subsample=subsample,
        encoder_reference_sha256=identity(encoder/'audio-encoder-reference.json')['sha256'],
        encoder_audit_sha256=identity(encoder/'audio-encoder-validation.json')['sha256'],
        subsample_reference_sha256=identity(subsample/'audio-layer-replay-report.json')['sha256'],
        weight_sha256=hashlib.sha256(weight.numpy().astype('<f4').tobytes()).hexdigest())
    return args,calls,weight,expected,identity


def test_conv_context_uses_saved_original_parameter_and_records_cpu_only_scope(tmp_path,monkeypatch):
    run=required();args,calls,weight,expected,_=fixture(tmp_path,monkeypatch)
    before=weight.clone();result=run(**args)
    assert len(calls)==1 and torch.equal(weight,before)
    assert result['context_validated'] and result['exact_differences']==0 and result['mismatched_elements']==0
    assert result['expected_unchanged'] and not result['gpu_validated'] and not result['full_encoder_validated']
    assert result['runtime']['cpu_capability'] and result['runtime']['torch_config']


@pytest.mark.parametrize('problem',['manifest','input','graph','weight'])
def test_conv_context_rejects_corrupt_provenance_and_weights(tmp_path,monkeypatch,problem):
    run=required();args,calls,weight,expected,_=fixture(tmp_path,monkeypatch)
    if problem=='manifest':args['subsample_reference_sha256']='0'*64
    elif problem=='input':(args['subsample']/'audio-layer-replay/layer0.act.bin').write_bytes(b'wrong')
    elif problem=='graph':(args['encoder']/'native.pt2').write_bytes(b'wrong')
    elif problem=='weight':args['weight_sha256']='0'*64
    with pytest.raises(ValueError):run(**args)
    if problem!='weight':assert not calls,'Refuse corrupt provenance before loading a model'


def test_conv_context_preserves_numeric_failure_without_replacing_expected(tmp_path,monkeypatch):
    run=required();args,calls,weight,expected,identity=fixture(tmp_path,monkeypatch)
    path=args['subsample']/'audio-layer-replay/layer1.conv.bin'
    (expected+1).numpy().astype('<f4').tofile(path)
    replay_path=args['subsample']/'audio-layer-replay-report.json';replay=json.loads(replay_path.read_text())
    replay['parts']['subsample.layer1.conv'].update(identity(path));replay_path.write_text(json.dumps(replay))
    args['subsample_reference_sha256']=identity(replay_path)['sha256'];before=path.read_bytes()
    result=run(**args)
    assert not result['context_validated'] and not result['success'] and result['mismatched_elements']>0
    assert path.read_bytes()==before and result['expected_unchanged']


def test_cli_failure_retains_requested_provenance_and_actual_runtime(tmp_path,monkeypatch):
    from embeddinggemma_tools import conv_reference_context
    args,calls,weight,expected,_=fixture(tmp_path,monkeypatch)
    args['subsample_reference_sha256']='0'*64;output=tmp_path/'failed.json'
    argv=[]
    for name,value in args.items():argv.extend(['--'+name.replace('_','-'),str(value)])
    argv.extend(['--output',str(output)]);monkeypatch.setenv('GITHUB_SHA','a'*40)
    assert conv_reference_context.main(argv)==1
    report=json.loads(output.read_text())
    assert report['requested_reference']['subsample_reference_sha256']=='0'*64
    assert report['runtime']['cpu_capability'] and report['job_source']=='a'*40
    assert not report['success'] and not report['gpu_validated'] and 'exact_differences' not in report
