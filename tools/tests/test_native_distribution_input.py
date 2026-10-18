"""The explicit clean-graph route records executed bytes separately from source identity."""
import hashlib
import json
import zipfile

import pytest
import torch

from test_audio_generation_candidate import candidate
from embeddinggemma_tools import exported_program_distribution as exported
from embeddinggemma_tools import native_regression_distribution as distribution
from embeddinggemma_tools.native_regression_release_privacy import _canonical


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def clean_input(candidate,tmp_path,monkeypatch):
    raw,current,original,contract,_=candidate
    graph=current/'audio.pt2';source_sha=sha(graph);clean=tmp_path/'clean.pt2'
    result=exported.sanitize(graph,clean,expected_source_sha256=source_sha)
    (tmp_path/'preserved-raw.pt2').write_bytes(graph.read_bytes());graph.write_bytes(clean.read_bytes())
    members={}
    with zipfile.ZipFile(clean) as archive:
        names,model=exported._members(archive)
        for info in archive.infolist():
            if info.filename!=model:members[info.filename]=dict(bytes=info.file_size,sha256=hashlib.sha256(archive.read(info)).hexdigest())
    entry=dict(kind='pt2',bytes=graph.stat().st_size,sha256=sha(graph),source_graph_sha256=source_sha,
               semantic_model_sha256=result['semantic_model_sha256'],non_model_member_manifest_sha256=_canonical(members),member_count=len(names))
    config=dict(inventory_sha256='b'*64)
    inventory=dict(candidate_contract_sha256=raw.CONTRACT_SHA,trace_removal_producer='c'*40,files={'candidate-1119.pt2':entry})
    monkeypatch.setattr(distribution,'configuration',lambda:(config,inventory));monkeypatch.setattr(distribution,'CONFIG_SHA','f'*64)
    marker=dict(schema=1,transport=distribution.TRANSPORT,case_id='silence',profile='full',graph_path='audio.pt2',
                source_graph_sha256=source_sha,executed_graph_sha256=entry['sha256'],inventory_sha256=config['inventory_sha256'],
                release_contract_sha256=distribution.CONFIG_SHA,distribution_producer=inventory['trace_removal_producer'])
    path=current/'native-distribution-source.json';path.write_text(json.dumps(marker))
    return raw,current,original,entry,path,marker


def test_explicit_loader_executes_clean_bytes_with_unchanged_original_expected_values(clean_input):
    _,current,original,entry,_,_=clean_input
    program,values,expected,provenance=distribution.load_distribution_input(current,original,case_id='silence')
    assert torch.equal(expected,torch.ones(2,32)) and torch.isfinite(program(values)).all()
    assert provenance['graph_sha256']==provenance['executed_graph_sha256']==entry['sha256']
    assert provenance['source_graph_sha256']==entry['source_graph_sha256']!=entry['sha256']
    assert provenance['graph_transport']==distribution.TRANSPORT
    assert provenance['graph_transport_contract_sha256']=='f'*64
    assert provenance['original_graph_bytes'] is False


def test_existing_raw_candidate_loader_never_silently_accepts_clean_graph_bytes(clean_input,monkeypatch):
    raw,current,original,_,_,_=clean_input
    monkeypatch.setattr(torch.export,'load',lambda *a:pytest.fail('untrusted graph reached the raw loader'))
    with pytest.raises(ValueError):raw.load_candidate_input(current,original,case_id='silence')


@pytest.mark.parametrize('problem',['graph','input','candidate-reference','original-reference','marker-missing',
                                   'marker-executed','marker-source','marker-contract','marker-case'])
def test_untrusted_distribution_inputs_are_rejected_before_deserialization(clean_input,monkeypatch,problem):
    _,current,original,_,path,marker=clean_input
    if problem=='graph':(current/'audio.pt2').write_bytes(b'changed')
    elif problem=='input':(current/'input.bin').write_bytes(b'changed')
    elif problem=='candidate-reference':(current/'audio-encoder-reference.json').write_text('{}')
    elif problem=='original-reference':(original/'audio-encoder-reference.json').write_text('{}')
    elif problem=='marker-missing':path.unlink()
    else:
        key={'marker-executed':'executed_graph_sha256','marker-source':'source_graph_sha256',
             'marker-contract':'release_contract_sha256','marker-case':'case_id'}[problem]
        marker[key]='changed';path.write_text(json.dumps(marker))
    monkeypatch.setattr(torch.export,'load',lambda *a:pytest.fail('untrusted graph was deserialized'))
    with pytest.raises(ValueError):distribution.load_distribution_input(current,original,case_id='silence')
