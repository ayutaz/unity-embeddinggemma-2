"""Pinned public release acquisition must preserve exact goldens and executed graph identity."""
import hashlib
import importlib
import io
import json
from pathlib import Path

import pytest

from test_native_regression_release_privacy import reviewed


def sha(raw):return hashlib.sha256(raw).hexdigest()
def module():return importlib.import_module('embeddinggemma_tools.native_regression_distribution')


@pytest.fixture
def distribution(tmp_path,monkeypatch):
    _,root,_=reviewed(tmp_path,monkeypatch)
    for name in ('LICENSE.txt','NOTICE.txt','MODIFICATIONS.txt'):(root/name).write_text('Public source and license information')
    inventory=json.loads((root/'native-regression-assets.json').read_text())
    assets={p.name:dict(bytes=p.stat().st_size,sha256=sha(p.read_bytes()),asset_id=i+1) for i,p in enumerate(sorted(root.iterdir()))}
    config=dict(schema=1,repository='ayutaz/unity-embeddinggemma-2',tag='native-regression-preview-e9de158',release_id=123,
                publication_target='728a3d004caab0afb51009009e801d932fedb64b',inventory_sha256=sha((root/'native-regression-assets.json').read_bytes()),assets=assets)
    config_path=tmp_path/'transport.json';raw=json.dumps(config).encode();config_path.write_bytes(raw)
    m=module();monkeypatch.setattr(m,'CONFIG_PATH',config_path);monkeypatch.setattr(m,'CONFIG_SHA',sha(raw));monkeypatch.setattr(m,'INVENTORY_PATH',root/'native-regression-assets.json')
    calls=[]
    class Response(io.BytesIO):
        def __init__(self,raw,url):super().__init__(raw);self.headers={'Content-Length':str(len(raw))};self.url=url
        def geturl(self):return self.url
    def fetch(url):
        calls.append(url);return Response((root/url.rsplit('/',1)[-1]).read_bytes(),url)
    monkeypatch.setattr(m,'_open',fetch)
    return m,root,inventory,config,config_path,calls,Response


@pytest.mark.parametrize('case',['silence','padded_noise'])
def test_fetches_only_selected_exact_graph_input_and_unchanged_originals_with_legal_metadata(distribution,tmp_path,case):
    m,root,inventory,_,_,calls,_=distribution
    output=tmp_path/'candidate';original=tmp_path/'original'
    result=m.fetch_candidate(output,original,case_id=case)
    profile='839' if case=='padded_noise' else '1119'
    graph=output/'audio-runtime-reference'/('frames-1119-valid-'+profile)/'audio.pt2'
    assert graph.read_bytes()==(root/('candidate-'+profile+'.pt2')).read_bytes()
    assert (original/'audio-encoder-reference.json').read_bytes()==(root/'original-reference.json').read_bytes()
    assert (original/'audio-encoder-validation.json').read_bytes()==(root/'original-audit.json').read_bytes()
    assert (output/'audio-runtime-reference'/('frames-1119-valid-'+profile)/(case+'-features-fp32.bin')).read_bytes()==(root/('input-'+case+'.bin')).read_bytes()
    names=[url.rsplit('/',1)[-1] for url in calls]
    assert names.count('candidate-'+profile+'.pt2')==1 and sum(name.endswith('.pt2') for name in names)==1
    assert not any(name.endswith('.zip') for name in names)
    for directory in (output,original):
        for name in ('LICENSE.txt','NOTICE.txt','MODIFICATIONS.txt','native-regression-assets.json'):assert (directory/name).is_file()
    marker=json.loads((output/'audio-runtime-reference/native-distribution-source.json').read_text())
    entry=inventory['files']['candidate-'+profile+'.pt2']
    assert marker['executed_graph_sha256']==entry['sha256'] and marker['source_graph_sha256']==entry['source_graph_sha256']
    assert marker['executed_graph_sha256']!=marker['source_graph_sha256']
    assert result['graph_executed'] is False and result['original_golden_changed'] is False
    assert (output/'candidate-range-receipt.json').is_file() and (original/'original-range-receipt.json').is_file()


def test_original_only_route_never_downloads_a_candidate_graph(distribution,tmp_path):
    m,root,_,_,_,calls,_=distribution;m.fetch_original(tmp_path/'original')
    assert not any(url.endswith('.pt2') for url in calls)


@pytest.mark.parametrize('problem',['config','inventory','unknown-case','nonempty','symlink'])
def test_untrusted_configuration_or_output_is_rejected_before_network(distribution,tmp_path,problem):
    m,root,_,_,path,calls,_=distribution;output=tmp_path/'candidate'
    if problem=='config':path.write_bytes(path.read_bytes()+b'changed')
    elif problem=='inventory':(root/'native-regression-assets.json').write_text('{}')
    elif problem=='nonempty':output.mkdir();(output/'keep.txt').write_text('User content')
    elif problem=='symlink':
        real=tmp_path/'real';real.mkdir()
        try:output.symlink_to(real,target_is_directory=True)
        except OSError:pytest.skip('Host does not permit synthetic symlinks')
    with pytest.raises(ValueError):m.fetch_candidate(output,tmp_path/'original',case_id='missing' if problem=='unknown-case' else 'silence')
    assert not calls


@pytest.mark.parametrize('problem',['changed','truncated','oversized','length','redirect'])
def test_downloaded_payload_is_verified_before_it_gets_its_final_name(distribution,tmp_path,monkeypatch,problem):
    m,root,_,_,_,_,Response=distribution
    def fetch(url):
        raw=(root/url.rsplit('/',1)[-1]).read_bytes()
        if problem=='changed':raw=bytes([raw[0]^1])+raw[1:]
        elif problem=='truncated':raw=raw[:-1]
        elif problem=='oversized':raw+=b'x'
        response=Response(raw,'http://invalid.example/file' if problem=='redirect' else url)
        if problem=='length':response.headers['Content-Length']='999999'
        return response
    monkeypatch.setattr(m,'_open',fetch);output=tmp_path/'original'
    with pytest.raises(ValueError):m.fetch_original(output)
    assert not (output/'audio-encoder-reference.json').exists()
    assert not list(output.rglob('*.part'))


def test_fetch_never_overwrites_existing_evidence(distribution,tmp_path):
    m,_,_,_,_,_,_=distribution;output=tmp_path/'original';output.mkdir();file=output/'keep.txt';file.write_bytes(b'original')
    with pytest.raises(ValueError):m.fetch_original(output)
    assert file.read_bytes()==b'original'


def test_current_production_contract_binds_the_complete_public_asset_cohort():
    config,inventory=module().configuration()
    assert len(config['assets'])==16 and len(inventory['files'])==12
    assert config['publication_target']=='728a3d004caab0afb51009009e801d932fedb64b'
    assert config['inventory_sha256']=='e4da37c53cc9b97d421b17caf56eacb6003fdca9160ac6e8f6a86a5314299d5e'


@pytest.mark.parametrize('layout',['same','original-child','candidate-child'])
def test_candidate_and_original_evidence_roots_must_not_overlap(distribution,tmp_path,layout):
    m,_,_,_,_,calls,_=distribution;output=tmp_path/'shared';original=output
    if layout=='original-child':original=output/'original'
    elif layout=='candidate-child':output=original/'candidate'
    with pytest.raises(ValueError):m.fetch_candidate(output,original,case_id='silence')
    assert not calls
