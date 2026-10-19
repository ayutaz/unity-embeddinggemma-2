import hashlib
import importlib
import json
from pathlib import Path

import pytest

from test_audio_cpu_cohort import cohort

SOURCE='a'*40


def implementation():
    return importlib.import_module('embeddinggemma_tools.audio_cpu_inputs')


@pytest.fixture
def prepared(cohort,monkeypatch,tmp_path):
    module=implementation()
    refs,pre,audit,encoder,_,document,save=cohort
    oracle=json.loads((Path(__file__).parents[1]/'references/audio-cpu-original/audio-validation.json').read_bytes())
    pre['metadata']['versions']=oracle['metadata']['versions']
    fields={'waveform':'pcm-fp32','sample_mask':'samples-int32','frame_mask':'frames-int32'}
    for row in pre['cases']:
        if row['id'] in refs.CPU_CASES:
            for key,suffix in fields.items():
                audit['files'][row[key]]=dict(oracle['files'][row['id']+'-'+suffix+'.bin'])
    roots=save()
    (roots[0]/'audio-embedding-reference.json').write_bytes(roots[2].read_bytes())
    monkeypatch.setattr(module,'REFERENCE_DIR',roots[0])
    monkeypatch.setattr(module,'_validate_execution_source',lambda _: {'synthetic.py':'0'*64})
    output=tmp_path/'materialized'
    return module,refs,pre,audit,encoder,roots,output,oracle


def test_committed_small_JSON_candidates_retain_all_three_original_pins():
    module=implementation()
    pins={'audio-reference.json':module.refs.PREPROCESS_REFERENCE_SHA256,
          'audio-validation.json':module.refs.PREPROCESS_AUDIT_SHA256,
          'audio-embedding-reference.json':module.refs.DOCUMENT_REFERENCE_SHA256}
    assert {v.name for v in module.REFERENCE_DIR.iterdir()}==set(pins)
    for name,digest in pins.items():
        raw=(module.REFERENCE_DIR/name).read_bytes()
        assert len(raw)<1024**2 and hashlib.sha256(raw).hexdigest()==digest


def test_procedural_inputs_match_external_original_hash_oracles_without_model_execution(prepared):
    module,refs,pre,_,_,roots,output,oracle=prepared
    result=module.materialize_audio_cpu_inputs(roots[1],output,source=SOURCE)
    assert result['success'] and result['procedural_buffers_reconstructed']==6
    assert result['reference_feature_buffers_copied_verbatim']==2 and result['model_executed'] is False
    assert result['original_reference_or_model_regenerated'] is False
    assert result['CPU_comparison_qualified'] is result['GPU_validated'] is False
    loaded=refs.load_audio_cpu_reference_cohort(output,roots[1],output/'audio-embedding-reference.json')
    assert len(loaded.cases)==2
    fields={'waveform':'pcm-fp32','sample_mask':'samples-int32','frame_mask':'frames-int32'}
    for row in pre['cases']:
        if row['id'] in refs.CPU_CASES:
            for key,suffix in fields.items():
                raw=(output/row[key]).read_bytes()
                expected=oracle['files'][row['id']+'-'+suffix+'.bin']
                assert len(raw)==expected['bytes'] and hashlib.sha256(raw).hexdigest()==expected['sha256']
    assert len(list(output.glob('*.json')))==4


def test_missing_raw_reference_features_require_explicit_fetch_selection(prepared):
    module,_,_,_,encoder,roots,output,_=prepared
    (roots[1]/encoder['cases'][0]['input_features']).unlink()
    with pytest.raises(ValueError):module.materialize_audio_cpu_inputs(roots[1],output,source=SOURCE)
    assert not output.exists()


def test_missing_reference_fetch_selects_only_two_original_numerical_assets(prepared,monkeypatch):
    module,refs,_,_,encoder,roots,output,_=prepared
    selected=[v for v in encoder['cases'] if v['id'] in refs.CPU_CASES]
    config={'assets':{}}
    inventory={'files':{}}
    original={}
    for row in selected:
        path=roots[1]/row['input_features']
        raw=path.read_bytes()
        path.unlink()
        asset='input-'+row['id']+'.bin'
        entry=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
        config['assets'][asset]=entry
        inventory['files'][asset]=dict(entry,original_member=row['input_features'])
        original[row['input_features']]=raw
    monkeypatch.setattr(module.transport,'configuration',lambda:(config,inventory))
    requests=[]
    def fetch(root,mappings,actual_config,actual_inventory):
        assert actual_config is config and actual_inventory is inventory
        requests.append(mappings)
        for asset,name in mappings.items():
            assert asset in ('input-silence.bin','input-padded_noise.bin')
            path=root/name
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(original[name])
        return {}
    monkeypatch.setattr(module.transport,'_fetch',fetch)
    result=module.materialize_audio_cpu_inputs(roots[1],output,source=SOURCE,fetch_missing_inputs=True)
    assert result['success'] and len(requests)==1
    assert set(requests[0])=={'input-silence.bin','input-padded_noise.bin'}


def test_changed_existing_reference_input_is_not_overwritten_by_fetch(prepared,monkeypatch):
    module,_,_,_,encoder,roots,output,_=prepared
    path=roots[1]/encoder['cases'][0]['input_features']
    path.write_bytes(b'changed')
    monkeypatch.setattr(module.transport,'_fetch',lambda *_:pytest.fail('Existing unsafe file was overwritten'))
    with pytest.raises(ValueError):module.materialize_audio_cpu_inputs(roots[1],output,source=SOURCE,fetch_missing_inputs=True)
    assert path.read_bytes()==b'changed' and not output.exists()


@pytest.mark.parametrize('filename',['audio-reference.json','audio-validation.json','audio-embedding-reference.json'])
def test_changed_original_JSON_is_rejected_without_regenerating_reference(prepared,filename):
    module,_,_,_,_,roots,output,_=prepared
    path=roots[0]/filename
    with path.open('ab') as stream:stream.write(b' ')
    with pytest.raises(ValueError):module.materialize_audio_cpu_inputs(roots[1],output,source=SOURCE)
    assert not output.exists()


def test_changed_torch_recipe_context_fails_closed(prepared,monkeypatch):
    module,_,_,_,_,roots,output,_=prepared
    monkeypatch.setattr(module.torch,'__version__','9.0.0')
    with pytest.raises(ValueError):module.materialize_audio_cpu_inputs(roots[1],output,source=SOURCE)
    assert not output.exists()


def test_changed_reconstruction_bytes_do_not_change_expected_hashes(prepared,monkeypatch):
    module,_,_,_,_,roots,output,_=prepared
    actual=module._procedural_inputs
    def changed():
        values=actual()
        raw=bytearray(values['padded_noise']['waveform'])
        raw[0]^=1
        values['padded_noise']['waveform']=bytes(raw)
        return values
    monkeypatch.setattr(module,'_procedural_inputs',changed)
    with pytest.raises(ValueError):module.materialize_audio_cpu_inputs(roots[1],output,source=SOURCE)
    assert not output.exists()


def test_existing_output_is_preserved(prepared):
    module,_,_,_,_,roots,output,_=prepared
    module.materialize_audio_cpu_inputs(roots[1],output,source=SOURCE)
    original=(output/'audio-cpu-inputs.json').read_bytes()
    with pytest.raises(ValueError):module.materialize_audio_cpu_inputs(roots[1],output,source=SOURCE)
    assert (output/'audio-cpu-inputs.json').read_bytes()==original


@pytest.mark.parametrize('source',[None,'main','A'*40])
def test_nonimmutable_source_cannot_authorize_input_replay(prepared,source):
    module,_,_,_,_,roots,output,_=prepared
    with pytest.raises(ValueError):module.materialize_audio_cpu_inputs(roots[1],output,source=source)


def test_CLI_reports_no_local_paths_or_model_execution(prepared,capsys):
    module,_,_,_,_,roots,output,_=prepared
    args=['--encoder-reference',str(roots[1]),'--output',str(output),'--source',SOURCE]
    assert module.main(args)==0
    assert module.main(args)==1
    text=capsys.readouterr().out
    assert str(output.parent) not in text and 'model_executed' in text
