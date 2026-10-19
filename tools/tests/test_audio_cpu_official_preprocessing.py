import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import Gemma4AudioFeatureExtractor


ORIGINAL_FEATURE_SHA256={
    'silence':'366de71cb4e45e4feda58a605babe4ea9d8693a7e37e84eac0735a7c42d0ad24',
    'padded_noise':'6065504b6b2d244526df3a79cb5b88e9463e8e576ff877b5b8fa90b9bf18b1b0'}


@pytest.fixture(autouse=True)
def cpu_threads():
    previous=torch.get_num_threads()
    torch.set_num_threads(4)
    yield
    torch.set_num_threads(previous)


def inputs(case_id='silence'):
    samples=torch.arange(179200)[None]
    valid=134400 if case_id=='padded_noise' else 179200
    mask=(samples<valid).long()
    if case_id=='silence':wave=torch.zeros(1,179200)
    else:wave=torch.rand((1,179200),generator=torch.Generator().manual_seed(17))*1.6-.8
    return wave*mask,mask


def implementation():
    return importlib.import_module('embeddinggemma_tools.audio_cpu_reference').OfficialAudioCpuPreprocessor


@pytest.mark.parametrize('case_id',['silence','padded_noise'])
def test_recomputes_original_feature_bytes_from_actual_PCM_without_loading_goldens(case_id,monkeypatch):
    wave,mask=inputs(case_id)
    original=json.loads((Path(__file__).parents[1]/'references/audio-cpu-original/audio-validation.json').read_bytes())
    for tensor,suffix in ((wave,'pcm-fp32'),(mask.int(),'samples-int32')):
        assert hashlib.sha256(tensor.numpy().tobytes()).hexdigest()==original['files'][case_id+'-'+suffix+'.bin']['sha256']
    before=(wave.clone(),mask.clone())
    module=implementation()(Gemma4AudioFeatureExtractor()).eval()
    def forbidden(*args,**kwargs):raise AssertionError('Preprocessing must compute from PCM, never read a golden')
    monkeypatch.setattr(Path,'read_bytes',forbidden)
    actual,frames=module(wave,mask)
    assert actual.device.type==frames.device.type=='cpu'
    assert actual.dtype==torch.float32 and actual.shape==(1,1119,128)
    assert frames.dtype==torch.int32 and frames.shape==(1,1119)
    assert hashlib.sha256(actual.numpy().tobytes()).hexdigest()==ORIGINAL_FEATURE_SHA256[case_id]
    assert torch.equal(frames,torch.arange(1119)[None]<(839 if case_id=='padded_noise' else 1119))
    assert torch.equal(wave,before[0]) and torch.equal(mask,before[1])
    actual.zero_()
    again,_=module(wave,mask)
    assert hashlib.sha256(again.numpy().tobytes()).hexdigest()==ORIGINAL_FEATURE_SHA256[case_id]


@pytest.mark.parametrize('invalid',[
    'wave_dtype','wave_shape','wave_meta','nan','amplitude','mask_dtype','mask_shape',
    'mask_meta','binary_mask','hole','short_prefix','nonzero_padding','wave_type','mask_type'])
def test_rejects_invalid_actual_CPU_inputs(invalid):
    module=implementation()(Gemma4AudioFeatureExtractor())
    wave,mask=inputs()
    if invalid=='wave_dtype':wave=wave.double()
    elif invalid=='wave_shape':wave=wave[0]
    elif invalid=='wave_meta':wave=torch.empty(wave.shape,device='meta')
    elif invalid=='nan':wave[0,0]=float('nan')
    elif invalid=='amplitude':wave[0,0]=1.1
    elif invalid=='mask_dtype':mask=mask.int()
    elif invalid=='mask_shape':mask=mask[0]
    elif invalid=='mask_meta':mask=torch.empty(mask.shape,dtype=torch.int64,device='meta')
    elif invalid=='binary_mask':mask[0,0]=2
    elif invalid=='hole':mask[0,100]=0
    elif invalid=='short_prefix':mask[:,160:]=0
    elif invalid=='nonzero_padding':mask[0,-1]=0;wave[0,-1]=1
    elif invalid=='wave_type':wave=None
    elif invalid=='mask_type':mask=None
    with pytest.raises(ValueError):module(wave,mask)


def test_rejects_processor_configuration_outside_the_pinned_contract():
    extractor=Gemma4AudioFeatureExtractor()
    extractor.sampling_rate=8000
    with pytest.raises(ValueError):implementation()(extractor)


@pytest.mark.parametrize('bad_output',['nan','shape'])
def test_rejects_invalid_official_extractor_output(monkeypatch,bad_output):
    extractor=Gemma4AudioFeatureExtractor()
    module=implementation()(extractor)
    raw=np.full((1119,128),np.nan,dtype=np.float32) if bad_output=='nan' else np.zeros((1118,128),np.float32)
    monkeypatch.setattr(extractor,'_extract_spectrogram',lambda *_:(raw,np.ones(1119,dtype=bool)))
    with pytest.raises(ValueError):module(*inputs())


def test_loaded_CPU_pipeline_uses_official_computation_instead_of_export_lowering(tmp_path,monkeypatch):
    driver=importlib.import_module('embeddinggemma_tools.audio_cpu_reference')
    export=importlib.import_module('embeddinggemma_tools.export')
    model=importlib.import_module('embeddinggemma_tools.model')
    (tmp_path/'processor_config.json').write_text(json.dumps({'feature_extractor':{}}))
    case=SimpleNamespace(id='padded_noise',profile='frames-1119-valid-839')
    class Encoder:
        configuration={'case_id':case.id,'profile':case.profile}
        def __init__(self,*args,**kwargs):pass
        def close(self):pass
    monkeypatch.setattr(driver,'AuditedAudioCpuEncoder',Encoder)
    monkeypatch.setattr(model,'load_text_model',lambda _:torch.nn.Identity())
    monkeypatch.setattr(export,'MultimodalEmbeddingExport',lambda *args,**kwargs:torch.nn.Identity())
    preprocessor,_,_=driver._load_modules(tmp_path,tmp_path,tmp_path/'chain.json','a'*40,case,
        {'image_token_id':1,'audio_token_id':2},{'sequence_length':4,'audio_feature_count':210})
    features,_=preprocessor(*inputs('padded_noise'))
    assert hashlib.sha256(features.numpy().tobytes()).hexdigest()==ORIGINAL_FEATURE_SHA256['padded_noise']
