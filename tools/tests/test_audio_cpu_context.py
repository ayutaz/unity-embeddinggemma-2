import hashlib
import json

import pytest
import torch

from embeddinggemma_tools import prepare
from test_audio_encoder_prepare import fixture


def test_cpu_math_context_is_recorded_before_weight_loading_even_when_loading_fails(tmp_path,monkeypatch):
    snapshot,inputs=fixture(tmp_path,monkeypatch);destination=tmp_path/'out'
    monkeypatch.setenv('MKL_ENABLE_INSTRUCTIONS','AVX2')
    monkeypatch.setenv('UNRELATED_PRIVATE_ENV','must-not-be-recorded')
    def load(path):
        p=destination/'audio-cpu-context.json'
        assert p.is_file(), 'Failure artifacts need CPU math context before loading real weights'
        r=json.loads(p.read_text())
        assert r['source_commit']=='b'*40 and r['cpu_capability']==torch.backends.cpu.get_cpu_capability()
        assert r['environment']['MKL_ENABLE_INSTRUCTIONS']=='AVX2'
        assert 'UNRELATED_PRIVATE_ENV' not in r['environment']
        assert r['torch_threads']==torch.get_num_threads() and r['torch_version']==str(torch.__version__)
        assert r['torch_build_configuration'] and r['system']
        assert r['numpy_version'] and r['numpy_blas']['name'] and r['numpy_blas']['version']
        assert 'include directory' not in r['numpy_blas'] and 'lib directory' not in r['numpy_blas']
        assert not r['encoder_validated'] and not r['gpu_validated']
        raise RuntimeError('unit-test stop before weights')
    monkeypatch.setattr(prepare,'load_audio_models',load)
    with pytest.raises(RuntimeError,match='stop before weights'):
        prepare.prepare_audio_encoder(snapshot,inputs,destination,source_commit='b'*40)
    assert not (destination/'audio-encoder-validation.json').exists()


def test_success_audits_context_file_without_changing_reference_tolerances_or_native_gate(tmp_path,monkeypatch):
    snapshot,inputs=fixture(tmp_path,monkeypatch);destination=tmp_path/'out'
    result=prepare.prepare_audio_encoder(snapshot,inputs,destination,source_commit='b'*40)
    p=destination/'audio-cpu-context.json'
    assert p.is_file(), 'Successful and failed producer contexts must use the same artifact contract'
    assert result['files'][p.name]==dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    assert result['metadata']['feature_atol']==1e-5 and result['metadata']['feature_rtol']==1e-4
    assert result['case_count']==6 and result['native_reference_complete']


def test_invalid_preprocessing_removes_stale_host_context(tmp_path):
    destination=tmp_path/'out';destination.mkdir()
    p=destination/'audio-cpu-context.json';p.write_text('{"source_commit":"stale"}')
    with pytest.raises(FileNotFoundError):
        prepare.prepare_audio_encoder(tmp_path/'missing',tmp_path/'missing',destination)
    assert not p.exists(), 'An earlier host must not be attributed to the new failed attempt'
