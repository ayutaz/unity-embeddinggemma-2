import hashlib
import io
import json
from pathlib import Path
import zipfile

import pytest
import yaml

from embeddinggemma_tools import artifact_zip
from embeddinggemma_tools import audio_generation_candidate as candidate
from embeddinggemma_tools import audio_candidate_distribution as distribution
from embeddinggemma_tools import audio_candidate_onnx_identity as identity
from embeddinggemma_tools import selected_audio_generation as selected


def archive(tmp_path, files=None):
    path = tmp_path/'original.zip'
    with zipfile.ZipFile(path, 'w') as stream:
        for name, data in (files or {'audio-runtime-reference/input.bin': b'original'}).items():
            stream.writestr(name, data)
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def test_local_reader_verifies_complete_bytes_without_faking_fresh_provider_metadata(tmp_path):
    path, digest = archive(tmp_path)
    before = path.read_bytes()
    reader, metadata = artifact_zip.verified_local_artifact_reader(path, digest)
    with reader:
        report = artifact_zip.extract_members(reader, tmp_path/'out', ['audio-runtime-reference/input.bin'])
    assert reader.closed and report['selected_count'] == 1
    assert (tmp_path/'out/audio-runtime-reference/input.bin').read_bytes() == b'original'
    assert path.read_bytes() == before
    assert metadata == {'transport': 'verified_local_archive', 'archive_sha256': digest,
                        'archive_bytes': len(before), 'provider_metadata_revalidated': False}
    assert 'workflow_run' not in metadata and 'id' not in metadata


@pytest.mark.parametrize('problem', ['digest', 'missing', 'empty'])
def test_unverified_local_archive_fails_without_echoing_local_path(tmp_path, problem):
    path, digest = archive(tmp_path)
    if problem == 'digest': digest = '0'*64
    if problem == 'missing': path.unlink()
    if problem == 'empty': path.write_bytes(b''); digest = hashlib.sha256(b'').hexdigest()
    with pytest.raises(ValueError) as caught:
        artifact_zip.verified_local_artifact_reader(path, digest)
    assert str(tmp_path) not in str(caught.value)
    assert not (tmp_path/'out').exists()


@pytest.mark.parametrize('route', ['candidate', 'distribution', 'identity'])
def test_retired_raw_routes_require_local_archive_before_network_or_output(monkeypatch, tmp_path, route):
    def forbidden(*args, **kwargs): pytest.fail('Retired raw source unexpectedly used a hosted reader')
    monkeypatch.setattr(artifact_zip, 'github_artifact_reader', forbidden)
    monkeypatch.setattr(candidate, 'fetch_original', forbidden)
    if route == 'candidate': call = lambda: candidate.fetch_candidate(tmp_path/'raw', tmp_path/'golden', case_id='silence')
    elif route == 'distribution': call = lambda: distribution.fetch_inputs(tmp_path/'raw', tmp_path/'golden')
    else: call = lambda: identity.fetch_native(tmp_path/'raw', profile='frames-1119-valid-1119')
    with pytest.raises(ValueError, match='local archive'): call()
    assert not (tmp_path/'raw').exists() and not (tmp_path/'golden').exists()


def test_raw_candidate_fetch_uses_exact_local_archive_and_preserves_raw_loader(monkeypatch, tmp_path):
    contract = candidate.candidate_contract().copy()
    profile = 'frames-1119-valid-839'
    files = {f'audio-runtime-reference/{name}': data for name, data in {
        'audio-encoder-reference.json': b'{}', 'audio-encoder-validation.json': b'{}',
        profile+'/audio.pt2': b'synthetic original raw graph', profile+'/padded_noise-features-fp32.bin': b'input'}.items()}
    path, digest = archive(tmp_path, files); contract['artifact_sha256'] = digest
    monkeypatch.setattr(candidate, 'candidate_contract', lambda: contract)
    monkeypatch.setattr(candidate, 'fetch_original', lambda output: pytest.fail('Archived fetch must use prepared local goldens'))
    monkeypatch.setattr(candidate, 'original_reference', lambda output: ({}, {}))
    checks = []
    monkeypatch.setattr(candidate, '_candidate_input_files', lambda *args, **kwargs: checks.append(kwargs['case_id']))
    monkeypatch.setattr(artifact_zip, 'github_artifact_reader', lambda *args: pytest.fail('Unexpected hosted source'))
    result = candidate.fetch_candidate(tmp_path/'raw', tmp_path/'golden', case_id='padded_noise', source_archive=path)
    assert checks == ['padded_noise']
    assert result['source_archive_transport'] == 'verified_local_archive'
    assert result['source_archive_sha256'] == digest and result['provider_metadata_revalidated'] is False
    assert result['generation_qualified'] is False
    assert (tmp_path/'raw/audio-runtime-reference'/profile/'audio.pt2').read_bytes() == b'synthetic original raw graph'


@pytest.mark.parametrize('route', ['candidate', 'distribution'])
def test_archived_fetch_requires_verified_local_goldens_before_extraction(monkeypatch, tmp_path, route):
    path, digest = archive(tmp_path); contract = candidate.candidate_contract().copy(); contract['artifact_sha256'] = digest
    monkeypatch.setattr(candidate, 'candidate_contract', lambda: contract)
    monkeypatch.setattr(candidate, 'fetch_original', lambda *a: pytest.fail('Archived fetch unexpectedly used hosted goldens'))
    def invalid(*a): raise ValueError('Prepared original golden bytes are required')
    monkeypatch.setattr(candidate, 'original_reference', invalid)
    monkeypatch.setattr(artifact_zip, 'extract_members', lambda *a, **kw: pytest.fail('Unverified local goldens reached extraction'))
    call = (lambda: candidate.fetch_candidate(tmp_path/'raw', tmp_path/'golden', case_id='silence', source_archive=path)) if route == 'candidate' else (lambda: distribution.fetch_inputs(tmp_path/'raw', tmp_path/'golden', source_archive=path))
    with pytest.raises(ValueError, match='golden'): call()
    assert not (tmp_path/'raw').exists() and not (tmp_path/'golden').exists()


@pytest.mark.parametrize('entry', ['raw', 'selected', 'distribution', 'identity'])
def test_cli_routes_explicit_archive_only_to_archived_fetch(monkeypatch, tmp_path, entry):
    source = tmp_path/'private-original.zip'; calls = []
    if entry in ('raw', 'selected'):
        monkeypatch.setattr(candidate, 'fetch_candidate', lambda *a, **kw: calls.append(kw['source_archive']) or {'model_generation': 'synthetic'})
        module = candidate if entry == 'raw' else selected
        args = ['fetch', '--output', str(tmp_path/'raw'), '--original', str(tmp_path/'golden'), '--case', 'silence']
    elif entry == 'distribution':
        monkeypatch.setattr(distribution, 'fetch_inputs', lambda *a, **kw: calls.append(kw['source_archive']))
        monkeypatch.setattr(distribution, 'prepare', lambda *a, **kw: {'case_count': 6, 'profile_count': 2})
        module = distribution
        args = ['--fetch', '--candidate', str(tmp_path/'raw'), '--original', str(tmp_path/'golden'), '--output', str(tmp_path/'output'), '--source', 'a'*40]
    else:
        monkeypatch.setattr(identity, 'fetch_native', lambda *a, **kw: (calls.append(kw['source_archive']) or tmp_path/'raw.onnx', 'b'*64, {}))
        monkeypatch.setattr(identity, 'inspect_native', lambda *a, **kw: {'complete': True, 'computation_identity_matches_original': True})
        module = identity
        args = ['--fetch', '--input', str(tmp_path/'raw'), '--profile', 'frames-1119-valid-1119', '--source', 'a'*40, '--output', str(tmp_path/'identity.json')]
    assert module.main(args+['--source-archive', str(source)]) == 0
    assert calls == [source]


def test_hosted_historical_raw_modes_fail_explicitly_without_models_or_uploads():
    root = Path(__file__).resolve().parents[2]
    workflow = yaml.load((root/'.github/workflows/audio-runtime-reference.yml').read_text(encoding='utf-8'), Loader=yaml.BaseLoader)
    assert 'candidate-native-identity' not in workflow['jobs'] and 'preserve-candidate' not in workflow['jobs']
    retired = workflow['jobs']['retired-raw-source']
    assert "github.event_name == 'workflow_dispatch'" in retired['if']
    assert 'inputs.diagnose_candidate_onnx_identity' in retired['if'] and 'inputs.preserve_candidate_distribution' in retired['if']
    assert not retired.get('continue-on-error')
    assert len(retired['steps']) == 1 and 'exit 1' in retired['steps'][0]['run']
    assert not any('uses' in step for step in retired['steps'])
    assert 'embeddinggemma_tools.native_regression_distribution' in '\n'.join(step.get('run', '') for step in workflow['jobs']['candidate-execute']['steps'])
    assert 'embeddinggemma_tools.original_audio_input_preservation' in '\n'.join(step.get('run', '') for step in workflow['jobs']['preserve-inputs']['steps'])
