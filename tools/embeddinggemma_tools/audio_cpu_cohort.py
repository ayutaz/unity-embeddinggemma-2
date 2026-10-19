"""Pinned original PCM and references for the two matched audio CPU cases.

Loading proves input/reference identity only, not inference or GPU acceptance.
No model graph is read and no reference array is substituted in execution.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
import re

import numpy as np


MODEL_REVISION = '914f7f89142e33e77833254d9c9b90c3cef7303b'
PREPROCESS_REFERENCE_SHA256 = '1209e2f95df2d0527db23d9185e01c38289b365c46e623e03d3e9ce26f8dfcf8'
PREPROCESS_AUDIT_SHA256 = '4e325f708e8d103a9de358a411380ddbb51cfe953984e9dcb73e291d919630fa'
ENCODER_REFERENCE_SHA256 = 'ac23f39904b8149dd5ccb6485afcc93d96047f0977b5faf1d640c8bd018c13c9'
ENCODER_AUDIT_SHA256 = '68d29c96012ef0a44114f2d3edbbff8d14e116ac697abee1a36d0b212155cfa1'
DOCUMENT_REFERENCE_SHA256 = '281d12573b431667e8841033db1c6a10dce4382ef98c9a5345c18044a1194001'
CASE_IDS = frozenset(('silence','impulse','sine','nyquist','noise','padded_noise'))
CPU_CASES = ('silence','padded_noise')


@dataclass(frozen=True)
class AudioCpuReferenceCase:
    id: str
    profile: str
    waveform: np.ndarray
    sample_mask: np.ndarray
    preprocessing_features: np.ndarray
    frame_mask: np.ndarray
    encoder_features: np.ndarray
    embedding: np.ndarray
    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    input_sha256: str


@dataclass(frozen=True)
class AudioCpuReferenceCohort:
    cases: tuple[AudioCpuReferenceCase, ...]
    snapshot_input_sha256: tuple[tuple[str,str], ...]
    preprocessing_reference_sha256: str
    encoder_reference_sha256: str
    document_reference_sha256: str

    def case(self, case_id):
        for case in self.cases:
            if case.id == case_id:
                return case
        raise ValueError('Unknown audio CPU reference case')


def _require(condition):
    if not condition:
        raise ValueError('Original audio CPU reference contract differs')


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _json(path, expected):
    path = Path(path)
    _require(path.is_file() and not path.is_symlink() and 0 < path.stat().st_size <= 32*1024**2)
    raw = path.read_bytes()
    _require(_sha(raw) == expected)
    value = json.loads(raw)
    _require(isinstance(value,dict))
    return value, raw


def _relative(name):
    _require(isinstance(name,str) and bool(name) and not any(c in name for c in ('\\',':','\x00')))
    path = PurePosixPath(name)
    _require(not path.is_absolute() and str(path) == name and not any(v in ('','.','..') for v in name.split('/')))
    return name


def _file(root, name, manifest):
    name = _relative(name)
    path = Path(root)/name
    spec = manifest[name]
    _require(type(spec['bytes']) is int and 0 < spec['bytes'] <= 2*1024**2)
    _require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(Path(root).resolve()))
    raw = path.read_bytes()
    _require(len(raw) == spec['bytes'] and _sha(raw) == spec['sha256'])
    return raw


def _array(value, shape, dtype='<f4'):
    array = np.asarray(value,dtype=dtype)
    _require(array.shape == shape and bool(np.isfinite(array).all()))
    # Bytes-backed arrays cannot have writeability re-enabled by a caller.
    return np.frombuffer(array.tobytes(),dtype=dtype).reshape(shape)


def _raw_array(raw, shape, dtype):
    _require(len(raw) == int(np.prod(shape))*4)
    value = np.frombuffer(raw,dtype=dtype).reshape(shape)
    _require(bool(np.isfinite(value).all()))
    return value


def _rows(value):
    rows = value['cases']
    _require(isinstance(rows,list) and len(rows) == 6)
    result = {v['id']:v for v in rows}
    _require(len(result) == 6 and set(result) == CASE_IDS)
    return result


def load_audio_cpu_reference_cohort(preprocessing, encoder_directory, document_reference):
    """Read exact original metadata and actual bytes with generic path-free errors."""
    try:
        preprocessing, encoder_directory = Path(preprocessing),Path(encoder_directory)
        pre, pre_raw = _json(preprocessing/'audio-reference.json',PREPROCESS_REFERENCE_SHA256)
        pre_audit,_ = _json(preprocessing/'audio-validation.json',PREPROCESS_AUDIT_SHA256)
        encoder,_ = _json(encoder_directory/'audio-encoder-reference.json',ENCODER_REFERENCE_SHA256)
        enc_audit,_ = _json(encoder_directory/'audio-encoder-validation.json',ENCODER_AUDIT_SHA256)
        document,_ = _json(document_reference,DOCUMENT_REFERENCE_SHA256)
        pm, em, dm = pre['metadata'],encoder['metadata'],document['metadata']
        _require(pre_audit['success'] is True and pre_audit['stage'] == 'audio_preprocessing_only')
        _require(pre_audit['metadata'] == pm and enc_audit['metadata'] == em)
        # The original audit includes historical ONNX failures. Its native
        # reference completion is distinct from that audit's overall success.
        _require(enc_audit['native_reference_complete'] is True and em['dtype'] == 'float32')
        for metadata in (pm,em,dm):
            _require(metadata['model_revision'] == MODEL_REVISION and re.fullmatch('[0-9a-f]{40}',metadata['source_commit']) is not None)
        _require(pm['fixture_provenance'] == 'procedurally_generated_pcm_no_user_audio')
        _require(pm['num_samples'] == 179200 and pm['sampling_rate'] == 16000)
        _require(pm['feature_shape'] == [1,1119,128] and pm['frame_mask_shape'] == [1,1119])
        _require(pm['feature_atol'] == .004 and pm['feature_rtol'] == 1e-4)
        _require(em['feature_atol'] == 1e-5 and em['feature_rtol'] == 1e-4)
        _require(em['preprocessing_reference_sha256'] == PREPROCESS_REFERENCE_SHA256)
        _require(dm['encoder_reference_sha256'] == ENCODER_REFERENCE_SHA256)
        _require(('role' not in dm or dm['role'] == 'document') and dm['embedding_dimension'] == 768)
        for name in ('config.json','model.safetensors'):
            _require(em['input_sha256'][name] == dm['input_sha256'][name])
        _require(pm['processor_sha256'] == em['input_sha256']['processor_config.json'])
        snapshot_hashes = dict(em['input_sha256'])
        for name,digest in dm['input_sha256'].items():
            _require(name not in snapshot_hashes or snapshot_hashes[name] == digest)
            snapshot_hashes[name] = digest
        for name,digest in snapshot_hashes.items():
            _relative(name)
            _require(isinstance(digest,str) and re.fullmatch('[0-9a-f]{64}',digest) is not None)
        spec = pre_audit['files']['audio-reference.json']
        _require(spec['bytes'] == len(pre_raw) and spec['sha256'] == PREPROCESS_REFERENCE_SHA256)
        pre_cases,enc_cases,final_cases = _rows(pre),_rows(encoder),_rows(document)
        _require(all(row['role'] == 'document' for row in final_cases.values()))
        result = []
        for case_id in CPU_CASES:
            valid_samples,frames,tokens = (134400,839,210) if case_id == 'padded_noise' else (179200,1119,280)
            profile = f'frames-1119-valid-{frames}'
            pr,er,dr = pre_cases[case_id],enc_cases[case_id],final_cases[case_id]
            ep,dp = encoder['profiles'][profile],document['profiles'][profile]
            _require(pr['valid_samples'] == valid_samples and er['profile'] == dr['profile'] == profile and dr['role'] == 'document')
            _require(ep['feature_count'] == dp['audio_feature_count'] == tokens and ep['feature_width'] == dp['feature_width'] == 512 and dp['image_feature_count'] == 0)
            raw = {name:_file(preprocessing,pr[name],pre_audit['files']) for name in ('waveform','sample_mask','features','frame_mask')}
            wave = _raw_array(raw['waveform'],(1,179200),'<f4')
            samples = _raw_array(raw['sample_mask'],(1,179200),'<i4')
            features = _raw_array(raw['features'],(1,1119,128),'<f4')
            frame_mask = _raw_array(raw['frame_mask'],(1,1119),'<i4')
            _require(bool((np.abs(wave) <= 1).all()) and not np.any(wave[samples == 0] != 0))
            _require(np.array_equal(samples,np.arange(179200)[None] < valid_samples))
            _require(np.array_equal(frame_mask,np.arange(1119)[None] < frames))
            _require(_file(encoder_directory,er['input_features'],enc_audit['files']) == raw['features'])
            _require(enc_audit['files'][ep['frame_mask']]['sha256'] == _sha(raw['frame_mask']))
            expected_encoder = _array(er['features'],(tokens,512))
            embedding = _array(dr['embedding'],(768,))
            _require(abs(float(np.linalg.norm(embedding.astype(np.float64))) - 1) <= .002)
            ids,mask = dr['input_ids'],dr['attention_mask']
            _require(len(ids) == len(mask) == dp['sequence_length'] and len(ids) > 0)
            _require(all(type(v) is int and 0 <= v <= 2**63-1 for v in ids))
            _require(all(type(v) is int and v in (0,1) for v in mask) and any(mask))
            result.append(AudioCpuReferenceCase(case_id,profile,wave,samples,features,frame_mask,
                expected_encoder,embedding,tuple(ids),tuple(mask),_sha(raw['waveform']+raw['sample_mask'])))
        return AudioCpuReferenceCohort(tuple(result),tuple(sorted(snapshot_hashes.items())),
            PREPROCESS_REFERENCE_SHA256,ENCODER_REFERENCE_SHA256,DOCUMENT_REFERENCE_SHA256)
    except (OSError,ValueError,TypeError,KeyError,IndexError,OverflowError):
        raise ValueError('Pinned audio CPU inputs or references are invalid') from None
