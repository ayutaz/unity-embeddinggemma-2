"""Bind existing image GPU encoder goldens and original Document final goldens.

The Document producer used a different encoder generation. Keep its provenance
separate; loading these references proves no model execution or CPU qualification.
No graph or image file is opened or certified by this metadata loader.
"""
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re

import numpy as np


ENCODER_REFERENCE_SHA256 = '3dfccd1a487ce04df3f8d640483ca045f7a7a56231bc1096ad0f845c923a29c6'
ENCODER_AUDIT_SHA256 = 'ab1aa775b553f33a5e2892abed2cdcef7768e35fbad6fa5af9ec3286a7ba90ba'
DOCUMENT_REFERENCE_SHA256 = '90692ad38c6d7d844dc6b6e79c7d5cfa719ea59866946dbf94a40bca10ac9eac'
MODEL_REVISION = '914f7f89142e33e77833254d9c9b90c3cef7303b'
PROFILES = ('square', 'landscape', 'portrait')
CASE_IDS = tuple(f'{profile}-{index}' for profile in PROFILES for index in range(2))


@dataclass(frozen=True)
class ImageReferenceCase:
    id: str
    profile: str
    image: str
    pixels: str
    pixels_sha256: str
    document_features_file: str
    encoder_features: np.ndarray
    embedding: np.ndarray
    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]


@dataclass(frozen=True)
class ImageReferenceCohort:
    cases: tuple[ImageReferenceCase, ...]
    encoder_reference_sha256: str
    document_reference_sha256: str
    document_encoder_source_sha256: str

    def case(self, case_id: str) -> ImageReferenceCase:
        for case in self.cases:
            if case.id == case_id:
                return case
        raise ValueError('Unknown image reference case')


def _require(condition):
    if not condition:
        raise ValueError('Inconsistent image reference cohort')


def _digest(value):
    _require(isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None)
    return value


def _relative(value):
    _require(isinstance(value, str) and bool(value))
    path = PurePosixPath(value)
    _require(not path.is_absolute() and not any(part in ('.', '..') for part in value.split('/'))
             and not any(char in value for char in ('\\', ':', '\x00'))
             and str(path) == value)
    return value


def _read(path, expected):
    try:
        raw = Path(path).read_bytes()
    except OSError:
        raise ValueError('Pinned image reference unavailable') from None
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError('Pinned image reference hash mismatch')
    value = json.loads(raw)
    _require(isinstance(value, dict))
    return value, len(raw)


def _positive(value):
    _require(type(value) is int and value > 0)
    return value


def _shape(value):
    _require(isinstance(value, list) and bool(value))
    return tuple(_positive(item) for item in value)


def _array(value, shape):
    result = np.array(value, dtype=np.float32)
    _require(result.shape == shape and bool(np.isfinite(result).all()))
    result.setflags(write=False)
    return result


def _cases(value):
    _require(isinstance(value, list) and len(value) == len(CASE_IDS))
    result = {case['id']: case for case in value}
    _require(len(result) == len(CASE_IDS) and set(result) == set(CASE_IDS))
    return result


def load_image_reference_cohort(encoder_reference_path, encoder_audit_path,
                                document_reference_path) -> ImageReferenceCohort:
    """Load only the pinned metadata cohort, preserving both encoder generations.

    Errors omit caller paths and data values. Manifest hashes bind declarations;
    callers still need to verify actual graph/input bytes before executing them.
    """
    try:
        encoder, encoder_size = _read(encoder_reference_path, ENCODER_REFERENCE_SHA256)
        audit, _ = _read(encoder_audit_path, ENCODER_AUDIT_SHA256)
        document, _ = _read(document_reference_path, DOCUMENT_REFERENCE_SHA256)
        _require(audit['success'] is True and audit['stage'] == 'vision_encoder_projector_only')
        metadata, final_metadata = encoder['metadata'], document['metadata']
        _require(audit['metadata'] == metadata and metadata['dtype'] == 'float32')
        _require(metadata['model_revision'] == final_metadata['model_revision'] == MODEL_REVISION)
        _require(final_metadata['role'] == 'document' and final_metadata['embedding_dimension'] == 768)
        for item in (metadata, final_metadata):
            _require(re.fullmatch('[0-9a-f]{40}', item['source_commit']) is not None)
        generation = _digest(final_metadata['vision_reference_sha256'])
        for name in ('config.json', 'processor_config.json', 'model.safetensors'):
            _require(_digest(metadata['input_sha256'][name]) == final_metadata['input_sha256'][name])
        for table in (metadata['input_sha256'], final_metadata['input_sha256']):
            for name, digest in table.items():
                _relative(name)
                _digest(digest)
        files = audit['files']
        for name, entry in files.items():
            _relative(name)
            _positive(entry['bytes'])
            _digest(entry['sha256'])
        _require(files['vision-reference.json'] == dict(bytes=encoder_size, sha256=ENCODER_REFERENCE_SHA256))
        _require(set(encoder['profiles']) == set(document['profiles']) == set(PROFILES))
        for name in PROFILES:
            profile, final = encoder['profiles'][name], document['profiles'][name]
            _positive(profile['feature_count'])
            _require(profile['feature_width'] == final['feature_width'] == 512)
            _require(final['image_feature_count'] == profile['feature_count'] and final['audio_feature_count'] == 0)
            _positive(final['sequence_length'])
            pixel_shape, positions_shape = _shape(profile['pixel_shape']), _shape(profile['positions_shape'])
            _require(len(pixel_shape) == len(positions_shape) == 3 and pixel_shape[0] == positions_shape[0] == 1)
            _relative(final['graph'])
            _require(_relative(profile['graph']) in files)
            _require(files[_relative(profile['positions'])]['bytes'] == math.prod(positions_shape) * 4)
        encoders, documents = _cases(encoder['cases']), _cases(document['cases'])
        result = []
        for case_id in CASE_IDS:
            source, final = encoders[case_id], documents[case_id]
            profile_name = case_id.rsplit('-', 1)[0]
            _require(source['profile'] == final['profile'] == profile_name and final['role'] == 'document')
            profile, final_profile = encoder['profiles'][profile_name], document['profiles'][profile_name]
            pixels = _relative(source['pixels'])
            _require(final['pixels'] == pixels and final['pixels_sha256'] == files[pixels]['sha256'])
            _require(files[pixels]['bytes'] == math.prod(profile['pixel_shape']) * 4)
            image = _relative(source['image'])
            _require(image in files)
            features = _array(source['features'], (profile['feature_count'], 512))
            embedding = _array(final['embedding'], (768,))
            _require(abs(float(np.linalg.norm(embedding.astype(np.float64))) - 1) <= .002)
            ids, mask = final['input_ids'], final['attention_mask']
            _require(len(ids) == len(mask) == final_profile['sequence_length'])
            _require(all(type(token) is int and token >= 0 for token in ids))
            _require(all(type(token) is int and token in (0, 1) for token in mask) and any(mask))
            result.append(ImageReferenceCase(case_id, profile_name, image, pixels, final['pixels_sha256'],
                _relative(final['features_file']), features, embedding, tuple(ids), tuple(mask)))
        return ImageReferenceCohort(tuple(result), ENCODER_REFERENCE_SHA256,
                                    DOCUMENT_REFERENCE_SHA256, generation)
    except (KeyError, TypeError, OverflowError, ValueError):
        # Do not leak filesystem paths or invalid input values in public diagnostics.
        raise ValueError('Pinned image reference cohort invalid or unavailable') from None
