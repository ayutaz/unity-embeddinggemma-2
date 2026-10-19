import hashlib
import importlib
import json
from pathlib import Path
import re

import numpy as np
import pytest


def implementation():
    return importlib.import_module('embeddinggemma_tools.image_reference_cohort')


@pytest.fixture
def cohort(tmp_path, monkeypatch):
    module = implementation()
    snapshot = {name: str(i + 1) * 64 for i, name in enumerate(
        ('config.json', 'processor_config.json', 'model.safetensors'))}
    metadata = dict(model_revision='914f7f89142e33e77833254d9c9b90c3cef7303b',
        source_commit='a' * 40, dtype='float32', input_sha256=snapshot)
    vision = dict(metadata=metadata, profiles={}, cases=[])
    document = dict(metadata=dict(model_revision=metadata['model_revision'], role='document',
        source_commit='b' * 40, vision_reference_sha256='7' * 64,
        input_sha256={**snapshot, 'tokenizer.json': '9' * 64}, embedding_dimension=768), profiles={}, cases=[])
    audit = dict(success=True, stage='vision_encoder_projector_only', metadata=metadata, files={})
    for profile in ('square', 'landscape', 'portrait'):
        positions = f'{profile}/positions-int32.bin'
        graph = f'{profile}/vision.pt2'
        vision['profiles'][profile] = dict(pixel_shape=[1, 2, 3], feature_count=1,
            feature_width=512, positions_shape=[1, 2, 2], positions=positions, graph=graph)
        document['profiles'][profile] = dict(sequence_length=4, image_feature_count=1,
            audio_feature_count=0, feature_width=512, graph=f'{profile}/backbone.pt2')
        audit['files'][positions] = dict(bytes=16, sha256='4' * 64)
        audit['files'][graph] = dict(bytes=100, sha256='5' * 64)
        for index in range(2):
            case = f'{profile}-{index}'
            pixels, image = f'{profile}/{index}-pixels-fp32.bin', f'{profile}/{index}.png'
            vision['cases'].append(dict(id=case, profile=profile, pixels=pixels, image=image,
                features=[[float(index)] * 512]))
            audit['files'][pixels] = dict(bytes=24, sha256='6' * 64)
            audit['files'][image] = dict(bytes=100, sha256='8' * 64)
            document['cases'].append(dict(id=case, profile=profile, role='document', pixels=pixels,
                pixels_sha256='6' * 64, features_file=f'{profile}/{index}-features-fp32.bin',
                input_ids=[1, 2, 3, 0], attention_mask=[1, 1, 1, 0],
                embedding=[1.0] + [0.0] * 767, formatted_text='document image'))

    def save():
        paths = [tmp_path / name for name in ('vision.json', 'audit.json', 'document.json')]
        raw = json.dumps(vision).encode()
        paths[0].write_bytes(raw)
        vision_sha = hashlib.sha256(raw).hexdigest()
        audit['files']['vision-reference.json'] = dict(bytes=len(raw), sha256=vision_sha)
        paths[1].write_text(json.dumps(audit), encoding='utf-8')
        paths[2].write_text(json.dumps(document), encoding='utf-8')
        for name, path in zip(('ENCODER_REFERENCE_SHA256', 'ENCODER_AUDIT_SHA256', 'DOCUMENT_REFERENCE_SHA256'), paths, strict=True):
            monkeypatch.setattr(module, name, hashlib.sha256(path.read_bytes()).hexdigest())
        return paths
    return module, vision, audit, document, save


def test_cpu_encoder_pin_is_the_existing_gpu_encoder_pin():
    module = implementation()
    fixture = Path(__file__).resolve().parents[2] / 'Assets/Validation/Runtime/PlayerMediaFixture.cs'
    gpu_pin = re.search(r'Pinned\("vision/vision-reference.json", "([a-f0-9]{64})"', fixture.read_text(encoding='utf-8')).group(1)
    assert module.ENCODER_REFERENCE_SHA256 == gpu_pin


def test_binds_same_gpu_encoder_and_original_final_without_substituting_document_features(cohort):
    module, _, _, _, save = cohort
    bound = module.load_image_reference_cohort(*save())
    case = bound.case('square-1')
    np.testing.assert_array_equal(case.encoder_features, np.ones((1, 512), dtype=np.float32))
    np.testing.assert_array_equal(case.embedding, np.asarray([1.] + [0.] * 767, dtype=np.float32))
    assert bound.document_encoder_source_sha256 == '7' * 64
    assert bound.encoder_reference_sha256 != bound.document_encoder_source_sha256
    assert len(bound.cases) == 6
    with pytest.raises(ValueError):
        case.encoder_features[0, 0] = 9


def test_regenerated_or_different_encoder_reference_is_rejected_even_if_numeric_values_match(cohort):
    module, _, _, _, save = cohort
    paths = save()
    value = json.loads(paths[0].read_text())
    value['metadata']['source_commit'] = 'c' * 40
    paths[0].write_text(json.dumps(value), encoding='utf-8')
    with pytest.raises(ValueError, match='Pinned'):
        module.load_image_reference_cohort(*paths)


@pytest.mark.parametrize('problem', ['snapshot', 'role', 'missing_case', 'duplicate_case',
    'profile', 'encoder_shape', 'nonfinite_encoder', 'embedding_shape', 'embedding_norm',
    'token_count', 'pixel_hash', 'manifest', 'unsafe_image'])
def test_inconsistent_approved_cohort_metadata_is_rejected(cohort, problem):
    module, vision, audit, document, save = cohort
    if problem == 'snapshot': document['metadata']['input_sha256']['model.safetensors'] = 'f' * 64
    elif problem == 'role': document['cases'][0]['role'] = 'query'
    elif problem == 'missing_case': document['cases'].pop()
    elif problem == 'duplicate_case': document['cases'].append(document['cases'][0])
    elif problem == 'profile': document['profiles']['square']['image_feature_count'] = 2
    elif problem == 'encoder_shape': vision['cases'][0]['features'] = [[0.0]]
    elif problem == 'nonfinite_encoder': vision['cases'][0]['features'][0][0] = float('nan')
    elif problem == 'embedding_shape': document['cases'][0]['embedding'].pop()
    elif problem == 'embedding_norm': document['cases'][0]['embedding'][0] = 2.
    elif problem == 'token_count': document['cases'][0]['input_ids'].pop()
    elif problem == 'pixel_hash': document['cases'][0]['pixels_sha256'] = 'f' * 64
    elif problem == 'manifest': audit['success'] = False
    elif problem == 'unsafe_image': vision['cases'][0]['image'] = '../fixture.png'
    with pytest.raises(ValueError):
        module.load_image_reference_cohort(*save())


def test_unknown_case_and_missing_reference_fail_without_local_paths(cohort, tmp_path):
    module, _, _, _, save = cohort
    paths = save()
    bound = module.load_image_reference_cohort(*paths)
    with pytest.raises(ValueError): bound.case('unknown')
    paths[0].unlink()
    with pytest.raises(ValueError) as error:
        module.load_image_reference_cohort(*paths)
    assert str(tmp_path) not in str(error.value)
