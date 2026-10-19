import hashlib
import importlib
import json

import numpy as np
from PIL import Image
import pytest
import torch

from test_image_reference_cohort import cohort


def implementation():
    return importlib.import_module('embeddinggemma_tools.image_cpu_reference')


@pytest.fixture
def pipeline(cohort, tmp_path, monkeypatch):
    module = implementation()
    _, vision, audit, document, save = cohort
    snapshot, fixtures, refs = [tmp_path / name for name in ('snapshot', 'fixtures', 'references')]
    for path in (snapshot, fixtures / 'square', refs): path.mkdir(parents=True)
    for name in document['metadata']['input_sha256']:
        raw = b'{}'
        (snapshot / name).write_bytes(raw)
        digest = hashlib.sha256(raw).hexdigest()
        document['metadata']['input_sha256'][name] = digest
        if name in vision['metadata']['input_sha256']: vision['metadata']['input_sha256'][name] = digest
    vision['metadata'].update(feature_rtol=1e-4, feature_atol=1e-5, minimum_cosine=.999999)
    document['metadata'].update(image_token_id=100, audio_token_id=101)
    image = fixtures / 'square/0.png'
    Image.new('RGB', (2, 1)).save(image)
    pixels = np.zeros((1, 2, 3), dtype='<f4').tobytes()
    positions = np.zeros((1, 2, 2), dtype='<i4').tobytes()
    for name, raw in [('square/0.png', image.read_bytes()), ('square/0-pixels-fp32.bin', pixels),
                      ('square/positions-int32.bin', positions)]:
        audit['files'][name] = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    document['cases'][0]['pixels_sha256'] = hashlib.sha256(pixels).hexdigest()
    for name, path in zip(('vision-reference.json', 'vision-validation.json', 'image-embedding-reference.json'), save(), strict=True):
        (refs / name).write_bytes(path.read_bytes())
    from embeddinggemma_tools import reference_cpu_performance
    monkeypatch.setattr(reference_cpu_performance, 'IMAGE_REFERENCE', hashlib.sha256((refs / 'image-embedding-reference.json').read_bytes()).hexdigest())
    state = dict(pixel_delta=0., position_delta=0, encoder_delta=0., final_bad=False, calls=[], loads=0)

    class Processor:
        def __call__(self, *, images, return_tensors):
            assert images.device.type == 'cpu' and return_tensors == 'pt'
            value = dict(pixel_values=torch.zeros(1, 2, 3) + state['pixel_delta'],
                         image_position_ids=torch.zeros(1, 2, 2, dtype=torch.int64) + state['position_delta'])
            state['calls'].append(('preprocessing', value['pixel_values']))
            return value

    class Encoder(torch.nn.Module):
        def forward(self, value):
            assert value['pixel_values'] is state['calls'][-1][1]
            if state.get('mutate_pixels'): value['pixel_values'].zero_()
            result = torch.zeros(1, 512) + value['pixel_values'].sum() + state['encoder_delta']
            state['calls'].append(('encoder', result))
            return result

    class Backbone(torch.nn.Module):
        def forward(self, ids, mask, features, empty):
            assert features is state['calls'][-1][1] and tuple(empty.shape) == (0, 512)
            if state.get('mutate_encoder'): features.zero_()
            result = torch.zeros(768)
            if not state['final_bad']: result[0] = 1 + features.sum()
            state['calls'].append(('backbone', result))
            return torch.nn.functional.normalize(result, dim=-1)

    def load(*args):
        state['loads'] += 1
        return Processor(), Encoder(), Backbone()

    monkeypatch.setattr(module, '_load_pipeline', load)
    monkeypatch.setattr(module, '_validate_execution_source', lambda source: {'fixture_source_checked': True})
    return module, snapshot, fixtures, refs, state


def run(pipeline, output):
    module, snapshot, fixtures, refs, _ = pipeline
    return module.measure_image_cpu_reference(snapshot, refs, 'square', output,
                                             source='a' * 40, fixture_root=fixtures)


def test_four_actual_chained_calls_preserve_all_intermediate_and_final_captures(pipeline, tmp_path):
    module, _, _, refs, state = pipeline
    output = tmp_path / 'output'
    result = run(pipeline, output)
    assert result['success'] and len(result['rounds']) == 4
    assert [name for name, _ in state['calls']] == ['preprocessing', 'encoder', 'backbone'] * 4
    assert len(list(output.glob('*.bin'))) == 16
    assert result['CPU_comparison_qualified'] is False and result['complete_performance_accepted'] is False
    audit = module.audit_image_cpu_reference(output, refs, source='a' * 40)
    assert audit['all_four_chains_match_fixed_references'] is True
    assert audit['complete_performance_accepted'] is False


@pytest.mark.parametrize('problem', ['pixel_delta', 'position_delta', 'encoder_delta', 'final_bad'])
def test_final_vector_success_does_not_hide_invalid_intermediates_or_partial_failure(pipeline, tmp_path, problem):
    module, _, _, refs, state = pipeline
    state[problem] = True if problem == 'final_bad' else 1
    output = tmp_path / 'failed'
    with pytest.raises(ValueError): run(pipeline, output)
    result = json.loads((output / 'image-cpu-reference.json').read_text())
    assert result['success'] is False
    assert list(output.glob('*-encoder.bin'))
    with pytest.raises(ValueError): module.audit_image_cpu_reference(output, refs, source='a' * 40)


def test_changed_snapshot_fails_before_loading_and_existing_evidence_is_preserved(pipeline, tmp_path):
    _, snapshot, _, _, state = pipeline
    (snapshot / 'model.safetensors').write_bytes(b'changed')
    with pytest.raises(ValueError): run(pipeline, tmp_path / 'output')
    assert state['loads'] == 0
    existing = tmp_path / 'existing'
    existing.mkdir()
    marker = existing / 'keep.json'
    marker.write_text('preserve')
    with pytest.raises(ValueError): run(pipeline, existing)
    assert marker.read_text() == 'preserve'


@pytest.mark.parametrize('target', ['encoder', 'final', 'path', 'source', 'module', 'case', 'phase', 'round_state', 'source_journal'])
def test_independent_auditor_rechecks_captures_even_when_self_report_hash_is_updated(pipeline, tmp_path, target):
    module, _, _, refs, _ = pipeline
    output = tmp_path / 'output'
    run(pipeline, output)
    if target in ('encoder', 'path', 'source'):
        path = output / 'image-cpu-reference.json'
        result = json.loads(path.read_text())
        if target == 'source': result['source'] = 'b' * 40
        else:
            entry = result['rounds'][0]['captures']['encoder']
            if target == 'path': entry['file'] = '../outside.bin'
            else:
                capture = output / entry['file']
                capture.write_bytes(np.ones((1, 512), dtype='<f4').tobytes())
                entry['sha256'] = hashlib.sha256(capture.read_bytes()).hexdigest()
        path.write_text(json.dumps(result), encoding='utf-8')
    elif target == 'source_journal':
        path = output / 'image-cpu-source.json'
        result = json.loads(path.read_text())
        result['source'] = 'b' * 40
        path.write_text(json.dumps(result), encoding='utf-8')
    else:
        path = output / 'pipeline.json'
        result = json.loads(path.read_text())
        if target == 'final':
            entry = result['rounds'][0]
            capture = output / entry['capture']
            raw = -np.frombuffer(capture.read_bytes(), dtype='<f4')
            capture.write_bytes(raw.tobytes())
            entry['capture_sha256'] = hashlib.sha256(capture.read_bytes()).hexdigest()
        elif target == 'module': result['module'] = 'image-portrait'
        elif target == 'case': result['reference_case_id'] = 'square-1'
        elif target == 'phase': result['phase'] = 'failed'
        elif target == 'round_state': result['input_signatures_checked_before_and_after_each_round'] = False
        path.write_text(json.dumps(result), encoding='utf-8')
    with pytest.raises(ValueError): module.audit_image_cpu_reference(output, refs, source='a' * 40)


def test_loading_failure_preserves_phase_and_safe_error_class(pipeline, tmp_path, monkeypatch):
    module, _, _, _, _ = pipeline
    def fail(*args): raise MemoryError('private fixture details')
    monkeypatch.setattr(module, '_load_pipeline', fail)
    output = tmp_path / 'failed-load'
    with pytest.raises(ValueError): run(pipeline, output)
    result = json.loads((output / 'image-cpu-source.json').read_text())
    assert result['phase'] == 'failed_loading' and result['error_class'] == 'MemoryError'
    assert result['model_execution_complete'] is False and 'private fixture details' not in json.dumps(result)


@pytest.mark.parametrize('problem', ['source', 'existing_output'])
def test_cli_admission_rejects_invalid_work_before_model_download(pipeline, tmp_path, monkeypatch, problem):
    module, _, _, refs, _ = pipeline
    import huggingface_hub
    downloads = []
    def fetch(*args, **kwargs): downloads.append(True); raise RuntimeError('no download allowed')
    monkeypatch.setattr(huggingface_hub, 'snapshot_download', fetch)
    monkeypatch.setattr(torch, 'set_num_threads', lambda value: None)
    monkeypatch.setattr(torch, 'set_num_interop_threads', lambda value: None)
    output = tmp_path / 'output'
    if problem == 'source':
        def invalid(source): raise ValueError('source invalid')
        monkeypatch.setattr(module, '_validate_execution_source', invalid)
    else:
        output.mkdir()
        (output / 'keep.txt').write_text('preserve')
    assert module.main(['measure', '--references', str(refs), '--output', str(output),
                        '--profile', 'square', '--source', 'a' * 40]) == 1
    assert not downloads


def test_position_capture_never_truncates_actual_int64_values_into_a_false_match(pipeline, tmp_path):
    module, _, _, refs, state = pipeline
    state['position_delta'] = 2**32
    output = tmp_path / 'overflow'
    with pytest.raises(ValueError): run(pipeline, output)
    assert not (output / 'independent-image-CPU-audit.json').exists()
    with pytest.raises(ValueError): module.audit_image_cpu_reference(output, refs, source='a' * 40)


@pytest.mark.parametrize('stage', ['pixels', 'encoder'])
def test_later_inplace_operations_cannot_hide_the_original_stage_output(pipeline, tmp_path, stage):
    _, _, _, _, state = pipeline
    state['pixel_delta' if stage == 'pixels' else 'encoder_delta'] = 1
    state['mutate_' + stage] = True
    with pytest.raises(ValueError): run(pipeline, tmp_path / 'output')
