import hashlib
import importlib
import json

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import pytest
import torch

pytest.importorskip('onnxruntime', reason='Install the cpu-comparison group for actual CPU session tests')

SOURCE = 'a' * 40


def implementation():
    return importlib.import_module('embeddinggemma_tools.audio_cpu_encoder')


def cohort(tmp_path, monkeypatch, *, padded=False):
    impl = implementation()
    audited = dict(original_tolerance_passed=True, source=SOURCE,
        case_id='padded_noise' if padded else 'silence',
        profile='frames-1119-valid-' + ('839' if padded else '1119'))
    monkeypatch.setattr(impl, 'audit_distribution_report', lambda *_: audited)
    parts = []
    shapes = [(1, 1119, 128)] + [(1, 280, 1024)]*4 + [(210 if padded else 280, 512)]
    for index in range(5):
        if index in (0, 4):
            constants = [numpy_helper.from_array(np.asarray(shapes[index+1], dtype=np.int64), 'shape')]
            nodes = [helper.make_node('ReduceMean', ['input'], ['mean'], keepdims=0),
                     helper.make_node('Expand', ['mean', 'shape'], ['output'])]
        else:
            constants = []
            nodes = [helper.make_node('Neg', ['input'], ['output'])]
        model = helper.make_model(helper.make_graph(nodes, 'cpu-part',
            [helper.make_tensor_value_info('input', TensorProto.FLOAT, shapes[index])],
            [helper.make_tensor_value_info('output', TensorProto.FLOAT, shapes[index+1])], constants),
            opset_imports=[helper.make_opsetid('', 24)], ir_version=8)
        graph = tmp_path / f'part{index}.onnx'
        onnx.save(model, graph)
        parts.append(dict(index=index, graph=graph.name, graph_sha256=hashlib.sha256(graph.read_bytes()).hexdigest()))
    data = dict(cases=[dict(variant='ordered_original_midpoint_bmm', parts=parts)])
    report = tmp_path / 'chain.json'
    report.write_text(json.dumps(data), encoding='utf-8')
    return impl, audited, report, data


def save(report, data):
    report.write_text(json.dumps(data), encoding='utf-8')


@pytest.mark.parametrize('padded', [False, True])
def test_real_five_cpu_parts_carry_actual_tensor_and_remain_usable_until_closed(tmp_path, monkeypatch, padded):
    impl, audited, report, _ = cohort(tmp_path, monkeypatch, padded=padded)
    with impl.AuditedAudioCpuEncoder(report, tmp_path, source=SOURCE) as encoder:
        assert isinstance(encoder, torch.nn.Module)
        value = torch.ones((1, 1119, 128), dtype=torch.float32)
        first = encoder(value)
        second = encoder(value * 2)
        assert first.shape == (210 if padded else 280, 512)
        torch.testing.assert_close(first, -torch.ones_like(first), rtol=0, atol=0)
        torch.testing.assert_close(second, first * 2, rtol=0, atol=0)
        torch.testing.assert_close(value, torch.ones_like(value), rtol=0, atol=0)
        assert first.device.type == 'cpu' and first.dtype == torch.float32
        metadata = encoder.configuration
        assert metadata['case_id'] == audited['case_id'] and metadata['source'] == SOURCE
        assert metadata['chain_report_sha256'] == hashlib.sha256(report.read_bytes()).hexdigest()
        assert metadata['references_used_as_inference_inputs'] is False
        assert metadata['GPU_validated'] is metadata['complete_performance_accepted'] is False
        assert len(metadata['sessions']) == 5
        assert all(v['providers'] == ['CPUExecutionProvider'] and v['node_execution_order'] == 'PRIORITY_BASED'
                   and v['graph_optimizations'] == 'disabled' for v in metadata['sessions'])
    with pytest.raises(ValueError):
        encoder(value)
    encoder.close()


@pytest.mark.parametrize('mutation', ['source', 'failed', 'profile', 'case'])
def test_original_audit_admission_fails_before_any_session(tmp_path, monkeypatch, mutation):
    impl, audited, report, _ = cohort(tmp_path, monkeypatch)
    if mutation == 'source': audited['source'] = 'b' * 40
    if mutation == 'failed': audited['original_tolerance_passed'] = False
    if mutation == 'profile': audited['profile'] = 'frames-1119-valid-839'
    if mutation == 'case': audited['case_id'] = 'unknown'
    monkeypatch.setattr(impl, 'CpuOnnxSession', lambda *_a, **_k: pytest.fail('session opened before admission'))
    with pytest.raises(ValueError):
        impl.AuditedAudioCpuEncoder(report, tmp_path, source=SOURCE)


@pytest.mark.parametrize('source', [None, '', 'main', 'A'*40, 'a'*39])
def test_immutable_source_is_checked_before_audit(tmp_path, monkeypatch, source):
    impl = implementation()
    monkeypatch.setattr(impl, 'audit_distribution_report', lambda *_: pytest.fail('audit started before source check'))
    with pytest.raises(ValueError):
        impl.AuditedAudioCpuEncoder(tmp_path/'absent.json', tmp_path, source=source)


@pytest.mark.parametrize('mutation', ['four', 'six', 'duplicate', 'order', 'escape', 'changed_hash', 'wrong_shape'])
def test_changed_cohort_cannot_be_used_as_reviewed_encoder(tmp_path, monkeypatch, mutation):
    impl, _, report, data = cohort(tmp_path, monkeypatch)
    row = data['cases'][0]
    if mutation == 'four': row['parts'].pop()
    if mutation == 'six': row['parts'].append(dict(row['parts'][-1]))
    if mutation == 'duplicate': data['cases'].append(dict(row))
    if mutation == 'order': row['parts'][1]['index'] = 2
    if mutation == 'escape': row['parts'][0]['graph'] = '../part0.onnx'
    if mutation == 'changed_hash': row['parts'][0]['graph_sha256'] = '0'*64
    if mutation == 'wrong_shape':
        path = tmp_path/'part4.onnx'
        model = onnx.load(path)
        model.graph.output[0].type.tensor_type.shape.dim[0].dim_value = 210
        model.graph.initializer[0].CopyFrom(numpy_helper.from_array(np.asarray([210,512], dtype=np.int64), 'shape'))
        onnx.save(model, path)
        row['parts'][-1]['graph_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    save(report, data)
    with pytest.raises(ValueError):
        impl.AuditedAudioCpuEncoder(report, tmp_path, source=SOURCE)


@pytest.mark.parametrize('kind', ['dtype', 'shape', 'nan', 'numpy'])
def test_invalid_actual_preprocessor_tensor_is_rejected(tmp_path, monkeypatch, kind):
    impl, _, report, _ = cohort(tmp_path, monkeypatch)
    value = torch.ones((1,1119,128), dtype=torch.float32)
    if kind == 'dtype': value = value.double()
    if kind == 'shape': value = value[:, :-1]
    if kind == 'nan': value[0,0,0] = float('nan')
    if kind == 'numpy': value = value.numpy()
    with impl.AuditedAudioCpuEncoder(report, tmp_path, source=SOURCE) as encoder:
        with pytest.raises(ValueError):
            encoder(value)


def test_partial_load_failure_releases_already_owned_sessions(tmp_path, monkeypatch):
    impl, _, report, _ = cohort(tmp_path, monkeypatch)
    actual_factory = impl.CpuOnnxSession
    opened = []
    def factory(*args, **kwargs):
        if len(opened) == 2:
            raise ValueError('synthetic loading failure')
        session = actual_factory(*args, **kwargs)
        opened.append(session)
        return session
    monkeypatch.setattr(impl, 'CpuOnnxSession', factory)
    with pytest.raises(ValueError):
        impl.AuditedAudioCpuEncoder(report, tmp_path, source=SOURCE)
    assert len(opened) == 2 and all(v.session is None for v in opened)


def test_source_report_cannot_change_during_session_loading(tmp_path, monkeypatch):
    impl, _, report, _ = cohort(tmp_path, monkeypatch)
    actual_factory = impl.CpuOnnxSession
    opened = []
    def factory(*args, **kwargs):
        session = actual_factory(*args, **kwargs)
        opened.append(session)
        report.write_text('{}', encoding='utf-8')
        return session
    monkeypatch.setattr(impl, 'CpuOnnxSession', factory)
    with pytest.raises(ValueError):
        impl.AuditedAudioCpuEncoder(report, tmp_path, source=SOURCE)
    assert opened and all(v.session is None for v in opened)


def test_boolean_part_index_cannot_impersonate_an_original_integer(tmp_path, monkeypatch):
    impl, _, report, data = cohort(tmp_path, monkeypatch)
    data['cases'][0]['parts'][1]['index'] = True
    save(report, data)
    with pytest.raises(ValueError):
        impl.AuditedAudioCpuEncoder(report, tmp_path, source=SOURCE)


def test_adapter_connects_actual_preprocessing_and_backbone_through_reference_protocol(tmp_path, monkeypatch):
    from embeddinggemma_tools.reference_cpu_performance import (
        AUDIO_REFERENCE, DESCRIPTOR, CpuReferenceCase, measure_reference_pipeline)
    impl, _, report, _ = cohort(tmp_path, monkeypatch)
    class Preprocessor(torch.nn.Module):
        def forward(self, wave):
            return wave.mean().expand(1,1119,128).contiguous()
    class Backbone(torch.nn.Module):
        def forward(self, features):
            value = features.new_zeros(768)
            value[0] = features.mean()
            return value / value.norm()
    preprocessor, backbone = Preprocessor(), Backbone()
    wave = torch.ones(1,179200)
    expected = torch.zeros(768)
    expected[0] = -1
    captures = []
    def observe(stage, actual):
        if stage == 'encoder':
            captures.append(actual.detach().clone())
    with impl.AuditedAudioCpuEncoder(report, tmp_path, source=SOURCE) as encoder:
        case = CpuReferenceCase(module='audio-1119', precision='fp32', source=SOURCE,
            descriptor_sha256=DESCRIPTOR, reference_sha256=AUDIO_REFERENCE,
            input_sha256='b'*64, reference_case_id='silence', expected=expected,
            input_signature=lambda:hashlib.sha256(wave.numpy().tobytes()).hexdigest(),
            preprocess=lambda:preprocessor(wave), encoder=encoder, backbone=backbone,
            active_modules=(preprocessor,encoder,backbone), after_stage=observe)
        measured = measure_reference_pipeline(case, tmp_path/'pipeline.json')
    assert measured['success'] and len(measured['rounds']) == len(captures) == 4
    assert measured['complete_performance_accepted'] is False
    assert all(torch.equal(v,-torch.ones((280,512))) for v in captures)
    assert all(v['cosine'] == 1 and v['stage_observer_wall_seconds'] >= 0 for v in measured['rounds'])
