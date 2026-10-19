import hashlib
import importlib
import json

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import pytest

pytest.importorskip('onnxruntime', reason='Install the cpu-comparison group to execute real CPU backend tests')


def implementation():
    return importlib.import_module('embeddinggemma_tools.onnx_cpu_comparison')


def graph(tmp_path, name='model', *, operation='Neg', factor=1e6, shape=(2, 3)):
    nodes = [helper.make_node(operation, ['input'], ['output'])]
    constants = []
    if operation == 'Mul':
        constants = [numpy_helper.from_array(np.asarray(factor, dtype=np.float32), 'factor')]
        nodes = [helper.make_node('Mul', ['input', 'factor'], ['output'])]
    if operation == 'rounding':
        constants = [numpy_helper.from_array(np.asarray(2**24, dtype=np.float32), 'large')]
        nodes = [helper.make_node('Add', ['input', 'large'], ['sum']),
                 helper.make_node('Sub', ['sum', 'large'], ['output'])]
    model = helper.make_model(helper.make_graph(nodes, name,
        [helper.make_tensor_value_info('input', TensorProto.FLOAT, shape)],
        [helper.make_tensor_value_info('output', TensorProto.FLOAT, shape)], constants),
        opset_imports=[helper.make_opsetid('', 24)], ir_version=8)
    path = tmp_path / (name + '.onnx')
    onnx.save(model, path)
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def session(tmp_path, **kwargs):
    path, sha = graph(tmp_path, **kwargs)
    return implementation().CpuOnnxSession(path, sha)


def test_explicit_cpu_session_preserves_rounding_and_input_bytes(tmp_path):
    runner = session(tmp_path, operation='rounding')
    value = np.ones((2, 3), dtype=np.float32)
    before = value.tobytes()
    np.testing.assert_array_equal(runner.run(value), np.zeros_like(value))
    assert value.tobytes() == before
    assert runner.configuration['providers'] == ['CPUExecutionProvider']
    assert runner.configuration['graph_optimizations'] == 'disabled'


def test_modified_graph_hash_is_rejected_before_loading(tmp_path):
    path, _ = graph(tmp_path)
    with pytest.raises(ValueError, match='hash'):
        implementation().CpuOnnxSession(path, '0' * 64)


@pytest.mark.parametrize('shape', [(None, 3), (2, 'sequence')])
def test_dynamic_input_shapes_are_rejected(tmp_path, shape):
    path, sha = graph(tmp_path, shape=shape)
    with pytest.raises(ValueError, match='static|shape'):
        implementation().CpuOnnxSession(path, sha)


@pytest.mark.parametrize('value', [
    np.zeros((2, 3), dtype=np.float64),
    np.zeros((1, 3), dtype=np.float32),
    np.full((2, 3), np.nan, dtype=np.float32),
    np.full((2, 3), np.inf, dtype=np.float32),
])
def test_invalid_runtime_input_is_rejected(tmp_path, value):
    runner = session(tmp_path)
    with pytest.raises(ValueError, match='FP32|shape|finite'):
        runner.run(value)


def test_actual_carry_is_used_instead_of_reference_intermediate(tmp_path):
    runners = [session(tmp_path, name='first'), session(tmp_path, name='second', operation='Mul')]
    initial = np.zeros((2, 3), dtype=np.float32)
    references = [np.full_like(initial, 5e-6), np.zeros_like(initial)]
    output = tmp_path / 'replay.json'
    result = implementation().replay_chain(runners, initial, references, output)
    assert result['success'] and len(result['rounds']) == 4
    assert all(r['tolerance_failures'] == 0 for r in result['rounds'])
    assert result['GPU_validated'] is False
    assert result['complete_performance_accepted'] is False
    assert result['CPU_reference_comparison_qualified'] is False
    assert json.loads(output.read_text()) == result
    for row in result['rounds']:
        assert len(row['parts']) == 2
        assert all(p['wall_seconds'] >= 0 and p['process_cpu_seconds'] >= 0 for p in row['parts'])
        for part in row['parts']:
            raw = (output.parent / part['capture']).read_bytes()
            assert hashlib.sha256(raw).hexdigest() == part['sha256']


def test_numerical_failure_preserves_report_and_actual_capture(tmp_path):
    runner = session(tmp_path)
    initial = np.zeros((2, 3), dtype=np.float32)
    output = tmp_path / 'failed.json'
    with pytest.raises(ValueError, match='original|tolerance'):
        implementation().replay_chain([runner], initial, [np.ones_like(initial)], output)
    report = json.loads(output.read_text())
    assert not report['success'] and report['phase'] == 'numerical_failure'
    assert report['rounds'][0]['tolerance_failures'] == 6
    capture = output.parent / report['rounds'][0]['parts'][0]['capture']
    np.testing.assert_array_equal(np.fromfile(capture, dtype='<f4'), np.zeros(6, dtype=np.float32))


def test_existing_report_or_capture_is_never_overwritten(tmp_path):
    runner = session(tmp_path)
    initial = np.zeros((2, 3), dtype=np.float32)
    output = tmp_path / 'existing.json'
    output.write_text('preserved')
    with pytest.raises(FileExistsError):
        implementation().replay_chain([runner], initial, [initial], output)
    assert output.read_text() == 'preserved'
    output.unlink()
    capture = tmp_path / 'existing-first-part0.bin'
    capture.write_bytes(b'preserved')
    with pytest.raises(FileExistsError):
        implementation().replay_chain([runner], initial, [initial], output)
    assert capture.read_bytes() == b'preserved' and not output.exists()


def test_reference_shape_is_checked_before_any_report_is_written(tmp_path):
    runner = session(tmp_path)
    output = tmp_path / 'bad-reference.json'
    with pytest.raises(ValueError, match='reference|shape'):
        implementation().replay_chain([runner], np.zeros((2, 3), dtype=np.float32),
            [np.zeros((1, 3), dtype=np.float32)], output)
    assert not output.exists()


def test_external_weight_data_is_rejected_before_session_creation(tmp_path):
    path, _ = graph(tmp_path, operation='Mul')
    model = onnx.load(path)
    onnx.save_model(model, path, save_as_external_data=True, all_tensors_to_one_file=True,
                    location='weights.data', size_threshold=0)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match='external|inline'):
        implementation().CpuOnnxSession(path, sha)


@pytest.fixture
def audited_chain(tmp_path, monkeypatch):
    from embeddinggemma_tools import native_regression_distribution
    parts = []
    initial = np.zeros((2, 3), dtype='<f4')
    for index in range(5):
        path, sha = graph(tmp_path, name=f'part{index}')
        capture = tmp_path / f'reference{index}.bin'
        initial.tofile(capture)
        parts.append(dict(index=index, graph=path.name, graph_sha256=sha,
            output_capture=capture.name, output_sha256=hashlib.sha256(capture.read_bytes()).hexdigest()))
    initial.tofile(tmp_path / 'raw-input-fp32.bin')
    initial.tofile(tmp_path / 'reference-output-fp32.bin')
    report = dict(input_sha256=hashlib.sha256(initial.tobytes()).hexdigest(),
        reference_sha256=hashlib.sha256(initial.tobytes()).hexdigest(),
        cases=[dict(variant='native', parts=[]),
               dict(variant='ordered_original_midpoint_bmm', parts=parts)])
    source = tmp_path / 'frozen-audio-chained-onnx.json'
    source.write_text(json.dumps(report))
    audit = dict(original_tolerance_passed=True, source='a' * 40, case_id='silence',
                 profile='frames-1119-valid-1119', values_compared=6)
    calls = []
    def checked(path, original):
        calls.append((Path(path), Path(original)))
        return audit
    from pathlib import Path
    monkeypatch.setattr(native_regression_distribution, 'audit_distribution_report', checked)
    return source, audit, calls


def test_distribution_entry_requires_audited_originals_and_binds_source(tmp_path, audited_chain):
    source, _, calls = audited_chain
    output = tmp_path / 'cpu.json'
    result = implementation().replay_distribution(source, tmp_path / 'original', output, source='a' * 40)
    assert len(calls) == 1
    assert result['source'] == 'a' * 40 and result['case_id'] == 'silence'
    assert result['original_chain_report_sha256'] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert result['original_reference_audit_passed'] is True
    assert result['encoder_replay_qualified'] is True
    assert result['CPU_reference_comparison_qualified'] is False
    assert len(result['sessions']) == 5 and len(result['rounds']) == 4


@pytest.mark.parametrize('change', ['failed-original-gate', 'different-source', 'invalid-source'])
def test_distribution_entry_rejects_failed_or_different_original_context(tmp_path, audited_chain, change):
    source, audit, _ = audited_chain
    expected = 'a' * 40
    if change == 'failed-original-gate':
        audit['original_tolerance_passed'] = False
    elif change == 'different-source':
        audit['source'] = 'b' * 40
    else:
        expected = 'not-a-source'
    output = tmp_path / 'rejected.json'
    with pytest.raises(ValueError, match='original|source|Immutable'):
        implementation().replay_distribution(source, tmp_path / 'original', output, source=expected)
    assert not output.exists()


@pytest.mark.parametrize('change', ['escape', 'changed-reference', 'missing-part'])
def test_distribution_entry_rejects_changed_chain_bytes(tmp_path, audited_chain, change):
    source, _, _ = audited_chain
    report = json.loads(source.read_text())
    if change == 'escape':
        report['cases'][1]['parts'][0]['graph'] = '../outside.onnx'
        source.write_text(json.dumps(report))
    elif change == 'changed-reference':
        (tmp_path / 'reference0.bin').write_bytes(b'changed')
    else:
        report['cases'][1]['parts'].pop()
        source.write_text(json.dumps(report))
    with pytest.raises(ValueError, match='relative|capture|five|hash'):
        implementation().replay_distribution(source, tmp_path / 'original', tmp_path / 'rejected.json', source='a' * 40)


def test_distribution_entry_never_decorates_an_existing_report(tmp_path, audited_chain):
    source, _, _ = audited_chain
    output = tmp_path / 'preserved.json'
    original = b'{"owned_by_this_run": false}\n'
    output.write_bytes(original)
    with pytest.raises(FileExistsError):
        implementation().replay_distribution(source, tmp_path / 'original', output, source='a' * 40)
    assert output.read_bytes() == original


def test_graph_mutation_after_loading_is_a_recorded_failure(tmp_path, audited_chain, monkeypatch):
    source, _, _ = audited_chain
    module = implementation()
    execute = module.CpuOnnxSession.run
    changed = False
    def mutate_after_actual_execution(session, value):
        nonlocal changed
        actual = execute(session, value)
        if not changed:
            session.graph.write_bytes(session.graph.read_bytes() + b'changed')
            changed = True
        return actual
    monkeypatch.setattr(module.CpuOnnxSession, 'run', mutate_after_actual_execution)
    output = tmp_path / 'mutated.json'
    with pytest.raises(ValueError, match='hash'):
        module.replay_distribution(source, tmp_path / 'original', output, source='a' * 40)
    report = json.loads(output.read_text())
    assert not report['success'] and not report['encoder_replay_qualified']
    assert report['phase'] == 'failed_cpu_replay'


def test_provider_readback_must_be_cpu_only(tmp_path, monkeypatch):
    import onnxruntime as ort
    monkeypatch.setattr(ort.InferenceSession, 'get_providers', lambda _: ['CUDAExecutionProvider'])
    path, sha = graph(tmp_path)
    with pytest.raises(ValueError, match='CPU execution provider'):
        implementation().CpuOnnxSession(path, sha)
