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


@pytest.mark.parametrize('field', ['enable_cpu_mem_arena', 'enable_mem_pattern'])
def test_actual_cpu_session_disables_speculative_memory_retention(tmp_path, field):
    runner = session(tmp_path)
    assert getattr(runner.session.get_session_options(), field) is False


def test_actual_cpu_session_disables_extra_constant_weight_prepacking(tmp_path):
    runner = session(tmp_path)
    assert runner.session.get_session_options().get_session_config_entry('session.disable_prepacking') == '1'


def test_cpu_report_records_actual_memory_configuration(tmp_path):
    runner = session(tmp_path)
    assert runner.configuration['cpu_memory_arena'] == 'disabled'
    assert runner.configuration['memory_pattern'] == 'disabled'
    assert runner.configuration['constant_prepacking'] == 'disabled'


@pytest.mark.parametrize('optimization', ['disabled', 'basic'])
def test_dense_cpu_carries_keep_original_elementwise_gate_in_every_round(tmp_path, optimization):
    rng = np.random.default_rng(729)
    initial = rng.uniform(-1, 1, (3, 32)).astype(np.float32)
    actual = initial
    runners, references = [], []
    for index, columns in enumerate((17, 7)):
        weights = rng.uniform(-1, 1, (actual.shape[1], columns)).astype(np.float32)
        model = helper.make_model(helper.make_graph(
            [helper.make_node('MatMul', ['input', 'weights'], ['output'])], f'dense{index}',
            [helper.make_tensor_value_info('input', TensorProto.FLOAT, actual.shape)],
            [helper.make_tensor_value_info('output', TensorProto.FLOAT, (3, columns))],
            [numpy_helper.from_array(weights, 'weights')]),
            opset_imports=[helper.make_opsetid('', 24)], ir_version=8)
        path = tmp_path / f'dense{index}.onnx'
        onnx.save(model, path)
        runners.append(implementation().CpuOnnxSession(path, hashlib.sha256(path.read_bytes()).hexdigest(),
            graph_optimizations=optimization))
        actual = actual @ weights
        references.append(actual)
    result = implementation().replay_chain(runners, initial, references, tmp_path / 'dense.json')
    assert result['success'] and len(result['rounds']) == 4
    assert result['original_atol'] == 1e-5 and result['original_rtol'] == 1e-4
    assert all(row['tolerance_failures'] == 0 and len(row['parts']) == 2 for row in result['rounds'])
    assert result['references_used_as_inference_inputs'] is False


def test_basic_cpu_optimization_is_explicit_and_keeps_fp32_rounding(tmp_path):
    import onnxruntime as ort
    path, sha = graph(tmp_path, operation='rounding')
    runner = implementation().CpuOnnxSession(path, sha, graph_optimizations='basic')
    assert runner.session.get_session_options().graph_optimization_level == ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    assert runner.configuration['graph_optimizations'] == 'basic'
    initial = np.ones((2, 3), dtype=np.float32)
    result = implementation().replay_chain([runner], initial, [np.zeros_like(initial)], tmp_path / 'basic.json')
    assert result['success'] and len(result['rounds']) == 4
    assert all(row['tolerance_failures'] == 0 for row in result['rounds'])


@pytest.mark.parametrize('optimization', ['disabled', 'basic'])
def test_actual_unreviewed_optimization_readback_is_rejected(tmp_path, monkeypatch, optimization):
    import onnxruntime as ort
    read = ort.InferenceSession.get_session_options
    def changed(self):
        actual = read(self)
        actual.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        return actual
    monkeypatch.setattr(ort.InferenceSession, 'get_session_options', changed)
    path, sha = graph(tmp_path)
    with pytest.raises(ValueError, match='optimization'):
        implementation().CpuOnnxSession(path, sha, graph_optimizations=optimization)


@pytest.mark.parametrize('optimization', ['extended', 'all', 'unknown'])
def test_unreviewed_cpu_optimization_levels_are_rejected(tmp_path, optimization):
    path, sha = graph(tmp_path)
    with pytest.raises(ValueError, match='optimization'):
        implementation().CpuOnnxSession(path, sha, graph_optimizations=optimization)


def test_distribution_basic_choice_reaches_all_five_actual_sessions(tmp_path, audited_chain):
    source, _, _ = audited_chain
    result = implementation().replay_distribution(source, tmp_path / 'original', tmp_path / 'basic-distribution.json',
        source='a' * 40, graph_optimizations='basic')
    assert result['success'] and len(result['sessions']) == 5 and len(result['rounds']) == 4
    assert all(row['graph_optimizations'] == 'basic' for row in result['sessions'])
    assert result['CPU_reference_comparison_qualified'] is False


def test_cli_forwards_the_explicit_basic_mode(tmp_path, monkeypatch):
    module = implementation()
    seen = {}
    def execute(chain, original, output, *, source, graph_optimizations, execution_schedule='chain-major', node_order='default'):
        assert execution_schedule == 'chain-major'
        assert node_order == 'default'
        seen.update(source=source, optimization=graph_optimizations)
        return dict(success=True, case_id='silence', encoder_replay_qualified=True)
    monkeypatch.setattr(module, 'replay_distribution', execute)
    assert module.main(['--chain-report', str(tmp_path / 'chain.json'), '--original', str(tmp_path),
        '--output', str(tmp_path / 'out.json'), '--source', 'a' * 40, '--graph-optimizations', 'basic']) == 0
    assert seen == dict(source='a' * 40, optimization='basic')


def test_cli_forwards_the_partition_major_numerical_schedule(tmp_path, monkeypatch):
    module = implementation()
    seen = {}
    def execute(chain, original, output, *, source, graph_optimizations, execution_schedule, node_order='default'):
        assert node_order == 'default'
        seen.update(optimization=graph_optimizations, schedule=execution_schedule)
        return dict(success=True, case_id='silence', encoder_replay_qualified=True)
    monkeypatch.setattr(module, 'replay_distribution', execute)
    assert module.main(['--chain-report', str(tmp_path / 'chain.json'), '--original', str(tmp_path),
        '--output', str(tmp_path / 'out.json'), '--source', 'a' * 40,
        '--execution-schedule', 'partition-major']) == 0
    assert seen == dict(optimization='disabled', schedule='partition-major')


def test_cli_forwards_priority_node_order_with_full_chain_schedule(tmp_path, monkeypatch):
    module = implementation()
    seen = {}
    def execute(chain, original, output, *, source, graph_optimizations, execution_schedule, node_order):
        seen.update(optimization=graph_optimizations, schedule=execution_schedule, order=node_order)
        return dict(success=True, case_id='silence', encoder_replay_qualified=True)
    monkeypatch.setattr(module, 'replay_distribution', execute)
    assert module.main(['--chain-report', str(tmp_path / 'chain.json'), '--original', str(tmp_path),
        '--output', str(tmp_path / 'out.json'), '--source', 'a' * 40,
        '--node-order', 'priority', '--execution-schedule', 'chain-major']) == 0
    assert seen == dict(optimization='disabled', schedule='chain-major', order='priority')


def test_priority_node_order_is_actual_and_preserves_fp32_dependencies(tmp_path):
    import onnxruntime as ort
    path, sha = graph(tmp_path, operation='rounding')
    runner = implementation().CpuOnnxSession(path, sha, node_order='priority')
    assert runner.session.get_session_options().execution_order == ort.ExecutionOrder.PRIORITY_BASED
    assert runner.configuration['node_execution_order'] == 'PRIORITY_BASED'
    actual = runner.run(np.ones((2, 3), dtype=np.float32))
    np.testing.assert_array_equal(actual, np.zeros((2, 3), dtype=np.float32))


@pytest.mark.parametrize('node_order', ['unknown', 'memory-efficient'])
def test_unreviewed_node_orders_are_rejected(tmp_path, node_order):
    path, sha = graph(tmp_path)
    with pytest.raises(ValueError, match='node order'):
        implementation().CpuOnnxSession(path, sha, node_order=node_order)


@pytest.mark.parametrize('node_order', ['default', 'priority'])
def test_actual_node_order_readback_mismatch_is_rejected(tmp_path, monkeypatch, node_order):
    import onnxruntime as ort
    read = ort.InferenceSession.get_session_options
    def mismatch(self):
        options = read(self)
        options.execution_order = ort.ExecutionOrder.DEFAULT if node_order == 'priority' else ort.ExecutionOrder.PRIORITY_BASED
        return options
    monkeypatch.setattr(ort.InferenceSession, 'get_session_options', mismatch)
    path, sha = graph(tmp_path)
    with pytest.raises(ValueError, match='node order'):
        implementation().CpuOnnxSession(path, sha, node_order=node_order)


def test_priority_distribution_runs_twenty_calls_with_all_five_sessions_owned(tmp_path, audited_chain, monkeypatch):
    import weakref
    source, _, _ = audited_chain
    module = implementation()
    create, execute = module.CpuOnnxSession.__init__, module.CpuOnnxSession.run
    owners, calls = [], []
    def count_create(self, *args, **kwargs):
        create(self, *args, **kwargs)
        owners.append(weakref.ref(self))
    def observed_run(self, value):
        assert len(owners) == 5 and all(owner() is not None and owner().session is not None for owner in owners)
        calls.append(self.graph.name)
        return execute(self, value)
    monkeypatch.setattr(module.CpuOnnxSession, '__init__', count_create)
    monkeypatch.setattr(module.CpuOnnxSession, 'run', observed_run)
    result = module.replay_distribution(source, tmp_path / 'original', tmp_path / 'priority-chain.json',
        source='a' * 40, node_order='priority', execution_schedule='chain-major')
    assert result['success'] and result['execution_schedule'] == 'chain-major'
    assert result['maximum_resident_sessions'] == 5
    assert result['end_to_end_warm_timing_qualified'] is False
    assert calls == [f'part{i}.onnx' for i in range(5)] * 4
    assert all(row['node_execution_order'] == 'PRIORITY_BASED' for row in result['sessions'])


def test_priority_profile_finishes_shared_tensor_consumers_before_next_branch(tmp_path, monkeypatch):
    import onnxruntime as ort
    nodes, constants, sums, maxima = [], [], [], []
    for i in range(4):
        nodes.extend((helper.make_node('Mul', ['input', f'factor{i}'], [f'large{i}'], name=f'large{i}'),
            helper.make_node('ReduceSum', [f'large{i}'], [f'sum{i}'], name=f'sum{i}', keepdims=1),
            helper.make_node('ReduceMax', [f'large{i}'], [f'max{i}'], name=f'max{i}', keepdims=1)))
        constants.append(numpy_helper.from_array(np.asarray(i + 1, dtype=np.float32), f'factor{i}'))
        sums.append(f'sum{i}'); maxima.append(f'max{i}')
    nodes.extend((helper.make_node('Concat', sums, ['sums'], name='sums', axis=0),
        helper.make_node('Concat', maxima, ['maxima'], name='maxima', axis=0),
        helper.make_node('Concat', ['sums', 'maxima'], ['output'], name='output', axis=0)))
    model = helper.make_model(helper.make_graph(nodes, 'shared-branch',
        [helper.make_tensor_value_info('input', TensorProto.FLOAT, (2, 3))],
        [helper.make_tensor_value_info('output', TensorProto.FLOAT, (8, 1))], constants),
        opset_imports=[helper.make_opsetid('', 24)], ir_version=8)
    path = tmp_path / 'shared.onnx'; onnx.save(model, path)
    create = ort.InferenceSession
    def profiled(*args, sess_options, **kwargs):
        sess_options.enable_profiling = True
        sess_options.profile_file_prefix = str(tmp_path / 'node-profile')
        return create(*args, sess_options=sess_options, **kwargs)
    monkeypatch.setattr(ort, 'InferenceSession', profiled)
    runner = implementation().CpuOnnxSession(path, hashlib.sha256(path.read_bytes()).hexdigest(), node_order='priority')
    try:
        value = np.full((2, 3), np.float32(2**-20), dtype=np.float32)
        actual = runner.run(value)
        expected = np.concatenate((np.arange(1, 5, dtype=np.float32) * np.float32(6 * 2**-20),
            np.arange(1, 5, dtype=np.float32) * np.float32(2**-20))).reshape(8, 1)
        np.testing.assert_array_equal(actual, expected)
        from pathlib import Path
        events = json.loads(Path(runner.session.end_profiling()).read_text())
        order = [e['name'].removesuffix('_kernel_time') for e in events
                 if e.get('cat') == 'Node' and e.get('name', '').endswith('_kernel_time')]
        assert len(order) == 15
        for i in range(3):
            assert order.index(f'max{i}') < order.index(f'large{i+1}')
    finally:
        runner.close()


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


def test_cpu_session_close_releases_native_owner_and_rejects_further_calls(tmp_path):
    runner = session(tmp_path)
    runner.close()
    runner.close()
    assert runner.session is None
    with pytest.raises(ValueError, match='closed'):
        runner.run(np.zeros((2, 3), dtype=np.float32))


def test_partition_major_keeps_only_one_session_and_all_twenty_actual_captures(tmp_path, audited_chain, monkeypatch):
    source, _, _ = audited_chain
    module = implementation()
    create, close = module.CpuOnnxSession.__init__, module.CpuOnnxSession.close
    live = [0]
    order = []
    def owned_create(self, *args, **kwargs):
        assert live[0] == 0
        create(self, *args, **kwargs)
        live[0] += 1
        order.append('load')
    def owned_close(self):
        close(self)
        live[0] -= 1
        order.append('close')
    monkeypatch.setattr(module.CpuOnnxSession, '__init__', owned_create)
    monkeypatch.setattr(module.CpuOnnxSession, 'close', owned_close)
    output = tmp_path / 'partitioned.json'
    result = module.replay_distribution(source, tmp_path / 'original', output, source='a' * 40,
        execution_schedule='partition-major')
    assert live[0] == 0 and order == ['load', 'close'] * 5
    assert result['success'] and result['encoder_replay_qualified']
    assert result['execution_schedule'] == 'partition-major'
    assert result['maximum_resident_sessions'] == 1
    assert result['end_to_end_warm_timing_qualified'] is False
    assert result['complete_performance_accepted'] is False
    assert result['references_used_as_inference_inputs'] is False
    assert len(result['sessions']) == 5 and len(result['rounds']) == 4
    assert sum(len(r['parts']) for r in result['rounds']) == 20
    assert all(r['tolerance_failures'] == 0 for r in result['rounds'])
    for row in result['rounds']:
        for part in row['parts']:
            path = output.parent / part['capture']
            assert hashlib.sha256(path.read_bytes()).hexdigest() == part['sha256']
    journal = json.loads(output.with_name('partitioned-progress.json').read_text())
    assert journal['success'] and journal['terminal']
    assert len(journal['events']) == 82
    for index in range(5):
        events = [e for e in journal['events'] if e['part'] == index]
        assert [e['phase'] for e in events] == [
            'load_part_begin', 'onnx_parse_begin', 'onnx_parse_end',
            'session_create_begin', 'session_create_end', 'load_part_end',
            *['inference_part_begin', 'inference_part_end'] * 4,
            'release_part_begin', 'release_part_end']
    assert [e['part'] for e in journal['events'] if e['phase'] == 'release_part_end'] == list(range(5))


def test_partition_major_uses_actual_carries_not_the_accepted_reference(tmp_path, audited_chain):
    source, _, _ = audited_chain
    report = json.loads(source.read_text())
    accepted_first = np.full((2, 3), 5e-6, dtype='<f4')
    accepted_first.tofile(tmp_path / 'reference0.bin')
    report['cases'][1]['parts'][0]['output_sha256'] = hashlib.sha256(accepted_first.tobytes()).hexdigest()
    path, sha = graph(tmp_path, name='amplification', operation='Mul')
    report['cases'][1]['parts'][1].update(graph=path.name, graph_sha256=sha)
    source.write_text(json.dumps(report))
    result = implementation().replay_distribution(source, tmp_path / 'original', tmp_path / 'actual.json',
        source='a' * 40, execution_schedule='partition-major')
    assert result['success'] and all(r['tolerance_failures'] == 0 for r in result['rounds'])
    for row in result['rounds']:
        actual = np.fromfile(tmp_path / row['parts'][1]['capture'], dtype='<f4')
        np.testing.assert_array_equal(actual, np.zeros(6, dtype=np.float32))


def test_partition_major_preserves_each_rounds_distinct_actual_carry(tmp_path, audited_chain, monkeypatch):
    source, _, _ = audited_chain
    module = implementation()
    report = json.loads(source.read_text())
    path, sha = graph(tmp_path, name='multiply-two', operation='Mul', factor=2)
    report['cases'][1]['parts'][1].update(graph=path.name, graph_sha256=sha)
    source.write_text(json.dumps(report))
    execute = module.CpuOnnxSession.run
    first_calls = [0]
    def distinguish(self, value):
        actual = execute(self, value)
        if self.graph.name == 'part0.onnx':
            first_calls[0] += 1
            actual = actual + np.float32(first_calls[0]) * np.float32(1e-6)
        return actual
    monkeypatch.setattr(module.CpuOnnxSession, 'run', distinguish)
    result = module.replay_distribution(source, tmp_path / 'original', tmp_path / 'streams.json',
        source='a' * 40, execution_schedule='partition-major')
    assert result['success'] and first_calls[0] == 4
    for index, row in enumerate(result['rounds']):
        expected = np.float32(index + 1) * np.float32(1e-6) * np.float32(2)
        for part_index, sign in ((1, 1), (4, -1)):
            actual = np.fromfile(tmp_path / row['parts'][part_index]['capture'], dtype='<f4')
            np.testing.assert_array_equal(actual, np.full(6, expected * sign, dtype=np.float32))


@pytest.mark.parametrize('changed', ['initial-during-load', 'final-during-inference'])
def test_partition_major_rechecks_deferred_input_and_final_reference_hash(tmp_path, audited_chain, monkeypatch, changed):
    source, _, _ = audited_chain
    module = implementation()
    create, execute = module.CpuOnnxSession.__init__, module.CpuOnnxSession.run
    def replace(name):
        np.full((2, 3), 5e-6, dtype='<f4').tofile(tmp_path / name)
    def mutated_create(self, *args, **kwargs):
        create(self, *args, **kwargs)
        if changed == 'initial-during-load' and self.graph.name == 'part0.onnx':
            replace('raw-input-fp32.bin')
    def mutated_run(self, value):
        actual = execute(self, value)
        if changed == 'final-during-inference' and self.graph.name == 'part0.onnx':
            replace('reference-output-fp32.bin')
        return actual
    monkeypatch.setattr(module.CpuOnnxSession, '__init__', mutated_create)
    monkeypatch.setattr(module.CpuOnnxSession, 'run', mutated_run)
    output = tmp_path / 'deferred.json'
    with pytest.raises(ValueError, match='hash'):
        module.replay_distribution(source, tmp_path / 'original', output, source='a' * 40,
            execution_schedule='partition-major')
    report = json.loads(output.read_text())
    assert not report['success'] and not report['encoder_replay_qualified']


def test_partition_major_excludes_progress_io_from_inference_timing(tmp_path, audited_chain, monkeypatch):
    source, _, _ = audited_chain
    module = implementation()
    execute, record = module.CpuOnnxSession.run, module.ProgressJournal.record
    clock = [0.0]
    def timed_call(self, value):
        actual = execute(self, value)
        clock[0] += 1
        return actual
    def costly_record(self, *args, **kwargs):
        clock[0] += 1000
        return record(self, *args, **kwargs)
    monkeypatch.setattr(module.CpuOnnxSession, 'run', timed_call)
    monkeypatch.setattr(module.ProgressJournal, 'record', costly_record)
    monkeypatch.setattr(module.time, 'perf_counter', lambda: clock[0])
    result = module.replay_distribution(source, tmp_path / 'original', tmp_path / 'timing.json',
        source='a' * 40, execution_schedule='partition-major')
    assert all(part['wall_seconds'] == 1 for row in result['rounds'] for part in row['parts'])
    assert not result['end_to_end_warm_timing_qualified']


@pytest.mark.parametrize('failure', ['numerical', 'native'])
def test_partition_major_failure_releases_session_and_preserves_partial_evidence(tmp_path, audited_chain, monkeypatch, failure):
    source, _, _ = audited_chain
    module = implementation()
    close, execute = module.CpuOnnxSession.close, module.CpuOnnxSession.run
    released = []
    def counted_close(self):
        close(self)
        released.append(self.session)
    monkeypatch.setattr(module.CpuOnnxSession, 'close', counted_close)
    if failure == 'numerical':
        report = json.loads(source.read_text())
        wrong = np.ones((2, 3), dtype='<f4')
        wrong.tofile(tmp_path / 'reference0.bin')
        report['cases'][1]['parts'][0]['output_sha256'] = hashlib.sha256(wrong.tobytes()).hexdigest()
        source.write_text(json.dumps(report))
        expected_error = ValueError
    else:
        def failed_run(self, value):
            execute(self, value)
            raise RuntimeError('unpublished-synthetic-detail')
        monkeypatch.setattr(module.CpuOnnxSession, 'run', failed_run)
        expected_error = RuntimeError
    output = tmp_path / 'failed-partitioned.json'
    with pytest.raises(expected_error):
        module.replay_distribution(source, tmp_path / 'original', output, source='a' * 40,
            execution_schedule='partition-major')
    assert released == [None]
    actual = json.loads(output.read_text())
    assert not actual['success'] and not actual['encoder_replay_qualified']
    assert actual['original_reference_audit_passed']
    assert 'unpublished-synthetic-detail' not in output.read_text()
    if failure == 'numerical':
        assert actual['phase'] == 'numerical_failure'
        assert actual['rounds'][0]['parts'][0]['tolerance_failures'] == 6
        assert (tmp_path / actual['rounds'][0]['parts'][0]['capture']).is_file()


def test_partition_major_never_overwrites_a_later_round_capture(tmp_path, audited_chain):
    source, _, _ = audited_chain
    output = tmp_path / 'preserved-partitioned.json'
    capture = tmp_path / 'preserved-partitioned-repeat-2-part4.bin'
    capture.write_bytes(b'preserved')
    with pytest.raises(FileExistsError):
        implementation().replay_distribution(source, tmp_path / 'original', output, source='a' * 40,
            execution_schedule='partition-major')
    assert capture.read_bytes() == b'preserved' and not output.exists()


def test_partition_major_checks_graph_hashes_again_after_later_parts(tmp_path, audited_chain, monkeypatch):
    source, _, _ = audited_chain
    module = implementation()
    execute = module.CpuOnnxSession.run
    calls = [0]
    def mutate_previous(self, value):
        actual = execute(self, value)
        calls[0] += 1
        if calls[0] == 5:
            path = tmp_path / 'part0.onnx'
            path.write_bytes(path.read_bytes() + b'changed')
        return actual
    monkeypatch.setattr(module.CpuOnnxSession, 'run', mutate_previous)
    output = tmp_path / 'changed-partitioned.json'
    with pytest.raises(ValueError, match='hash'):
        module.replay_distribution(source, tmp_path / 'original', output, source='a' * 40,
            execution_schedule='partition-major')
    result = json.loads(output.read_text())
    assert not result['success'] and not result['encoder_replay_qualified']


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


def test_native_session_failure_retains_the_preallocation_journal(tmp_path, audited_chain, monkeypatch, capsys):
    source, _, _ = audited_chain
    module = implementation()
    output = tmp_path / 'cpu-load.json'
    progress_path = tmp_path / 'cpu-load-progress.json'
    def failed_constructor(self, path, sha, *, progress=None, graph_optimizations='disabled', node_order='default'):
        assert graph_optimizations == 'disabled'
        assert node_order == 'default'
        assert json.loads(progress_path.read_text())['events'][-1]['phase'] == 'load_part_begin'
        progress('session_create_begin')
        raise MemoryError('unpublished-synthetic-detail')
    monkeypatch.setattr(module.CpuOnnxSession, '__init__', failed_constructor)
    with pytest.raises(MemoryError):
        module.replay_distribution(source, tmp_path / 'original', output, source='a' * 40)
    journal = json.loads(progress_path.read_text())
    assert journal['terminal'] and not journal['success'] and journal['error_class'] == 'MemoryError'
    assert journal['events'][-2]['phase'] == 'session_create_begin'
    assert journal['events'][-2]['part'] == 0
    assert not output.exists()
    assert 'unpublished-synthetic-detail' not in progress_path.read_text() + capsys.readouterr().out


def test_progress_io_is_outside_the_actual_inference_timing(tmp_path, monkeypatch):
    module = implementation()
    runner = session(tmp_path)
    value = np.zeros((2, 3), dtype=np.float32)
    clock = [0.0]
    execute = runner.run
    def actual_call(array):
        result = execute(array)
        clock[0] += 1
        return result
    class SlowJournal:
        def record(self, *args, **kwargs):
            clock[0] += 1000
    monkeypatch.setattr(runner, 'run', actual_call)
    monkeypatch.setattr(module.time, 'perf_counter', lambda: clock[0])
    result = module.replay_chain([runner], value, [value], tmp_path / 'timed.json', progress=SlowJournal())
    assert all(row['parts'][0]['wall_seconds'] == 1 for row in result['rounds'])
