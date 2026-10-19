"""Optional CPU-only ONNX replay; never a Sentis runtime or release acceptance."""
import hashlib
import json
from pathlib import Path
import re
import time

import numpy as np

ROUNDS = ('first', 'warm', 'repeat-1', 'repeat-2')


def _external_tensor(message):
    if message.DESCRIPTOR.full_name == 'onnx.TensorProto':
        return bool(message.external_data) or message.data_location == 1
    for field, value in message.ListFields():
        if field.message_type is not None:
            children = value if field.is_repeated else (value,)
            if any(_external_tensor(child) for child in children):
                return True
    return False


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _array(value, shape):
    if not isinstance(value, np.ndarray) or value.dtype != np.float32:
        raise ValueError('A real FP32 CPU array is required')
    if tuple(value.shape) != tuple(shape):
        raise ValueError('CPU tensor shape differs from the static graph')
    if not np.isfinite(value).all():
        raise ValueError('CPU tensor values must be finite')
    return value


def _shape(port):
    if (port.type != 'tensor(float)' or not port.shape
            or any(type(n) is not int or n < 1 for n in port.shape)):
        raise ValueError('Only a positive static FP32 shape is supported')
    return tuple(port.shape)


class CpuOnnxSession:
    def __init__(self, graph, expected_sha256):
        import onnx
        import onnxruntime as ort
        self.graph = Path(graph)
        if (re.fullmatch('[a-f0-9]{64}', expected_sha256 or '') is None
                or digest(self.graph) != expected_sha256):
            raise ValueError('The actual ONNX graph hash differs')
        model = onnx.load_model(self.graph, load_external_data=False)
        if _external_tensor(model):
            raise ValueError('Only inline weights bound by the complete graph hash are supported; external data is rejected')
        del model
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.intra_op_num_threads = 4
        options.inter_op_num_threads = 1
        options.log_severity_level = 3
        self.session = ort.InferenceSession(str(self.graph), sess_options=options,
                                           providers=['CPUExecutionProvider'])
        self.session.disable_fallback()
        if self.session.get_providers() != ['CPUExecutionProvider']:
            raise ValueError('Exactly the explicit CPU execution provider is required')
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError('Exactly one input and output are required')
        self.input_shape, self.output_shape = _shape(inputs[0]), _shape(outputs[0])
        self.input_name, self.output_name = inputs[0].name, outputs[0].name
        if digest(self.graph) != expected_sha256:
            raise ValueError('The ONNX graph hash changed during session creation')
        self.configuration = dict(providers=self.session.get_providers(),
            graph_sha256=expected_sha256, graph_optimizations='disabled',
            execution_mode='sequential', intra_op_threads=4, inter_op_threads=1,
            fallback_enabled=False, onnxruntime_version=ort.__version__)

    def run(self, value):
        _array(value, self.input_shape)
        before = hashlib.sha256(value.tobytes()).digest()
        result = self.session.run([self.output_name], {self.input_name: value})[0]
        if hashlib.sha256(value.tobytes()).digest() != before:
            raise ValueError('An actual CPU execution mutated its input')
        return _array(result, self.output_shape)


def replay_chain(sessions, initial, references, output):
    """Run four actual carries. References are compared only after computation."""
    if (not sessions or len(sessions) != len(references)
            or not all(isinstance(s, CpuOnnxSession) for s in sessions)):
        raise ValueError('Actual CPU sessions and matching reference parts are required')
    _array(initial, sessions[0].input_shape)
    for index, (session, expected) in enumerate(zip(sessions, references, strict=True)):
        try:
            _array(expected, session.output_shape)
        except ValueError as error:
            raise ValueError('A reference shape/dtype/finite contract differs') from error
        if index and session.input_shape != sessions[index - 1].output_shape:
            raise ValueError('Adjacent CPU graph shapes do not connect')
    output = Path(output)
    captures = {(r, i): output.with_name(f'{output.stem}-{r}-part{i}.bin')
                for r in ROUNDS for i in range(len(sessions))}
    if output.exists() or any(p.exists() for p in captures.values()):
        raise FileExistsError('Preserve all existing CPU reports and captures')
    output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(schema=1, success=False, phase='starting',
        scope='Explicit CPU ONNX encoder carry diagnostic only',
        actual_backend='CPUExecutionProvider', sessions=[s.configuration for s in sessions],
        original_atol=1e-5, original_rtol=1e-4, rounds=[],
        GPU_validated=False, CPU_reference_comparison_qualified=False,
        complete_performance_accepted=False, references_used_as_inference_inputs=False,
        timing_scope='CPU session including input/output integrity checks; excludes reference comparison, capture and report writes')

    def save():
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')

    with output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    for round_name in ROUNDS:
        row = dict(round=round_name, parts=[], tolerance_failures=0)
        report['rounds'].append(row)
        actual = initial
        for index, (session, expected) in enumerate(zip(sessions, references, strict=True)):
            wall, cpu = time.perf_counter(), time.process_time()
            actual = session.run(actual)
            elapsed, used = time.perf_counter() - wall, time.process_time() - cpu
            capture = captures[(round_name, index)]
            with capture.open('xb') as stream:
                stream.write(actual.astype('<f4', copy=False).tobytes())
            difference = np.abs(actual - expected)
            failures = int(np.count_nonzero(difference > np.float32(1e-5) + np.float32(1e-4) * np.abs(expected)))
            row['parts'].append(dict(index=index, wall_seconds=elapsed, process_cpu_seconds=used,
                capture=capture.name, sha256=digest(capture), shape=list(actual.shape),
                tolerance_failures=failures, maximum_absolute_error=float(difference.max())))
            row['tolerance_failures'] += failures
            report['phase'] = 'numerical_failure' if failures else round_name
            save()
            if failures:
                raise ValueError('Actual CPU output exceeds the original tolerance')
    report.update(success=True, phase='complete')
    save()
    return report


def replay_distribution(chain_report, original, output, *, source):
    """Require the unchanged original audit before loading any CPU graph."""
    from .native_regression_distribution import audit_distribution_report
    if re.fullmatch('[a-f0-9]{40}', source or '') is None:
        raise ValueError('Immutable execution source is required')
    output = Path(output)
    if output.exists():
        raise FileExistsError('Preserve the existing CPU replay report')
    audited = audit_distribution_report(chain_report, original)
    if audited['original_tolerance_passed'] is not True or audited['source'] != source:
        raise ValueError('The original reference gate and exact execution source must match')
    path = Path(chain_report)
    source_sha = digest(path)
    data = json.loads(path.read_text(encoding='utf-8'))
    selected = [row for row in data['cases'] if row['variant'] == 'ordered_original_midpoint_bmm']
    if len(selected) != 1 or len(selected[0]['parts']) != 5:
        raise ValueError('Exactly five audited ordered ONNX parts are required')
    parts = selected[0]['parts']

    def checked(name, expected_sha):
        if not isinstance(name, str) or '/' in name or '\\' in name or ':' in name or name in ('', '.', '..'):
            raise ValueError('Only a relative capture/graph basename is supported')
        target = path.parent / name
        if not target.is_file() or target.is_symlink() or digest(target) != expected_sha:
            raise ValueError('An actual graph/capture hash differs')
        return target

    load_wall, load_cpu = time.perf_counter(), time.process_time()
    sessions, references = [], []
    for index, part in enumerate(parts):
        if part['index'] != index:
            raise ValueError('The five original graph parts must retain their order')
        graph = checked(part['graph'], part['graph_sha256'])
        session = CpuOnnxSession(graph, part['graph_sha256'])
        ref = checked(part['output_capture'], part['output_sha256'])
        array = np.fromfile(ref, dtype='<f4')
        if array.size != int(np.prod(session.output_shape)):
            raise ValueError('An actual reference capture shape differs')
        references.append(array.reshape(session.output_shape))
        sessions.append(session)
    initial_path = checked('raw-input-fp32.bin', data['input_sha256'])
    final_path = checked('reference-output-fp32.bin', data['reference_sha256'])
    initial = np.fromfile(initial_path, dtype='<f4').reshape(sessions[0].input_shape)
    references[-1] = np.fromfile(final_path, dtype='<f4').reshape(sessions[-1].output_shape)
    load_seconds, load_used = time.perf_counter() - load_wall, time.process_time() - load_cpu
    try:
        result = replay_chain(sessions, initial, references, output)
        if digest(path) != source_sha or any(digest(s.graph) != s.configuration['graph_sha256'] for s in sessions):
            raise ValueError('Original chain/graph hashes changed during CPU replay')
    except FileExistsError:
        raise
    except Exception as error:
        if output.is_file():
            failed = json.loads(output.read_text(encoding='utf-8'))
            failed.update(success=False, encoder_replay_qualified=False, source=source,
                case_id=audited['case_id'], profile=audited['profile'],
                original_chain_report_sha256=source_sha, original_reference_audit_passed=True,
                phase='numerical_failure' if failed['phase'] == 'numerical_failure' else 'failed_cpu_replay',
                error_class=type(error).__name__)
            output.write_text(json.dumps(failed, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        raise
    result.update(source=source, case_id=audited['case_id'], profile=audited['profile'],
        original_chain_report_sha256=source_sha, original_reference_audit_passed=True,
        encoder_replay_qualified=True, load_wall_seconds=load_seconds,
        load_process_cpu_seconds=load_used)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return result


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--chain-report', type=Path, required=True)
    parser.add_argument('--original', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source', required=True)
    args = parser.parse_args(argv)
    result = replay_distribution(args.chain_report, args.original, args.output, source=args.source)
    print(json.dumps(dict(success=result['success'], case_id=result['case_id'],
        encoder_replay_qualified=result['encoder_replay_qualified'],
        CPU_reference_comparison_qualified=False, complete_performance_accepted=False)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
