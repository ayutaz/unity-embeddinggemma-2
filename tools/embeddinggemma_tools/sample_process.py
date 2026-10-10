"""Check a saved sample report against a completed external process receipt."""

import argparse
import hashlib
import json
from pathlib import Path

from .android import _hex, _require
from .android_runs import _date, _object


def audit_sample_process(report, process):
    report_path, process_path = Path(report), Path(process)
    runtime, receipt = _object(report_path), _object(process_path)
    run_id = runtime.get('run_id')
    build = runtime.get('build')
    _require(runtime.get('schema_version') == 1 and isinstance(build, dict), 'A sample runtime report is required')
    source = build.get('codeCommit')
    _require(_hex(run_id, 32) and _hex(source, 40)
             and receipt.get('runId') == run_id and receipt.get('source') == source,
             'Process receipt and runtime provenance must match')
    exit_code = receipt.get('exitCode')
    _require(type(exit_code) is int, 'A completed process exit code is required')
    process_start, process_end = _date(receipt.get('startedUtc')), _date(receipt.get('completedUtc'))
    runtime_start, runtime_end = _date(runtime.get('started_utc')), _date(runtime.get('completed_utc'))
    _require(process_start <= runtime_start <= runtime_end <= process_end,
             'Runtime timestamps must be inside the completed process interval')
    runtime_passed = (all(runtime.get(field) is True for field in
                         ('success', 'gpu_verified', 'worker_released', 'font_destroyed',
                          'blank_rejected', 'missing_model_rejected'))
                      and runtime.get('is_editor') is False and runtime.get('batch_mode') is False
                      and runtime.get('phase') == 'completed' and runtime.get('validation_mode') == 'real_model'
                      and runtime.get('console') == [] and runtime.get('error') is None)
    passed = exit_code == 0 and runtime_passed
    return {
        'schema_version': 1, 'run_id': run_id, 'source_commit': source,
        'runtime_report_success': runtime.get('success') is True,
        'runtime_gpu_verified': runtime.get('gpu_verified') is True,
        'runtime_completion_checks_passed': runtime_passed,
        'process_exit_code': exit_code, 'process_exit_zero': exit_code == 0,
        'process_seconds': (process_end - process_start).total_seconds(),
        'process_gate_passed': passed,
        'reason': 'completed' if passed else 'nonzero_process_exit' if exit_code != 0 else 'runtime_checks_incomplete',
        'report_sha256': hashlib.sha256(report_path.read_bytes()).hexdigest(),
        'process_receipt_sha256': hashlib.sha256(process_path.read_bytes()).hexdigest(),
        'execution_performed_by_auditor': False, 'reference_reaudited': False, 'm2_release_verified': False,
        'scope': 'Completed receipt consistency and runtime completion flags only. Numeric reference, screenshots, '
                 'model provenance and two independent launches require separate evidence. Saved exit codes are '
                 'external launcher observations, not operating-system execution performed by this auditor.',
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', required=True, type=Path)
    parser.add_argument('--process', required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        result = audit_sample_process(args.report, args.process)
    except (ValueError, OSError) as error:
        print(json.dumps({'process_gate_passed': False, 'reason': 'invalid_process_receipt',
                          'error_type': type(error).__name__}))
        return 1
    print(json.dumps(result, indent=2))
    return 0 if result['process_gate_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
