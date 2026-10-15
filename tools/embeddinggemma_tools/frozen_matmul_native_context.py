"""Compare native BLAS contexts using one audited matrix, never a model gate.

MKL_ENABLE_INSTRUCTIONS is set before importing Torch in each fresh child.
The default reference/export and ATen/oneDNN settings are not changed.
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import torch

from .frozen_audio_prefix_onnx import file_digest
from .frozen_matmul_probe import load_capture, metrics


def context_environments(source):
    base = dict(source)
    base.pop('MKL_ENABLE_INSTRUCTIONS', None)
    base['MKL_VERBOSE'] = '1'
    return {'default': dict(base), 'avx2': {**base, 'MKL_ENABLE_INSTRUCTIONS': 'AVX2'}}


def worker(directory, output, context):
    source = os.environ.get('GITHUB_SHA', '')
    if re.fullmatch('[0-9a-f]{40}', source) is None:
        raise ValueError('Native matrix context requires immutable job provenance')
    if context not in ('default', 'avx2') or os.environ.get('MKL_ENABLE_INSTRUCTIONS') != (
            'AVX2' if context == 'avx2' else None):
        raise ValueError('Fresh native worker environment differs from requested context')
    output = Path(output)
    receipt = output / 'native-context.json'
    if receipt.exists():
        raise ValueError('Refusing to overwrite native context evidence')
    a, b, expected, provenance = load_capture(directory)
    torch.set_num_threads(2)
    with torch.no_grad():
        actual = torch.mm(a, b)
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'native-fp32.bin'
    actual.numpy().astype('<f4').tofile(path)
    keys = ('MKL_ENABLE_INSTRUCTIONS', 'MKL_CBWR', 'ATEN_CPU_CAPABILITY', 'ONEDNN_MAX_CPU_ISA', 'MKL_VERBOSE')
    report = dict(complete=True, context=context, job_source=source, capture_provenance=provenance,
        execution='native_torch_mm_only_not_onnx_candidate', torch_version=str(torch.__version__),
        cpu_capability=torch.backends.cpu.get_cpu_capability(), threads=torch.get_num_threads(),
        environment={key: os.environ.get(key) for key in keys}, torch_build_config=torch.__config__.show(),
        metrics=metrics(actual, expected), capture=path.name, capture_sha256=file_digest(path),
        full_encoder_validated=False, gpu_validated=False)
    receipt.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return report


def probe_native_contexts(directory, output):
    directory, output = Path(directory).resolve(), Path(output).resolve()
    receipt = output / 'native-matmul-contexts.json'
    if receipt.exists():
        raise ValueError('Refusing to overwrite native context evidence')
    output.mkdir(parents=True, exist_ok=True)
    report = dict(scope='same_original_qualified_single_matrix_native_contexts_only', complete=False,
        cases=[], full_encoder_validated=False, gpu_validated=False)
    def save():
        receipt.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    save()
    try:
        for context, environment in context_environments(os.environ).items():
            target = output / context
            command = [sys.executable, '-m', 'embeddinggemma_tools.frozen_matmul_native_context',
                '--input', str(directory), '--output', str(target), '--worker', '--context', context]
            process = subprocess.run(command, env=environment, check=True, capture_output=True, text=True, timeout=120)
            log = output / f'{context}.log'
            log.write_text(process.stdout + process.stderr, encoding='utf-8')
            row = json.loads((target / 'native-context.json').read_text(encoding='utf-8'))
            if row['context'] != context or row['complete'] is not True:
                raise ValueError('Native child evidence does not match its requested context')
            row.update(log=log.name, log_sha256=file_digest(log))
            report['cases'].append(row)
            save()
        report['same_original_fixture'] = report['cases'][0]['capture_provenance'] == report['cases'][1]['capture_provenance']
        if not report['same_original_fixture']:
            raise ValueError('Native contexts did not consume the same audited fixture')
        report['complete'] = True
        save()
    except Exception as error:
        report['diagnostic_error'] = dict(type=type(error).__name__, message=str(error))
        save()
        raise
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--context', choices=('default', 'avx2'), default='default')
    args = parser.parse_args(argv)
    report = worker(args.input, args.output, args.context) if args.worker else probe_native_contexts(args.input, args.output)
    print(json.dumps(dict(complete=report['complete'], full_encoder_validated=False, gpu_validated=False)))
    return 0 if report['complete'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
