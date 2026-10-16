"""Observe the untouched pinned encoder on all six original inputs, not ONNX/GPU."""
import argparse
import gc
import json
import os
from pathlib import Path
import platform
import re

import torch

from .audio_native_norm_replay import digest, load_frozen_input
from .frozen_matmul_probe import metrics

CASES = ('silence', 'impulse', 'sine', 'nyquist', 'noise', 'padded_noise')


def probe(directory, output):
    source = os.environ.get('GITHUB_SHA', '')
    if re.fullmatch('[0-9a-f]{40}', source) is None:
        raise ValueError('Immutable job source is required')
    output = Path(output)
    receipt = output/'audio-native-context.json'
    if receipt.exists() or any(output.glob('*-native-fp32.bin')):
        raise ValueError('Refusing to overwrite native context evidence')
    output.mkdir(parents=True, exist_ok=True)
    report = dict(scope='untouched_saved_native_encoder_original_six_cases_only', source=source,
        complete=False, original_tolerance_passed=False, onnx_validated=False,
        gpu_validated=False, fresh_official_validated=False, cases=[],
        runtime=dict(system=platform.system(), processor=platform.processor(),
            torch_version=str(torch.__version__), cpu_capability=torch.backends.cpu.get_cpu_capability(),
            threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(),
            torch_build_configuration=torch.__config__.show(),
            environment={k:os.environ.get(k) for k in ('MKL_ENABLE_INSTRUCTIONS','MKL_CBWR',
                'ATEN_CPU_CAPABILITY','ONEDNN_MAX_CPU_ISA','DNNL_MAX_CPU_ISA',
                'OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')}))

    def save():
        receipt.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')

    save()
    try:
        for case in CASES:
            # The existing loader audits pinned hashes, paths, sizes and Torch.
            # No graph/kernel is replaced; free each model before the next load.
            program, values, expected, provenance = load_frozen_input(directory, case_id=case)
            with torch.no_grad():
                actual = program(values)
            if isinstance(actual, tuple) and len(actual) == 1:
                actual = actual[0]
            if (not isinstance(actual, torch.Tensor) or actual.dtype != torch.float32
                    or actual.device.type != 'cpu' or actual.shape != expected.shape
                    or not torch.isfinite(actual).all()):
                raise ValueError('Native output shape/dtype/finite contract differs')
            capture = output/f'{case}-native-fp32.bin'
            actual.detach().contiguous().numpy().astype('<f4').tofile(capture)
            report['cases'].append(dict(case_id=case, profile=provenance['profile'],
                provenance=provenance, metrics=metrics(actual,expected), shape=list(actual.shape),
                capture=capture.name, capture_sha256=digest(capture)))
            save()
            del program, values, expected, actual
            gc.collect()
        report['values_compared'] = sum(int(torch.tensor(row['shape']).prod()) for row in report['cases'])
        report['original_tolerance_passed'] = all(row['metrics']['mismatched_elements'] == 0 for row in report['cases'])
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
    args = parser.parse_args(argv)
    r = probe(args.input,args.output)
    print(json.dumps(dict(complete=r['complete'], original_tolerance_passed=r['original_tolerance_passed'],
        cases=[dict(case_id=x['case_id'],metrics=x['metrics']) for x in r['cases']],
        onnx_validated=False, gpu_validated=False, fresh_official_validated=False)))
    return 0 if r['complete'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
