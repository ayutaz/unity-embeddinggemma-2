"""Weights-free GPU probes; expectations use independent native FP32 kernels."""
import hashlib
import json
from pathlib import Path
import sys

import torch
from .export import export_audio_native_onnx, fp32_fma, TorchWelfordLayerNorm


class FmaProbe(torch.nn.Module):
    def __init__(self, emulate=True):
        super().__init__()
        self.emulate = emulate

    def forward(self, values):
        a, b, c = values[0], values[1], values[2]
        return fp32_fma(a, b, c) if self.emulate else a * b + c


def prepare(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'precision-probe.json').exists():
        raise FileExistsError('preserve each probe generation separately')
    cases, files = {}, {}
    def remember(path):
        files[path.name] = dict(bytes=path.stat().st_size,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    def save(name, model, values, expected):
        path = output / f'{name}.onnx'
        export_audio_native_onnx(model, values, path)
        remember(path)
        for suffix, array in (('input', values), ('expected', expected)):
            path = output / f'{name}-{suffix}.bin'
            array.detach().numpy().astype('<f4').tofile(path)
            remember(path)
        cases[name] = dict(input_shape=list(values.shape), output_shape=list(expected.shape))
    values = torch.tensor([[1.0000001192092896, 4097., -4097.],
                           [0.9999998807907104, 4097., 4097.],
                           [-1., -16785408., 16785408.]])
    expected = (values[0].double()*values[1].double()+values[2].double()).float()
    assert not torch.equal(values[0]*values[1]+values[2], expected)
    save('fma', FmaProbe(), values, expected)
    baseline = output / 'baseline-fma.onnx'
    export_audio_native_onnx(FmaProbe(False), values, baseline)
    remember(baseline)
    for width in (32, 128):
        source = torch.nn.LayerNorm(width, eps=1e-6, bias=False).float().eval()
        source.weight.data.copy_(torch.linspace(0.8, 1.2, width))
        values = torch.randn(3, width, generator=torch.Generator().manual_seed(14)) * 10
        with torch.no_grad(): expected = source(values)
        save(f'welford{width}', TorchWelfordLayerNorm(source), values, expected)
    report = dict(success=True, scope='weights_free_precision_probe_artifacts_only',
        gpu_validated=False, full_encoder_validated=False, end_to_end_validated=False,
        feature_atol=1e-5, feature_rtol=1e-4, cases=cases, files=files,
        torch_version=torch.__version__, torch_commit=torch.version.git_version,
        source_weights='synthetic_linear_scale_no_downloaded_weights',
        single_rounding_reference='float64 product+sum rounded once to FP32 on three bounded exact-product cases')
    (output/'precision-probe.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    return report


if __name__ == '__main__':
    report = prepare(Path(sys.argv[1]))
    print(json.dumps(dict(success=report['success'], case_count=len(report['cases']), gpu_validated=False)))
