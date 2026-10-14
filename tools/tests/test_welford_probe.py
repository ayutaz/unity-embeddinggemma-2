import json
import hashlib

from embeddinggemma_tools import welford_probe


def test_weights_free_probe_has_independent_expectations_and_no_gpu_success_claim(tmp_path):
    report = welford_probe.prepare(tmp_path)
    assert report['success'] and report['gpu_validated'] is False
    assert report['full_encoder_validated'] is False
    assert set(report['cases']) == {'fma', 'welford32', 'welford128'}
    assert set(report['files']) >= {'fma.onnx', 'baseline-fma.onnx', 'welford32.onnx', 'welford128.onnx'}
    for name, spec in report['files'].items():
        path = tmp_path / name
        assert path.stat().st_size == spec['bytes']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == spec['sha256']
    assert json.loads((tmp_path/'precision-probe.json').read_text()) == report
