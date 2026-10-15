import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from embeddinggemma_tools import frozen_matmul_probe as probe


def fixture(tmp_path, monkeypatch):
    x = torch.randn(2, 4, generator=torch.Generator().manual_seed(281))
    w = torch.randn(3, 4, generator=torch.Generator().manual_seed(282)).t() * 0.1
    expected = x @ w
    files = []
    for name, value in [("input0", x), ("input1", w), ("native", expected)]:
        p = tmp_path / f"worst-MatMul-{name}-fp32.bin"
        p.write_bytes(value.contiguous().numpy().tobytes())
        files.append(dict(file=p.name, shape=list(value.shape), native_strides=list(value.stride()),
                          sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    r = dict(complete=True, original_context_qualified=True, native_interpreter_control_bits_equal=True,
        coefficients_unchanged=True, native_segment_original_metrics=dict(exact_difference_count=0),
        provenance=dict(job_source=probe.CAPTURE_SOURCE),
        worst_operator_captures=dict(MatMul=dict(node="synthetic_mm", files=files)))
    p = tmp_path / "audio-primitive-replay.json"
    p.write_text(json.dumps(r))
    monkeypatch.setattr(probe, "CAPTURE_REPORT_SHA", hashlib.sha256(p.read_bytes()).hexdigest())
    return x, w, expected, r


def test_loader_audits_pinned_qualified_capture_and_restores_original_matrix_layout(tmp_path, monkeypatch):
    x, w, expected, _ = fixture(tmp_path, monkeypatch)
    actual_x, actual_w, actual_expected, info = probe.load_capture(tmp_path)
    assert torch.equal(actual_x, x) and torch.equal(actual_w, w) and torch.equal(actual_expected, expected)
    assert actual_w.stride() == w.stride()
    assert info["capture_source"] == probe.CAPTURE_SOURCE


@pytest.mark.parametrize("problem", ["report", "tensor", "escape", "unqualified"])
def test_loader_rejects_tampering_path_escape_or_unqualified_context(tmp_path, monkeypatch, problem):
    _, _, _, r = fixture(tmp_path, monkeypatch)
    report = tmp_path / "audio-primitive-replay.json"
    if problem == "tensor":
        p = tmp_path / r["worst_operator_captures"]["MatMul"]["files"][0]["file"]
        raw = bytearray(p.read_bytes()); raw[0] ^= 1; p.write_bytes(raw)
    else:
        if problem == "escape": r["worst_operator_captures"]["MatMul"]["files"][0]["file"] = "../outside.bin"
        if problem == "unqualified": r["original_context_qualified"] = False
        report.write_text(json.dumps(r) + " ")
        if problem != "report":
            monkeypatch.setattr(probe, "CAPTURE_REPORT_SHA", hashlib.sha256(report.read_bytes()).hexdigest())
    with pytest.raises(ValueError):
        probe.load_capture(tmp_path)


def test_actual_onnx_results_are_separate_from_native_passthrough_and_whole_encoder_gates(tmp_path, monkeypatch):
    folder = tmp_path / "input"; folder.mkdir()
    _, _, expected, _ = fixture(folder, monkeypatch)
    output = tmp_path / "output"
    r = probe.probe_capture(folder, output)
    assert r["complete"] and r["weights_unchanged"]
    assert r["native_execution"] == "original_MatMul_passthrough_not_fixed_order_execution"
    assert r["onnx_metrics"]["mismatched_elements"] == 0
    actual = np.fromfile(output / r["onnx_capture"], dtype="<f4").reshape(expected.shape)
    np.testing.assert_allclose(actual, expected.numpy(), atol=1e-5, rtol=1e-4)
    assert not r["full_encoder_validated"] and not r["gpu_validated"] and not r["end_to_end_validated"]
    with pytest.raises(ValueError, match="overwrite"):
        probe.probe_capture(folder, output)


def test_workflow_downloads_only_pinned_primitive_fixture_and_never_whole_model():
    p = Path(__file__).resolve().parents[2] / ".github/workflows/frozen-matmul-onnx.yml"
    text = p.read_text()
    assert '12001831037' in text and 'frozen_matmul_probe' in text
    assert 'audio.pt2' not in text and 'snapshot_download' not in text
    assert 'timeout-minutes: 10' in text and 'workflow_dispatch:' in text
