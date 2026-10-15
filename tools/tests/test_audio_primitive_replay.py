import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from embeddinggemma_tools import audio_primitive_replay as probe
from embeddinggemma_tools.frozen_audio_prefix_onnx import FrozenStageReference


def program():
    root = torch.nn.Module()
    root.register_buffer("weight", torch.randn(8, 8, generator=torch.Generator().manual_seed(18)))
    root.register_buffer("kernel", torch.randn(8, 1, 3, generator=torch.Generator().manual_seed(19)))
    g = torch.fx.Graph()
    x = g.placeholder("x")
    cut = g.call_function(torch.ops.aten.mul.Tensor, (x, 2.0))
    weight = g.get_attr("weight")
    mm = g.call_function(torch.ops.aten.mm.default, (cut, weight))
    square = g.call_function(torch.ops.aten.pow.Tensor_Scalar, (mm, 2))
    mean = g.call_function(torch.ops.aten.mean.dim, (square, [-1], True))
    epsilon = g.call_function(torch.ops.aten.add.Tensor, (mean, 1e-6))
    inv = g.call_function(torch.ops.aten.pow.Tensor_Scalar, (epsilon, -0.5))
    normalized = g.call_function(torch.ops.aten.mul.Tensor, (mm, inv))
    scores = g.call_function(torch.ops.aten._softmax.default, (normalized, -1, False))
    activated = g.call_function(torch.ops.aten.silu.default, (scores,))
    transposed = g.call_function(torch.ops.aten.permute.default, (activated, [1, 0]))
    expanded = g.call_function(torch.ops.aten.unsqueeze.default, (transposed, 0))
    kernel = g.get_attr("kernel")
    conv = g.call_function(torch.ops.aten.convolution.default,
        (expanded, kernel, None, [1], [1], [1], False, [0], 8))
    g.output(conv)
    return torch.fx.GraphModule(root, g), cut.name, conv.name


def frozen(values, stage):
    return FrozenStageReference(values, dict(stage=stage, expected_sha256="synthetic-test-only"))


def test_primitive_replay_is_noninjecting_and_keeps_native_control_and_original_coefficients(tmp_path):
    source, start, end = program()
    x = torch.randn(4, 8, generator=torch.Generator().manual_seed(20))
    expected = source(x)
    before, weight = str(source.graph), source.weight.clone()
    r = probe.probe_primitives(source, frozen(x * 2, "subsample"), frozen(expected, "layer_0"),
        tmp_path, from_boundary=start, to_boundary=end)
    assert r["complete"] and r["native_interpreter_control_bits_equal"]
    assert r["native_segment_original_metrics"]["exact_difference_count"] == 0
    assert set(row["operator"] for row in r["cases"]) == {"MatMul", "Pow", "ReduceMean", "Softmax", "SiLU", "Conv"}
    assert any(row["variant"] == "contiguous_inputs" for row in r["cases"])
    assert all(row["metrics"]["mismatched_elements"] == 0 for row in r["cases"])
    assert str(source.graph) == before and torch.equal(source.weight, weight)
    assert not r["full_encoder_validated"] and not r["gpu_validated"] and not r["end_to_end_validated"]
    for capture in r["worst_operator_captures"].values():
        for spec in capture["files"]:
            path = tmp_path / spec["file"]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == spec["sha256"]
            assert path.stat().st_size == int(np.prod(spec["shape"])) * 4


def test_native_reference_context_failure_is_retained_while_local_comparisons_use_native_nodes(tmp_path):
    source, start, end = program()
    x = torch.zeros(4, 8)
    r = probe.probe_primitives(source, frozen(x, "subsample"), frozen(torch.ones(1, 8, 4), "layer_0"),
        tmp_path, from_boundary=start, to_boundary=end)
    assert r["complete"] and not r["original_context_qualified"]
    assert r["native_segment_original_metrics"]["mismatched_elements"] > 0
    assert all(row["metrics"]["mismatched_elements"] == 0 for row in r["cases"])
    assert not r["full_encoder_validated"]


def test_failed_isolated_operator_never_changes_downstream_native_replay(tmp_path, monkeypatch):
    source, start, end = program()
    x = torch.randn(4, 8, generator=torch.Generator().manual_seed(27))
    real = probe.primitive_outputs
    def incorrect(target, args, kwargs):
        if target == torch.ops.aten.mm.default:
            return [("deliberately_wrong_unit_test", torch.zeros(4, 8), {})]
        return real(target, args, kwargs)
    monkeypatch.setattr(probe, "primitive_outputs", incorrect)
    r = probe.probe_primitives(source, frozen(x * 2, "subsample"), frozen(source(x), "layer_0"),
        tmp_path, from_boundary=start, to_boundary=end)
    assert r["native_interpreter_control_bits_equal"] and r["original_context_qualified"]
    assert r["cases"][0]["metrics"]["mismatched_elements"] > 0
    assert r["complete"] and not r["full_encoder_validated"]


def test_partial_failure_receipt_retains_context_without_claiming_collection_success(tmp_path, monkeypatch):
    source, start, end = program()
    x = torch.zeros(4, 8)
    def fail(*args):
        raise ValueError("unit test rejected primitive")
    monkeypatch.setattr(probe, "primitive_outputs", fail)
    with pytest.raises(ValueError, match="rejected primitive"):
        probe.probe_primitives(source, frozen(x, "subsample"), frozen(source(x), "layer_0"), tmp_path,
            from_boundary=start, to_boundary=end)
    r = json.loads((tmp_path / "audio-primitive-replay.json").read_text())
    assert not r["complete"] and r["diagnostic_error"]["type"] == "ValueError"
    assert r["native_segment_original_metrics"]["mismatched_elements"] == 0


def test_batched_matmul_is_compared_to_the_same_native_inputs_without_copying_weights():
    a = torch.randn(2, 3, 4, generator=torch.Generator().manual_seed(23))
    b = torch.randn(2, 5, 4, generator=torch.Generator().manual_seed(24)).transpose(1, 2)
    expected = torch.bmm(a, b)
    rows = probe.primitive_outputs(torch.ops.aten.bmm.default, (a, b), {})
    assert {row[0] for row in rows} == {"native_layout", "contiguous_inputs"}
    for _, result, _ in rows:
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("target,args", [
    (torch.ops.aten._softmax.default, (torch.ones(2, 3), -1, True)),
    (torch.ops.aten.convolution.default, (torch.ones(1, 2, 4), torch.ones(2, 1, 3), None, [1], [1], [1], True, [0], 2)),
    (torch.ops.aten.mm.default, (torch.ones(2, 3, dtype=torch.float64), torch.ones(3, 4, dtype=torch.float64))),
])
def test_rejects_unqualified_dtype_or_operator_contract(target, args):
    with pytest.raises(ValueError):
        probe.primitive_outputs(target, args, {})


def test_refuses_unaudited_inputs_and_overwriting_prior_results(tmp_path):
    source, start, end = program()
    x = torch.zeros(4, 8)
    with pytest.raises(ValueError, match="audited"):
        probe.probe_primitives(source, x, frozen(torch.zeros(1, 8, 4), "layer_0"), tmp_path,
            from_boundary=start, to_boundary=end)
    path = tmp_path / "audio-primitive-replay.json"
    path.write_text("original evidence")
    with pytest.raises(ValueError, match="overwrite"):
        probe.probe_primitives(source, frozen(x, "subsample"), frozen(torch.zeros(1, 8, 4), "layer_0"),
            tmp_path, from_boundary=start, to_boundary=end)
    assert path.read_text() == "original evidence"


def test_workflow_requires_explicit_primitive_mode_and_uses_original_saved_program():
    text = (Path(__file__).resolve().parents[2] / ".github/workflows/frozen-audio-prefix-onnx.yml").read_text()
    assert 'primitive_replay:' in text and 'PRIMITIVE_REPLAY' in text
    assert 'embeddinggemma_tools.audio_primitive_replay' in text
    assert '11814567883' in text and 'timeout-minutes: 20' in text
