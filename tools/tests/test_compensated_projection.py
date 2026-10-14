from types import SimpleNamespace

import pytest
import torch

from embeddinggemma_tools import audio_layer_replay


def run():
    function = getattr(audio_layer_replay, "linear_fp32_compensated", None)
    assert callable(function), "FP32 compensated projection is missing"
    return function


def module(width=3, clipped=False):
    linear = torch.nn.Linear(width, 1, bias=False)
    with torch.no_grad():
        linear.weight.fill_(1)
    return SimpleNamespace(linear=linear, use_clipped_linears=clipped,
        input_min=torch.tensor(-2.), input_max=torch.tensor(2.),
        output_min=torch.tensor(-0.5), output_max=torch.tensor(0.5))


@pytest.mark.parametrize("values", [[1e8, 1., -1e8], [1., 1e8, -1e8], [1e8, -1e8, 1.]])
def test_compensation_preserves_cancellation_term_using_only_fp32(values):
    layer = module()
    hidden = torch.tensor([values], dtype=torch.float32)
    original, weight = hidden.clone(), layer.linear.weight.detach().clone()
    actual = run()(layer, hidden, chunk_size=1)
    assert actual.dtype == torch.float32 and actual.item() == 1.
    assert torch.equal(hidden, original) and torch.equal(layer.linear.weight, weight)


def test_odd_tail_and_batched_shape_follow_official_clipping():
    layer = module(5, clipped=True)
    hidden = torch.tensor([[[3., 1., 3., -3., -1.]]])
    actual = run()(layer, hidden, chunk_size=2)
    assert actual.shape == (1, 1, 1) and actual.item() == 0.5


@pytest.mark.parametrize("chunk", [0, -1, True, 1.5])
def test_invalid_chunk_budget_is_rejected(chunk):
    with pytest.raises(ValueError, match="chunk"):
        run()(module(), torch.ones(1, 3), chunk_size=chunk)


@pytest.mark.parametrize("problem", ["dtype", "shape", "nonfinite_input", "nonfinite_weight", "bias"])
def test_invalid_projection_context_is_rejected(problem):
    layer, hidden = module(), torch.ones(1, 3)
    if problem == "dtype": hidden = hidden.half()
    elif problem == "shape": hidden = torch.ones(1, 4)
    elif problem == "nonfinite_input": hidden[0, 0] = float("nan")
    elif problem == "nonfinite_weight":
        with torch.no_grad(): layer.linear.weight[0, 0] = float("inf")
    else: layer.linear.bias = torch.nn.Parameter(torch.zeros(1))
    with pytest.raises(ValueError):
        run()(layer, hidden, chunk_size=2)


def compare():
    function = getattr(audio_layer_replay, "compensated_projection_comparison", None)
    assert callable(function), "Fixed-context FP32 projection comparison is missing"
    return function


def test_candidate_compares_three_outputs_and_keeps_frozen_baselines():
    from test_audio_projection_oracle import detailed_context
    layer, parts = detailed_context()
    before = {key: value.clone() for key, value in parts.items()}
    report, values = compare()(layer, parts, chunk_size=2)
    assert report["fp32_baselines_exact_match"] is True
    assert set(values) == {"linear1", "linear2", "linear2_post_norm"}
    assert report["fp32_compensated_gate_passed"] == all(row["elementwise_passed"] for row in report["metrics"].values())
    assert all(torch.equal(parts[key], value) for key, value in before.items())


def test_bad_candidate_is_recorded_as_failure(monkeypatch):
    from test_audio_projection_oracle import detailed_context
    layer, parts = detailed_context()
    original = audio_layer_replay.linear_fp32_compensated
    monkeypatch.setattr(audio_layer_replay, "linear_fp32_compensated", lambda *args, **kwargs: original(*args, **kwargs) + 0.1)
    report, _ = compare()(layer, parts, chunk_size=2)
    assert report["fp32_compensated_gate_passed"] is False


def test_changed_baseline_is_rejected():
    from test_audio_projection_oracle import detailed_context
    layer, parts = detailed_context()
    parts["feed_forward1.ffw_layer_1"] += 0.1
    with pytest.raises(ValueError, match="baseline"):
        compare()(layer, parts, chunk_size=2)


def test_cli_returns_failure_for_bad_compensated_candidate(tmp_path, monkeypatch):
    monkeypatch.setattr(audio_layer_replay, "prepare", lambda *args, **kwargs: dict(
        context_validated=True, native_final_exact_match=True, fp32_compensated_gate_passed=False))
    code = audio_layer_replay.main(["--snapshot", str(tmp_path), "--context", str(tmp_path),
        "--output", str(tmp_path / "out"), "--reference-source", "a" * 40,
        "--diagnostic-source", "c" * 40, "--report", str(tmp_path / "report.json"), "--compensated-projection"])
    import json
    report = json.loads((tmp_path / "report.json").read_text())
    assert code == 1 and report["success"] is False


def test_producer_preserves_original_parts_and_writes_separate_candidates(tmp_path):
    from test_audio_layer_replay import bundle
    snapshot, context = bundle(tmp_path)
    base = audio_layer_replay.prepare(snapshot, context, tmp_path / "base", reference_source="a" * 40,
        diagnostic_source="c" * 40, detail_feed_forward=True)
    candidate = audio_layer_replay.prepare(snapshot, context, tmp_path / "candidate", reference_source="a" * 40,
        diagnostic_source="c" * 40, with_compensated_projection=True)
    assert candidate["parts"] == base["parts"]
    assert len(candidate["fp32_compensated_projection"]["files"]) == 3
    assert candidate["native_final_exact_match"] is True
    assert candidate["full_encoder_validated"] is False and candidate["unity_validated"] is False


def exported_layer(source, chunk_size=1):
    kind = getattr(audio_layer_replay, "CompensatedClippedLinear", None)
    assert kind is not None, "Exportable compensated FP32 layer is missing"
    return kind(source, chunk_size=chunk_size)


def test_saved_torch_program_preserves_compensation_and_borrowed_weights(tmp_path):
    source = module()
    hidden = torch.tensor([[1e8, 1., -1e8]], dtype=torch.float32)
    before = source.linear.weight.detach().clone()
    candidate = exported_layer(source)
    exported = torch.export.export(candidate, (hidden,))
    path = tmp_path / "candidate.pt2"
    torch.export.save(exported, path)
    loaded = torch.export.load(path).module()
    assert loaded(hidden).item() == 1.
    assert torch.equal(source.linear.weight, before)


def test_onnx_candidate_uses_standard_fp32_operators_and_fixed_clips(tmp_path):
    import numpy as np
    import onnx
    from onnx.reference import ReferenceEvaluator
    candidate = exported_layer(module(clipped=True))
    hidden = torch.tensor([[3., 1., 3.]])
    path = tmp_path / "candidate.onnx"
    torch.onnx.export(candidate, (hidden,), path, dynamo=False, opset_version=18, input_names=["input"])
    model = onnx.load(path)
    assert all(node.domain == "" for node in model.graph.node)
    actual = ReferenceEvaluator(model).run(None, {"input": hidden.numpy()})[0]
    assert actual.dtype == np.float32 and actual.item() == 0.5


@pytest.mark.parametrize("problem", ["nonfinite", "dtype", "shape"])
def test_exportable_layer_rejects_invalid_eager_input(problem):
    candidate = exported_layer(module())
    hidden = torch.ones(1, 3)
    if problem == "nonfinite": hidden[0, 0] = float("nan")
    elif problem == "dtype": hidden = hidden.double()
    else: hidden = torch.ones(1, 4)
    with pytest.raises(ValueError):
        candidate(hidden)
