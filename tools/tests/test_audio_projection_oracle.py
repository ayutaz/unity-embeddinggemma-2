from types import SimpleNamespace

import pytest
import torch

from embeddinggemma_tools import audio_layer_replay
from test_audio_layer_replay import fixture, bundle


def oracle():
    function = getattr(audio_layer_replay, "linear_fp64", None)
    assert callable(function), "Audited FP64 projection oracle is missing"
    return function


def module(clipped=False):
    linear = torch.nn.Linear(3, 1, bias=False)
    with torch.no_grad():
        linear.weight.fill_(1)
    return SimpleNamespace(linear=linear, use_clipped_linears=clipped,
                           input_min=torch.tensor(-2.), input_max=torch.tensor(2.),
                           output_min=torch.tensor(-0.5), output_max=torch.tensor(0.5))


def test_fp64_accumulation_retains_small_cancellation_term_without_mutating_fp32_context():
    run = oracle()
    layer = module()
    hidden = torch.tensor([[1e8, 1., -1e8]], dtype=torch.float32)
    before, weight = hidden.clone(), layer.linear.weight.detach().clone()
    actual = run(layer, hidden)
    assert actual.dtype == torch.float32 and actual.item() == 1.0
    assert torch.equal(hidden, before) and torch.equal(layer.linear.weight, weight)


def test_oracle_applies_official_input_and_output_clips():
    actual = oracle()(module(True), torch.tensor([[3., 1., 3.]], dtype=torch.float32))
    assert actual.item() == 0.5


@pytest.mark.parametrize("problem", ["dtype", "shape", "nonfinite_input", "nonfinite_weight"])
def test_projection_oracle_rejects_invalid_context(problem):
    run = oracle()
    layer = module()
    hidden = torch.tensor([[1., 2., 3.]], dtype=torch.float32)
    if problem == "dtype":
        hidden = hidden.half()
    elif problem == "shape":
        hidden = torch.ones(1, 4)
    elif problem == "nonfinite_input":
        hidden[0, 0] = float("nan")
    else:
        with torch.no_grad():
            layer.linear.weight[0, 0] = float("inf")
    with pytest.raises(ValueError):
        run(layer, hidden)


def comparison():
    function = getattr(audio_layer_replay, "projection_oracle", None)
    assert callable(function), "Frozen-context projection comparison is missing"
    return function


def detailed_context():
    layer, hidden, blocked, positions, final = fixture()
    parts = audio_layer_replay._capture_named(layer, hidden, blocked, positions, final,
        (*audio_layer_replay.PARTS, *audio_layer_replay.FEED_FORWARD_PATHS))
    return layer, parts


def test_comparison_requires_exact_fp32_baselines_and_preserves_native_parts():
    run = comparison()
    layer, parts = detailed_context()
    before = {name: value.clone() for name, value in parts.items()}
    report, outputs = run(layer, parts)
    assert report["fp32_baselines_exact_match"] is True
    assert set(outputs) == {"linear1", "linear2", "linear2_post_norm"}
    assert report["fp64_projection_gate_passed"] == all(row["elementwise_passed"] for row in report["metrics"].values())
    assert all(torch.equal(parts[name], value) for name, value in before.items())


def test_changed_fp32_baseline_is_rejected_instead_of_replacing_native_reference():
    run = comparison()
    layer, parts = detailed_context()
    parts["feed_forward1.ffw_layer_2"] += 0.1
    with pytest.raises(ValueError, match="FP32.*baseline"):
        run(layer, parts)


def test_bad_fp64_candidate_is_recorded_as_numeric_failure(monkeypatch):
    run = comparison()
    layer, parts = detailed_context()
    original = audio_layer_replay.linear_fp64
    monkeypatch.setattr(audio_layer_replay, "linear_fp64", lambda module, hidden: original(module, hidden) + 0.1)
    report, _ = run(layer, parts)
    assert report["fp64_projection_gate_passed"] is False
    assert report["metrics"]["linear1"]["mismatches"] > 0


def test_cli_never_turns_failed_fp64_candidate_into_success(tmp_path, monkeypatch):
    monkeypatch.setattr(audio_layer_replay, "prepare", lambda *args, **kwargs: dict(
        context_validated=True, native_final_exact_match=True, fp64_projection_gate_passed=False,
        full_encoder_validated=False, unity_validated=False))
    code = audio_layer_replay.main(["--snapshot", str(tmp_path), "--context", str(tmp_path),
        "--output", str(tmp_path / "out"), "--reference-source", "a" * 40,
        "--diagnostic-source", "c" * 40, "--report", str(tmp_path / "report.json"), "--projection-oracle"])
    import json
    report = json.loads((tmp_path / "report.json").read_text())
    assert code == 1 and report["success"] is False
    assert report["context_validated"] is True and report["full_encoder_validated"] is False


def test_producer_oracle_preserves_every_native_part_and_records_separate_candidates(tmp_path):
    snapshot, context = bundle(tmp_path)
    base = audio_layer_replay.prepare(snapshot, context, tmp_path / "base", reference_source="a" * 40,
        diagnostic_source="c" * 40, detail_feed_forward=True)
    candidate = audio_layer_replay.prepare(snapshot, context, tmp_path / "candidate", reference_source="a" * 40,
        diagnostic_source="c" * 40, with_projection_oracle=True)
    assert candidate["parts"] == base["parts"]
    assert len(candidate["fp64_projection_oracle"]["files"]) == 3
    assert candidate["native_final_exact_match"] is True
    assert candidate["full_encoder_validated"] is False and candidate["unity_validated"] is False
