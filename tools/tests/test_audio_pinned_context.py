import copy
import importlib

import pytest


def compare():
    try:
        module = importlib.import_module("embeddinggemma_tools.audio_pinned_context")
    except ModuleNotFoundError:
        pytest.fail("Pinned audio reference context auditor is missing")
    return module.compare_references


def reference():
    profiles = {"full": dict(feature_count=2, feature_width=2, graph="full.onnx", frame_mask="full-mask.bin"),
                "partial": dict(feature_count=1, feature_width=2, graph="partial.onnx", frame_mask="partial-mask.bin")}
    signals = ["silence", "impulse", "sine", "nyquist", "noise", "padded_noise"]
    rows = [dict(id=name, profile="partial" if name == "padded_noise" else "full",
                 input_features=name + ".bin", features=[[1., 2.]] if name == "padded_noise" else [[1., 2.], [3., 4.]])
            for name in signals]
    value = dict(metadata=dict(model_revision="a" * 40, source_commit="b" * 40, input_sha256={"weights": "c" * 64},
                               feature_rtol=1e-4, feature_atol=1e-5, versions={"torch": "pinned"}),
                 profiles=profiles, cases=rows)
    audit = dict(files={name: dict(sha256="d" * 64) for name in ["full.onnx", "partial.onnx", "full-mask.bin", "partial-mask.bin", *[row["input_features"] for row in rows]]})
    return value, audit


def test_exact_native_outputs_and_inputs_validate_context_across_producer_ids():
    function = compare()
    old, audit = reference()
    new = copy.deepcopy(old)
    new["metadata"]["source_commit"] = "e" * 40
    report = function(new, audit, old, audit)
    assert report["context_validated"] is True
    assert report["case_count"] == 6
    assert all(row["native_exact_match"] for row in report["cases"])
    assert report["full_encoder_validated"] is False and report["unity_validated"] is False


def test_native_drift_refuses_localization_even_if_same_workflow_stage_context_passed():
    function = compare()
    old, audit = reference()
    new = copy.deepcopy(old)
    new["cases"][0]["features"][0][0] += 0.1
    report = function(new, audit, old, audit)
    assert report["context_validated"] is False
    assert report["cases"][0]["native_fixed_matches"] is False
    assert report["cases"][0]["mismatched_elements"] == 1


def test_changed_input_hash_refuses_context_despite_equal_native_vectors():
    function = compare()
    old, audit = reference()
    changed = copy.deepcopy(audit)
    changed["files"]["sine.bin"]["sha256"] = "e" * 64
    report = function(old, changed, old, audit)
    assert report["context_validated"] is False
    assert report["cases"][2]["input_sha_equal"] is False


def test_incomplete_original_signals_cannot_validate_context():
    function = compare()
    old, audit = reference()
    missing = copy.deepcopy(old)
    missing["cases"].pop()
    with pytest.raises(ValueError, match="six original signals"):
        function(missing, audit, old, audit)


@pytest.mark.parametrize("missing", ["versions", "frame_mask"])
def test_missing_context_metadata_cannot_become_matching_null_values(missing):
    function = compare()
    broken, audit = reference()
    if missing == "versions":
        broken["metadata"].pop("versions")
    else:
        broken["profiles"]["partial"].pop("frame_mask")
    with pytest.raises(ValueError, match="pinned audio context.*missing"):
        function(broken, audit, broken, audit)
