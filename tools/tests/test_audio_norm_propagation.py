import numpy as np
import pytest

from embeddinggemma_tools.audio_norm_propagation import norm_error_budget, rms_norm_oracle


def test_independent_scaled_rms_oracle_uses_last_axis_and_epsilon():
    values = np.array([[[3., 4.], [0., 0.]]], dtype=np.float32)
    weight = np.array([.5, 1.], dtype=np.float32)
    actual = rms_norm_oracle(values, weight, .5)
    expected = np.array([[[3 / np.sqrt(13), 8 / np.sqrt(13)], [0., 0.]]])
    np.testing.assert_allclose(actual, expected, rtol=1e-15, atol=0)
    assert actual.dtype == np.float64


@pytest.mark.parametrize("values,weight,epsilon,restore", [
    (np.array([3., 4.], np.float32), np.ones(2, np.float32), .5, 2),
    (np.ones((1, 2), np.float32), np.ones(3, np.float32), .5, 2),
    (np.ones((1, 2), np.float64), np.ones(2, np.float32), .5, 2),
    (np.array([[np.nan, 0]], np.float32), np.ones(2, np.float32), .5, 2),
    (np.ones((1, 2), np.float32), np.array([np.inf, 0], np.float32), .5, 2),
    (np.ones((1, 2), np.float32), np.ones(2, np.float32), 0, 2),
    (np.ones((1, 2), np.float32), np.ones(2, np.float32), np.nan, 2),
    (np.ones((1, 2), np.float32), np.ones(2, np.float32), .5, 0),
])
def test_oracle_rejects_invalid_context(values, weight, epsilon, restore):
    with pytest.raises(ValueError):
        rms_norm_oracle(values, weight, epsilon, restore_multiplier=restore)


def test_budget_separates_propagated_input_native_rounding_and_backend_error():
    native_input = np.array([[0., .001]], np.float32)
    propagated_input = native_input.copy()
    propagated_input[0, 0] = 5e-6  # Input within tolerance; normalization amplifies it.
    weight = np.ones(2, np.float32) / 2
    native_output = rms_norm_oracle(native_input, weight, 1e-6).astype(np.float32)
    actual = rms_norm_oracle(propagated_input, weight, 1e-6).astype(np.float32)
    report = norm_error_budget(propagated_input, actual, native_input, native_output, weight, 1e-6)
    assert report["input_vs_native"]["mismatches"] == 0
    assert report["output_vs_native"]["mismatches"] == 1
    assert report["propagation_vs_native_oracle"]["mismatches"] == 1
    assert report["backend_vs_propagated_oracle"]["mismatches"] == 0
    assert report["native_vs_native_oracle"]["mismatches"] == 0
    assert report["decomposition_verified"]
    assert not report["numeric_gate_passed"]
    assert not report["full_encoder_validated"]


def test_budget_does_not_hide_backend_error_behind_good_input():
    values = np.array([[3., 4.]], np.float32)
    weight = np.ones(2, np.float32) / 2
    native = np.array([[3 / np.sqrt(13), 4 / np.sqrt(13)]], np.float32)
    actual = native.copy()
    actual[0, 1] += .01
    report = norm_error_budget(values, actual, values, native, weight, .5)
    assert report["propagation_vs_native_oracle"]["mismatches"] == 0
    assert report["backend_vs_propagated_oracle"]["mismatches"] == 1
    assert not report["numeric_gate_passed"]


def test_budget_rejects_output_shape_mismatch():
    values = np.ones((1, 2), np.float32)
    with pytest.raises(ValueError):
        norm_error_budget(values, np.ones((2, 1), np.float32), values, values,
                          np.ones(2, np.float32), .5)
