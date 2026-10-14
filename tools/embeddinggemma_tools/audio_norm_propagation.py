"""Independent diagnostic error budget; never changes the pinned reference."""

import numpy as np


def _fp32(values):
    result = np.asarray(values)
    if result.dtype != np.float32 or not result.size or not np.isfinite(result).all():
        raise ValueError("diagnosis requires finite nonempty FP32 arrays")
    return result


def rms_norm_oracle(values, folded_weight, epsilon, *, restore_multiplier=2):
    values, weight = _fp32(values), _fp32(folded_weight)
    if (values.ndim not in (2, 3) or weight.ndim != 1 or values.shape[-1] != weight.size
            or not np.isfinite(epsilon) or epsilon <= 0
            or not np.isfinite(restore_multiplier) or restore_multiplier <= 0):
        raise ValueError("RMSNorm shape, epsilon, or coefficient scale is invalid")
    working = values.astype(np.float64)
    return (working / np.sqrt(np.mean(working ** 2, axis=-1, keepdims=True) + epsilon)
            * weight.astype(np.float64) * restore_multiplier)


def _metrics(actual, expected):
    difference = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    allowed = 1e-5 + 1e-4 * np.abs(expected.astype(np.float64))
    return {"mismatches": int(np.count_nonzero(difference > allowed)),
            "maximum_absolute_error": float(difference.max()),
            "maximum_tolerance_ratio": float((difference / allowed).max())}


def norm_error_budget(propagated_input, propagated_output, native_input, native_output,
                      folded_weight, epsilon, *, restore_multiplier=2):
    arrays = tuple(_fp32(value) for value in
                   (propagated_input, propagated_output, native_input, native_output))
    propagated_input, propagated_output, native_input, native_output = arrays
    if any(value.shape != native_input.shape for value in arrays):
        raise ValueError("input and output shape identities differ")
    propagated_oracle = rms_norm_oracle(propagated_input, folded_weight, epsilon,
                                      restore_multiplier=restore_multiplier)
    native_oracle = rms_norm_oracle(native_input, folded_weight, epsilon,
                                  restore_multiplier=restore_multiplier)
    propagated_delta = propagated_oracle - native_oracle
    backend_delta = propagated_output.astype(np.float64) - propagated_oracle
    native_delta = native_oracle - native_output.astype(np.float64)
    total = propagated_output.astype(np.float64) - native_output.astype(np.float64)
    remainder = total - (propagated_delta + backend_delta + native_delta)
    output_metrics = _metrics(propagated_output, native_output)
    return {
        "input_vs_native": _metrics(propagated_input, native_input),
        "output_vs_native": output_metrics,
        "propagation_vs_native_oracle": _metrics(propagated_oracle, native_oracle),
        "backend_vs_propagated_oracle": _metrics(propagated_output, propagated_oracle),
        "native_vs_native_oracle": _metrics(native_output, native_oracle),
        "components_peak_absolute": {
            "propagated_input": float(np.abs(propagated_delta).max()),
            "backend": float(np.abs(backend_delta).max()),
            "native_reference": float(np.abs(native_delta).max()),
        },
        "decomposition_maximum_remainder": float(np.abs(remainder).max()),
        "decomposition_verified": bool(np.allclose(remainder, 0, atol=1e-12, rtol=0)),
        "numeric_gate_passed": output_metrics["mismatches"] == 0,
        "full_encoder_validated": False,
    }
