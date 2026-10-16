import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch


def candidate():
    from embeddinggemma_tools.native_tanh_onnx import fp32_native_tanh
    return fp32_native_tanh


def assert_original_and_current(actual, original, current):
    np.testing.assert_array_equal(actual.numpy().view(np.uint32), original.numpy().view(np.uint32))
    # oneMKL HA permits processor-specific last-bit rounding. The original
    # capture remains bit-exact; current-host compatibility retains the same
    # atol/rtol as the encoder instead of claiming universal native bits.
    np.testing.assert_allclose(actual.numpy(), current.numpy(), rtol=1e-4, atol=1e-5, equal_nan=False)
    return int(torch.count_nonzero(actual.view(torch.int32) != current.view(torch.int32)))


def test_recorded_ci_native_rounding_difference_does_not_replace_original_bits():
    # Five displayed values from CI38825766738/job116732753417; none is
    # synthesized from the candidate during this test.
    original = torch.from_numpy(np.array([3204377643, 1060752127, 1063778551,
        3210347835, 3201919855], dtype=np.uint32).view(np.float32))
    current = torch.from_numpy(np.array([3204377642, 1060752128, 1063778552,
        3210347834, 3201919856], dtype=np.uint32).view(np.float32))
    assert assert_original_and_current(original, original, current) == 5


def test_current_native_cannot_replace_a_changed_original_output():
    original = torch.tensor([0.5, -0.5])
    changed = torch.nextafter(original, torch.ones_like(original))
    with pytest.raises(AssertionError):
        assert_original_and_current(changed, original, changed)


def test_original_bit_match_does_not_hide_native_tolerance_failure():
    original = torch.tensor([0.5, -0.5])
    with pytest.raises(AssertionError):
        assert_original_and_current(original, original, original + 1e-3)


def frozen_case(case_id):
    directory = Path(__file__).parent/'fixtures/native_tanh'
    manifest = json.loads((directory/'reference.json').read_text())
    assert manifest['source'] == 'd838ecb7d1bac9169f5afc525d823de0d4a578ae'
    assert manifest['run'] == 38823855948 and manifest['artifact_id'] == 12013054484
    row = next(c for c in manifest['cases'] if c['id'] == case_id)
    values = []
    for kind in ('input', 'native'):
        spec = row[kind]
        path = directory/spec['file']
        raw = path.read_bytes()
        assert len(raw) == spec['bytes']
        assert hashlib.sha256(raw).hexdigest() == spec['sha256']
        values.append(torch.from_numpy(np.frombuffer(raw,dtype='<f4').reshape(row['shape']).copy()))
    return values


@pytest.mark.parametrize('case_id', ['dense32771', 'random514'])
def test_paired_fp32_candidate_matches_fixed_original_bits_and_current_native_tolerance(case_id):
    values, original = frozen_case(case_id)
    assert_original_and_current(candidate()(values), original, torch.tanh(values))


def test_preserves_signed_zero_and_small_finite_values():
    values = torch.tensor([-0., 0., -1e-40, 1e-40, -1e-30, 1e-30, -1e-10, 1e-10])
    actual = candidate()(values)
    assert torch.equal(actual.view(torch.int32), torch.tanh(values).view(torch.int32))


@pytest.mark.parametrize('case_id', ['dense32771', 'random514'])
def test_actual_standard_fp32_onnx_output_matches_fixed_original_bits_independently(tmp_path,case_id):
    from embeddinggemma_tools.audio_candidate_primitives import run_candidate
    values, original = frozen_case(case_id)
    actual, metadata = run_candidate(candidate(), (values,), tmp_path)
    assert metadata['execution'] == 'actual_standard_fp32_onnx_on_same_native_inputs'
    assert_original_and_current(actual, original, torch.tanh(values))


@pytest.mark.parametrize('values', [torch.ones(2, dtype=torch.float64), torch.empty(0),
    torch.tensor([float('nan')]), torch.tensor([float('inf')]), torch.tensor([float('-inf')])])
def test_rejects_unqualified_inputs(values):
    with pytest.raises(ValueError, match='finite CPU FP32'):
        candidate()(values)
