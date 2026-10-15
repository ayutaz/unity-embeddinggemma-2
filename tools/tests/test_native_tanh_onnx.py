import numpy as np
import pytest
import torch


def candidate():
    from embeddinggemma_tools.native_tanh_onnx import fp32_native_tanh
    return fp32_native_tanh


def test_paired_fp32_candidate_matches_native_cpu_tanh_across_attention_domain_and_saturation():
    values = torch.cat((torch.linspace(-8.7, 8.7, 32769), torch.tensor([-100., 100.])))
    actual, expected = candidate()(values), torch.tanh(values)
    assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))


def test_preserves_signed_zero_and_small_finite_values():
    values = torch.tensor([-0., 0., -1e-40, 1e-40, -1e-30, 1e-30, -1e-10, 1e-10])
    actual = candidate()(values)
    assert torch.equal(actual.view(torch.int32), torch.tanh(values).view(torch.int32))


def test_actual_standard_fp32_onnx_output_is_checked_independently(tmp_path):
    from embeddinggemma_tools.audio_candidate_primitives import run_candidate
    values = torch.randn(2, 257, generator=torch.Generator().manual_seed(161024)) * 1.5
    actual, metadata = run_candidate(candidate(), (values,), tmp_path)
    assert metadata['execution'] == 'actual_standard_fp32_onnx_on_same_native_inputs'
    np.testing.assert_array_equal(actual.numpy().view(np.uint32), torch.tanh(values).numpy().view(np.uint32))


@pytest.mark.parametrize('values', [torch.ones(2, dtype=torch.float64), torch.empty(0),
    torch.tensor([float('nan')]), torch.tensor([float('inf')]), torch.tensor([float('-inf')])])
def test_rejects_unqualified_inputs(values):
    with pytest.raises(ValueError, match='finite CPU FP32'):
        candidate()(values)
