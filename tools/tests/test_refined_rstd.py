import pytest
import torch
import onnx
from onnx.reference import ReferenceEvaluator

from embeddinggemma_tools import export


def required():
    function = getattr(export, 'fp32_refined_rstd', None)
    assert callable(function), 'Bounded FP32 sqrt / reciprocal refinement is missing'
    return function


def test_refined_inverse_keeps_scalar_two_rounding_order_without_fp64_runtime():
    values = torch.linspace(-12, 12, 2048).exp().float()
    before = values.clone()
    # Native LayerNorm uses scalar std::sqrt, not Torch's vectorized sqrt seed.
    # The independent double oracle rounds to FP32 after each scalar operation.
    # This is test-only FP64; exported candidate ops must remain entirely FP32.
    import numpy as np
    roots = np.sqrt(values.numpy().astype(np.float64)).astype(np.float32)
    expected = torch.from_numpy((1 / roots.astype(np.float64)).astype(np.float32))
    actual = required()(values)
    assert actual.dtype == torch.float32 and torch.equal(actual, expected)
    assert torch.equal(values, before)


def test_refinement_does_not_move_correct_quotient_to_a_rounded_newton_midpoint():
    import numpy as np
    # Original layer four, token 173, query head 9. Its first reciprocal is
    # correctly rounded; one residual/Newton step lands on the lower midpoint.
    bits=np.arange(1081870327-32,1081870327+33,dtype='u4')
    values=torch.from_numpy(bits.view('f4').copy())
    roots=np.sqrt(values.numpy().astype('f8')).astype('f4')
    expected=torch.from_numpy((1/roots.astype('f8')).astype('f4'))
    actual=required()(values)
    assert torch.equal(actual.view(torch.int32),expected.view(torch.int32))


@pytest.mark.parametrize('seed_offset', [-1, 0, 1])
def test_reciprocal_correction_recovers_adjacent_gpu_seed_on_both_midpoint_sides(seed_offset):
    import numpy as np
    correct = getattr(export, '_fp32_correct_reciprocal', None)
    assert callable(correct), 'GPU reciprocal correction must consider adjacent representable candidates'
    rng = np.random.default_rng(41016)
    statistic_bits = rng.integers(np.float32(1e-6).view('u4'), np.float32(2e6).view('u4'), 65536, dtype='u4')
    witness_bits = np.arange(1081870327-32, 1081870327+33, dtype='u4')
    roots = np.sqrt(np.concatenate((statistic_bits, witness_bits)).view('f4').astype('f8')).astype('f4')
    # Include exact powers of two and their neighbors: a relative half-ULP
    # increment must reach an adjacent value even at a binade boundary.
    powers = np.exp2(np.arange(-9, 11)).astype('f4')
    roots = np.concatenate((roots, powers, np.nextafter(powers, np.float32(0)), np.nextafter(powers, np.float32(np.inf))))
    expected = (1 / roots.astype('f8')).astype('f4')
    seeds = (expected.view('u4').astype('i8') + seed_offset).astype('u4').view('f4')
    actual = correct(torch.from_numpy(roots.copy()), torch.from_numpy(seeds.copy()))
    assert np.array_equal(actual.numpy().view('u4'), expected.view('u4'))


@pytest.mark.parametrize('values', [
    torch.tensor([1.], dtype=torch.float16),
    torch.tensor([0., 1.]),
    torch.tensor([float('nan')]),
    torch.tensor([1e8]),
])
def test_unproven_inverse_domains_are_rejected(values):
    with pytest.raises(ValueError, match='bounded.*FP32'):
        required()(values)


def test_opt_in_norm_refinement_keeps_native_source_and_exported_standard_fp32_ops(tmp_path):
    required()
    source = torch.nn.LayerNorm(128, eps=1e-6, bias=False).float().eval()
    values = torch.randn(6, 128, generator=torch.Generator().manual_seed(37)) * 3
    before = source(values).detach().clone()
    model = export.TorchWelfordLayerNorm(source, refine_inverse=True)
    path = tmp_path / 'refined.onnx'
    export.export_audio_native_onnx(model, values, path)
    graph = onnx.load(path)
    onnx.checker.check_model(graph)
    assert all(w.data_type != onnx.TensorProto.DOUBLE for w in graph.graph.initializer)
    assert all(n.domain in ('', 'ai.onnx') for n in graph.graph.node)
    actual = ReferenceEvaluator(graph).run(None, {'input': values.numpy()})[0]
    __import__('numpy').testing.assert_allclose(actual, before.numpy(), rtol=1e-4, atol=1e-5)
    assert model.weight is source.weight and torch.equal(source(values), before)
