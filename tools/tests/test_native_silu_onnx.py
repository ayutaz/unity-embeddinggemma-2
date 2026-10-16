import numpy as np
import onnx
import pytest
import torch
from onnx.reference import ReferenceEvaluator

from embeddinggemma_tools import native_silu_onnx as candidate
from embeddinggemma_tools.export import export_audio_native_onnx
from embeddinggemma_tools.streaming_reference import run_flat_reference


@pytest.mark.parametrize("values", [torch.linspace(-20, 20, 256).reshape(4, 64),
                                    torch.randn(16, 128, generator=torch.Generator().manual_seed(73)) * 6])
def test_bounded_fp32_silu_matches_native_vector_rounding(values):
    actual = candidate.fp32_native_silu(values)
    expected = torch.nn.functional.silu(values)
    assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))


def test_silu_preserves_signed_zero():
    values = torch.tensor([-0.0, 0.0], dtype=torch.float32)
    assert torch.equal(candidate.fp32_native_silu(values).view(torch.int32), values.view(torch.int32))


@pytest.mark.parametrize("values", [torch.tensor([float("nan")]), torch.tensor([float("inf")]),
                                    torch.tensor([1.0], dtype=torch.float64), torch.empty(0)])
def test_silu_rejects_unqualified_inputs(values):
    with pytest.raises(ValueError, match="bounded|FP32"):
        candidate.fp32_native_silu(values)


def test_silu_uses_original_shader_mathematical_tail_outside_polynomial_domain():
    values = torch.tensor([-1e10, -100., -89., -88., -85., -81., -80., 80., 81., 100., 1e10])
    actual = candidate.fp32_native_silu(values)
    expected = torch.nn.functional.silu(values)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
    # Positive tails round to their input, negative overflow tails preserve -0.
    assert torch.equal(actual[-4:].view(torch.int32), values[-4:].view(torch.int32))
    assert torch.equal(actual[:3].view(torch.int32), expected[:3].view(torch.int32))


def test_silu_exports_only_standard_fp32_operators_and_matches_cpu(tmp_path):
    class Module(torch.nn.Module):
        def forward(self, values):
            return candidate.fp32_native_silu(values)
    values = torch.linspace(-16, 16, 128).reshape(2, 64)
    path = tmp_path / "silu.onnx"
    export_audio_native_onnx(Module(), values, path)
    model = onnx.load(path)
    onnx.checker.check_model(model)
    assert all(node.domain in ("", "ai.onnx") for node in model.graph.node)
    assert all(t.data_type != onnx.TensorProto.DOUBLE for t in model.graph.initializer)
    for node in model.graph.node:
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.TENSOR:
                assert attr.t.data_type != onnx.TensorProto.DOUBLE
    evaluator = ReferenceEvaluator(model)
    actual = evaluator.run(None, {"input": values.numpy()})[0]
    streamed = run_flat_reference(evaluator, {"input": values.numpy()})[0]
    np.testing.assert_array_equal(actual.view('u4'), streamed.view('u4'))
    expected = Module()(values).detach().numpy()
    np.testing.assert_array_equal(actual.view('u4'), expected.view('u4'))
    tails = values.clone()
    tails[0, :6] = torch.tensor([-1e10, -100., -89., -88., -85., -81.])
    tails[1, :3] = torch.tensor([81., 100., 1e10])
    with np.errstate(over="ignore"):
        actual_tails = run_flat_reference(evaluator, {"input": tails.numpy()})[0]
    assert np.isfinite(actual_tails).all()
    np.testing.assert_allclose(actual_tails, torch.nn.functional.silu(tails).numpy(), atol=1e-5, rtol=1e-4)
