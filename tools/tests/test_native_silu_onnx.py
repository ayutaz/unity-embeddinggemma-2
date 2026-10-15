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


@pytest.mark.parametrize("values", [torch.tensor([81.0]), torch.tensor([-81.0]),
                                    torch.tensor([float("nan")]), torch.tensor([float("inf")]),
                                    torch.tensor([1.0], dtype=torch.float64), torch.empty(0)])
def test_silu_rejects_unqualified_inputs(values):
    with pytest.raises(ValueError, match="bounded|FP32"):
        candidate.fp32_native_silu(values)


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
