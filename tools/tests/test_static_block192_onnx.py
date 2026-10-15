import numpy as np
import onnx
import pytest
import torch
from onnx.reference import ReferenceEvaluator

from embeddinggemma_tools.static_block192_onnx import static_block192_export
from embeddinggemma_tools.export import export_audio_native_onnx
from embeddinggemma_tools.streaming_reference import run_flat_reference


def oracle(x, weight):
    total = None
    for start in range(0, weight.shape[0], 192):
        part = torch.zeros(x.shape[0], weight.shape[1])
        for k in range(start, min(start + 192, weight.shape[0])):
            part = (x[:, k:k+1].double() * weight[k].double() + part.double()).float()
        total = part if total is None else total + part
    return total


def test_native_forward_is_explicitly_only_original_matmul_passthrough(monkeypatch):
    from embeddinggemma_tools import export
    def forbidden(*args):
        pytest.fail("Native passthrough must not eagerly execute the expensive exported rounding order")
    monkeypatch.setattr(export, "fp32_block192_matmul", forbidden)
    x = torch.randn(3, 200, generator=torch.Generator().manual_seed(271))
    w = torch.randn(200, 7, generator=torch.Generator().manual_seed(272))
    assert torch.equal(static_block192_export(x, w).view(torch.int32), (x @ w).view(torch.int32))


@pytest.mark.parametrize("width", [1, 192, 200, 1024])
def test_standard_fp32_onnx_lowering_matches_independent_fma_and_tail_order(width, tmp_path):
    class Module(torch.nn.Module):
        def __init__(self, w):
            super().__init__()
            self.register_buffer("weight", w)
        def forward(self, x):
            return static_block192_export(x, self.weight)
    x = torch.randn(3, width, generator=torch.Generator().manual_seed(273))
    w = torch.randn(width, 5, generator=torch.Generator().manual_seed(274)) * 0.1
    before = w.clone()
    path = tmp_path / "static.onnx"
    export_audio_native_onnx(Module(w), x, path)
    model = onnx.load(path)
    onnx.checker.check_model(model)
    assert all(n.domain in ("", "ai.onnx") for n in model.graph.node)
    assert all(t.data_type != onnx.TensorProto.DOUBLE for t in model.graph.initializer)
    for node in model.graph.node:
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.TENSOR:
                assert attr.t.data_type != onnx.TensorProto.DOUBLE
    actual = run_flat_reference(ReferenceEvaluator(model), {"input": x.numpy()})[0]
    np.testing.assert_array_equal(actual.view('u4'), oracle(x, w).numpy().view('u4'))
    assert torch.equal(w, before)


@pytest.mark.parametrize("problem", ["dtype", "rank", "shape", "nonfinite", "bound"])
def test_rejects_unqualified_native_contract(problem):
    x, w = torch.ones(2, 200), torch.ones(200, 3)
    if problem == "dtype": x = x.double()
    if problem == "rank": x = x.unsqueeze(0)
    if problem == "shape": w = w[:-1]
    if problem == "nonfinite": w[0, 0] = float("nan")
    if problem == "bound": x[0, 0] = 2e6
    with pytest.raises(ValueError):
        static_block192_export(x, w)
