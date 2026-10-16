import numpy as np
import onnx
import pytest
import torch
from onnx.reference import ReferenceEvaluator

from embeddinggemma_tools import static_block192_onnx as lowering
from embeddinggemma_tools.export import export_audio_native_onnx
from embeddinggemma_tools.streaming_reference import run_flat_reference


@pytest.mark.parametrize("width", [24, 128, 200, 1024])
def test_batched_standard_fp32_graph_matches_independent_fma_order_without_batch_mixing(width, tmp_path):
    function = getattr(lowering, "static_block192_bmm_export", None)
    assert callable(function), "Actual fixed-order batched ONNX lowering is missing"
    left = torch.randn(2, 3, width, generator=torch.Generator().manual_seed(330))
    right = torch.randn(2, 5, width, generator=torch.Generator().manual_seed(331)).transpose(1, 2)
    total = None
    for start in range(0, width, 192):
        part = torch.zeros(2, 3, 5)
        for k in range(start, min(start + 192, width)):
            part = (left[:, :, k:k+1].double() * right[:, k:k+1, :].double() + part.double()).float()
        total = part if total is None else total + part

    class Module(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("right", right)
        def forward(self, values):
            return function(values, self.right)

    wrapper = Module()
    assert torch.equal(wrapper(left).view(torch.int32), torch.bmm(left, right).view(torch.int32))
    graph = tmp_path / "batched.onnx"
    export_audio_native_onnx(wrapper, left, graph)
    model = onnx.load(graph)
    onnx.checker.check_model(model)
    assert all(n.domain in ("", "ai.onnx") for n in model.graph.node)
    assert all(t.data_type != onnx.TensorProto.DOUBLE for t in model.graph.initializer)
    for node in model.graph.node:
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.TENSOR:
                assert attribute.t.data_type != onnx.TensorProto.DOUBLE
    actual = run_flat_reference(ReferenceEvaluator(model), {"input": left.numpy()})[0]
    np.testing.assert_array_equal(actual.view('u4'), total.numpy().view('u4'))


@pytest.mark.parametrize("problem", ["rank", "batch", "width", "dtype", "nonfinite", "bound"])
def test_batched_lowering_rejects_unaudited_contracts(problem):
    function = getattr(lowering, "static_block192_bmm_export", None)
    assert callable(function), "Batched lowering validator is missing"
    left, right = torch.ones(2, 3, 4), torch.ones(2, 4, 5)
    if problem == "rank": left = left[0]
    if problem == "batch": right = right[:1]
    if problem == "width": right = right[:, :3]
    if problem == "dtype": left = left.double()
    if problem == "nonfinite": right[0, 0, 0] = float('nan')
    if problem == "bound": right[0, 0, 0] = 2e6
    with pytest.raises(ValueError):
        function(left, right)
