import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator
import pytest

from embeddinggemma_tools.streaming_reference import run_flat_reference


def model(nodes, *, outputs=("out",), initializers=()):
    graph = helper.make_graph(nodes, "flat", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])],
        [helper.make_tensor_value_info(name, TensorProto.FLOAT, [2, 3]) for name in outputs], list(initializers))
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 24)])


def test_last_use_release_keeps_branch_and_requested_early_output_bits():
    nodes = [helper.make_node("Mul", ["x", "x"], ["square"]),
             helper.make_node("Add", ["square", "x"], ["branch"]),
             helper.make_node("Sub", ["branch", "square"], ["out"])]
    evaluator = ReferenceEvaluator(model(nodes, outputs=("square", "out")))
    x = np.array([[-0., 0., 1e-4], [-1e-4, 2.375, -13.25]], dtype="float32")
    observations = []
    actual = run_flat_reference(evaluator, {"x": x}, observer=lambda names: observations.append(set(names)))
    expected = evaluator.run(None, {"x": x})
    for a, b in zip(actual, expected):
        np.testing.assert_array_equal(a.view('u4'), b.view('u4'))
    assert observations[-1] == {"", "square", "out"}
    assert "x" not in observations[1]
    assert "branch" not in observations[-1]


def test_long_chain_retains_only_live_values_instead_of_all_activations():
    nodes = []
    name = "x"
    for index in range(100):
        target = "out" if index == 99 else f"step{index}"
        nodes.append(helper.make_node("Neg", [name], [target]))
        name = target
    evaluator = ReferenceEvaluator(model(nodes))
    observations = []
    x = np.arange(6, dtype="float32").reshape(2, 3)
    actual = run_flat_reference(evaluator, {"x": x}, observer=lambda names: observations.append(set(names)))[0]
    np.testing.assert_array_equal(actual.view('u4'), x.view('u4'))
    assert max(map(len, observations)) <= 2  # optional input plus one live tensor


def test_initializers_and_repeated_runs_keep_original_coefficient_bytes():
    weight = numpy_helper.from_array(np.arange(6, dtype="float32").reshape(2, 3), "weight")
    evaluator = ReferenceEvaluator(model([helper.make_node("Add", ["x", "weight"], ["out"])], initializers=[weight]))
    before = evaluator.rt_inits_["weight"].tobytes()
    for value in (1.0, 2.0):
        x = np.full((2, 3), value, dtype="float32")
        actual = run_flat_reference(evaluator, {"x": x})[0]
        np.testing.assert_array_equal(actual, evaluator.run(None, {"x": x})[0])
    assert evaluator.rt_inits_["weight"].tobytes() == before


def test_missing_input_and_unknown_output_fail_explicitly():
    evaluator = ReferenceEvaluator(model([helper.make_node("Neg", ["x"], ["out"])]))
    with pytest.raises(RuntimeError, match="input"):
        run_flat_reference(evaluator, {})
    with pytest.raises(RuntimeError, match="output"):
        run_flat_reference(evaluator, {"x": np.zeros((2, 3), dtype="float32")}, output_names=["missing"])


def test_context_dependent_nodes_are_rejected_before_execution(monkeypatch):
    evaluator = ReferenceEvaluator(model([helper.make_node("Neg", ["x"], ["out"])]))
    monkeypatch.setattr(evaluator.rt_nodes_[0], "need_context", lambda: True)
    with pytest.raises(ValueError, match="context|flat"):
        run_flat_reference(evaluator, {"x": np.zeros((2, 3), dtype="float32")})
