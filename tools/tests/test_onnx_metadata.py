import hashlib
import io

import onnx
import pytest
from onnx import TensorProto, helper

from embeddinggemma_tools import onnx_metadata


def scanner():
    value = getattr(onnx_metadata, "scan", None)
    assert callable(value), "Bounded ONNX metadata scanner is missing"
    return value


class LimitedReader(io.BytesIO):
    def __init__(self, data):
        super().__init__(data)
        self.read_bytes = 0

    def read(self, count=-1):
        assert 0 <= count <= 1 << 20, "Never request an entire model or an unbounded read"
        value = super().read(count)
        self.read_bytes += len(value)
        return value


def model_bytes():
    large = TensorProto(name="large", data_type=TensorProto.FLOAT, dims=[1 << 19], raw_data=b"a" * (2 << 20))
    small = TensorProto(name="audio.layers.0.norm_out.weight", data_type=TensorProto.FLOAT, dims=[2], raw_data=b"\x00\x00\x80?" * 2)
    graph = helper.make_graph([helper.make_node("Mul", ["input", small.name], ["hidden"], name="norm_out"),
                              helper.make_node("Clip", ["hidden"], ["output"], name="clamp")],
                             "fixture", [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 2])],
                             [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2])], [large, small])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 24)]).SerializeToString()


def test_names_structure_and_small_weight_identity_without_reading_large_weights():
    run = scanner()
    stream = LimitedReader(model_bytes())
    report = run(stream, small_tensor_bytes=4096)
    assert stream.read_bytes < 20000
    assert report["nodes"][0] == dict(name="norm_out", op_type="Mul", domain="", inputs=["input", "audio.layers.0.norm_out.weight"], outputs=["hidden"])
    assert report["outputs"] == ["output"]
    weights = {row["name"]: row for row in report["initializers"]}
    assert weights["large"]["raw_bytes"] == 2 << 20 and weights["large"].get("sha256") is None
    assert weights["audio.layers.0.norm_out.weight"]["sha256"] == hashlib.sha256(b"\x00\x00\x80?" * 2).hexdigest()
    assert weights["audio.layers.0.norm_out.weight"]["dims"] == [2]
    assert report["model_executed"] is False and report["weights_fully_read"] is False


@pytest.mark.parametrize("payload", [b"", b"\x3a\x80", b"\x3a\x05\x01"])
def test_truncated_or_missing_graph_is_rejected(payload):
    with pytest.raises(ValueError):
        scanner()(LimitedReader(payload))


def test_metadata_field_budget_is_enforced():
    run = scanner()
    graph = helper.make_graph([helper.make_node("Identity", ["x"], ["y"], name="x" * 1000)], "g", [], [])
    data = helper.make_model(graph).SerializeToString()
    with pytest.raises(ValueError, match="budget"):
        run(LimitedReader(data), metadata_field_bytes=128)
