import pytest

from embeddinggemma_tools import onnx_metadata


def extractor():
    function = getattr(onnx_metadata, "audio_feed_forward_norm_anchors", None)
    assert callable(function), "Feed-forward norm coefficient anchors are missing"
    return function


def fixture():
    names = ("pre_layer_norm", "post_layer_norm")
    return dict(nodes=[dict(name=f"/feed_forward1/{name}/Mul_1", op_type="Mul", inputs=["h", name + "_weight"], outputs=[name + "_out"])
                       for name in names],
                initializers=[dict(name=name + "_weight", data_type=1, dims=[2], raw_bytes=8, sha256=letter * 64)
                              for name, letter in zip(names, "ab")])


def test_feed_forward_scale_bytes_bind_pre_and_post_normalization():
    anchors = extractor()(fixture())
    assert [value["stage"] for value in anchors] == ["pre_layer_norm", "post_layer_norm"]
    assert [value["coefficient_sha256"] for value in anchors] == [letter * 64 for letter in "ab"]


@pytest.mark.parametrize("problem", ["missing", "duplicate", "unhashed"])
def test_feed_forward_norm_anchor_rejects_missing_or_ambiguous_identity(problem):
    run = extractor()
    metadata = fixture()
    if problem == "missing":
        metadata["nodes"].pop(0)
    elif problem == "duplicate":
        metadata["nodes"].append(dict(metadata["nodes"][0]))
    else:
        metadata["initializers"][0].pop("sha256")
    with pytest.raises(ValueError):
        run(metadata)
