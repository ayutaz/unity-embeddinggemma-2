import pytest

from embeddinggemma_tools import onnx_metadata


def extractor():
    function = getattr(onnx_metadata, "audio_layer_zero_norm_anchors", None)
    assert callable(function), "Native layer zero normalization anchors are missing"
    return function


def fixture():
    parts = ("norm_pre_attn", "norm_post_attn", "norm_out")
    return dict(nodes=[dict(name=f"/{name}/Mul_1", op_type="Mul", inputs=["hidden", name + "_weight"], outputs=[name + "_output"])
                       for name in parts],
                initializers=[dict(name=name + "_weight", data_type=1, dims=[2], raw_bytes=8, sha256=letter * 64)
                              for name, letter in zip(parts, "abc")])


def test_three_internal_norms_bind_to_exact_original_coefficients():
    values = extractor()(fixture())
    assert [row["stage"] for row in values] == ["norm_pre_attn", "norm_post_attn", "norm_out"]
    assert [row["original_output"] for row in values] == [name + "_output" for name in ("norm_pre_attn", "norm_post_attn", "norm_out")]
    assert [row["coefficient_sha256"] for row in values] == [letter * 64 for letter in "abc"]


@pytest.mark.parametrize("problem", ["missing", "duplicate", "unhashed"])
def test_internal_norm_mapping_rejects_incomplete_or_ambiguous_identity(problem):
    run = extractor()
    metadata = fixture()
    if problem == "missing":
        metadata["nodes"].pop(1)
    elif problem == "duplicate":
        metadata["nodes"].append(dict(metadata["nodes"][0]))
    else:
        metadata["initializers"][0].pop("sha256")
    with pytest.raises(ValueError):
        run(metadata)
