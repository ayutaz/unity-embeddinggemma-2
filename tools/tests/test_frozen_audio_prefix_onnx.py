import operator
from pathlib import Path

import numpy as np
import onnx
import pytest
import torch
import yaml

from embeddinggemma_tools import frozen_audio_prefix_onnx as probe


def program(*, mask=False):
    root = torch.nn.Module()
    root.register_buffer("scale", torch.linspace(0.75, 1.25, 32))
    root.register_buffer("matrix", torch.randn(32, 8, generator=torch.Generator().manual_seed(51)))
    graph = torch.fx.Graph()
    value = graph.placeholder("value")
    if mask:
        root.register_buffer("fixed_mask", torch.arange(32, dtype=torch.int64) % 2)
        fixed_mask = graph.get_attr("fixed_mask")
        boolean = graph.call_function(torch.ops.aten._to_copy.default, (fixed_mask,), {"dtype": torch.bool})
        value = graph.call_function(torch.ops.aten.mul.Tensor, (value, boolean))
    scale = graph.get_attr("scale")
    norm = graph.call_function(torch.ops.aten.native_layer_norm.default, (value, [32], scale, None, 1e-6))
    normalized = graph.call_function(operator.getitem, (norm, 0))
    matrix = graph.get_attr("matrix")
    boundary = graph.call_function(torch.ops.aten.mm.default, (normalized, matrix))
    tail = graph.call_function(torch.ops.aten.mul.Tensor, (boundary, 0.5))
    graph.output(tail)
    return torch.fx.GraphModule(root, graph), boundary.name


def test_prefix_uses_only_ancestors_and_borrows_original_coefficients():
    source, boundary = program()
    original = str(source.graph)
    prefix = probe.extract_prefix(source, boundary)
    values = torch.randn(3, 32, generator=torch.Generator().manual_seed(7))
    assert prefix.scale is source.scale and prefix.matrix is source.matrix
    assert all(node.target != torch.ops.aten.mul.Tensor for node in prefix.graph.nodes)
    assert torch.equal(prefix(values) * 0.5, source(values))
    assert str(source.graph) == original


@pytest.mark.parametrize("boundary", ["missing", "value", "scale"])
def test_prefix_rejects_missing_or_nonoperation_boundary(boundary):
    source, _ = program()
    with pytest.raises(ValueError, match="boundary"):
        probe.extract_prefix(source, boundary)


def test_coefficient_audit_keeps_native_dtype_and_bytes():
    source, _ = program()
    source.register_buffer("fixed_mask", torch.ones(3, dtype=torch.int32))
    before = probe.weight_digests(source)
    source.fixed_mask = source.fixed_mask.float()
    after = probe.weight_digests(source)
    assert before["fixed_mask"] != after["fixed_mask"]


def test_qualified_prefix_exports_standard_fp32_variants_and_preserves_source(tmp_path):
    source, boundary = program()
    values = torch.randn(2, 32, generator=torch.Generator().manual_seed(120)) * 7
    expected = source(values).detach()
    old_scale, old_matrix = source.scale.clone(), source.matrix.clone()
    report = probe.probe_prefix(source, values, expected, tmp_path, boundary=boundary)
    assert report["context_qualified"] and report["complete"]
    assert [row["variant"] for row in report["cases"]] == list(probe.VARIANTS)
    assert report["cases"][0]["native_wrapper_bits_equal"]
    assert report["cases"][0]["native_whole_encoder_exact_difference_count"] == 0
    assert not report["full_encoder_validated"] and not report["gpu_validated"]
    assert torch.equal(source.scale, old_scale) and torch.equal(source.matrix, old_matrix)
    for row in report["cases"]:
        model = onnx.load(tmp_path / row["graph"])
        onnx.checker.check_model(model)
        assert all(node.domain in ("", "ai.onnx") for node in model.graph.node)
        assert all(value.data_type != onnx.TensorProto.DOUBLE for value in model.graph.initializer)
        actual = np.fromfile(tmp_path / row["capture"], dtype="<f4")
        native = np.fromfile(tmp_path / "native-prefix-fp32.bin", dtype="<f4")
        difference = np.abs(actual.astype(np.float64) - native)
        assert int((difference > 1e-5 + 1e-4 * np.abs(native.astype(np.float64))).sum()) == row["metrics"]["mismatched_elements"]


def test_original_final_bits_are_a_prerequisite_before_export(tmp_path, monkeypatch):
    source, boundary = program()
    values = torch.zeros(1, 32)
    def forbidden(*args, **kwargs):
        pytest.fail("unqualified native context must not be exported as a frozen reference")
    monkeypatch.setattr(probe, "export_audio_native_onnx", forbidden)
    report = probe.probe_prefix(source, values, torch.ones(1, 8), tmp_path, boundary=boundary)
    assert not report["context_qualified"] and not report["complete"]
    assert report["cases"] == [] and not report["full_encoder_validated"]


def test_saved_int64_mask_bool_copy_exports_without_changing_native_bits(tmp_path):
    source, boundary = program(mask=True)
    values = torch.randn(2, 32, generator=torch.Generator().manual_seed(81))
    report = probe.probe_prefix(source, values, source(values).detach(), tmp_path, boundary=boundary)
    assert report["complete"] and report["cases"][0]["native_wrapper_bits_equal"]
    assert "_to_copy" in str(source.graph)


def test_copy_translation_rejects_unqualified_dtype_changes():
    source, boundary = program(mask=True)
    copy = next(node for node in source.graph.nodes if node.target == torch.ops.aten._to_copy.default)
    copy.kwargs = {"dtype": torch.float32}
    source.recompile()
    prefix = probe.extract_prefix(source, boundary)
    with pytest.raises(ValueError, match="mask"):
        probe.PrefixExport(prefix, "native")(torch.zeros(1, 32))


def test_refuses_overwriting_prior_evidence(tmp_path):
    path = tmp_path / "frozen-audio-prefix-onnx.json"
    path.write_text("original evidence")
    source, boundary = program()
    with pytest.raises(ValueError, match="overwrite"):
        probe.probe_prefix(source, torch.zeros(1, 32), torch.zeros(1, 8), tmp_path, boundary=boundary)
    assert path.read_text() == "original evidence"


def test_workflow_is_opt_in_and_downloads_only_frozen_graph_and_one_fixture():
    path = Path(__file__).resolve().parents[2] / ".github/workflows/frozen-audio-prefix-onnx.yml"
    text = path.read_text()
    job = yaml.safe_load(text)["jobs"]["probe"]
    assert "frozen-audio-prefix-onnx" in job["if"] and job["timeout-minutes"] <= 20
    assert "11814567883" in text and "frozen_audio_prefix_onnx" in text
    assert "HF_TOKEN" not in text and "snapshot_download" not in text
    assert "if: always()" in text
