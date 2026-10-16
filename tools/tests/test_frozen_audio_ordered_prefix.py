"""Opt-in original-input coverage; these small tests do not qualify a model."""
from pathlib import Path

import numpy as np
import pytest
import torch

from embeddinggemma_tools import frozen_audio_prefix_onnx as probe


VARIANT = "ordered_static_linear_refined"


def fixture(rows=280):
    class Module(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("scale", torch.linspace(0.8, 1.2, 32))
            self.register_buffer("weight", torch.randn(32, 3, generator=torch.Generator().manual_seed(320)) * 0.1)

        def forward(self, values):
            hidden = torch.ops.aten.native_layer_norm.default(values, [32], self.scale, None, 1e-6)[0]
            return torch.mm(hidden, self.weight)

    values = torch.randn(rows, 32, generator=torch.Generator().manual_seed(321))
    exported = torch.export.export(Module(), (values,)).run_decompositions({}).module()
    source = torch.fx.GraphModule(exported, exported.graph)
    boundary = next(n.name for n in source.graph.nodes if n.target == torch.ops.aten.mm.default)
    return source, values, boundary


def test_ordered_prefix_exports_actual_fp32_graph_and_uses_streaming_without_source_changes(tmp_path, monkeypatch):
    source, values, boundary = fixture()
    graph, weights = str(source.graph), probe.weight_digests(source)
    runs = []
    original = getattr(probe, "run_flat_reference", None)
    assert callable(original), "Ordered prefix requires bounded-memory reference execution"

    def record(evaluator, feeds):
        runs.append(evaluator)
        return original(evaluator, feeds)

    monkeypatch.setattr(probe, "run_flat_reference", record)
    report = probe.probe_prefix(source, values, source(values).detach(), tmp_path,
                                boundary=boundary, variants=("native", VARIANT))
    assert report["complete"] and report["context_qualified"] and len(runs) == 2
    assert not report["full_encoder_validated"] and not report["gpu_validated"]
    row = report["cases"][1]
    assert row["linear_replacements"] == 1 and row["normalization_replacements"] == 1
    assert row["rms_mean_replacements"] == 0
    assert row["ordered_linear_native_execution"] == "original_MatMul_addmm_passthrough_not_fixed_order_execution"
    wrapper = probe.PrefixExport(probe.extract_prefix(source, boundary), VARIANT)
    expected = wrapper(values).detach().numpy()
    actual = np.fromfile(tmp_path / row["capture"], dtype="<f4").reshape(expected.shape)
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-4)
    assert str(source.graph) == graph and probe.weight_digests(source) == weights


def test_ordered_prefix_remains_opt_in_and_rejects_unqualified_row_count():
    assert probe.VARIANTS == ("native", "refined_moment", "refined_mean_moment")
    assert VARIANT in probe.EXPERIMENTAL_VARIANTS
    source, _, boundary = fixture(rows=13)
    with pytest.raises(ValueError, match="280-row"):
        probe.PrefixExport(probe.extract_prefix(source, boundary), VARIANT)
    path = Path(__file__).resolve().parents[2] / ".github/workflows/frozen-audio-prefix-onnx.yml"
    assert VARIANT in path.read_text(encoding="utf-8")


@pytest.mark.parametrize("stage,count", [("subsample", 1), ("layer_0", 10), ("layer_11", 109), ("projected_features", 111)])
def test_ordered_prefix_pinned_coverage_includes_original_stem_projection(stage, count):
    assert probe.expected_linear_count(stage) == count


def test_ordered_prefix_does_not_bypass_original_context_gate(tmp_path, monkeypatch):
    source, values, boundary = fixture()
    def forbidden(*args, **kwargs):
        pytest.fail("The new lowering must retain the original native-context prerequisite")
    monkeypatch.setattr(probe, "export_audio_native_onnx", forbidden)
    report = probe.probe_prefix(source, values, torch.ones(280, 3), tmp_path,
                                boundary=boundary, variants=(VARIANT,))
    assert not report["complete"] and not report["reference_qualified"]
    assert not report["context_qualified"] and report["cases"] == []


def test_ordered_prefix_refuses_overwriting_experimental_graph(tmp_path):
    path = tmp_path / (VARIANT + ".onnx")
    path.write_bytes(b"original evidence")
    source, values, boundary = fixture()
    with pytest.raises(ValueError, match="overwrite"):
        probe.probe_prefix(source, values, source(values).detach(), tmp_path,
                           boundary=boundary, variants=(VARIANT,))
    assert path.read_bytes() == b"original evidence"
