import operator
from pathlib import Path

import pytest
import torch
import yaml

from embeddinggemma_tools import audio_native_norm_replay as replay
from embeddinggemma_tools.export import TorchWelfordLayerNorm


def graph(*, width=32, statistic=0, epsilon=1e-6):
    root = torch.nn.Module()
    root.register_buffer("scale", torch.linspace(0.5, 1.5, width))
    nodes = torch.fx.Graph()
    value = nodes.placeholder("value")
    scale = nodes.get_attr("scale")
    norm = nodes.call_function(torch.ops.aten.native_layer_norm.default,
                               (value, [width], scale, None, epsilon))
    selected = nodes.call_function(operator.getitem, (norm, statistic))
    # A downstream operation must still execute, rather than returning golden values.
    output = nodes.call_function(torch.ops.aten.mul.Tensor, (selected, 0.375))
    nodes.output(output)
    return torch.fx.GraphModule(root, nodes)


def test_native_control_and_interventions_keep_original_graph_weights_and_expected(tmp_path):
    program = graph()
    values = torch.randn(8, 32, generator=torch.Generator().manual_seed(120)) * 7
    expected = program(values).detach().clone()
    before_graph = str(program.graph)
    before_weight = program.scale.clone()
    report = replay.replay_norm_variants(program, values, expected, tmp_path)
    assert [row["variant"] for row in report["cases"]] == list(replay.VARIANTS)
    assert report["cases"][0]["native_program_bits_equal"]
    assert report["cases"][0]["metrics"]["exact_difference_count"] == 0
    assert not report["full_encoder_validated"] and not report["gpu_validated"]
    assert torch.equal(expected, program(values))
    assert str(program.graph) == before_graph and torch.equal(program.scale, before_weight)
    source = type("Norm", (), dict(normalized_shape=(32,), weight=program.scale, bias=None, eps=1e-6))()
    wanted = TorchWelfordLayerNorm(source, refine_inverse=True, fuse_scalar_moment=True)(values) * 0.375
    capture = torch.frombuffer(bytearray((tmp_path / "refined_moment-final-fp32.bin").read_bytes()), dtype=torch.float32).reshape_as(wanted)
    assert torch.equal(capture, wanted)
    assert len(report["normalizations"]) == 1


def test_reference_is_only_compared_and_never_used_as_an_inference_output(tmp_path):
    program = graph()
    values = torch.zeros(1, 32)
    report = replay.replay_norm_variants(program, values, torch.ones(1, 32), tmp_path)
    assert report["cases"][0]["metrics"]["mismatched_elements"] == 32
    assert report["cases"][0]["native_program_bits_equal"]


def test_real_exported_tuple_output_is_unwrapped_without_changing_graph(tmp_path):
    class TwoNorms(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.first = torch.nn.LayerNorm(128, eps=1e-6, bias=False)
            self.second = torch.nn.LayerNorm(32, eps=1e-6, bias=False)
        def forward(self, values):
            first = torch.ops.aten.native_layer_norm.default(values, [128], self.first.weight, None, 1e-6)[0]
            return torch.ops.aten.native_layer_norm.default(first[..., :32], [32], self.second.weight, None, 1e-6)[0]
    values = torch.randn(2, 128, generator=torch.Generator().manual_seed(10))
    program = torch.export.export(TwoNorms(), (values,)).module()
    report = replay.replay_norm_variants(program, values, program(values).detach(), tmp_path)
    assert sorted(row["width"] for row in report["normalizations"]) == [32, 128]
    assert report["cases"][0]["native_program_bits_equal"]


def test_unpinned_reference_is_rejected_before_graph_deserialization(tmp_path, monkeypatch):
    (tmp_path / "audio-encoder-reference.json").write_text("{}")
    def unexpected(*args):
        pytest.fail("untrusted graph must never be loaded")
    monkeypatch.setattr(torch.export, "load", unexpected)
    with pytest.raises(ValueError, match="hash"):
        replay.load_frozen_input(tmp_path)


@pytest.mark.parametrize("statistic", [1, 2])
def test_rejects_live_mean_or_inverse_before_any_inference(tmp_path, statistic):
    with pytest.raises(ValueError, match="statistic"):
        replay.replay_norm_variants(graph(statistic=statistic), torch.zeros(1, 32), torch.zeros(1, 1), tmp_path)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("width,epsilon", [(64, 1e-6), (32, 1e-5)])
def test_rejects_unqualified_normalization_contracts(tmp_path, width, epsilon):
    with pytest.raises(ValueError, match="normalization"):
        replay.replay_norm_variants(graph(width=width, epsilon=epsilon), torch.zeros(1, width), torch.zeros(1, width), tmp_path)


def test_partial_failure_is_saved_and_not_accepted(tmp_path, monkeypatch):
    program = graph()
    def fail(*args):
        raise RuntimeError("injected candidate failure")
    monkeypatch.setattr(replay, "candidate_output", fail)
    with pytest.raises(RuntimeError, match="injected"):
        replay.replay_norm_variants(program, torch.zeros(1, 32), program(torch.zeros(1, 32)), tmp_path)
    import json
    report = json.loads((tmp_path / "audio-native-norm-replay.json").read_text())
    assert not report["complete"] and len(report["cases"]) == 1
    assert not report["full_encoder_validated"]


def test_rejects_output_directory_overwriting_previous_evidence(tmp_path):
    path = tmp_path / "audio-native-norm-replay.json"
    path.write_text("original evidence")
    with pytest.raises(ValueError, match="overwrite"):
        replay.replay_norm_variants(graph(), torch.zeros(1, 32), torch.zeros(1, 32), tmp_path)
    assert path.read_text() == "original evidence"


def test_workflow_is_opt_in_and_audits_original_graph_without_hf_download():
    path = Path(__file__).resolve().parents[2] / ".github/workflows/audio-native-norm-replay.yml"
    source = path.read_text()
    workflow = yaml.safe_load(source)
    job = workflow["jobs"]["probe"]
    assert "audio-native-norm-replay" in job["if"]
    assert job["timeout-minutes"] <= 20
    assert "11814567883" in source and "audio.pt2" in source
    assert "audio_native_norm_replay" in source and "if: always()" in source
    assert "snapshot_download" not in source and "HF_TOKEN" not in source
