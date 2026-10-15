import json
from pathlib import Path
import numpy as np
import pytest
import torch

from embeddinggemma_tools.frozen_audio_prefix_onnx import FrozenStageReference


def candidate():
    from embeddinggemma_tools import frozen_audio_chained_onnx
    return frozen_audio_chained_onnx


def fixture():
    class Module(torch.nn.Module):
        def __init__(self):
            super().__init__()
            for i in range(3): self.register_buffer(f'w{i}', torch.eye(4) * (i + 1))
        def forward(self, x):
            x = x * 2
            x = torch.mm(x, self.w0)
            x = torch.mm(x, self.w1)
            return torch.mm(x, self.w2)
    x = torch.arange(1, 17, dtype=torch.float32).reshape(4, 4)
    exported = torch.export.export(Module(), (x,)).run_decompositions({}).module()
    source = torch.fx.GraphModule(exported, exported.graph)
    boundaries = [next(n.name for n in source.graph.nodes if n.target == torch.ops.aten.mul.Tensor)]
    boundaries += [n.name for n in source.graph.nodes if n.target == torch.ops.aten.mm.default]
    return source, x, boundaries


def frozen(value):
    return FrozenStageReference(value, dict(stage='synthetic_unit_only'))


def test_connected_partitions_keep_actual_onnx_carry_and_compare_only_original_final_reference(tmp_path):
    probe = candidate()
    source, x, boundaries = fixture()
    before = str(source.graph)
    report = probe.probe_chained_segments(source, frozen(x * 2), frozen(source(x)), tmp_path, boundaries=boundaries)
    assert report['complete'] and report['native_chain_control_bits_equal'] and report['context_qualified']
    assert report['reference_sha256'] == report['native_sha256']
    np.testing.assert_array_equal(np.fromfile(tmp_path / 'reference-output-fp32.bin', dtype='<f4'), source(x).numpy().reshape(-1))
    assert not report['whole_encoder_validated'] and not report['gpu_validated']
    row = report['cases'][0]
    assert row['metrics']['mismatched_elements'] == 0 and len(row['parts']) == 3
    for a, b in zip(row['parts'], row['parts'][1:]):
        assert a['output_sha256'] == b['input_sha256']
    assert row['actual_carry'] == 'previous_onnx_output_without_frozen_stage_reset'
    assert str(source.graph) == before


def test_wrong_first_onnx_output_is_carried_to_end_without_replacing_it_with_native_or_goldens(tmp_path, monkeypatch):
    probe = candidate()
    source, x, boundaries = fixture()
    real = probe.run_flat_reference
    calls = []
    def wrong_first(evaluator, feeds):
        calls.append(feeds['input'].copy())
        return [np.zeros((4, 4), dtype=np.float32)] if len(calls) == 1 else real(evaluator, feeds)
    monkeypatch.setattr(probe, 'run_flat_reference', wrong_first)
    report = probe.probe_chained_segments(source, frozen(x * 2), frozen(source(x)), tmp_path, boundaries=boundaries)
    assert report['complete'] and report['native_chain_control_bits_equal']
    assert report['cases'][0]['metrics']['mismatched_elements'] == 16
    assert np.count_nonzero(calls[1]) == 0 and np.count_nonzero(calls[2]) == 0


def test_chained_failure_preserves_partial_receipt_without_success_claim(tmp_path, monkeypatch):
    probe = candidate()
    source, x, boundaries = fixture()
    real = probe.run_flat_reference
    calls = []
    def fail_second(evaluator, feeds):
        calls.append(1)
        if len(calls) == 2: raise ValueError('unit-test stopped second partition')
        return real(evaluator, feeds)
    monkeypatch.setattr(probe, 'run_flat_reference', fail_second)
    with pytest.raises(ValueError, match='second partition'):
        probe.probe_chained_segments(source, frozen(x * 2), frozen(source(x)), tmp_path, boundaries=boundaries)
    report = json.loads((tmp_path / 'frozen-audio-chained-onnx.json').read_text())
    assert not report['complete'] and len(report['cases'][0]['parts']) == 1
    assert report['diagnostic_error']['type'] == 'ValueError'


def test_chained_probe_refuses_overwrite_and_unaudited_endpoints(tmp_path):
    probe = candidate()
    source, x, boundaries = fixture()
    with pytest.raises(ValueError, match='audited'):
        probe.probe_chained_segments(source, x, frozen(source(x)), tmp_path, boundaries=boundaries)
    path = tmp_path / 'frozen-audio-chained-onnx.json'; path.write_text('original evidence')
    with pytest.raises(ValueError, match='overwrite'):
        probe.probe_chained_segments(source, frozen(x * 2), frozen(source(x)), tmp_path, boundaries=boundaries)
    assert path.read_text() == 'original evidence'


def test_native_control_preserves_transposed_boundary_layout(tmp_path):
    probe = candidate()
    class Module(torch.nn.Module):
        def __init__(self): super().__init__(); self.register_buffer('weight', torch.eye(4) * 2)
        def forward(self, x):
            x = x * 2
            x = torch.mm(x, self.weight).permute(1, 0)
            return torch.mm(x, self.weight)
    x = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    exported = torch.export.export(Module(), (x,)).run_decompositions({}).module()
    source = torch.fx.GraphModule(exported, exported.graph)
    start = next(n.name for n in source.graph.nodes if n.target == torch.ops.aten.mul.Tensor)
    middle = next(n.name for n in source.graph.nodes if n.target == torch.ops.aten.permute.default)
    end = [n.name for n in source.graph.nodes if n.target == torch.ops.aten.mm.default][-1]
    report = probe.probe_chained_segments(source, frozen(x * 2), frozen(source(x)), tmp_path,
                                        boundaries=[start, middle, end])
    assert report['native_chain_control_bits_equal']
    assert report['cases'][0]['parts'][1]['native_input_strides'] == [1, 4]
    assert report['cases'][0]['metrics']['exact_difference_count'] == 0


def test_workflow_chain_mode_keeps_full_scope_and_does_not_fetch_middle_stage_goldens():
    p = Path(__file__).resolve().parents[2] / '.github/workflows/frozen-audio-prefix-onnx.yml'
    text = p.read_text(encoding='utf-8')
    assert 'chained_probe:' in text and 'CHAINED_PROBE' in text
    assert 'embeddinggemma_tools.frozen_audio_chained_onnx' in text
    assert 'timeout-minutes: 20' in text
    assert '--member "audio-runtime-stages/frames-1119-valid-1119/silence-stage-${stage_index}-fp32.bin"' in text


@pytest.mark.parametrize('variant', ['native', 'ordered_static_linear_all_bmm', 'ordered_original_primitives'])
def test_chain_cli_requires_immutable_job_source(tmp_path, monkeypatch, variant):
    monkeypatch.delenv('GITHUB_SHA', raising=False)
    with pytest.raises(ValueError, match='provenance'):
        candidate().main(['--input', str(tmp_path), '--output', str(tmp_path / 'out'),
                          '--frozen-stage-source', str(tmp_path), '--variant', variant])


def test_numeric_capture_upload_precedes_optional_models_and_chain_omits_model_archive():
    import yaml
    p = Path(__file__).resolve().parents[2] / '.github/workflows/frozen-audio-prefix-onnx.yml'
    steps = yaml.safe_load(p.read_text(encoding='utf-8'))['jobs']['probe']['steps']
    uploads = [s for s in steps if s.get('with', {}).get('name', '').startswith('frozen-audio-prefix-onnx-')]
    assert '*.onnx' not in uploads[0]['with']['path'], 'Numeric evidence must not wait for the large model archive'
    assert len(uploads) == 2 and '*.onnx' in uploads[1]['with']['path']
    assert "env.CHAINED_PROBE != 'true'" in uploads[1]['if']
