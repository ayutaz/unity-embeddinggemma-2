import numpy as np
import pytest
import torch

from embeddinggemma_tools import frozen_audio_prefix_onnx as probe


VARIANT = 'ordered_original_stem'


def fixture():
    class Module(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer('first', torch.randn(128, 1, 3, 3, generator=torch.Generator().manual_seed(391)) * 0.1)
            self.register_buffer('second', torch.randn(32, 128, 3, 3, generator=torch.Generator().manual_seed(392)) * 0.01)
            self.register_buffer('scale0', torch.linspace(0.8, 1.2, 128))
            self.register_buffer('scale1', torch.linspace(0.8, 1.2, 32))
            self.register_buffer('projection', torch.randn(256, 4, generator=torch.Generator().manual_seed(393)) * 0.1)
        def forward(self, x):
            x = x.unsqueeze(1)
            for weight, scale, width in [(self.first, self.scale0, 128), (self.second, self.scale1, 32)]:
                x = torch.ops.aten.convolution.default(x, weight, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
                x = torch.ops.aten.native_layer_norm.default(x.permute(0, 2, 3, 1), [width], scale, None, 1e-6)[0]
                x = x.permute(0, 3, 1, 2).contiguous().relu()
            x = x.permute(0, 2, 3, 1).contiguous().reshape(8, 256).repeat(35, 1)
            return torch.mm(x, self.projection)
    x = torch.randn(1, 31, 32, generator=torch.Generator().manual_seed(394)) * 0.1
    exported = torch.export.export(Module(), (x,)).run_decompositions({}).module()
    source = torch.fx.GraphModule(exported, exported.graph)
    boundary = next(n.name for n in source.graph.nodes if n.target == torch.ops.aten.mm.default)
    return source, x, boundary


def test_explicit_stem_combines_only_qualified_conv_norm_and_projection_with_actual_onnx(tmp_path):
    source, x, boundary = fixture()
    before, weights = str(source.graph), probe.weight_digests(source)
    report = probe.probe_prefix(source, x, probe.tensor(source(x)), tmp_path, boundary=boundary,
                                variants=('native', VARIANT))
    assert report['complete'] and report['native_wrapper_control_bits_equal']
    candidate = report['cases'][1]
    assert candidate['convolution_replacements'] == candidate['normalization_replacements'] == 2
    assert candidate['linear_replacements'] == 1 and candidate['rms_mean_replacements'] == 0
    assert candidate['convolution_native_execution'] == 'original_conv2d_passthrough_not_fixed_order_execution'
    a = np.fromfile(tmp_path / candidate['capture'], dtype='<f4').reshape(280, 4)
    np.testing.assert_allclose(a, probe.tensor(source(x)).numpy(), atol=1e-5, rtol=1e-4)
    assert str(source.graph) == before and probe.weight_digests(source) == weights
    assert not report['full_encoder_validated'] and not report['gpu_validated']


def test_new_stem_candidate_is_not_a_default_or_a_partial_unaudited_convolution_rewrite():
    assert VARIANT not in probe.VARIANTS and VARIANT in probe.EXPERIMENTAL_VARIANTS
    source, _, _ = fixture()
    conv = next(n for n in source.graph.nodes if n.target == torch.ops.aten.convolution.default)
    conv.args = (*conv.args[:3], [1, 1], *conv.args[4:])
    with pytest.raises(ValueError, match='stem convolution'):
        probe.PrefixExport(source, VARIANT)


def test_stem_cli_rejects_any_later_boundary_before_loading_large_model(tmp_path, monkeypatch):
    def forbidden(*args): pytest.fail('Invalid stem boundary must be rejected before model download/load')
    monkeypatch.setattr(probe, 'load_frozen_input', forbidden)
    monkeypatch.setenv('GITHUB_SHA', 'a' * 40)
    with pytest.raises(ValueError, match='subsample'):
        probe.main(['--input', str(tmp_path), '--output', str(tmp_path / 'out'),
                    '--variant', VARIANT, '--stage', 'layer_0'])
