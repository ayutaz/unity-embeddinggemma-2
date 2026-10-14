import pytest
import torch
import onnx
from onnx.reference import ReferenceEvaluator

from embeddinggemma_tools import export


def required(name):
    value = getattr(export, name, None)
    assert callable(value), f"Faithful FP32 LayerNorm arithmetic missing: {name}"
    return value


def test_fma_emulation_preserves_the_single_rounding_counterexample():
    fma = required("fp32_fma")
    a = torch.tensor([1.0000001192092896, 4097., -4097.])
    b = torch.tensor([0.9999998807907104, 4097., 4097.])
    c = torch.tensor([-1., -16785408., 16785408.])
    expected = (a.double() * b.double() + c.double()).float()
    assert not torch.equal(a * b + c, expected), "Expose the separate-rounding failure"
    assert torch.equal(fma(a, b, c), expected)


@pytest.mark.parametrize("width", [32, 128])
def test_welford8_layernorm_matches_independent_native_statistics_and_keeps_source(width):
    build = required("TorchWelfordLayerNorm")
    source = torch.nn.LayerNorm(width, eps=1e-6, bias=False).float().eval()
    values = torch.randn(3, width, generator=torch.Generator().manual_seed(14)) * 10
    source.weight.data.copy_(torch.linspace(0.8, 1.2, width))
    before = source.weight.detach().clone()
    candidate = build(source)
    actual = candidate(values)
    with torch.no_grad(): expected = source(values)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
    assert candidate.weight is source.weight and torch.equal(source.weight, before)
    assert actual.dtype == torch.float32


@pytest.mark.parametrize("width", [7, 64, 256])
def test_welford_candidate_rejects_unproven_normalized_widths(width):
    with pytest.raises(ValueError, match="32 or 128"):
        required("TorchWelfordLayerNorm")(torch.nn.LayerNorm(width))


def test_fma_counterexamples_cover_cancellation_and_finite_product_range():
    fma = required("fp32_fma")
    generator = torch.Generator().manual_seed(18)
    a = torch.randn(2048, generator=generator) * 4096
    b = torch.randn(2048, generator=generator) / 16
    c = -(a * b)
    expected = (a.double() * b.double() + c.double()).float()
    assert torch.equal(fma(a, b, c), expected)


@pytest.mark.parametrize("width", [32, 128])
def test_welford_export_uses_only_standard_fp32_ops_and_matches_native(tmp_path, width):
    build = required("TorchWelfordLayerNorm")
    source = torch.nn.LayerNorm(width, eps=1e-6, bias=False).float().eval()
    values = torch.randn(3, width, generator=torch.Generator().manual_seed(14)) * 10
    path = tmp_path / f"welford{width}.onnx"
    export.export_audio_native_onnx(build(source), values, path)
    graph = onnx.load(path)
    onnx.checker.check_model(graph)
    assert all(n.domain in ("", "ai.onnx") for n in graph.graph.node)
    assert all(w.data_type != onnx.TensorProto.DOUBLE for w in graph.graph.initializer)
    actual = ReferenceEvaluator(graph).run(None, {"input": values.numpy()})[0]
    with torch.no_grad(): expected = source(values).numpy()
    __import__('numpy').testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("fail", [False, True])
def test_opt_in_export_replaces_only_onnx_norms_and_restores_native_modules(tmp_path, monkeypatch, fail):
    class Probe(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = torch.nn.LayerNorm(32, eps=1e-6, bias=False)
        def forward(self, values): return self.norm(values)
    source = Probe().eval()
    norm = source.norm
    values = torch.randn(3,32,generator=torch.Generator().manual_seed(14))
    before = source(values).detach().clone()
    if fail:
        def broken(*args, **kwargs): raise RuntimeError("export failed")
        monkeypatch.setattr(torch.onnx, "export", broken)
        with pytest.raises(RuntimeError, match="export failed"):
            export.export_audio_native_onnx(source, values, tmp_path/'candidate.onnx', layernorm_algorithm='torch_welford8')
    else:
        path = tmp_path/'candidate.onnx'
        export.export_audio_native_onnx(source, values, path, layernorm_algorithm='torch_welford8')
        graph = onnx.load(path)
        assert not any(n.op_type == 'LayerNormalization' for n in graph.graph.node)
        actual = ReferenceEvaluator(graph).run(None, {'input': values.numpy()})[0]
        __import__('numpy').testing.assert_allclose(actual, before.numpy(), rtol=1e-4, atol=1e-5)
    assert source.norm is norm and torch.equal(source(values), before)


def test_complete_tiny_encoder_candidate_keeps_native_path_and_mask_contract(tmp_path):
    from transformers import Gemma4AudioConfig, Gemma4AudioModel
    from transformers.models.embedding_gemma2.modeling_embedding_gemma2 import EmbeddingGemma2MultimodalEmbedder
    from test_model import tiny_model
    config=Gemma4AudioConfig(hidden_size=16,num_hidden_layers=2,num_attention_heads=2,
        subsampling_conv_channels=[128,32],output_proj_dims=12,attention_chunk_size=4,
        attention_context_left=5,attention_context_right=2,conv_kernel_size=3)
    torch.manual_seed(29)
    audio=Gemma4AudioModel(config).float().eval()
    projection=EmbeddingGemma2MultimodalEmbedder(config,tiny_model().config).eval()
    mask=(torch.arange(25)[None]<17).long()
    values=torch.randn(1,25,128,generator=torch.Generator().manual_seed(14))*mask[:,:,None]
    norms=(audio.subsample_conv_projection.layer0.norm,audio.subsample_conv_projection.layer1.norm)
    with torch.no_grad():
        native=audio(input_features=values,attention_mask=mask.bool())
        expected=projection(native.last_hidden_state)[native.attention_mask].numpy()
    path=tmp_path/'audio-candidate.onnx'
    wrapper=export.AudioEncoderExport(audio,projection,mask)
    export.export_audio_native_onnx(wrapper,values,path,layernorm_algorithm='torch_welford8')
    assert audio.subsample_conv_projection.layer0.norm is norms[0]
    assert audio.subsample_conv_projection.layer1.norm is norms[1]
    graph=onnx.load(path)
    assert not any(n.op_type=='LayerNormalization' for n in graph.graph.node)
    actual=ReferenceEvaluator(graph).run(None,{'input':values.numpy()})[0]
    __import__('numpy').testing.assert_allclose(actual,expected,rtol=1e-4,atol=1e-5)
