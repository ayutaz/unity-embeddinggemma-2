import onnx
import numpy as np
import pytest
import torch
from onnx.reference import ReferenceEvaluator

from embeddinggemma_tools import export
from test_audio_encoder import audio_models, mask


def native_export():
    value=getattr(export,"export_audio_native_pt2",None)
    assert callable(value),"Native audio export preserving SiLU is missing."
    return value


def onnx_export():
    value=getattr(export,"export_audio_native_onnx",None)
    assert callable(value),"Standard Swish audio ONNX export is missing."
    return value


@pytest.mark.parametrize("shape",[(17,),(1,2,4096)])
def test_native_audio_graph_preserves_vector_silu_exactly(tmp_path,shape):
    save=native_export()
    class Probe(torch.nn.Module):
        def validate_inputs(self,x):pass
        def forward(self,x):return torch.nn.functional.silu(x)
    x=torch.randn(shape,generator=torch.Generator().manual_seed(412))*2
    save(Probe(),x,tmp_path/"native.pt2")
    graph=torch.export.load(tmp_path/"native.pt2")
    assert 'aten.silu.default' in {str(node.target) for node in graph.graph.nodes if node.op=='call_function'}
    assert torch.equal(graph.module()(x),torch.nn.functional.silu(x))


def test_standard_division_silu_export_supports_legacy_limit_without_custom_runtime_domain(tmp_path):
    save=onnx_export()
    class Probe(torch.nn.Module):
        def forward(self,x):return torch.nn.functional.silu(x)+x*0.25
    x=torch.randn(17,generator=torch.Generator().manual_seed(412))*2
    path=tmp_path/"probe.onnx";save(Probe(),x,path)
    model=onnx.load(path);onnx.checker.check_model(model)
    assert all(node.domain in ('','ai.onnx') for node in model.graph.node)
    assert not any(node.op_type=='Swish' for node in model.graph.node)
    assert {'Neg', 'Exp', 'Add', 'Div'} <= {node.op_type for node in model.graph.node}
    assert next(op.version for op in model.opset_import if op.domain=='')==24
    expected=Probe()(x).detach().numpy()
    actual=ReferenceEvaluator(model).run(None,{'input':x.numpy()})[0]
    __import__('numpy').testing.assert_allclose(actual,expected,rtol=1e-4,atol=1e-5)


def test_native_full_tiny_encoder_onnx_matches_the_official_features(tmp_path):
    save=onnx_export()
    audio,projection=audio_models()
    frame_mask=mask(25,17)
    wrapper=export.AudioEncoderExport(audio,projection,frame_mask)
    values=torch.randn(1,25,8,generator=torch.Generator().manual_seed(9))*frame_mask[:,:,None]
    path=tmp_path/"audio.onnx";save(wrapper,values,path)
    model=onnx.load(path);onnx.checker.check_model(model)
    assert all(node.domain in ('','ai.onnx') for node in model.graph.node)
    assert not any(node.op_type=='Swish' for node in model.graph.node)
    with torch.no_grad():
        native=audio(input_features=values,attention_mask=frame_mask.bool())
        expected=projection(native.last_hidden_state)[native.attention_mask].numpy()
    actual=ReferenceEvaluator(model).run(None,{'input':values.numpy()})[0]
    __import__('numpy').testing.assert_allclose(actual,expected,rtol=1e-4,atol=1e-5)


def test_default_audio_export_keeps_native_silu_for_the_standard_swish_route(tmp_path):
    class Probe(torch.nn.Module):
        def validate_inputs(self,x):pass
        def forward(self,x):return torch.nn.functional.silu(x)
    x=torch.randn(17,generator=torch.Generator().manual_seed(412))*2
    path=tmp_path/"default.pt2"
    export.export_audio_pt2(Probe(),x,path)
    targets={str(node.target) for node in torch.export.load(path).graph.nodes if node.op=='call_function'}
    assert 'aten.silu.default' in targets,"Audio math must keep the native kernel; Sentis uses its standard ONNX Swish representation."


def test_onnx_activation_preserves_fp32_division_instead_of_reciprocal_multiplication(tmp_path):
    class Probe(torch.nn.Module):
        def forward(self, x):
            return torch.nn.functional.silu(x)
    values = torch.randn(257, generator=torch.Generator().manual_seed(412)) * 3
    source = values.numpy()
    denominator = np.float32(1) + np.exp(-source)
    expected = source / denominator
    reciprocal_multiply = source * (np.float32(1) / denominator)
    assert np.any(expected.view(np.uint32) != reciprocal_multiply.view(np.uint32)), "Fixture must expose the rounding distinction"
    path = tmp_path / 'division-silu.onnx'
    onnx_export()(Probe(), values, path)
    actual = ReferenceEvaluator(str(path)).run(None, {'input': source})[0]
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
