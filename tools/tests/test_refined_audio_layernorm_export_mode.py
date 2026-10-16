import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator
import pytest
import torch

from embeddinggemma_tools import export,__main__ as cli


@pytest.mark.parametrize("width",[32,128])
def test_refined_moment_export_mode_preserves_borrowed_source_and_standard_fp32_math(tmp_path,width):
    norm=torch.nn.LayerNorm(width,eps=1e-6,bias=False).float().eval();wrapper=torch.nn.Sequential(norm)
    values=torch.randn(2,3,width,generator=torch.Generator().manual_seed(121))*5
    before=norm.weight.detach().clone();path=tmp_path/"refined.onnx"
    export.export_audio_native_onnx(wrapper,values,path,layernorm_algorithm="torch_welford8_refined_moment")
    assert wrapper[0] is norm and torch.equal(norm.weight,before)
    graph=onnx.load(path);onnx.checker.check_model(graph)
    assert all(node.domain in ("","ai.onnx") for node in graph.graph.node)
    assert all(weight.data_type!=onnx.TensorProto.DOUBLE for weight in graph.graph.initializer)
    expected=export.TorchWelfordLayerNorm(norm,refine_inverse=True,fuse_scalar_moment=True)(values).detach().numpy()
    actual=ReferenceEvaluator(graph).run(None,{"input":values.numpy()})[0]
    np.testing.assert_array_equal(actual,expected)


def test_cli_forwards_explicit_refined_moment_mode_without_changing_defaults(tmp_path,monkeypatch):
    calls=[]
    def run(*args,**kwargs):
        calls.append(kwargs)
        return dict(success=True,stage="audio_encoder_projector_only",profile_count=2,case_count=6,minimum_cosine=1,unity_validated=False,end_to_end_validated=False)
    monkeypatch.setattr(cli,"prepare_audio_encoder",run)
    for extra in ([],["--onnx-layernorm","torch_welford8_refined_moment"]):
        cli.main(["prepare-audio-encoder","--snapshot",str(tmp_path/"snapshot"),"--preprocessing",str(tmp_path/"pre"),"--output",str(tmp_path/"out"),"--native-silu",*extra])
    assert calls[0]["onnx_layernorm"]=="native"
    assert calls[1]["onnx_layernorm"]=="torch_welford8_refined_moment"
