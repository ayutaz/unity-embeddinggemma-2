import numpy as np
import onnx
from onnx import numpy_helper
from onnx.reference import ReferenceEvaluator
import pytest
import torch
from embeddinggemma_tools import export


def required():
    from embeddinggemma_tools import refine_layernorm_onnx
    function = getattr(refine_layernorm_onnx, 'refine_welford_scalar_means', None)
    assert callable(function), 'Audited scalar mean graph rewrite is missing'
    return function


def graphs(tmp_path,*,width=32):
    values = torch.randn(2, width, generator=torch.Generator().manual_seed(120))
    source = torch.nn.LayerNorm(width, eps=1e-6, bias=False)
    path = tmp_path/'norm.onnx'
    export.export_audio_native_onnx(export.TorchWelfordLayerNorm(source, refine_inverse=True), values, path)
    model = onnx.load(path)
    class Fma(torch.nn.Module):
        def forward(self, a, b, c): return export.fp32_fma(a, b, c)
    path = tmp_path/'fma.onnx'
    torch.onnx.export(Fma(), (torch.ones(2,1),)*3, path, dynamo=False, opset_version=20,
                      input_names=['a','b','c'], output_names=['output'])
    template = onnx.load(path)
    return model, template, values, source


def test_scalar_mean_rewrite_preserves_weights_and_matches_opt_in_fused_graph(tmp_path):
    model, template, values, source = graphs(tmp_path)
    before = [i.SerializeToString() for i in model.graph.initializer]
    report = required()(model, template, prefix='/', expected_count=8)
    onnx.checker.check_model(model)
    assert report['replaced_scalar_mean_count'] == 8
    assert [i.SerializeToString() for i in model.graph.initializer[:len(before)]] == before
    actual = ReferenceEvaluator(model).run(None, {'input':values.numpy()})[0]
    expected = export.TorchWelfordLayerNorm(source, refine_inverse=True, fuse_scalar_mean=True)(values)
    np.testing.assert_array_equal(actual, expected.detach().numpy())


@pytest.mark.parametrize('problem', ['count','prefix','coefficient','fp64','axis'])
def test_scalar_mean_rewrite_rejects_unproven_graph_before_mutation(tmp_path, problem):
    model, template, _, _ = graphs(tmp_path)
    prefix = '/' if problem != 'prefix' else '/missing/'
    count = 7 if problem == 'count' else 8
    if problem == 'coefficient':
        # All scalar 1/7 coefficients are part of the seventh lane update.
        found = False
        for node in model.graph.node:
            if node.op_type == 'Constant':
                for attribute in node.attribute:
                    if attribute.name == 'value':
                        value = numpy_helper.to_array(attribute.t)
                        if value.size == 1 and value.dtype == np.float32 and value.item() == np.float32(1/7):
                            attribute.t.CopyFrom(numpy_helper.from_array(np.array(.2,dtype=np.float32)))
                            found = True
        assert found
    elif problem == 'fp64':
        template.graph.input[0].type.tensor_type.elem_type = onnx.TensorProto.DOUBLE
    elif problem == 'axis':
        producers = {v:n for n in model.graph.node for v in n.output}
        sliced = next(n for n in model.graph.node if n.op_type == 'Slice'
                      and numpy_helper.to_array(producers[n.input[1]].attribute[0].t).item() == 7)
        producers[sliced.input[3]].attribute[0].t.CopyFrom(numpy_helper.from_array(np.array([0],dtype=np.int64)))
    before = model.SerializeToString()
    with pytest.raises(ValueError): required()(model, template, prefix=prefix, expected_count=count)
    assert model.SerializeToString() == before
