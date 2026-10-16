import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto
from onnx.reference import ReferenceEvaluator
import pytest
import torch

from embeddinggemma_tools import export


def required():
    from embeddinggemma_tools import refine_layernorm_onnx
    function = getattr(refine_layernorm_onnx, 'refine_layernorm_inverses', None)
    assert callable(function), 'Audited in-place ONNX inverse rewrite is missing'
    return function


def model():
    nodes = []
    for name in ('norm0', 'norm1'):
        nodes.extend([
            helper.make_node('Sqrt', ['input'], [name+'-root'], name=name+'/Sqrt'),
            helper.make_node('Reciprocal', [name+'-root'], [name+'-inverse'], name=name+'/Reciprocal'),
            helper.make_node('Mul', [name+'-inverse', 'weight'], [name+'-output'], name=name+'/Mul'),
        ])
    graph = helper.make_graph(nodes, 'two-norms', [helper.make_tensor_value_info('input', TensorProto.FLOAT, [1,2])],
        [helper.make_tensor_value_info(name+'-output', TensorProto.FLOAT, [1,2]) for name in ('norm0','norm1')],
        [numpy_helper.from_array(np.array([.5, 1.25], np.float32), name='weight')])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid('',24)])


def template(tmp_path):
    class Inverse(torch.nn.Module):
        def forward(self, values): return export.fp32_refined_rstd(values)
    path = tmp_path/'template.onnx'
    export.export_audio_native_onnx(Inverse(), torch.tensor([[5.]],dtype=torch.float32), path)
    return onnx.load(path)


def test_only_selected_inverses_change_and_all_existing_coefficients_stay_identical(tmp_path):
    run = required()
    original = model()
    weight = original.graph.initializer[0].SerializeToString()
    kept = {n.name:n.SerializeToString() for n in original.graph.node if n.name in ('norm0/Mul','norm1/Mul')}
    report = run(original, template(tmp_path), prefixes=('norm0','norm1'))
    onnx.checker.check_model(original)
    assert report['replaced_inverse_count'] == 2 and report['existing_initializer_bytes_preserved']
    assert original.graph.initializer[0].SerializeToString() == weight
    assert {n.name:n.SerializeToString() for n in original.graph.node if n.name in kept} == kept
    values = np.array([[4.93749,5.17808]],np.float32)
    roots = np.sqrt(values.astype(np.float64)).astype(np.float32)
    expected = (1/roots.astype(np.float64)).astype(np.float32)*np.array([.5,1.25],np.float32)
    for actual in ReferenceEvaluator(original).run(None,{'input':values}):
        np.testing.assert_array_equal(actual,expected)


@pytest.mark.parametrize('problem',['missing','shared_root','duplicate','bad_template','double_input','double_cast'])
def test_invalid_context_is_rejected_before_any_model_mutation(tmp_path,problem):
    run = required()
    original, replacement = model(), template(tmp_path)
    if problem=='missing':original.graph.node[4].name='not-a-target'
    elif problem=='shared_root':original.graph.node.append(helper.make_node('Identity',['norm0-root'],['extra']))
    elif problem=='duplicate':original.graph.node[4].name='norm0/Reciprocal'
    elif problem=='bad_template':replacement.graph.node[0].domain='untrusted.custom'
    elif problem=='double_input':replacement.graph.input[0].type.tensor_type.elem_type=TensorProto.DOUBLE
    else:replacement.graph.node.append(helper.make_node('Cast',[replacement.graph.input[0].name],['unexpected-double'],to=TensorProto.DOUBLE))
    before = original.SerializeToString()
    with pytest.raises(ValueError):run(original,replacement,prefixes=('norm0','norm1'))
    assert original.SerializeToString()==before


def test_template_tensor_names_cannot_collide_with_original_graph(tmp_path):
    run = required()
    original, replacement = model(), template(tmp_path)
    original.graph.initializer.append(numpy_helper.from_array(np.array(2.,np.float32),
        name='__eg_refined_norm0_'+replacement.graph.node[0].output[0]))
    run(original,replacement,prefixes=('norm0','norm1'))
    onnx.checker.check_model(original)
    names=[w.name for w in original.graph.initializer]
    assert len(names)==len(set(names))
