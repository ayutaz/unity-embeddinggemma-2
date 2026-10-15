import numpy as np
import onnx
from onnx import numpy_helper
from onnx.reference import ReferenceEvaluator
import pytest
import torch
from embeddinggemma_tools import export,refine_layernorm_onnx
from test_refine_scalar_mean_onnx import graphs


def required():
    function=getattr(refine_layernorm_onnx,'refine_welford_scalar_moments',None)
    assert callable(function),'Audited scalar moment FMA rewrite is missing'
    return function


@pytest.mark.parametrize('width',[32,128])
def test_scalar_moment_rewrite_keeps_originals_and_matches_independent_opt_in_export(tmp_path,width):
    model,template,values,source=graphs(tmp_path,width=width)
    before=[v.SerializeToString() for v in model.graph.initializer]
    report=required()(model,template,prefix='/',width=width,expected_count=8)
    assert report['replaced_scalar_moment_count']==8 and report['existing_initializer_bytes_preserved']
    assert [v.SerializeToString() for v in model.graph.initializer[:len(before)]]==before
    onnx.checker.check_model(model)
    actual=ReferenceEvaluator(model).run(None,{'input':values.numpy()})[0]
    expected=export.TorchWelfordLayerNorm(source,refine_inverse=True,fuse_scalar_moment=True)(values)
    np.testing.assert_array_equal(actual,expected.detach().numpy())


@pytest.mark.parametrize('problem',['count','prefix','width','fp64','coefficient','axis','previous_count'])
def test_scalar_moment_rewrite_rejects_unproved_scope_without_mutation(tmp_path,problem):
    model,template,_,_=graphs(tmp_path);kwargs=dict(prefix='/',width=32,expected_count=8)
    if problem=='count':kwargs['expected_count']=7
    elif problem=='prefix':kwargs['prefix']='/missing/'
    elif problem=='width':kwargs['width']=128
    elif problem=='fp64':template.graph.input[0].type.tensor_type.elem_type=onnx.TensorProto.DOUBLE
    else:
        producers={value:node for node in model.graph.node for value in node.output}
        if problem=='axis':
            sliced=next(n for n in model.graph.node if n.op_type=='Slice' and numpy_helper.to_array(producers[n.input[1]].attribute[0].t).item()==7)
            producers[sliced.input[3]].attribute[0].t.CopyFrom(numpy_helper.from_array(np.array([0],dtype=np.int64)))
        else:
            wanted=np.float32(1/7) if problem=='coefficient' else np.float32(28)
            matches=[a for n in model.graph.node if n.op_type=='Constant' for a in n.attribute if a.name=='value' and numpy_helper.to_array(a.t).dtype==np.float32 and numpy_helper.to_array(a.t).size==1 and numpy_helper.to_array(a.t).item()==wanted]
            assert matches
            for attribute in matches:attribute.t.CopyFrom(numpy_helper.from_array(np.array(.2,dtype=np.float32)))
    before=model.SerializeToString()
    with pytest.raises(ValueError):required()(model,template,**kwargs)
    assert model.SerializeToString()==before
