import hashlib
import importlib

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto
from onnx.reference import ReferenceEvaluator
import pytest
import torch

from embeddinggemma_tools import export


def rewrite():
    try: module=importlib.import_module("embeddinggemma_tools.refine_layernorm_node_onnx")
    except ModuleNotFoundError: pytest.fail("Audited original LayerNormalization node replacement is missing")
    return module.refine_layernorm_node


@pytest.fixture
def context(tmp_path):
    def build(width=32):
        values=torch.randn(1,2,width,generator=torch.Generator().manual_seed(71))
        scale=torch.randn(width,generator=torch.Generator().manual_seed(72))
        graph=helper.make_graph([helper.make_node("LayerNormalization",["input","scale","bias"],["output"],name="norm",axis=-1,epsilon=1e-6)],
            "norm",[helper.make_tensor_value_info("input",TensorProto.FLOAT,list(values.shape))],
            [helper.make_tensor_value_info("output",TensorProto.FLOAT,list(values.shape))],
            initializer=[numpy_helper.from_array(scale.numpy(),name="scale"),numpy_helper.from_array(np.zeros(width,dtype=np.float32),name="bias")])
        model=helper.make_model(graph,opset_imports=[helper.make_opsetid("",24)],ir_version=9)
        class Template(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.norm=export.TorchWelfordLayerNorm(torch.nn.LayerNorm(width,eps=1e-6,bias=False),refine_inverse=True,fuse_scalar_moment=True)
            def forward(self,x,weight):return self.norm(x)*weight
        module=Template().eval();path=tmp_path/(str(width)+".onnx")
        torch.onnx.export(module,(values,scale),path,dynamo=False,opset_version=20,input_names=["x","scale"],output_names=["out"])
        template=onnx.version_converter.convert_version(onnx.load(path),24)
        return model,template,values,scale,module,hashlib.sha256(scale.numpy().astype("<f4").tobytes()).hexdigest()
    return build


@pytest.mark.parametrize("width",[32,128])
def test_layernorm_replacement_preserves_original_coefficients_and_explicit_fp32_order(context,width):
    model,template,values,scale,module,digest=context(width)
    before=[weight.SerializeToString() for weight in model.graph.initializer]
    result=rewrite()(model,template,node_name="norm",expected_scale_sha256=digest,expected_epsilon=1e-6)
    assert result["replaced_layernorm_count"]==1 and result["existing_initializer_bytes_preserved"]
    assert [weight.SerializeToString() for weight in model.graph.initializer[:len(before)]]==before
    onnx.checker.check_model(model)
    actual=ReferenceEvaluator(model).run(None,{"input":values.numpy()})[0]
    np.testing.assert_array_equal(actual,module(values,scale).detach().numpy())


@pytest.mark.parametrize("problem",["missing","hash","shape","fp64","bias","epsilon","axis","duplicate"])
def test_unqualified_layernorm_node_or_template_is_rejected_before_mutation(context,problem):
    model,template,_,_,_,digest=context()
    name="missing" if problem=="missing" else "norm"
    if problem=="hash":digest="0"*64
    elif problem=="shape":template.graph.input[0].type.tensor_type.shape.dim[1].dim_value=3
    elif problem=="fp64":template.graph.input[0].type.tensor_type.elem_type=TensorProto.DOUBLE
    elif problem=="bias":model.graph.initializer[1].CopyFrom(numpy_helper.from_array(np.ones(32,dtype=np.float32),name="bias"))
    elif problem=="epsilon":next(a for a in model.graph.node[0].attribute if a.name=="epsilon").f=0.001
    elif problem=="axis":next(a for a in model.graph.node[0].attribute if a.name=="axis").i=0
    elif problem=="duplicate":model.graph.node.append(model.graph.node[0])
    before=model.SerializeToString()
    with pytest.raises(ValueError):rewrite()(model,template,node_name=name,expected_scale_sha256=digest,expected_epsilon=1e-6)
    assert model.SerializeToString()==before
