import importlib

import numpy as np
import onnx
from onnx import helper as h, TensorProto as T
from onnx.reference import ReferenceEvaluator
import pytest
import torch


def module():return importlib.import_module('embeddinggemma_tools.fixed_shape_metadata')


@pytest.fixture
def padded(tmp_path):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__();self.conv=torch.nn.Conv1d(2,2,3,groups=2,bias=False)
        def forward(self,x):
            y=self.conv(torch.nn.functional.pad(x,(2,0))).transpose(1,2).clamp(-30,30)
            return y.square().mean(-1,keepdim=True)
    torch.manual_seed(17);path=tmp_path/'pad.onnx'
    torch.onnx.export(Model().eval(),(torch.ones(1,2,3),),path,dynamo=False,opset_version=20,input_names=['input'],output_names=['out'])
    return onnx.load(path)


def test_constant_pad_program_proves_unknown_outer_rms_shape_without_running_model(padded):
    source=padded;before_nodes=[n.SerializeToString() for n in source.graph.node];before_weights=[w.SerializeToString() for w in source.graph.initializer]
    model,report=module().infer_with_fixed_pads(source)
    infos={v.name:v for v in model.graph.value_info}
    mean=next(n for n in model.graph.node if n.op_type=='ReduceMean')
    assert [d.dim_value for d in infos[mean.input[0]].type.tensor_type.shape.dim]==[1,3,2]
    assert report['bound_pad_count']>=1 and report['node_initializer_bytes_preserved'] is True
    assert report['model_executed'] is False and report['gpu_validated'] is False
    assert before_nodes==[n.SerializeToString() for n in model.graph.node]
    assert before_weights==[w.SerializeToString() for w in model.graph.initializer]
    values=np.arange(6,dtype=np.float32).reshape(1,2,3)
    expected=ReferenceEvaluator(source).run(None,{'input':values})[0]
    actual=ReferenceEvaluator(model).run(None,{'input':values})[0]
    np.testing.assert_array_equal(actual,expected)


@pytest.mark.parametrize('problem',['dynamic_pads','conflict'])
def test_unproven_pad_geometry_does_not_overwrite_metadata(padded,problem):
    model=padded;pad=next(n for n in model.graph.node if n.op_type=='Pad')
    if problem=='dynamic_pads':
        model.graph.input.append(h.make_tensor_value_info('runtime_pads',T.INT64,[6]));pad.input[1]='runtime_pads'
    else:model.graph.value_info.append(h.make_tensor_value_info(pad.output[0],T.FLOAT,[1,2,99]))
    before=model.SerializeToString()
    with pytest.raises(ValueError):module().bind_fixed_pad_shapes(model)
    assert model.SerializeToString()==before


def test_symbolic_input_is_reported_as_unresolved_without_invented_dimensions(padded):
    model=padded;dimension=model.graph.input[0].type.tensor_type.shape.dim[2];dimension.dim_param='time'
    before=model.SerializeToString();report=module().bind_fixed_pad_shapes(model)
    assert report['bound_pad_count']==0 and report['unresolved_pad_names']
    assert report['complete'] is False and model.SerializeToString()==before
