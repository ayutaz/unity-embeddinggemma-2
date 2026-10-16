import torch
import pytest
from embeddinggemma_tools import reference_context_failure as context


def fixture(count=7,ambiguous=False,gelus=1):
    root=torch.nn.Module();root.register_buffer('weight',torch.tensor([1.,2.,3.]))
    graph=torch.fx.Graph();value=graph.placeholder('hidden');weight=graph.get_attr('weight');expected=[]
    for index in range(count):
        incoming=value;square=graph.call_function(torch.ops.aten.pow.Tensor_Scalar,(incoming,2))
        mean=graph.call_function(torch.ops.aten.mean.dim,(square,[-1],True))
        statistic=graph.call_function(torch.ops.aten.add.Tensor,(mean,1e-6))
        inverse=graph.call_function(torch.ops.aten.pow.Tensor_Scalar,(statistic,-.5))
        normal=graph.call_function(torch.ops.aten.mul.Tensor,(incoming,inverse));value=normal
        if index!=3:value=graph.call_function(torch.ops.aten.mul.Tensor,(normal,weight))
        if ambiguous and index==0:graph.call_function(torch.ops.aten.mul.Tensor,(normal,weight))
        expected.append((incoming.name,value.name,index!=3))
    for _ in range(gelus):value=graph.call_function(torch.ops.aten.gelu.default,(value,),{'approximate':'tanh'})
    graph.output(value)
    return torch.fx.GraphModule(root,graph),expected


def describe(model):
    function=getattr(context,'describe_vision_layer_intermediates',None)
    assert callable(function),'Audited vision layer RMS/GELU descriptor is missing'
    return function(model)


def test_descriptor_identifies_exact_scaled_and_unscaled_boundaries_without_inference_or_mutation():
    model,expected=fixture();code=model.code;before=model.weight.clone();result=describe(model)
    assert len(result['rms_operations'])==7 and result['gelu']['output_node']
    for row,(incoming,outgoing,scaled) in zip(result['rms_operations'],expected,strict=True):
        assert row['input_node']==incoming and row['output_node']==outgoing
        assert (row['scale_target']=='weight') is scaled
        assert row['mean_node'] and row['statistic_node'] and row['inverse_node'] and row['normalized_node']
    assert model.code==code and torch.equal(model.weight,before)


@pytest.mark.parametrize('problem',['count','ambiguous','no_gelu','two_gelus'])
def test_descriptor_rejects_changed_or_ambiguous_layer_structure(problem):
    model,_=fixture(count=6 if problem=='count' else 7,ambiguous=problem=='ambiguous',gelus=0 if problem=='no_gelu' else 2 if problem=='two_gelus' else 1)
    with pytest.raises(ValueError):describe(model)
