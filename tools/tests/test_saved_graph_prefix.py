import pytest
import torch

from embeddinggemma_tools import reference_context_failure as context


def forbidden_tail(value):
    raise RuntimeError('The full graph tail must not execute')


def graph():
    value=torch.fx.Graph();input=value.placeholder('input')
    prefix=value.call_function(torch.add,(input,2.0));tail=value.call_function(forbidden_tail,(prefix,))
    value.output(tail)
    return torch.fx.GraphModule({},value),prefix.name


def run(*args):
    helper=getattr(context,'run_saved_graph_prefix',None)
    assert callable(helper),'Bounded saved-graph prefix runner is missing'
    return helper(*args)


def test_prefix_stops_before_tail_and_preserves_borrowed_input_and_graph():
    module,node=graph();values=torch.arange(8,dtype=torch.float32);before=values.clone();code=module.code
    result=run(module,(values,),node)
    assert torch.equal(result,before+2) and torch.equal(values,before) and module.code==code
    assert result.device.type=='cpu' and result.dtype==torch.float32 and not result.requires_grad
    result.fill_(19);assert torch.equal(values,before)


def test_missing_prefix_is_rejected_before_any_graph_execution():
    module,node=graph()
    with pytest.raises(ValueError,match='node'):
        run(module,(torch.ones(1),),'missing_node')


@pytest.mark.parametrize('values,error',[(torch.ones(2,dtype=torch.float64),TypeError),
    (torch.tensor([float('inf')]),ValueError),(torch.tensor([float('nan')]),ValueError)])
def test_prefix_refuses_unqualified_dtype_or_nonfinite_output(values,error):
    module,node=graph()
    with pytest.raises(error):run(module,(values,),node)


def test_prefix_refuses_nontensor_target_without_executing_tail():
    value=torch.fx.Graph();input=value.placeholder('input');prefix=value.call_function(len,(input,));value.output(prefix)
    with pytest.raises(TypeError):run(torch.fx.GraphModule({},value),(torch.ones(2),),prefix.name)


def test_independent_constant_prefix_does_not_execute_unrelated_earlier_model_nodes():
    value=torch.fx.Graph();input=value.placeholder('input');value.call_function(forbidden_tail,(input,))
    buffer=value.get_attr('positions');prefix=value.call_function(torch.add,(buffer,2.0));value.output(prefix)
    root=torch.nn.Module();root.register_buffer('positions',torch.arange(4,dtype=torch.float32))
    module=torch.fx.GraphModule(root,value);before=module.code
    result=run(module,(torch.ones(1),),prefix.name)
    assert torch.equal(result,torch.arange(4,dtype=torch.float32)+2) and module.code==before
