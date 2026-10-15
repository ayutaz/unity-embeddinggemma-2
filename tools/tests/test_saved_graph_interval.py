import torch
import pytest
from embeddinggemma_tools import reference_context_failure as context


def forbidden(value):raise RuntimeError('Unrelated encoder work must not execute')


def program(extra_input=False):
    root=torch.nn.Module();root.register_buffer('weight',torch.tensor([1.,2.,3.]))
    graph=torch.fx.Graph();pixels=graph.placeholder('pixels')
    boundary=graph.call_function(forbidden,(pixels,));weight=graph.get_attr('weight')
    product=graph.call_function(torch.ops.aten.mul.Tensor,(boundary,weight))
    other=graph.placeholder('other') if extra_input else boundary
    target=graph.call_function(torch.ops.aten.add.Tensor,(product,other))
    graph.output(graph.call_function(forbidden,(target,)))
    return torch.fx.GraphModule(root,graph),boundary.name,target.name


def extract(*args):
    function=getattr(context,'extract_saved_graph_interval',None)
    assert callable(function),'Saved-graph single-boundary interval extraction is missing'
    return function(*args)


def test_interval_cuts_upstream_and_downstream_work_owns_no_weights_and_keeps_source_unchanged():
    model,boundary,target=program();code=model.code;weights=model.weight.clone()
    interval=extract(model,boundary,target)
    values=torch.tensor([2.,3.,4.]);before=values.clone()
    assert torch.equal(interval(values),torch.tensor([4.,9.,16.]))
    assert torch.equal(values,before) and model.code==code and torch.equal(model.weight,weights)
    assert interval.weight.data_ptr()==model.weight.data_ptr()
    assert len([n for n in interval.graph.nodes if n.op=='placeholder'])==1


@pytest.mark.parametrize('problem',['missing_input','missing_output','same_boundary','uncovered_dynamic_input','not_module'])
def test_interval_rejects_unqualified_boundaries_without_running_any_encoder_work(problem):
    model,boundary,target=program(extra_input=problem=='uncovered_dynamic_input');code=model.code
    if problem=='missing_input':boundary='absent'
    elif problem=='missing_output':target='absent'
    elif problem=='same_boundary':target=boundary
    elif problem=='not_module':model=torch.nn.Identity()
    with pytest.raises(ValueError):extract(model,boundary,target)
    if isinstance(model,torch.fx.GraphModule):assert model.code==code


def test_interval_rejects_a_boundary_that_does_not_control_the_target():
    model,boundary,target=program();unrelated=next(n.name for n in model.graph.nodes if n.op=='get_attr')
    # The target still depends on pixels when a coefficient is supplied as the boundary.
    with pytest.raises(ValueError):extract(model,unrelated,target)
