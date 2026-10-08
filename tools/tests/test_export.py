import pytest
import torch
from torch.nn import functional as F

from embeddinggemma_tools.export import TextEmbeddingExport, export_pt2
from test_model import tiny_model


def inputs():
    return (torch.tensor([[2, 3, 4, 1, 0, 0], [2, 5, 6, 7, 8, 1]]),
            torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1]]))


def independent_reference(model, ids, mask):
    with torch.no_grad():
        hidden = model(input_ids=ids, attention_mask=mask).last_hidden_state
    rows = [hidden[i, mask[i].bool()].mean(0) for i in range(ids.shape[0])]
    return F.normalize(torch.stack(rows), dim=-1)


def test_export_wrapper_matches_official_masks_and_projection():
    model = tiny_model()
    ids, mask = inputs()
    actual = TextEmbeddingExport(model, sequence_length=6)(ids, mask)
    torch.testing.assert_close(actual, independent_reference(model, ids, mask), rtol=1e-5, atol=1e-6)
    assert actual.shape == (2, 24)


def test_saved_core_aten_program_handles_changed_tokens_and_padding(tmp_path):
    model = tiny_model()
    ids, mask = inputs()
    destination = tmp_path / "model.pt2"
    export_pt2(model, ids, mask, destination)
    program = torch.export.load(destination)
    changed_ids = torch.tensor([[2, 9, 1, 0, 0, 0], [2, 10, 11, 1, 0, 0]])
    changed_mask = torch.tensor([[1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 0, 0]])
    with torch.no_grad():
        actual = program.module()(changed_ids, changed_mask)
    torch.testing.assert_close(actual, independent_reference(model, changed_ids, changed_mask), rtol=1e-5, atol=1e-6)
    targets = {str(n.target) for n in program.graph.nodes if n.op == "call_function"}
    assert not any("scaled_dot_product" in target or "masked_scatter" in target for target in targets)
    # Sentis 2.6 deserializes graph scalar integers as System.Int32. Python's
    # open-ended slice sentinel (INT64_MAX) must not reach the saved graph.
    def integer_literals(value):
        if type(value) is int:
            yield value
        elif isinstance(value, (tuple, list)):
            for child in value:
                yield from integer_literals(child)
        elif isinstance(value, dict):
            for child in value.values():
                yield from integer_literals(child)

    for node in program.graph.nodes:
        for literal in integer_literals((node.args, node.kwargs)):
            assert -(2**31) <= literal <= 2**31 - 1, (node.target, literal)
    # This export-time assertion has no numerical output and is absent from
    # the Sentis 2.6 supported-operator list.
    assert "aten._assert_tensor_metadata.default" not in targets
    with pytest.raises((AssertionError, RuntimeError), match="size|shape"):
        program.module()(changed_ids[:, :5], changed_mask[:, :5])
