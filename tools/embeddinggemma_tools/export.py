"""Fixed-shape fp32 export of the official text backbone and sentence pooling."""

from pathlib import Path

import torch
from torch import nn
from torch.fx.node import map_aggregate

from .text import pool_and_normalize


class TextEmbeddingExport(nn.Module):
    def __init__(self, model: nn.Module, sequence_length: int):
        super().__init__()
        self.model = model
        positions = torch.arange(sequence_length)
        self.register_buffer("positions", positions.unsqueeze(0))
        self.register_buffer(
            "local_band",
            (positions.unsqueeze(0) - positions.unsqueeze(1)).abs() <= model.config.sliding_window,
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        valid_keys = attention_mask[:, None, None, :].bool()
        zero = torch.zeros((), dtype=torch.float32, device=input_ids.device)
        blocked = torch.full((), torch.finfo(torch.float32).min, device=input_ids.device)
        masks = {
            "full_attention": torch.where(valid_keys, zero, blocked),
            "sliding_attention": torch.where(valid_keys & self.local_band, zero, blocked),
        }
        output = self.model(
            input_ids=input_ids, attention_mask=masks, position_ids=self.positions, return_dict=True
        )
        return pool_and_normalize(output.last_hidden_state, attention_mask)


def export_pt2(
    model: nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor, destination: Path
) -> None:
    wrapper = TextEmbeddingExport(model, input_ids.shape[1]).eval()
    with torch.no_grad():
        program = torch.export.export(wrapper, (input_ids, attention_mask))
        program = program.run_decompositions()
    # Sentis' Torch JSON reader stores graph scalar integers as System.Int32.
    # Core ATen uses INT64_MAX for open-ended slices, even on fixed-size inputs.
    def clamp_scalar(value):
        return max(-(2**31), min(2**31 - 1, value)) if type(value) is int else value

    for node in program.graph.nodes:
        node.args = map_aggregate(node.args, clamp_scalar)
        node.kwargs = map_aggregate(node.kwargs, clamp_scalar)
    # Torch 2.14 emits metadata-only assertions that Sentis 2.6 cannot import.
    # erase_node refuses removal if a node has users; numerical operations stay intact.
    for node in list(program.graph.nodes):
        if node.op == "call_function" and node.target == torch.ops.aten._assert_tensor_metadata.default:
            program.graph.erase_node(node)
    program.graph.lint()
    program.graph_module.recompile()
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.export.save(program, destination)
