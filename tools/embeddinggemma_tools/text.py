"""Text prompting and masked mean pooling used by the official model."""

import torch
from torch.nn import functional as F


def format_text(text: str, role: str, title: str | None = None) -> str:
    if role not in ("query", "document", "raw"):
        raise ValueError(f"Unknown role: {role}")
    if title is not None and role != "document":
        raise ValueError("title is only supported for the document role")
    if role == "query":
        return "task: search result | query: " + text
    if role == "document":
        return f"title: {title if title is not None else 'none'} | text: {text}"
    return text


def pool_and_normalize(
    hidden: torch.Tensor, attention_mask: torch.Tensor, dimensions: int | None = None
) -> torch.Tensor:
    if dimensions is not None and not 1 <= dimensions <= hidden.shape[-1]:
        raise ValueError("dimensions must be positive and no greater than the hidden dimension")
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
    if dimensions is not None:
        pooled = pooled[:, :dimensions]
    return F.normalize(pooled, p=2, dim=-1)
