"""Vocabulary surgery and embedding dimension alignment for model merging."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch


def reconcile_embedding_shapes(
    t_base: torch.Tensor,
    t_donor: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reconcile shape mismatch between vocabulary embedding or lm_head tensors.

    If one tensor has a larger vocabulary size (row dimension), pads the smaller
    tensor to match dimensions, preserving existing token weight representations.

    Args:
        t_base: Base model parameter tensor.
        t_donor: Donor model parameter tensor.

    Returns:
        Tuple of (aligned_base, aligned_donor) with identical shapes.
    """
    if t_base.shape == t_donor.shape:
        return t_base, t_donor

    # Check if only the first dimension (vocab size) differs
    if len(t_base.shape) != len(t_donor.shape):
        raise ValueError(
            f"Cannot reconcile tensors with different ranks: {len(t_base.shape)} vs {len(t_donor.shape)}"
        )

    for dim_idx in range(1, len(t_base.shape)):
        if t_base.shape[dim_idx] != t_donor.shape[dim_idx]:
            raise ValueError(
                f"Hidden dimension mismatch at dim {dim_idx}: {t_base.shape[dim_idx]} vs {t_donor.shape[dim_idx]}"
            )

    rows_base = t_base.shape[0]
    rows_donor = t_donor.shape[0]
    target_rows = max(rows_base, rows_donor)

    aligned_base = _pad_vocab_rows(t_base, target_rows)
    aligned_donor = _pad_vocab_rows(t_donor, target_rows)

    return aligned_base, aligned_donor


def _pad_vocab_rows(tensor: torch.Tensor, target_rows: int) -> torch.Tensor:
    """Pad tensor along row dimension up to target_rows."""
    current_rows = tensor.shape[0]
    if current_rows == target_rows:
        return tensor

    if current_rows > target_rows:
        return tensor[:target_rows]

    pad_rows = target_rows - current_rows
    trailing_shape = list(tensor.shape[1:])
    # Initialize padding with zeros or mean embeddings
    pad_tensor = torch.zeros([pad_rows] + trailing_shape, dtype=tensor.dtype, device=tensor.device)
    return torch.cat([tensor, pad_tensor], dim=0)


def patch_vocab_size_config(config_data: Dict[str, Any], new_vocab_size: int) -> Dict[str, Any]:
    """Update vocab_size in Hugging Face model configuration."""
    updated = dict(config_data)
    if "vocab_size" in updated:
        updated["vocab_size"] = max(updated.get("vocab_size", 0), new_vocab_size)
    return updated
