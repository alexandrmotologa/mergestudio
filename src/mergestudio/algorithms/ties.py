"""TIES-Merging: Resolving Interference When Merging Models.

Implements the Trim, Elect Sign, and Disjoint Merge algorithm from
Yadav et al. (NeurIPS 2023).
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Union

import torch

from mergestudio.core.config import TiesConfig


def ties_merge(
    base: torch.Tensor,
    models: Sequence[torch.Tensor],
    density: float = 0.2,
    majority_sign_method: str = "total",
    weight: float = 1.0,
    normalize_signs: bool = True,
    model_weights: Optional[Sequence[float]] = None,
    negative_mask: Optional[Sequence[bool]] = None,
    config: Optional[TiesConfig] = None,
) -> torch.Tensor:
    """Merge fine tuned model tensors into a base tensor using TIES-Merging.

    Args:
        base: Base anchor model tensor.
        models: Sequence of fine-tuned model tensors to merge.
        density: Fraction of highest magnitude weights to keep during trimming (0.0 < density <= 1.0).
        majority_sign_method: 'total' (sum of trimmed deltas) or 'frequency' (sign vote count).
        weight: Global multiplier lambda applied to the aggregated task vector.
        normalize_signs: If True, zero out parameters whose sign conflicts with the elected sign.
        model_weights: Optional relative weighting coefficients for each donor model.
        negative_mask: Optional boolean sequence marking models whose task vectors should be negated (unlearned).
        config: Optional TiesConfig instance overriding discrete parameters.

    Returns:
        Merged tensor matching the shape and dtype of the base tensor.
    """
    if config is not None:
        density = config.density
        majority_sign_method = config.majority_sign_method
        weight = config.weight
        normalize_signs = config.normalize_signs

    if not models:
        return base.clone()

    for i, m in enumerate(models):
        if m.shape != base.shape:
            raise ValueError(
                f"Donor model tensor {i} shape mismatch: {tuple(m.shape)} vs base {tuple(base.shape)}"
            )

    orig_dtype = base.dtype
    orig_shape = base.shape

    calc_dtype = torch.float64 if orig_dtype == torch.float64 else torch.float32
    base_calc = base.to(calc_dtype)

    # Step 1: Compute task vectors (deltas with optional negation for unlearning)
    task_vectors = []
    for i, m in enumerate(models):
        delta = m.to(calc_dtype) - base_calc
        if negative_mask and i < len(negative_mask) and negative_mask[i]:
            delta = -delta
        task_vectors.append(delta)
    num_models = len(task_vectors)

    weights_tensor = (
        torch.tensor(model_weights, dtype=calc_dtype, device=base.device)
        if model_weights is not None
        else torch.ones(num_models, dtype=calc_dtype, device=base.device)
    )

    # Step 2: Trim task vectors based on top-k density
    trimmed_vectors = [_trim_task_vector(tv, density) for tv in task_vectors]
    stacked_trimmed = torch.stack(trimmed_vectors, dim=0)  # Shape: (N, *shape)

    # Step 3: Elect Sign
    elected_sign = _elect_sign(stacked_trimmed, majority_sign_method)  # Shape: (*shape)

    # Step 4: Disjoint Merge
    if normalize_signs:
        # Keep only values whose sign matches the elected sign (or when elected sign is 0)
        signs = torch.sign(stacked_trimmed)
        agreement_mask = (signs == elected_sign.unsqueeze(0)) | (elected_sign.unsqueeze(0) == 0)
        filtered_deltas = stacked_trimmed * agreement_mask.to(calc_dtype)
    else:
        filtered_deltas = stacked_trimmed

    # Calculate average across models that actively retain non-zero parameter
    nonzero_mask = filtered_deltas != 0  # Shape: (N, *shape)

    # Apply weights across model dimension
    expanded_weights = weights_tensor.view(num_models, *([1] * len(orig_shape)))
    weighted_deltas = filtered_deltas * expanded_weights

    weighted_sum = torch.sum(weighted_deltas, dim=0)

    # Count participating models per parameter index
    participating_weights = torch.sum(nonzero_mask.to(calc_dtype) * expanded_weights, dim=0)
    safe_denominator = torch.clamp(participating_weights, min=1.0)

    merged_task_vector = torch.where(
        participating_weights > 0,
        weighted_sum / safe_denominator,
        torch.zeros_like(weighted_sum),
    )

    # Step 5: Add scaled task vector back to base
    final_merged = base_calc + weight * merged_task_vector
    return final_merged.to(orig_dtype)


def _trim_task_vector(tv: torch.Tensor, density: float) -> torch.Tensor:
    """Retain top-k absolute values and zero the rest."""
    if density >= 1.0:
        return tv

    total_elements = tv.numel()
    k = max(1, int(round(total_elements * density)))

    if k >= total_elements:
        return tv

    flat_abs = torch.abs(tv).flatten()
    # topk returns the k largest elements
    threshold = torch.topk(flat_abs, k, largest=True).values[-1]

    mask = torch.abs(tv) >= threshold
    return tv * mask.to(tv.dtype)


def _elect_sign(stacked: torch.Tensor, method: str) -> torch.Tensor:
    """Determine majority consensus sign across models."""
    if method == "frequency":
        # Count positive vs negative non-zero signs
        pos_votes = torch.sum((stacked > 0).to(torch.int32), dim=0)
        neg_votes = torch.sum((stacked < 0).to(torch.int32), dim=0)
        diff = pos_votes - neg_votes
        return torch.sign(diff.to(stacked.dtype))

    # Default 'total': sum of trimmed deltas
    total_delta = torch.sum(stacked, dim=0)
    return torch.sign(total_delta)
