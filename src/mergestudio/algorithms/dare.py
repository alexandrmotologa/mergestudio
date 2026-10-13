"""DARE: Drop And REscale merging algorithm.

Implements random delta pruning with expectation preserving rescaling from
Yu et al. (2024) 'Language Models are Super Mario'.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch

from mergestudio.algorithms.ties import _elect_sign
from mergestudio.core.config import DareConfig, MergeMethod


def dare_merge(
    base: torch.Tensor,
    models: Sequence[torch.Tensor],
    drop_rate: float = 0.5,
    rescale: bool = True,
    base_method: MergeMethod = MergeMethod.DARE_TIES,
    seed: Optional[int] = None,
    model_weights: Optional[Sequence[float]] = None,
    config: Optional[DareConfig] = None,
) -> torch.Tensor:
    """Merge donor models into base using Drop And REscale (DARE).

    Args:
        base: Base anchor model tensor.
        models: Sequence of donor fine tuned model tensors.
        drop_rate: Probability p in [0.0, 1.0) of dropping each delta element.
        rescale: Whether to scale retained weights by 1 / (1 - drop_rate).
        base_method: Reduction variant, either DARE_TIES or DARE_LINEAR.
        seed: Optional random seed for reproducible Bernoulli sampling.
        model_weights: Optional relative weighting coefficients for donor models.
        config: Optional DareConfig instance overriding parameters.

    Returns:
        Merged tensor matching the shape and dtype of the base tensor.
    """
    if config is not None:
        drop_rate = config.drop_rate
        rescale = config.rescale
        base_method = config.base_method

    if not models:
        return base.clone()

    if drop_rate < 0.0 or drop_rate >= 1.0:
        raise ValueError(f"drop_rate must be in [0.0, 1.0), got {drop_rate}")

    orig_dtype = base.dtype
    orig_shape = base.shape

    calc_dtype = torch.float64 if orig_dtype == torch.float64 else torch.float32
    base_calc = base.to(calc_dtype)

    generator = None
    if seed is not None:
        generator = torch.Generator(device=base.device)
        generator.manual_seed(seed)

    # Step 1: Compute task vectors
    task_vectors = [m.to(calc_dtype) - base_calc for m in models]
    num_models = len(task_vectors)

    keep_prob = 1.0 - drop_rate
    scale_factor = (1.0 / keep_prob) if (rescale and keep_prob > 0.0) else 1.0

    # Step 2: Apply Bernoulli mask and rescale
    dropped_vectors = []
    for tv in task_vectors:
        if drop_rate > 0.0:
            rand_tensor = torch.rand(tv.shape, device=tv.device, generator=generator)
            mask = rand_tensor < keep_prob
            rescaled_tv = (tv * mask.to(calc_dtype)) * scale_factor
        else:
            rescaled_tv = tv
        dropped_vectors.append(rescaled_tv)

    weights_tensor = (
        torch.tensor(model_weights, dtype=calc_dtype, device=base.device)
        if model_weights is not None
        else torch.ones(num_models, dtype=calc_dtype, device=base.device)
    )

    expanded_weights = weights_tensor.view(num_models, *([1] * len(orig_shape)))

    if base_method == MergeMethod.DARE_LINEAR or base_method == MergeMethod.TASK_ARITHMETIC:
        # Weighted task arithmetic
        stacked = torch.stack(dropped_vectors, dim=0)
        weighted_sum = torch.sum(stacked * expanded_weights, dim=0)
        total_weight = torch.sum(weights_tensor)
        avg_delta = weighted_sum / max(float(total_weight), 1e-8)
        return (base_calc + avg_delta).to(orig_dtype)

    # Default DARE_TIES: Elect sign and disjoint merge over dropped deltas
    stacked = torch.stack(dropped_vectors, dim=0)
    elected_sign = _elect_sign(stacked, method="total")

    signs = torch.sign(stacked)
    agreement_mask = (signs == elected_sign.unsqueeze(0)) | (elected_sign.unsqueeze(0) == 0)
    filtered_deltas = stacked * agreement_mask.to(calc_dtype)

    nonzero_mask = filtered_deltas != 0
    weighted_deltas = filtered_deltas * expanded_weights
    weighted_sum = torch.sum(weighted_deltas, dim=0)

    participating_weights = torch.sum(nonzero_mask.to(calc_dtype) * expanded_weights, dim=0)
    safe_denominator = torch.clamp(participating_weights, min=1.0)

    merged_delta = torch.where(
        participating_weights > 0,
        weighted_sum / safe_denominator,
        torch.zeros_like(weighted_sum),
    )

    return (base_calc + merged_delta).to(orig_dtype)
