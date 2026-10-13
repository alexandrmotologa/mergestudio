"""Spherical Linear Interpolation (SLERP) for neural network tensors."""

from __future__ import annotations

from typing import Optional

import torch

from mergestudio.core.config import SlerpConfig


def slerp(
    v0: torch.Tensor,
    v1: torch.Tensor,
    alpha: float = 0.5,
    eps: float = 1e-7,
    per_row: bool = False,
    preserve_norm: bool = True,
    config: Optional[SlerpConfig] = None,
) -> torch.Tensor:
    """Perform spherical linear interpolation between two parameter tensors.

    Calculates:
        v(t) = [sin((1-t)*theta) / sin(theta)] * v0 + [sin(t*theta) / sin(theta)] * v1

    Args:
        v0: First source tensor (t=0).
        v1: Second source tensor (t=1).
        alpha: Interpolation parameter t in [0.0, 1.0].
        eps: Threshold for collinear / near-parallel vector detection.
        per_row: If True and tensor has >=2 dims, apply SLERP along row vectors.
        preserve_norm: If True, scale unit directional vector by interpolated norm.
        config: Optional SlerpConfig instance overriding discrete parameters.

    Returns:
        Interpolated tensor with same shape and dtype as input tensors.
    """
    if config is not None:
        alpha = config.alpha
        eps = config.eps
        per_row = config.per_row
        preserve_norm = config.preserve_norm

    if v0.shape != v1.shape:
        raise ValueError(
            f"Tensor shape mismatch for SLERP: {tuple(v0.shape)} vs {tuple(v1.shape)}"
        )

    if alpha <= 0.0:
        return v0.clone()
    if alpha >= 1.0:
        return v1.clone()

    orig_dtype = v0.dtype
    orig_shape = v0.shape

    # Upcast to float32 or float64 for stable trigonometric operations
    calc_dtype = torch.float64 if orig_dtype == torch.float64 else torch.float32
    t0 = v0.to(calc_dtype)
    t1 = v1.to(calc_dtype)

    if per_row and len(orig_shape) >= 2:
        return _slerp_per_row(t0, t1, alpha, eps, preserve_norm).to(orig_dtype)

    return _slerp_global(t0, t1, alpha, eps, preserve_norm).view(orig_shape).to(orig_dtype)


def _slerp_global(
    v0: torch.Tensor,
    v1: torch.Tensor,
    alpha: float,
    eps: float,
    preserve_norm: bool,
) -> torch.Tensor:
    """Compute SLERP treating the entire tensor as a single vector on a hypersphere."""
    u0 = v0.flatten()
    u1 = v1.flatten()

    norm0 = torch.linalg.norm(u0)
    norm1 = torch.linalg.norm(u1)

    # If any vector is degenerate, fall back to linear interpolation
    if norm0 < eps or norm1 < eps:
        return (1.0 - alpha) * u0 + alpha * u1

    unit0 = u0 / norm0
    unit1 = u1 / norm1

    dot = torch.dot(unit0, unit1)
    dot = torch.clamp(dot, -1.0, 1.0)

    # Check for collinearity or anti-parallel vectors
    if torch.abs(dot) > (1.0 - eps):
        res = (1.0 - alpha) * u0 + alpha * u1
        return res

    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)

    if torch.abs(sin_theta) < eps:
        return (1.0 - alpha) * u0 + alpha * u1

    scale0 = torch.sin((1.0 - alpha) * theta) / sin_theta
    scale1 = torch.sin(alpha * theta) / sin_theta

    res_unit = scale0 * unit0 + scale1 * unit1

    if preserve_norm:
        res_norm = (1.0 - alpha) * norm0 + alpha * norm1
        return res_unit * res_norm

    return res_unit


def _slerp_per_row(
    v0: torch.Tensor,
    v1: torch.Tensor,
    alpha: float,
    eps: float,
    preserve_norm: bool,
) -> torch.Tensor:
    """Compute SLERP across rows (first dimension) independently."""
    rows = v0.shape[0]
    cols = v0[0].numel()

    u0 = v0.reshape(rows, cols)
    u1 = v1.reshape(rows, cols)

    norm0 = torch.linalg.norm(u0, dim=1, keepdim=True)
    norm1 = torch.linalg.norm(u1, dim=1, keepdim=True)

    degenerate = (norm0 < eps) | (norm1 < eps)

    unit0 = torch.where(degenerate, u0, u0 / torch.clamp(norm0, min=eps))
    unit1 = torch.where(degenerate, u1, u1 / torch.clamp(norm1, min=eps))

    # Dot product per row: (rows, 1)
    dot = torch.sum(unit0 * unit1, dim=1, keepdim=True)
    dot = torch.clamp(dot, -1.0, 1.0)

    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)

    collinear = (torch.abs(dot) > (1.0 - eps)) | (torch.abs(sin_theta) < eps) | degenerate

    scale0 = torch.sin((1.0 - alpha) * theta) / torch.clamp(sin_theta, min=eps)
    scale1 = torch.sin(alpha * theta) / torch.clamp(sin_theta, min=eps)

    slerp_res = scale0 * unit0 + scale1 * unit1
    if preserve_norm:
        interp_norm = (1.0 - alpha) * norm0 + alpha * norm1
        slerp_res = slerp_res * interp_norm

    lerp_res = (1.0 - alpha) * u0 + alpha * u1
    out = torch.where(collinear, lerp_res, slerp_res)
    return out.reshape(v0.shape)
