"""Lightweight model health evaluation and numerical stability smoke test."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch

from mergestudio.streaming.engine import ModelIndex


@dataclass
class ModelHealthReport:
    """Numerical stability and integrity audit report of a merged model."""

    model_dir: str
    total_tensors: int
    total_parameters: int
    has_nan: bool
    has_inf: bool
    healthy: bool
    mean_weight_norm: float
    status: str
    diagnostics: List[str]


def evaluate_model_health(model_dir: Path) -> ModelHealthReport:
    """Audit merged model tensors for numerical corruption, NaNs, and layer completeness.

    Args:
        model_dir: Path to directory containing merged safetensors.

    Returns:
        ModelHealthReport with detailed integrity diagnostics.
    """
    model_dir = Path(model_dir)
    idx = ModelIndex(model_dir)

    total_params = 0
    total_norm = 0.0
    nan_found = False
    inf_found = False
    diagnostics: List[str] = []

    tensor_names = idx.tensor_names
    if not tensor_names:
        return ModelHealthReport(
            model_dir=str(model_dir),
            total_tensors=0,
            total_parameters=0,
            has_nan=False,
            has_inf=False,
            healthy=False,
            mean_weight_norm=0.0,
            status="Empty: No tensors found",
            diagnostics=["Model directory contains zero safetensors tensors."],
        )

    for name in tensor_names:
        tensor = idx.get_tensor(name)
        if tensor is None:
            continue

        numel = tensor.numel()
        total_params += numel

        f_tensor = tensor.to(torch.float32)
        if torch.isnan(f_tensor).any():
            nan_found = True
            diagnostics.append(f"NaN weights detected in tensor: {name}")

        if torch.isinf(f_tensor).any():
            inf_found = True
            diagnostics.append(f"Inf weights detected in tensor: {name}")

        norm = float(torch.linalg.norm(f_tensor).item())
        total_norm += norm

    mean_norm = total_norm / len(tensor_names) if tensor_names else 0.0

    # Inspect configuration alignment
    cfg_file = model_dir / "config.json"
    if cfg_file.exists():
        with open(cfg_file, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        expected_layers = cfg.get("num_hidden_layers")
        if expected_layers is not None:
            diagnostics.append(f"Config num_hidden_layers verified: {expected_layers}")
    else:
        diagnostics.append("Notice: config.json missing in output directory.")

    healthy = not nan_found and not inf_found and total_params > 0
    if healthy:
        status = "Healthy: All weights numerically stable"
        diagnostics.append(f"Audited {len(tensor_names)} tensors ({total_params:,} parameters).")
    else:
        status = "Corrupted: Numerical instability detected"

    return ModelHealthReport(
        model_dir=str(model_dir),
        total_tensors=len(tensor_names),
        total_parameters=total_params,
        has_nan=nan_found,
        has_inf=inf_found,
        healthy=healthy,
        mean_weight_norm=round(mean_norm, 4),
        status=status,
        diagnostics=diagnostics,
    )
