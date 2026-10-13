"""Layer divergence and cosine similarity analysis across model weights."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from mergestudio.algorithms.frankenmerge import LayerSurgeryPlanner
from mergestudio.streaming.engine import ModelIndex


@dataclass
class LayerMetric:
    """Quantitative divergence metrics for an individual transformer layer."""

    layer_idx: int
    cosine_similarity: float
    frobenius_drift: float
    angular_distance: float
    tensor_count: int


@dataclass
class LayerDivergenceReport:
    """Aggregate divergence report comparing two model architectures."""

    model_a: str
    model_b: str
    total_layers: int
    layer_metrics: List[LayerMetric]
    mean_cosine_similarity: float
    max_divergence_layer: int
    adaptive_schedule_weights: List[float] = field(default_factory=list)


def analyze_layer_divergence(
    model_a_path: Path,
    model_b_path: Path,
) -> LayerDivergenceReport:
    """Compute mathematical divergence and similarity across all layers of two models.

    Args:
        model_a_path: Path to first model directory.
        model_b_path: Path to second model directory.

    Returns:
        LayerDivergenceReport containing per-layer and aggregate drift statistics.
    """
    idx_a = ModelIndex(model_a_path)
    idx_b = ModelIndex(model_b_path)

    common_keys = sorted(set(idx_a.tensor_names).intersection(set(idx_b.tensor_names)))
    if not common_keys:
        raise ValueError("No common tensor keys found between the provided models")

    # Group common tensor keys by layer index
    layer_tensors: Dict[int, List[str]] = {}
    for key in common_keys:
        extracted = LayerSurgeryPlanner.extract_layer_index(key)
        if extracted:
            layer_idx = extracted[1]
            layer_tensors.setdefault(layer_idx, []).append(key)

    if not layer_tensors:
        raise ValueError("No indexed transformer layers identified in common tensors")

    sorted_layers = sorted(layer_tensors.keys())
    metrics_list: List[LayerMetric] = []
    total_cosine = 0.0

    for l_idx in sorted_layers:
        keys = layer_tensors[l_idx]
        layer_cosines = []
        layer_frob_drifts = []

        for k in keys:
            t_a = idx_a.get_tensor(k)
            t_b = idx_b.get_tensor(k)
            if t_a is None or t_b is None:
                continue

            f_a = t_a.flatten().to(torch.float32)
            f_b = t_b.flatten().to(torch.float32)

            norm_a = torch.linalg.norm(f_a)
            norm_b = torch.linalg.norm(f_b)

            if norm_a > 1e-8 and norm_b > 1e-8:
                dot = torch.dot(f_a, f_b)
                cos_sim = float(torch.clamp(dot / (norm_a * norm_b), -1.0, 1.0).item())
                layer_cosines.append(cos_sim)

                diff_norm = float(torch.linalg.norm(f_a - f_b).item())
                frob_drift = diff_norm / max(float(norm_a.item()), 1e-8)
                layer_frob_drifts.append(frob_drift)

        if layer_cosines:
            avg_cos = sum(layer_cosines) / len(layer_cosines)
            avg_frob = sum(layer_frob_drifts) / len(layer_frob_drifts)
            # Angular distance = acos(cos) / pi
            clamped_cos = max(-1.0, min(1.0, avg_cos))
            import math
            ang_dist = math.acos(clamped_cos) / math.pi
        else:
            avg_cos = 1.0
            avg_frob = 0.0
            ang_dist = 0.0

        metrics_list.append(
            LayerMetric(
                layer_idx=l_idx,
                cosine_similarity=avg_cos,
                frobenius_drift=avg_frob,
                angular_distance=ang_dist,
                tensor_count=len(keys),
            )
        )
        total_cosine += avg_cos

    mean_cos = total_cosine / len(metrics_list) if metrics_list else 1.0

    # Locate layer with highest divergence (lowest cosine similarity)
    min_cos = 1.0
    max_div_layer = 0
    for m in metrics_list:
        if m.cosine_similarity < min_cos:
            min_cos = m.cosine_similarity
            max_div_layer = m.layer_idx

    # Compute adaptive weight schedule based on drift
    divergence_values = [1.0 - m.cosine_similarity for m in metrics_list]
    schedule = compute_adaptive_schedule(divergence_values)

    return LayerDivergenceReport(
        model_a=model_a_path.name,
        model_b=model_b_path.name,
        total_layers=len(metrics_list),
        layer_metrics=metrics_list,
        mean_cosine_similarity=mean_cos,
        max_divergence_layer=max_div_layer,
        adaptive_schedule_weights=schedule,
    )


def compute_adaptive_schedule(
    divergence_scores: List[float],
    min_weight: float = 0.15,
    max_weight: float = 0.85,
) -> List[float]:
    """Generate normalized layer weights proportional to fine-tuning drift.

    Low drift layers receive lower donor weight to preserve base foundation parameters;
    high drift layers receive higher donor weight to transfer specialized abilities.
    """
    if not divergence_scores:
        return []

    max_score = max(divergence_scores)
    min_score = min(divergence_scores)
    spread = max_score - min_score

    if spread < 1e-7:
        # Uniform drift across layers
        return [0.5 for _ in divergence_scores]

    weights: List[float] = []
    for s in divergence_scores:
        norm_s = (s - min_score) / spread
        w = min_weight + norm_s * (max_weight - min_weight)
        weights.append(round(w, 4))

    return weights
