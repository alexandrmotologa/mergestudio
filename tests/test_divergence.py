"""Unit tests for layer divergence and cosine similarity analysis."""

import json
from pathlib import Path
import pytest
import torch
from safetensors.torch import save_file

from mergestudio.analysis.divergence import (
    analyze_layer_divergence,
    compute_adaptive_schedule,
)


@pytest.fixture
def divergence_models(tmp_path):
    m_a_dir = tmp_path / "model_a"
    m_b_dir = tmp_path / "model_b"
    m_a_dir.mkdir()
    m_b_dir.mkdir()

    # Layer 0: almost identical (cosine ~ 1.0)
    # Layer 1: orthogonal (cosine ~ 0.0)
    weights_a = {
        "model.layers.0.mlp.down_proj.weight": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        "model.layers.1.mlp.down_proj.weight": torch.tensor([[1.0, 0.0], [0.0, 0.0]]),
    }
    weights_b = {
        "model.layers.0.mlp.down_proj.weight": torch.tensor([[1.0, 0.01], [0.0, 1.0]]),
        "model.layers.1.mlp.down_proj.weight": torch.tensor([[0.0, 1.0], [0.0, 0.0]]),
    }

    save_file(weights_a, m_a_dir / "model.safetensors")
    save_file(weights_b, m_b_dir / "model.safetensors")

    return m_a_dir, m_b_dir


def test_analyze_layer_divergence(divergence_models):
    dir_a, dir_b = divergence_models
    report = analyze_layer_divergence(dir_a, dir_b)

    assert report.total_layers == 2
    assert len(report.layer_metrics) == 2

    # Layer 0 has high cosine similarity
    m0 = report.layer_metrics[0]
    assert m0.layer_idx == 0
    assert m0.cosine_similarity > 0.99

    # Layer 1 has low cosine similarity (orthogonal)
    m1 = report.layer_metrics[1]
    assert m1.layer_idx == 1
    assert m1.cosine_similarity == pytest.approx(0.0, abs=1e-5)

    assert report.max_divergence_layer == 1
    assert len(report.adaptive_schedule_weights) == 2


def test_compute_adaptive_schedule():
    divergence_scores = [0.01, 0.2, 0.5, 0.9]
    schedule = compute_adaptive_schedule(divergence_scores, min_weight=0.2, max_weight=0.8)

    assert len(schedule) == 4
    # Weights should be strictly increasing with divergence
    assert schedule[0] == pytest.approx(0.2)
    assert schedule[-1] == pytest.approx(0.8)
    assert schedule[1] < schedule[2] < schedule[3]
