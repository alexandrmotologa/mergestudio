"""Tests for model health smoke test and numerical stability auditor."""

import json
from pathlib import Path
import pytest
import torch
from safetensors.torch import save_file

from mergestudio.evaluation.smoke_test import evaluate_model_health


def test_evaluate_healthy_model(tmp_path):
    m_dir = tmp_path / "healthy_model"
    m_dir.mkdir()

    weights = {
        "model.layers.0.mlp.weight": torch.randn(4, 4),
        "model.layers.1.mlp.weight": torch.randn(4, 4),
    }
    save_file(weights, m_dir / "model.safetensors")

    cfg = {"num_hidden_layers": 2}
    with open(m_dir / "config.json", "w") as f:
        json.dump(cfg, f)

    report = evaluate_model_health(m_dir)
    assert report.healthy is True
    assert report.total_tensors == 2
    assert report.has_nan is False
    assert report.has_inf is False
    assert "Healthy" in report.status


def test_evaluate_corrupted_model_nan(tmp_path):
    m_dir = tmp_path / "corrupted_model"
    m_dir.mkdir()

    t = torch.tensor([1.0, float("nan"), 3.0])
    save_file({"weight": t}, m_dir / "model.safetensors")

    report = evaluate_model_health(m_dir)
    assert report.healthy is False
    assert report.has_nan is True
    assert "Corrupted" in report.status
