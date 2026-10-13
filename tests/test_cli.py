"""Unit tests for MergeStudio Typer CLI commands."""

import json
from pathlib import Path
import pytest
import torch
from safetensors.torch import save_file
from typer.testing import CliRunner

from mergestudio.cli.main import app

runner = CliRunner()


@pytest.fixture
def mock_cli_models(tmp_path):
    m1_dir = tmp_path / "m1"
    m2_dir = tmp_path / "m2"
    m1_dir.mkdir()
    m2_dir.mkdir()

    save_file({"layer1": torch.ones(2, 2)}, m1_dir / "model.safetensors")
    save_file({"layer1": torch.ones(2, 2) * 2}, m2_dir / "model.safetensors")

    cfg = {"num_hidden_layers": 4, "hidden_size": 128}
    with open(m1_dir / "config.json", "w") as f:
        json.dump(cfg, f)
    with open(m2_dir / "config.json", "w") as f:
        json.dump(cfg, f)

    return m1_dir, m2_dir


def test_cli_inspect(mock_cli_models):
    m1_dir, _ = mock_cli_models
    result = runner.invoke(app, ["inspect", str(m1_dir)])
    assert result.exit_code == 0
    assert "Model Inspection" in result.stdout
    assert "Hidden Layers" in result.stdout


def test_cli_slerp_merge(mock_cli_models, tmp_path):
    m1_dir, m2_dir = mock_cli_models
    out_dir = tmp_path / "cli_slerp_out"

    result = runner.invoke(
        app,
        [
            "slerp",
            "--model-a", str(m1_dir),
            "--model-b", str(m2_dir),
            "--alpha", "0.5",
            "--output", str(out_dir),
        ],
    )
    assert result.exit_code == 0
    assert (out_dir / "model.safetensors").exists()


def test_cli_ties_merge(mock_cli_models, tmp_path):
    m1_dir, m2_dir = mock_cli_models
    out_dir = tmp_path / "cli_ties_out"

    result = runner.invoke(
        app,
        [
            "ties",
            "--base", str(m1_dir),
            "-d", str(m2_dir),
            "--density", "0.5",
            "--output", str(out_dir),
        ],
    )
    assert result.exit_code == 0
    assert (out_dir / "model.safetensors").exists()
