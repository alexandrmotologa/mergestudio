"""Additional tests for CLI commands to ensure comprehensive coverage."""

import json
from pathlib import Path
import pytest
import torch
from safetensors.torch import save_file
from typer.testing import CliRunner

from mergestudio.cli.main import app

runner = CliRunner()


@pytest.fixture
def mock_cli_models_dare(tmp_path):
    m1_dir = tmp_path / "m1"
    m2_dir = tmp_path / "m2"
    m1_dir.mkdir()
    m2_dir.mkdir()

    save_file({"model.layers.0.mlp.weight": torch.tensor([1.0, 2.0, 3.0])}, m1_dir / "model.safetensors")
    save_file({"model.layers.0.mlp.weight": torch.tensor([2.0, 3.0, 4.0])}, m2_dir / "model.safetensors")

    cfg = {"num_hidden_layers": 2, "hidden_size": 3}
    with open(m1_dir / "config.json", "w") as f:
        json.dump(cfg, f)
    with open(m2_dir / "config.json", "w") as f:
        json.dump(cfg, f)

    return m1_dir, m2_dir


def test_cli_dare_merge(mock_cli_models_dare, tmp_path):
    m1_dir, m2_dir = mock_cli_models_dare
    out_dir = tmp_path / "cli_dare_out"

    result = runner.invoke(
        app,
        [
            "dare",
            "--base", str(m1_dir),
            "-d", str(m2_dir),
            "--drop-rate", "0.5",
            "--output", str(out_dir),
        ],
    )
    assert result.exit_code == 0
    assert (out_dir / "model.safetensors").exists()


def test_cli_merge_yaml_config(mock_cli_models_dare, tmp_path):
    m1_dir, m2_dir = mock_cli_models_dare
    out_dir = tmp_path / "cli_yaml_out"
    config_file = tmp_path / "merge_config.yaml"

    yaml_text = f"""
name: test-cli-yaml
method: slerp
models:
  - path: "{m1_dir.as_posix()}"
    name: m1
    is_base: true
  - path: "{m2_dir.as_posix()}"
    name: m2
output_dir: "{out_dir.as_posix()}"
slerp_config:
  alpha: 0.5
"""
    with open(config_file, "w", encoding="utf-8") as f:
        f.write(yaml_text)

    result = runner.invoke(app, ["merge", "-c", str(config_file)])
    assert result.exit_code == 0
    assert (out_dir / "model.safetensors").exists()


def test_cli_merge_missing_config():
    result = runner.invoke(app, ["merge", "-c", "non_existent_file.yaml"])
    assert result.exit_code == 1
    assert "not found" in result.stdout


def test_cli_ties_with_negative_unlearning(mock_cli_models_dare, tmp_path):
    m1_dir, m2_dir = mock_cli_models_dare
    out_dir = tmp_path / "cli_unlearn_out"

    result = runner.invoke(
        app,
        [
            "ties",
            "--base", str(m1_dir),
            "-d", str(m2_dir),
            "-n", str(m2_dir),
            "--output", str(out_dir),
        ],
    )
    assert result.exit_code == 0
    assert (out_dir / "model.safetensors").exists()


def test_cli_analyze_divergence(mock_cli_models_dare):
    m1_dir, m2_dir = mock_cli_models_dare
    result = runner.invoke(
        app,
        [
            "analyze-divergence",
            "--model-a", str(m1_dir),
            "--model-b", str(m2_dir),
        ],
    )
    assert result.exit_code == 0
    assert "Layer Divergence Report" in result.stdout


def test_cli_export_ollama_and_evaluate(mock_cli_models_dare):
    m1_dir, _ = mock_cli_models_dare
    res_export = runner.invoke(app, ["export-ollama", str(m1_dir)])
    assert res_export.exit_code == 0
    assert "Ollama & GGUF Artifacts" in res_export.stdout

    res_eval = runner.invoke(app, ["evaluate", str(m1_dir)])
    assert res_eval.exit_code == 0
    assert "Model Health Audit" in res_eval.stdout

