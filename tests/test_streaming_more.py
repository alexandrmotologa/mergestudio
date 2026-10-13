"""Tests for streaming TIES and DARE merges."""

import json
from pathlib import Path
import pytest
import torch
from safetensors.torch import load_file, save_file

from mergestudio.core.config import (
    DareConfig,
    DType,
    MergeMethod,
    MergePlan,
    ModelSource,
    StreamingConfig,
    TiesConfig,
)
from mergestudio.streaming.engine import StreamingMergeEngine


@pytest.fixture
def base_and_donors(tmp_path):
    base_dir = tmp_path / "base"
    d1_dir = tmp_path / "donor1"
    d2_dir = tmp_path / "donor2"
    base_dir.mkdir()
    d1_dir.mkdir()
    d2_dir.mkdir()

    weights_base = {
        "model.layers.0.mlp.down_proj.weight": torch.tensor([[10.0, 10.0], [10.0, 10.0]]),
    }
    weights_d1 = {
        "model.layers.0.mlp.down_proj.weight": torch.tensor([[12.0, 15.0], [10.0, 8.0]]),
    }
    weights_d2 = {
        "model.layers.0.mlp.down_proj.weight": torch.tensor([[13.0, 14.0], [11.0, 7.0]]),
    }

    save_file(weights_base, base_dir / "model.safetensors")
    save_file(weights_d1, d1_dir / "model.safetensors")
    save_file(weights_d2, d2_dir / "model.safetensors")

    cfg = {"num_hidden_layers": 1, "hidden_size": 2}
    with open(base_dir / "config.json", "w") as f:
        json.dump(cfg, f)

    return base_dir, d1_dir, d2_dir


def test_streaming_ties_merge(base_and_donors, tmp_path):
    base_dir, d1_dir, d2_dir = base_and_donors
    out_dir = tmp_path / "ties_stream_out"

    plan = MergePlan(
        name="ties-stream",
        method=MergeMethod.TIES,
        models=[
            ModelSource(path=str(base_dir), name="base", is_base=True),
            ModelSource(path=str(d1_dir), name="d1"),
            ModelSource(path=str(d2_dir), name="d2"),
        ],
        base_model="base",
        ties_config=TiesConfig(density=0.5),
        output_dir=str(out_dir),
        streaming_config=StreamingConfig(output_dtype=DType.FLOAT32),
    )

    engine = StreamingMergeEngine(plan)
    engine.run()

    assert (out_dir / "model.safetensors").exists()
    weights = load_file(out_dir / "model.safetensors")
    assert "model.layers.0.mlp.down_proj.weight" in weights


def test_streaming_dare_merge(base_and_donors, tmp_path):
    base_dir, d1_dir, d2_dir = base_and_donors
    out_dir = tmp_path / "dare_stream_out"

    plan = MergePlan(
        name="dare-stream",
        method=MergeMethod.DARE_TIES,
        models=[
            ModelSource(path=str(base_dir), name="base", is_base=True),
            ModelSource(path=str(d1_dir), name="d1"),
            ModelSource(path=str(d2_dir), name="d2"),
        ],
        base_model="base",
        dare_config=DareConfig(drop_rate=0.3, rescale=True),
        output_dir=str(out_dir),
        streaming_config=StreamingConfig(output_dtype=DType.FLOAT32),
    )

    engine = StreamingMergeEngine(plan)
    engine.run()

    assert (out_dir / "model.safetensors").exists()
    weights = load_file(out_dir / "model.safetensors")
    assert "model.layers.0.mlp.down_proj.weight" in weights
