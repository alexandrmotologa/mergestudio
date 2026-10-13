"""Tests for zero-RAM memory mapped streaming engine."""

import json
from pathlib import Path
import pytest
import torch
from safetensors.torch import load_file, save_file

from mergestudio.core.config import (
    DType,
    LayerSlice,
    MergeMethod,
    MergePlan,
    ModelSource,
    SlerpConfig,
    StreamingConfig,
    TiesConfig,
)
from mergestudio.streaming.engine import (
    ModelIndex,
    ShardedSafetensorsWriter,
    StreamingMergeEngine,
)


@pytest.fixture
def mock_models(tmp_path):
    """Create two mock model directories with safetensors and configs."""
    model_a_dir = tmp_path / "model_a"
    model_b_dir = tmp_path / "model_b"
    model_a_dir.mkdir()
    model_b_dir.mkdir()

    # Model A weights
    weights_a = {
        "model.embed_tokens.weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
        "model.layers.0.self_attn.q_proj.weight": torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32),
        "model.layers.1.self_attn.q_proj.weight": torch.tensor([[2.0, 0.0], [0.0, 2.0]], dtype=torch.float32),
        "lm_head.weight": torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float32),
    }
    save_file(weights_a, model_a_dir / "model.safetensors")

    config_a = {
        "architectures": ["LlamaForCausalLM"],
        "num_hidden_layers": 2,
        "hidden_size": 2,
    }
    with open(model_a_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_a, f)

    with open(model_a_dir / "tokenizer.json", "w", encoding="utf-8") as f:
        f.write('{"mock": "tokenizer"}')

    # Model B weights
    weights_b = {
        "model.embed_tokens.weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
        "model.layers.0.self_attn.q_proj.weight": torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float32),
        "model.layers.1.self_attn.q_proj.weight": torch.tensor([[4.0, 0.0], [0.0, 4.0]], dtype=torch.float32),
        "lm_head.weight": torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float32),
    }
    save_file(weights_b, model_b_dir / "model.safetensors")

    return model_a_dir, model_b_dir


def test_model_index(mock_models):
    dir_a, _ = mock_models
    index = ModelIndex(dir_a)

    assert "model.embed_tokens.weight" in index.tensor_names
    assert "model.layers.0.self_attn.q_proj.weight" in index.tensor_names

    tensor = index.get_tensor("model.embed_tokens.weight")
    assert tensor is not None
    assert tensor.shape == (2, 2)


def test_streaming_slerp_merge(mock_models, tmp_path):
    dir_a, dir_b = mock_models
    out_dir = tmp_path / "merged_slerp"

    plan = MergePlan(
        name="slerp-test",
        method=MergeMethod.SLERP,
        models=[
            ModelSource(path=str(dir_a), name="model_a"),
            ModelSource(path=str(dir_b), name="model_b"),
        ],
        slerp_config=SlerpConfig(alpha=0.5),
        streaming_config=StreamingConfig(output_dtype=DType.FLOAT32),
        output_dir=str(out_dir),
    )

    progress_events = []
    engine = StreamingMergeEngine(plan)
    res_dir = engine.run(progress_callback=lambda p: progress_events.append(p))

    assert res_dir == out_dir
    assert (out_dir / "model.safetensors").exists()
    assert (out_dir / "config.json").exists()
    assert (out_dir / "tokenizer.json").exists()
    assert len(progress_events) > 0

    merged_weights = load_file(out_dir / "model.safetensors")
    assert "model.layers.0.self_attn.q_proj.weight" in merged_weights

    # Orthogonal vectors at alpha=0.5 norm should be preserved
    merged_q = merged_weights["model.layers.0.self_attn.q_proj.weight"]
    assert merged_q.shape == (2, 2)


def test_streaming_frankenmerge(mock_models, tmp_path):
    dir_a, dir_b = mock_models
    out_dir = tmp_path / "merged_franken"

    # Splice: take layer 0 from model A, take layer 1 from model B
    plan = MergePlan(
        name="franken-test",
        method=MergeMethod.FRANKENMERGE,
        models=[
            ModelSource(path=str(dir_a), name="model_a"),
            ModelSource(path=str(dir_b), name="model_b"),
        ],
        slices=[
            LayerSlice(source_model="model_a", start_layer=0, end_layer=1),
            LayerSlice(source_model="model_b", start_layer=1, end_layer=2),
        ],
        streaming_config=StreamingConfig(output_dtype=DType.FLOAT32),
        output_dir=str(out_dir),
    )

    engine = StreamingMergeEngine(plan)
    engine.run()

    merged_weights = load_file(out_dir / "model.safetensors")
    # Output should contain model.layers.0... and model.layers.1...
    assert "model.layers.0.self_attn.q_proj.weight" in merged_weights
    assert "model.layers.1.self_attn.q_proj.weight" in merged_weights

    # Check config.json num_hidden_layers
    with open(out_dir / "config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)
    assert cfg["num_hidden_layers"] == 2


def test_sharded_safetensors_writer(tmp_path):
    out_dir = tmp_path / "sharded_out"
    # Small max shard size to force multiple shards
    writer = ShardedSafetensorsWriter(out_dir, max_shard_size_bytes=64)

    t1 = torch.ones((4, 4), dtype=torch.float32)  # 64 bytes
    t2 = torch.ones((4, 4), dtype=torch.float32)  # 64 bytes
    t3 = torch.ones((4, 4), dtype=torch.float32)  # 64 bytes

    writer.write_tensor("layer_1", t1)
    writer.write_tensor("layer_2", t2)
    writer.write_tensor("layer_3", t3)
    writer.finalize()

    index_path = out_dir / "model.safetensors.index.json"
    assert index_path.exists()
    with open(index_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "weight_map" in data
    assert "layer_1" in data["weight_map"]
    assert "layer_3" in data["weight_map"]
