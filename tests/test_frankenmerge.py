"""Tests for layer surgery planner and config patching."""

from mergestudio.algorithms.frankenmerge import LayerSurgeryPlanner
from mergestudio.core.config import LayerSlice


def test_layer_surgery_mapping():
    # Splicing:
    # Slice 0: Model A layers 0..16 -> Output layers 0..15
    # Slice 1: Model B layers 8..24 -> Output layers 16..31
    slices = [
        LayerSlice(source_model="model_a", start_layer=0, end_layer=16),
        LayerSlice(source_model="model_b", start_layer=8, end_layer=24),
    ]

    planner = LayerSurgeryPlanner(slices)
    assert planner.total_target_layers == 32

    # Map layer from Model A
    m_a_0 = planner.map_tensor_key("model_a", "model.layers.0.self_attn.q_proj.weight")
    assert m_a_0 == [("model.layers.0.self_attn.q_proj.weight", 0)]

    m_a_15 = planner.map_tensor_key("model_a", "model.layers.15.mlp.down_proj.weight")
    assert m_a_15 == [("model.layers.15.mlp.down_proj.weight", 15)]

    # Model A layer outside slice should not map
    m_a_16 = planner.map_tensor_key("model_a", "model.layers.16.mlp.down_proj.weight")
    assert m_a_16 == []

    # Map layer from Model B: source layer 8 should map to target layer 16
    m_b_8 = planner.map_tensor_key("model_b", "model.layers.8.self_attn.q_proj.weight")
    assert m_b_8 == [("model.layers.16.self_attn.q_proj.weight", 16)]

    # Non layer tensors (embed_tokens, norm, etc.) should map as-is with index -1
    norm_res = planner.map_tensor_key("model_a", "model.norm.weight")
    assert norm_res == [("model.norm.weight", -1)]


def test_patch_config():
    slices = [
        LayerSlice(source_model="m1", start_layer=0, end_layer=10),
        LayerSlice(source_model="m2", start_layer=10, end_layer=24),
    ]
    planner = LayerSurgeryPlanner(slices)
    assert planner.total_target_layers == 24

    base_config = {
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": 4096,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
    }

    patched = planner.patch_config(base_config)
    assert patched["num_hidden_layers"] == 24
    assert patched["hidden_size"] == 4096
