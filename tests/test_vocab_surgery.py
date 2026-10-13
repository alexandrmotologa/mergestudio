"""Tests for vocab surgery and embedding dimension alignment."""

import pytest
import torch

from mergestudio.surgery.vocab import patch_vocab_size_config, reconcile_embedding_shapes


def test_reconcile_embedding_shapes_padding():
    # Base vocab size 4, hidden size 2
    t_base = torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
    # Donor vocab size 2, hidden size 2
    t_donor = torch.tensor([[10.0, 10.0], [20.0, 20.0]])

    aligned_base, aligned_donor = reconcile_embedding_shapes(t_base, t_donor)

    assert aligned_base.shape == (4, 2)
    assert aligned_donor.shape == (4, 2)

    # Base retains original values
    assert torch.allclose(aligned_base, t_base)
    # Donor retains first 2 rows, padded with zeros
    assert torch.allclose(aligned_donor[:2], t_donor)
    assert torch.allclose(aligned_donor[2:], torch.zeros(2, 2))


def test_reconcile_rank_mismatch():
    t1 = torch.zeros(4, 2)
    t2 = torch.zeros(4, 2, 2)
    with pytest.raises(ValueError, match="different ranks"):
        reconcile_embedding_shapes(t1, t2)


def test_patch_vocab_size_config():
    cfg = {"architectures": ["LlamaForCausalLM"], "vocab_size": 128000}
    patched = patch_vocab_size_config(cfg, 128256)
    assert patched["vocab_size"] == 128256
