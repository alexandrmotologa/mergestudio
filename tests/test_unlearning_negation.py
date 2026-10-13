"""Tests for task vector negation and unlearning in TIES and DARE."""

import pytest
import torch

from mergestudio.algorithms.dare import dare_merge
from mergestudio.algorithms.ties import ties_merge
from mergestudio.core.config import MergeMethod


def test_ties_vector_negation_unlearning():
    base = torch.tensor([10.0, 10.0])
    # Fine-tuned model learned a shift of [+5.0, +5.0] -> tensor is [15.0, 15.0]
    m_learned = torch.tensor([15.0, 15.0])

    # Unlearning this behavior by negating its task vector
    unlearned = ties_merge(
        base=base,
        models=[m_learned],
        density=1.0,
        negative_mask=[True],
    )

    # Base (10.0) - Delta (5.0) = 5.0
    expected = torch.tensor([5.0, 5.0])
    assert torch.allclose(unlearned, expected)


def test_dare_vector_negation_unlearning():
    base = torch.tensor([20.0, 20.0])
    m_learned = torch.tensor([26.0, 28.0])

    # With drop_rate=0.0 and negative_mask=[True], delta [6.0, 8.0] is subtracted
    unlearned = dare_merge(
        base=base,
        models=[m_learned],
        drop_rate=0.0,
        base_method=MergeMethod.DARE_LINEAR,
        negative_mask=[True],
    )

    expected = torch.tensor([14.0, 12.0])
    assert torch.allclose(unlearned, expected)
