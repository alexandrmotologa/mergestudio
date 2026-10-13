"""Mathematical and algorithmic verification tests for DARE."""

import pytest
import torch

from mergestudio.algorithms.dare import dare_merge
from mergestudio.core.config import MergeMethod


def test_dare_zero_drop_rate():
    base = torch.zeros(4)
    m1 = torch.tensor([1.0, 2.0, 3.0, 4.0])
    m2 = torch.tensor([2.0, 4.0, 6.0, 8.0])

    # With drop_rate = 0, DARE-Linear equals standard average
    merged = dare_merge(
        base=base,
        models=[m1, m2],
        drop_rate=0.0,
        base_method=MergeMethod.DARE_LINEAR,
    )
    expected = torch.tensor([1.5, 3.0, 4.5, 6.0])
    assert torch.allclose(merged, expected)


def test_dare_rescaling_expectation():
    # Statistical expectation test:
    # Over 100,000 parameters, mean of rescaled deltas should equal true delta
    torch.manual_seed(42)
    n = 100_000
    base = torch.zeros(n)
    true_delta = 5.0
    m1 = torch.full((n,), true_delta)

    # Drop rate 0.5 with rescale=True -> scaling factor 2.0
    merged = dare_merge(
        base=base,
        models=[m1],
        drop_rate=0.5,
        rescale=True,
        seed=1337,
        base_method=MergeMethod.DARE_LINEAR,
    )

    empirical_mean = torch.mean(merged).item()
    # Expectation should be very close to true_delta (5.0) within 1.5%
    assert abs(empirical_mean - true_delta) / true_delta < 0.015


def test_dare_invalid_drop_rate():
    base = torch.zeros(2)
    m1 = torch.ones(2)

    with pytest.raises(ValueError):
        dare_merge(base, [m1], drop_rate=1.0)

    with pytest.raises(ValueError):
        dare_merge(base, [m1], drop_rate=-0.1)
