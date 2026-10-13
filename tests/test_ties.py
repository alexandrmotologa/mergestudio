"""Mathematical and algorithmic verification tests for TIES-Merging."""

import pytest
import torch

from mergestudio.algorithms.ties import _elect_sign, _trim_task_vector, ties_merge
from mergestudio.core.config import TiesConfig


def test_trim_task_vector():
    # 4 elements: magnitudes [1, 10, 2, 20]
    tv = torch.tensor([1.0, 10.0, -2.0, -20.0])

    # Density = 0.5 -> Keep top 2: 10.0 and -20.0, zero out 1.0 and -2.0
    trimmed = _trim_task_vector(tv, density=0.5)
    expected = torch.tensor([0.0, 10.0, 0.0, -20.0])
    assert torch.allclose(trimmed, expected)


def test_elect_sign_total_vs_frequency():
    # 3 models, 2 parameters
    # Param 0: deltas [+10, -2, -1] -> total sum is +7 (sign +1). frequency is +1 vs 2 negative (sign -1).
    # Param 1: deltas [-5, -4, +1] -> total sum is -8 (sign -1). frequency is 2 negative vs 1 positive (sign -1).
    stacked = torch.tensor([
        [10.0, -5.0],
        [-2.0, -4.0],
        [-1.0, 1.0],
    ])

    sign_total = _elect_sign(stacked, method="total")
    assert torch.allclose(sign_total, torch.tensor([1.0, -1.0]))

    sign_freq = _elect_sign(stacked, method="frequency")
    assert torch.allclose(sign_freq, torch.tensor([-1.0, -1.0]))


def test_ties_disjoint_merge_interference_resolution():
    base = torch.tensor([100.0, 100.0])
    # Model 1 has deltas: [+10.0, +20.0]
    # Model 2 has deltas: [+12.0, -25.0]
    # For param 0: both positive -> agreement (+10, +12) -> avg delta = +11.0 -> merged 111.0
    # For param 1: Model 1 has +20, Model 2 has -25.
    # Total sum delta for param 1 is -5 -> elected sign is -1.
    # Model 1 (+20) has discordant sign (+1 vs -1) -> zeroed out!
    # Model 2 (-25) matches elected sign -> retained!
    # Nonzero count is 1 (only Model 2) -> avg delta is -25.0 -> merged 100.0 - 25.0 = 75.0!
    m1 = torch.tensor([110.0, 120.0])
    m2 = torch.tensor([112.0, 75.0])

    merged = ties_merge(
        base=base,
        models=[m1, m2],
        density=1.0,  # no trimming to isolate sign election & disjoint merge
        majority_sign_method="total",
        normalize_signs=True,
    )

    expected = torch.tensor([111.0, 75.0])
    assert torch.allclose(merged, expected)


def test_ties_with_weights_and_scaling():
    base = torch.zeros(3)
    m1 = torch.tensor([2.0, 2.0, 2.0])
    m2 = torch.tensor([4.0, 4.0, 4.0])

    # Weights: m1=1.0, m2=3.0 -> weighted avg = (1*2 + 3*4) / 4 = 14 / 4 = 3.5
    # Global weight lambda = 2.0 -> final = 0 + 2.0 * 3.5 = 7.0
    merged = ties_merge(
        base=base,
        models=[m1, m2],
        density=1.0,
        weight=2.0,
        model_weights=[1.0, 3.0],
    )

    assert torch.allclose(merged, torch.tensor([7.0, 7.0, 7.0]))
