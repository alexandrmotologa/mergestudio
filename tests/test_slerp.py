"""Mathematical and algorithmic verification tests for SLERP."""

import math
import pytest
import torch

from mergestudio.algorithms.slerp import slerp
from mergestudio.core.config import SlerpConfig


def test_slerp_boundaries():
    v0 = torch.tensor([1.0, 0.0, 0.0])
    v1 = torch.tensor([0.0, 1.0, 0.0])

    res_0 = slerp(v0, v1, alpha=0.0)
    assert torch.allclose(res_0, v0)

    res_1 = slerp(v0, v1, alpha=1.0)
    assert torch.allclose(res_1, v1)


def test_slerp_orthogonal_midpoint():
    v0 = torch.tensor([1.0, 0.0, 0.0])
    v1 = torch.tensor([0.0, 1.0, 0.0])

    # At alpha = 0.5 for orthogonal unit vectors, angle theta = pi/2
    # Result should be [cos(pi/4), sin(pi/4), 0] = [sqrt(2)/2, sqrt(2)/2, 0]
    res = slerp(v0, v1, alpha=0.5)
    expected_val = math.sqrt(2.0) / 2.0
    expected = torch.tensor([expected_val, expected_val, 0.0])

    assert torch.allclose(res, expected, atol=1e-5)
    # Norm should be exactly 1.0
    assert torch.isclose(torch.linalg.norm(res), torch.tensor(1.0), atol=1e-5)


def test_slerp_norm_preservation():
    # v0 has norm 2.0, v1 has norm 4.0
    v0 = torch.tensor([2.0, 0.0, 0.0])
    v1 = torch.tensor([0.0, 4.0, 0.0])

    # At alpha = 0.5, expected norm is (2.0 + 4.0) / 2 = 3.0
    res = slerp(v0, v1, alpha=0.5, preserve_norm=True)
    norm = torch.linalg.norm(res)
    assert torch.isclose(norm, torch.tensor(3.0), atol=1e-5)


def test_slerp_collinear_fallback():
    # Identical direction vectors should seamlessly fall back to lerp
    v0 = torch.tensor([2.0, 4.0, 6.0])
    v1 = torch.tensor([4.0, 8.0, 12.0])

    res = slerp(v0, v1, alpha=0.5)
    expected = torch.tensor([3.0, 6.0, 9.0])
    assert torch.allclose(res, expected, atol=1e-5)


def test_slerp_per_row():
    # 2D matrix (2 rows, 3 columns)
    v0 = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 2.0],
    ])
    v1 = torch.tensor([
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 4.0],
    ])

    res = slerp(v0, v1, alpha=0.5, per_row=True, preserve_norm=True)

    # First row: orthogonal unit vectors
    val = math.sqrt(2.0) / 2.0
    assert torch.allclose(res[0], torch.tensor([val, val, 0.0]), atol=1e-5)

    # Second row: collinear along z-axis, norm (2+4)/2 = 3.0
    assert torch.allclose(res[1], torch.tensor([0.0, 0.0, 3.0]), atol=1e-5)


def test_slerp_shape_mismatch():
    v0 = torch.zeros((4, 4))
    v1 = torch.zeros((4, 8))

    with pytest.raises(ValueError, match="Tensor shape mismatch"):
        slerp(v0, v1, alpha=0.5)


def test_slerp_dtype_preservation():
    v0 = torch.randn(10, 10, dtype=torch.float32)
    v1 = torch.randn(10, 10, dtype=torch.float32)

    res = slerp(v0, v1, alpha=0.3)
    assert res.dtype == torch.float32
    assert res.shape == v0.shape
