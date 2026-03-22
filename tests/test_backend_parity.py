from __future__ import annotations

import numpy as np
import pytest

from fastiv import vectorized_black_scholes
from fastiv.backends import available_backends


def _inputs():
    flag = np.array(["c", "p", "c"])
    s = np.array([100.0, 100.0, 95.0])
    k = np.array([90.0, 110.0, 100.0])
    t = np.array([0.25, 0.25, 0.5])
    r = np.array([0.01, 0.01, 0.03])
    sigma = np.array([0.2, 0.2, 0.35])
    return flag, s, k, t, r, sigma


def test_numpy_backend_parity_baseline() -> None:
    flag, s, k, t, r, sigma = _inputs()
    values = vectorized_black_scholes(flag, s, k, t, r, sigma, backend="numpy", return_as="numpy")
    assert values.shape == (3,)


@pytest.mark.skipif("torch" not in available_backends(), reason="torch not installed")
def test_torch_backend_matches_numpy() -> None:
    flag, s, k, t, r, sigma = _inputs()
    base = vectorized_black_scholes(flag, s, k, t, r, sigma, backend="numpy", return_as="numpy")
    trial = vectorized_black_scholes(flag, s, k, t, r, sigma, backend="torch", return_as="numpy")
    assert np.allclose(base, trial, atol=1e-6)


@pytest.mark.skipif("jax" not in available_backends(), reason="jax not installed")
def test_jax_backend_matches_numpy() -> None:
    flag, s, k, t, r, sigma = _inputs()
    base = vectorized_black_scholes(flag, s, k, t, r, sigma, backend="numpy", return_as="numpy")
    trial = vectorized_black_scholes(flag, s, k, t, r, sigma, backend="jax", return_as="numpy")
    assert np.allclose(base, trial, atol=1e-6)
