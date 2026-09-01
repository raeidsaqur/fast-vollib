"""The namespace-generic pricing kernel agrees across array backends."""

from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose
import pytest

from fast_vollib.diagnostics import normalized_option_price


def _inputs():
    iv = np.array([0.15, 0.20, 0.35, 0.05])
    k = np.array([-0.30, 0.0, 0.25, 0.10])
    T = np.array([0.10, 0.50, 1.50, 2.00])
    is_call = np.array([True, False, True, False])
    return iv, k, T, is_call


def _numpy_reference():
    iv, k, T, is_call = _inputs()
    return np.asarray(normalized_option_price(iv, k, T, is_call))


def test_numpy_prices_are_finite_and_bounded():
    prices = _numpy_reference()
    assert np.all(np.isfinite(prices))
    calls = prices[[0, 2]]
    assert np.all((calls >= 0.0) & (calls < 1.0))


def test_torch_matches_numpy():
    torch = pytest.importorskip("torch", reason="torch not installed")
    iv, k, T, is_call = _inputs()
    priced = normalized_option_price(
        torch.as_tensor(iv, dtype=torch.float64),
        torch.as_tensor(k, dtype=torch.float64),
        torch.as_tensor(T, dtype=torch.float64),
        torch.as_tensor(is_call),
    )
    assert priced.dtype == torch.float64
    assert_allclose(priced.detach().cpu().numpy(), _numpy_reference(), rtol=1e-12, atol=1e-14)


def test_torch_keeps_the_autograd_tape():
    torch = pytest.importorskip("torch", reason="torch not installed")
    iv, k, T, is_call = _inputs()
    tensor = torch.as_tensor(iv, dtype=torch.float64).requires_grad_(True)
    priced = normalized_option_price(
        tensor,
        torch.as_tensor(k, dtype=torch.float64),
        torch.as_tensor(T, dtype=torch.float64),
        torch.as_tensor(is_call),
    )
    priced.sum().backward()
    assert tensor.grad is not None
    assert torch.all(torch.isfinite(tensor.grad))


def test_jax_matches_numpy():
    jax = pytest.importorskip("jax", reason="jax not installed")
    jnp = jax.numpy
    jax.config.update("jax_enable_x64", True)
    iv, k, T, is_call = _inputs()
    priced = normalized_option_price(
        jnp.asarray(iv, dtype=jnp.float64),
        jnp.asarray(k, dtype=jnp.float64),
        jnp.asarray(T, dtype=jnp.float64),
        jnp.asarray(is_call),
    )
    assert_allclose(np.asarray(priced), _numpy_reference(), rtol=1e-12, atol=1e-14)
