"""JAX backend parity & autograd for the surface penalty (best-effort, skips
without jax)."""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from fast_vollib.surface import IVSurface, arbitrage_penalty, validate_surface  # noqa: E402

jax.config.update("jax_enable_x64", True)


def _seed(violation: bool = True):
    k = np.linspace(-0.4, 0.4, 21)
    T = np.array([0.1, 0.25, 0.5, 1.0])
    a, b, rho, m, sig = 0.04, 0.4, -0.4, 0.0, 0.1
    w = np.outer(a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sig**2)), T / T[-1])
    iv = np.sqrt(w / T[None, :])
    if violation:
        iv = iv.copy()
        iv[10, 2] *= 0.6
    return k, T, iv


def test_jax_penalty_matches_numpy():
    k, T, iv = _seed()
    pen_np = float(arbitrage_penalty(iv, k, T, 1.0, 0.0))
    pen_jx = float(arbitrage_penalty(jnp.asarray(iv), jnp.asarray(k), jnp.asarray(T), 1.0, 0.0))
    assert pen_jx == pytest.approx(pen_np, rel=1e-10, abs=1e-12)


def test_jax_autograd_finite():
    k, T, iv = _seed()

    def loss(x):
        return arbitrage_penalty(x, jnp.asarray(k), jnp.asarray(T), 1.0, 0.0)

    grad = jax.grad(loss)(jnp.asarray(iv))
    assert bool(jnp.isfinite(grad).all())
    assert float(jnp.linalg.norm(grad)) > 0.0


def test_jax_surface_report_path():
    k, T, iv = _seed()
    rep = validate_surface(
        IVSurface.from_logmoneyness(jnp.asarray(k), jnp.asarray(T), jnp.asarray(iv))
    )
    assert not rep.passed  # the seeded violation is detected through the jax path
    assert 0.0 < rep.sas <= 1.0
