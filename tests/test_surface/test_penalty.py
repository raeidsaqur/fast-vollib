"""Differentiable penalty: backend parity, autograd correctness, NaN handling."""

from __future__ import annotations

import numpy as np
import pytest

from fast_vollib.surface import IVSurface, arbitrage_penalty
from fast_vollib.surface.penalty import penalty_from_surface

torch = pytest.importorskip("torch")


def _seed_surface():
    k = np.linspace(-0.4, 0.4, 21)
    T = np.array([0.1, 0.25, 0.5, 1.0])
    a, b, rho, m, sig = 0.04, 0.4, -0.4, 0.0, 0.1
    w_shape = a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sig**2))
    w = np.outer(w_shape, T / T[-1])
    iv = np.sqrt(w / T[None, :])
    return k, T, iv


def test_penalty_zero_on_clean_surface():
    k, T, iv = _seed_surface()
    pen = arbitrage_penalty(iv, k, T, 1.0, 0.0)
    assert float(pen) == pytest.approx(0.0, abs=1e-10)


def test_penalty_positive_on_violation():
    k, T, iv = _seed_surface()
    iv = iv.copy()
    iv[10, 2] *= 0.5
    pen = arbitrage_penalty(iv, k, T, 1.0, 0.0)
    assert float(pen) > 0.0


def test_numpy_torch_parity():
    k, T, iv = _seed_surface()
    iv = iv.copy()
    iv[10, 2] *= 0.6  # a real violation so the penalty is non-trivial
    pen_np = float(arbitrage_penalty(iv, k, T, 1.0, 0.0))
    pen_t = float(
        arbitrage_penalty(
            torch.tensor(iv, dtype=torch.float64),
            torch.tensor(k, dtype=torch.float64),
            torch.tensor(T, dtype=torch.float64),
            1.0,
            0.0,
        )
    )
    assert pen_t == pytest.approx(pen_np, rel=1e-10, abs=1e-12)


def test_autograd_gradient_finite_difference():
    k, T, iv = _seed_surface()
    iv = iv.copy()
    iv[10, 2] *= 0.5  # strong violation → smooth, away from relu kinks
    iv_t = torch.tensor(iv, dtype=torch.float64, requires_grad=True)
    kt = torch.tensor(k, dtype=torch.float64)
    Tt = torch.tensor(T, dtype=torch.float64)

    pen = arbitrage_penalty(iv_t, kt, Tt, 1.0, 0.0)
    pen.backward()
    grad = iv_t.grad.clone()
    assert torch.isfinite(grad).all()

    # Central finite difference at the most-sensitive node.
    i, j = np.unravel_index(int(torch.argmax(grad.abs())), grad.shape)
    eps = 1e-6
    iv_p = iv.copy()
    iv_p[i, j] += eps
    iv_m = iv.copy()
    iv_m[i, j] -= eps
    fd = (
        float(arbitrage_penalty(iv_p, k, T, 1.0, 0.0))
        - float(arbitrage_penalty(iv_m, k, T, 1.0, 0.0))
    ) / (2 * eps)
    assert fd == pytest.approx(float(grad[i, j]), rel=1e-4, abs=1e-6)


def test_gradcheck_small_surface():
    # torch.autograd.gradcheck on a strongly-violating surface (locally smooth).
    k, T, iv = _seed_surface()
    iv = iv.copy()
    iv[10, 2] *= 0.5
    iv_t = torch.tensor(iv, dtype=torch.float64, requires_grad=True)
    kt = torch.tensor(k, dtype=torch.float64)
    Tt = torch.tensor(T, dtype=torch.float64)
    assert torch.autograd.gradcheck(
        lambda x: arbitrage_penalty(x, kt, Tt, 1.0, 0.0),
        (iv_t,),
        eps=1e-6,
        atol=1e-4,
        rtol=1e-3,
    )


def test_reduction_modes():
    k, T, iv = _seed_surface()
    iv = iv.copy()
    iv[10, 2] *= 0.5
    pmean = float(arbitrage_penalty(iv, k, T, 1.0, 0.0, reduction="mean"))
    psum = float(arbitrage_penalty(iv, k, T, 1.0, 0.0, reduction="sum"))
    assert psum >= pmean > 0.0
    with pytest.raises(ValueError):
        arbitrage_penalty(iv, k, T, 1.0, 0.0, reduction="bogus")


def test_nan_quotes_contribute_no_penalty():
    k, T, iv = _seed_surface()
    iv = iv.copy()
    iv[0, 0] = np.nan
    pen = arbitrage_penalty(iv, k, T, 1.0, 0.0)
    assert np.isfinite(float(pen))


def test_penalty_from_surface_matches_direct():
    k, T, iv = _seed_surface()
    iv = iv.copy()
    iv[10, 2] *= 0.6
    surf = IVSurface.from_logmoneyness(k, T, iv)
    direct = float(arbitrage_penalty(iv, k, T, surf.forward, surf.r))
    via = float(penalty_from_surface(surf))
    assert via == pytest.approx(direct, rel=1e-12)
