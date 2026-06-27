"""IVSurface container & constructor tests."""

from __future__ import annotations

import numpy as np
import pytest

from fast_vollib.surface import IVSurface, SurfaceSequence


def test_from_logmoneyness_shapes(moneyness, maturities, flat_iv):
    surf = IVSurface.from_logmoneyness(moneyness, maturities, flat_iv)
    assert surf.Nk == moneyness.size
    assert surf.Nt == maturities.size
    assert surf.shared_k is True
    k2d, T2d, w, fwd2d, disc2d = surf.broadcast()
    for arr in (k2d, T2d, w, fwd2d, disc2d):
        assert arr.shape == (surf.Nk, surf.Nt)


def test_total_variance_matches_definition(moneyness, maturities, svi_iv):
    surf = IVSurface.from_logmoneyness(moneyness, maturities, svi_iv)
    w = surf.total_variance()
    np.testing.assert_allclose(w, svi_iv**2 * maturities[None, :])


def test_from_total_variance_roundtrips(moneyness, maturities, svi_iv):
    w = svi_iv**2 * maturities[None, :]
    surf = IVSurface.from_total_variance(moneyness, maturities, w)
    np.testing.assert_allclose(surf.iv, svi_iv, rtol=1e-12)


def test_from_strikes_flat_forward_is_shared_k(maturities):
    # r = q = 0 → forward = spot (flat across T) → shared 1-D moneyness axis.
    K = np.linspace(80, 120, 11)
    iv = np.full((K.size, maturities.size), 0.25)
    surf = IVSurface.from_strikes(K, maturities, iv, spot=100.0)
    assert surf.shared_k is True
    np.testing.assert_allclose(surf.k, np.log(K / 100.0))


def test_from_strikes_term_forward_is_fixed_strike(maturities):
    # r > 0 → forward varies with T → 2-D k, fixed-strike calendar form.
    K = np.linspace(80, 120, 11)
    iv = np.full((K.size, maturities.size), 0.25)
    surf = IVSurface.from_strikes(K, maturities, iv, spot=100.0, r=0.05)
    assert surf.shared_k is False
    assert surf.k.shape == (K.size, maturities.size)


def test_call_prices_consistent_with_models(moneyness, maturities, flat_iv):
    import fast_vollib as fv

    surf = IVSurface.from_logmoneyness(moneyness, maturities, flat_iv, forward=1.0)
    undisc = surf.call_prices(undiscounted=True)
    k2d = moneyness[:, None] + 0.0 * flat_iv
    T2d = maturities[None, :] + 0.0 * flat_iv
    K = np.exp(k2d)
    ref = fv.fast_black(
        "c",
        np.ones_like(K).ravel(),
        K.ravel(),
        T2d.ravel(),
        np.zeros(K.size),
        flat_iv.ravel(),
        return_as="numpy",
    ).reshape(K.shape)
    np.testing.assert_allclose(undisc, ref, atol=1e-13)


def test_from_call_prices_recovers_iv(maturities):
    import fast_vollib as fv

    K = np.linspace(80, 120, 11)
    spot, r, q = 100.0, 0.0, 0.0
    iv_true = np.full((K.size, maturities.size), 0.3)
    K2d = np.broadcast_to(K[:, None], iv_true.shape)
    T2d = np.broadcast_to(maturities[None, :], iv_true.shape)
    prices = fv.fast_black_scholes(
        "c",
        np.full(iv_true.size, spot),
        K2d.ravel(),
        T2d.ravel(),
        np.full(iv_true.size, r),
        iv_true.ravel(),
        return_as="numpy",
    ).reshape(iv_true.shape)
    surf = IVSurface.from_call_prices(K, maturities, prices, spot=spot, r=r, q=q)
    np.testing.assert_allclose(surf.iv, iv_true, atol=1e-6)


def test_surface_sequence(moneyness, maturities, flat_iv):
    frames = [
        IVSurface.from_logmoneyness(moneyness, maturities, flat_iv, t_index=i) for i in range(3)
    ]
    seq = SurfaceSequence(frames)
    assert len(seq) == 3
    reports = seq.validate(compute_trust=False)
    assert all(r.passed for r in reports)


def test_empty_sequence_raises():
    with pytest.raises(ValueError):
        SurfaceSequence([])


def test_iv_must_be_2d(moneyness, maturities):
    with pytest.raises(ValueError):
        IVSurface.from_logmoneyness(moneyness, maturities, np.full(moneyness.size, 0.2))
