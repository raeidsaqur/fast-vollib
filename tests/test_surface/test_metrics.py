"""Stencil accuracy, artifact separation, and round-trip trust mask."""

from __future__ import annotations

import numpy as np
import pytest

from fast_vollib.surface import IVSurface, validate_surface
from fast_vollib.surface._xp import numpy_namespace
from fast_vollib.surface.density import durrleman_g, parabolic_derivatives

from .conftest import (
    SVI_PARAMS,
    svi_g,
    svi_total_variance,
    svi_w_double_prime,
    svi_w_prime,
)


def test_parabolic_derivatives_match_svi_closed_form():
    xp = numpy_namespace()
    k = np.linspace(-0.6, 0.6, 241)  # dense → tight O(h²) truncation
    w = svi_total_variance(k, **SVI_PARAMS)[:, None]
    first, second = parabolic_derivatives(k, w, xp)
    ki = k[1:-1]
    np.testing.assert_allclose(first[:, 0], svi_w_prime(ki, **SVI_PARAMS), atol=2e-4)
    np.testing.assert_allclose(second[:, 0], svi_w_double_prime(ki, **SVI_PARAMS), atol=3e-3)


def test_durrleman_g_matches_svi_closed_form():
    xp = numpy_namespace()
    k = np.linspace(-0.6, 0.6, 241)
    w = svi_total_variance(k, **SVI_PARAMS)[:, None]
    g_num = durrleman_g(k, w, xp)[:, 0]
    np.testing.assert_allclose(g_num, svi_g(k[1:-1], **SVI_PARAMS), atol=2e-3)
    assert g_num.min() > 0.0  # the fixture really is butterfly-free


def test_stencils_converge_second_order():
    xp = numpy_namespace()
    errs = []
    for n in (61, 121, 241):
        k = np.linspace(-0.6, 0.6, n)
        w = svi_total_variance(k, **SVI_PARAMS)[:, None]
        _, second = parabolic_derivatives(k, w, xp)
        err = np.max(np.abs(second[:, 0] - svi_w_double_prime(k[1:-1], **SVI_PARAMS)))
        errs.append(err)
    # Halving h should cut the error by ≈4× (second order).
    assert errs[0] / errs[1] > 3.0
    assert errs[1] / errs[2] > 3.0


def test_ndm_zero_on_clean_positive_on_dip(moneyness, maturities, svi_iv):
    clean = validate_surface(IVSurface.from_logmoneyness(moneyness, maturities, svi_iv))
    assert clean.metrics["ndm"] == pytest.approx(0.0, abs=1e-9)
    iv = svi_iv.copy()
    iv[moneyness.size // 2, 1] *= 0.5
    dipped = validate_surface(IVSurface.from_logmoneyness(moneyness, maturities, iv))
    assert 0.0 < dipped.metrics["ndm"] <= 1.0


def test_sas_is_zero_for_clean_and_bounded(moneyness, maturities, flat_iv):
    rep = validate_surface(IVSurface.from_logmoneyness(moneyness, maturities, flat_iv))
    assert rep.sas == 0.0
    iv = flat_iv.copy()
    iv[:, -1] = 0.05
    bad = validate_surface(IVSurface.from_logmoneyness(moneyness, maturities, iv))
    assert 0.0 < bad.sas <= 1.0


def test_artifact_separation_labels_interpolated_nodes(moneyness, maturities, svi_iv):
    iv = svi_iv.copy()
    mid = moneyness.size // 2
    iv[mid, 1] *= 0.5
    native = np.ones_like(iv, dtype=bool)
    native[mid, 1] = False  # the dipped node is an interpolated (non-native) node
    surf = IVSurface.from_logmoneyness(moneyness, maturities, iv, native_mask=native)
    rep = validate_surface(surf)
    total_interp = sum(rep.interpolation_induced.values())
    assert total_interp > 0.0
    # At least one localized violation is tagged interpolation_induced.
    assert any(v.origin == "interpolation_induced" for v in rep.violations)


def test_trust_mask_clean_surface_all_trusted(moneyness, maturities, svi_iv):
    surf = IVSurface.from_logmoneyness(moneyness, maturities, svi_iv)
    rep = validate_surface(surf)
    # Well-posed quotes round-trip σ→C→σ' to machine precision.
    assert rep.trust_mask is not None
    assert float(np.mean(rep.trust_mask)) > 0.99


def test_trust_mask_oracle_residual_tiny(moneyness, maturities, svi_iv):
    from fast_vollib.jackel.jackel_iv import jackel_iv_black
    from fast_vollib.surface.transforms import undiscounted_call

    xp = numpy_namespace()
    surf = IVSurface.from_logmoneyness(moneyness, maturities, svi_iv, forward=1.0)
    k2d, T2d, w, fwd2d, _ = surf.broadcast(xp)
    c = undiscounted_call(k2d, w, fwd2d, xp)
    K = fwd2d * np.exp(k2d)
    sig = jackel_iv_black(c.ravel(), fwd2d.ravel(), K.ravel(), T2d.ravel(), True)
    resid = np.nanmax(np.abs(sig.reshape(svi_iv.shape) - svi_iv))
    assert resid < 1e-8


def test_compute_trust_can_be_disabled(moneyness, maturities, flat_iv):
    surf = IVSurface.from_logmoneyness(moneyness, maturities, flat_iv)
    rep = validate_surface(surf, compute_trust=False)
    assert rep.trust_mask is None
