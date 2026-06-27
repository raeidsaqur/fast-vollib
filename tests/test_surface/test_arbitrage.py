"""Arbitrage detection: clean surfaces pass, seeded violations are recovered."""

from __future__ import annotations

import numpy as np
import pytest

from fast_vollib.surface import IVSurface, validate_surface


def test_flat_surface_is_clean(moneyness, maturities, flat_iv):
    surf = IVSurface.from_logmoneyness(moneyness, maturities, flat_iv)
    rep = validate_surface(surf)
    assert rep.passed
    assert rep.sas == 0.0
    assert rep.n_violations == 0
    for v in rep.metrics.values():
        assert v == pytest.approx(0.0, abs=1e-9)


def test_svi_surface_is_clean(moneyness, maturities, svi_iv):
    surf = IVSurface.from_logmoneyness(moneyness, maturities, svi_iv)
    rep = validate_surface(surf)
    assert rep.passed, rep.violations[:3]
    assert rep.metrics["bfly_frac"] == 0.0
    assert rep.metrics["cal_frac"] == 0.0
    assert rep.metrics["ndm"] == pytest.approx(0.0, abs=1e-9)


def test_seeded_calendar_crossing_recovered(moneyness, maturities, flat_iv):
    iv = flat_iv.copy()
    # Drop total variance at the last maturity → w decreases in T (crossing).
    iv[:, -1] = 0.10
    surf = IVSurface.from_logmoneyness(moneyness, maturities, iv)
    rep = validate_surface(surf)
    assert not rep.passed
    assert rep.metrics["cal_frac"] > 0.0
    # Relative depth: (w_j − w_{j+1}) / w_j with w = σ²T.
    w_prev = 0.2**2 * maturities[-2]
    w_next = 0.10**2 * maturities[-1]
    expected = (w_prev - w_next) / w_prev
    assert rep.metrics["cal_depth_max"] == pytest.approx(expected, rel=1e-9)
    assert any(v.type == "calendar" for v in rep.violations)
    cal = next(v for v in rep.violations if v.type == "calendar")
    assert cal.location[1] == pytest.approx(maturities[-2])


def test_seeded_butterfly_violation_recovered(moneyness, maturities, svi_iv):
    iv = svi_iv.copy()
    mid = moneyness.size // 2
    iv[mid, 1] *= 0.5  # sharp ATM dip → negative risk-neutral density
    surf = IVSurface.from_logmoneyness(moneyness, maturities, iv)
    rep = validate_surface(surf)
    assert not rep.passed
    assert rep.metrics["ndm"] > 0.0
    assert rep.metrics["bfly_frac"] > 0.0
    assert any(v.type == "butterfly" for v in rep.violations)


def test_seeded_vertical_violation_recovered(maturities):
    # Inverted skew steep enough to break call monotonicity in strike.
    k = np.linspace(-0.5, 0.5, 41)
    iv = np.interp(k, [-0.5, 0.0, 0.5], [0.05, 1.5, 0.05])[:, None]
    surf = IVSurface.from_logmoneyness(k, np.array([0.5]), iv)
    rep = validate_surface(surf)
    assert rep.metrics["vert_frac"] > 0.0
    assert any(v.type == "vertical" for v in rep.violations)


def test_severity_classification(moneyness, maturities, flat_iv):
    iv = flat_iv.copy()
    iv[:, -1] = 0.10
    rep = validate_surface(IVSurface.from_logmoneyness(moneyness, maturities, iv), tolerance=1e-6)
    # A 0.5 relative crossing depth is ≫ 5× the 1e-6 tolerance → severe.
    assert all(v.severity == "severe" for v in rep.violations if v.type == "calendar")


def test_fixed_strike_calendar_form_used(maturities):
    K = np.linspace(80, 120, 11)
    iv = np.full((K.size, maturities.size), 0.25)
    surf = IVSurface.from_strikes(K, maturities, iv, spot=100.0, r=0.05)
    rep = validate_surface(surf)
    assert rep.context["calendar_form"] == "undiscounted_call"
    # A flat-IV surface with positive rates is still calendar-arbitrage-free.
    assert rep.metrics["cal_frac"] == 0.0


def test_nan_quotes_excluded_from_metrics(moneyness, maturities, svi_iv):
    iv = svi_iv.copy()
    iv[0, 0] = np.nan  # a single missing quote
    surf = IVSurface.from_logmoneyness(moneyness, maturities, iv)
    rep = validate_surface(surf)
    # One NaN must not poison the aggregate metrics into a false violation.
    assert rep.metrics["cal_frac"] == 0.0
    assert rep.metrics["bfly_frac"] == 0.0


def test_wide_wing_clean_no_spurious_butterfly(maturities):
    # A flat-vol surface is exactly arbitrage-free, but over a very wide
    # moneyness range the wing call prices underflow toward 0, so the density
    # stencil is dominated by O(h²)/fp noise. Gating butterfly on the raw
    # `density < 0` (rather than the normalized magnitude vs tolerance) would
    # manufacture spurious wing violations here; the harness must not.
    k = np.linspace(-3.0, 3.0, 121)
    iv = np.full((k.size, maturities.size), 0.2)
    rep = validate_surface(IVSurface.from_logmoneyness(k, maturities, iv))
    assert rep.by_condition["butterfly"]["count"] == 0
    assert not any(v.type == "butterfly" for v in rep.violations)
    assert rep.passed


def test_all_nan_surface_not_passed(moneyness, maturities, flat_iv):
    iv = np.full_like(flat_iv, np.nan)
    rep = validate_surface(IVSurface.from_logmoneyness(moneyness, maturities, iv))
    assert rep.context["coverage"] == 0.0
    assert not rep.passed  # nothing to certify ⇒ not clean


def test_severity_bands_are_reachable(moneyness, maturities, flat_iv):
    # A shallow calendar crossing yields a small relative depth → minor/moderate,
    # confirming the severity bands are informative (not all-severe).
    # T doubles from 1.0→2.0, so a shallow crossing needs iv_last well below
    # 0.2/√2 ≈ 0.1414: 0.139²·2.0 = 0.03864 < 0.2²·1.0 = 0.04 → ~3.4% depth.
    iv = flat_iv.copy()
    iv[:, -1] = 0.139
    rep = validate_surface(IVSurface.from_logmoneyness(moneyness, maturities, iv))
    cal = [v for v in rep.violations if v.type == "calendar"]
    sev = {v.severity for v in cal}
    assert cal and sev <= {"minor", "moderate"}
    assert "severe" not in sev


def test_return_as_dict_and_json(moneyness, maturities, flat_iv):
    import json

    surf = IVSurface.from_logmoneyness(moneyness, maturities, flat_iv)
    d = validate_surface(surf, return_as="dict")
    assert d["passed"] is True
    assert set(d["metrics"]) >= {"ndm", "bfly_frac", "cal_frac"}
    j = validate_surface(surf, return_as="json")
    assert json.loads(j)["passed"] is True


def test_invalid_return_as_raises(moneyness, maturities, flat_iv):
    surf = IVSurface.from_logmoneyness(moneyness, maturities, flat_iv)
    with pytest.raises(ValueError):
        validate_surface(surf, return_as="bogus")
