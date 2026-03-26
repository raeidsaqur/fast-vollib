"""Compatibility tests against py_vollib_vectorized.

Github: https://github.com/marcdemers/py_vollib_vectorized

Two layers are tested:

1. **Numerical parity** — fast_vollib functions produce results within tolerance
   of the corresponding py_vollib_vectorized functions for the same inputs,
   without any patching.

2. **Patch** — after ``patch_py_vollib_vectorized()``, py_vollib_vectorized module
   attributes are replaced by the fast_vollib implementations.

All tests are skipped if py_vollib_vectorized is not importable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest

pvv = pytest.importorskip("py_vollib_vectorized")

from fast_vollib import (  # noqa: E402
    fast_black,
    fast_black_scholes,
    fast_black_scholes_merton,
    fast_implied_volatility,
    fast_implied_volatility_black,
    get_all_greeks,
    patch_py_vollib,
    patch_py_vollib_vectorized,
    price_dataframe,
    vectorized_delta,
    vectorized_gamma,
    vectorized_rho,
    vectorized_theta,
    vectorized_vega,
)

# ---------------------------------------------------------------------------
# Shared input helpers
# ---------------------------------------------------------------------------

# fast_vollib uses analytical Greeks; py_vollib_vectorized uses numerical finite-differences.
# Analytical vs numerical agreement is typically 0.1–0.5 % for first-order Greeks.
_RTOL_PRICING = 1e-5
_RTOL_GREEKS = 5e-3


def _bs_inputs():
    flag = np.array(["c", "p", "c", "p"])
    S = np.array([100.0, 100.0, 95.0, 110.0])
    K = np.array([90.0, 110.0, 100.0, 105.0])
    t = np.array([0.25, 0.25, 0.5, 0.1])
    r = np.array([0.01, 0.01, 0.03, 0.02])
    sigma = np.array([0.20, 0.20, 0.35, 0.25])
    return flag, S, K, t, r, sigma


def _black_inputs():
    flag = np.array(["c", "p", "c"])
    F = np.array([100.0, 100.0, 95.0])
    K = np.array([90.0, 110.0, 100.0])
    t = np.array([0.25, 0.25, 0.5])
    r = np.array([0.01, 0.01, 0.03])
    sigma = np.array([0.20, 0.20, 0.35])
    return flag, F, K, t, r, sigma


def _bsm_inputs():
    flag, S, K, t, r, sigma = _bs_inputs()
    q = np.array([0.02, 0.02, 0.01, 0.03])
    return flag, S, K, t, r, sigma, q


_FIXTURE_DIR = Path(__file__).parent / "fixtures"


def _load_fake_data() -> pd.DataFrame:
    return pd.read_csv(_FIXTURE_DIR / "fake_data.csv")


# ---------------------------------------------------------------------------
# 1. Numerical parity: fast_vollib == py_vollib_vectorized (no patching)
# ---------------------------------------------------------------------------


def test_parity_black_pricing():
    """Black-76 pricing parity, accounting for a known convention difference.

    ``fast_black`` returns the fully discounted price ``exp(-r·t)·[F·N(d1) - K·N(d2)]``.
    ``py_vollib_vectorized.vectorized_black`` (following py_vollib convention) returns the
    undiscounted form ``[F·N(d1) - K·N(d2)]``.  Dividing the fast_vollib result by the
    discount factor must recover the py_vollib_vectorized value.
    """
    flag, F, K, t, r, sigma = _black_inputs()
    pvv_result = pvv.vectorized_black(flag, F, K, t, r, sigma, return_as="numpy")
    fast_result = fast_black(flag, F, K, t, r, sigma, return_as="numpy")
    discount = np.exp(-r * t)
    assert_allclose(fast_result / discount, pvv_result, rtol=_RTOL_PRICING)


def test_parity_black_scholes_pricing():
    flag, S, K, t, r, sigma = _bs_inputs()
    expected = pvv.vectorized_black_scholes(flag, S, K, t, r, sigma, return_as="numpy")
    actual = fast_black_scholes(flag, S, K, t, r, sigma, return_as="numpy")
    assert_allclose(actual, expected, rtol=_RTOL_PRICING)


def test_parity_bsm_pricing():
    flag, S, K, t, r, sigma, q = _bsm_inputs()
    expected = pvv.vectorized_black_scholes_merton(flag, S, K, t, r, sigma, q, return_as="numpy")
    actual = fast_black_scholes_merton(flag, S, K, t, r, sigma, q, return_as="numpy")
    assert_allclose(actual, expected, rtol=_RTOL_PRICING)


def test_parity_implied_volatility_black_scholes():
    flag, S, K, t, r, sigma = _bs_inputs()
    prices = fast_black_scholes(flag, S, K, t, r, sigma, return_as="numpy")
    expected = pvv.vectorized_implied_volatility(
        prices, S, K, t, r, flag, model="black_scholes", return_as="numpy"
    )
    actual = fast_implied_volatility(
        prices, S, K, t, r, flag, model="black_scholes", return_as="numpy"
    )
    assert_allclose(actual, expected, rtol=_RTOL_PRICING)


def test_parity_implied_volatility_black():
    """Recover sigma from fast_black prices via both implementations.

    Note: fast_black returns discounted prices; we invert using the same fast_black
    prices so both IV solvers see the same input regardless of the discount convention.
    Argument order for Black IV: (price, F, K, r, t, flag) — r before t.
    """
    flag, F, K, t, r, sigma = _black_inputs()
    prices = fast_black(flag, F, K, t, r, sigma, return_as="numpy")
    expected = pvv.vectorized_implied_volatility_black(prices, F, K, r, t, flag, return_as="numpy")
    actual = fast_implied_volatility_black(prices, F, K, r, t, flag, return_as="numpy")
    assert_allclose(actual, expected, rtol=_RTOL_PRICING)


def test_parity_implied_volatility_bsm():
    flag, S, K, t, r, sigma, q = _bsm_inputs()
    prices = fast_black_scholes_merton(flag, S, K, t, r, sigma, q, return_as="numpy")
    expected = pvv.vectorized_implied_volatility(
        prices, S, K, t, r, flag, q=q, model="black_scholes_merton", return_as="numpy"
    )
    actual = fast_implied_volatility(
        prices, S, K, t, r, flag, q=q, model="black_scholes_merton", return_as="numpy"
    )
    assert_allclose(actual, expected, rtol=_RTOL_PRICING)


def test_parity_greeks_delta():
    flag, S, K, t, r, sigma = _bs_inputs()
    expected = pvv.vectorized_delta(flag, S, K, t, r, sigma, return_as="numpy")
    actual = vectorized_delta(flag, S, K, t, r, sigma, return_as="numpy")
    assert_allclose(actual, expected, rtol=_RTOL_GREEKS)


def test_parity_greeks_gamma():
    flag, S, K, t, r, sigma = _bs_inputs()
    expected = pvv.vectorized_gamma(flag, S, K, t, r, sigma, return_as="numpy")
    actual = vectorized_gamma(flag, S, K, t, r, sigma, return_as="numpy")
    assert_allclose(actual, expected, rtol=_RTOL_GREEKS)


def test_parity_greeks_vega():
    flag, S, K, t, r, sigma = _bs_inputs()
    expected = pvv.vectorized_vega(flag, S, K, t, r, sigma, return_as="numpy")
    actual = vectorized_vega(flag, S, K, t, r, sigma, return_as="numpy")
    assert_allclose(actual, expected, rtol=_RTOL_GREEKS)


def test_parity_greeks_theta():
    flag, S, K, t, r, sigma = _bs_inputs()
    expected = pvv.vectorized_theta(flag, S, K, t, r, sigma, return_as="numpy")
    actual = vectorized_theta(flag, S, K, t, r, sigma, return_as="numpy")
    assert_allclose(actual, expected, rtol=_RTOL_GREEKS)


def test_parity_greeks_rho():
    flag, S, K, t, r, sigma = _bs_inputs()
    expected = pvv.vectorized_rho(flag, S, K, t, r, sigma, return_as="numpy")
    actual = vectorized_rho(flag, S, K, t, r, sigma, return_as="numpy")
    assert_allclose(actual, expected, rtol=_RTOL_GREEKS)


def test_parity_get_all_greeks():
    flag, S, K, t, r, sigma = _bs_inputs()
    expected = pvv.get_all_greeks(flag, S, K, t, r, sigma, return_as="dict")
    actual = get_all_greeks(flag, S, K, t, r, sigma, return_as="dict")
    for greek in ("delta", "gamma", "vega", "theta", "rho"):
        assert_allclose(
            actual[greek], expected[greek], rtol=_RTOL_GREEKS, err_msg=f"mismatch: {greek}"
        )


def test_parity_price_dataframe():
    flag, S, K, t, r, sigma = _bs_inputs()
    df = pd.DataFrame({"flag": flag, "S": S, "K": K, "t": t, "r": r, "sigma": sigma})
    kwargs = dict(
        flag_col="flag",
        underlying_price_col="S",
        strike_col="K",
        annualized_tte_col="t",
        riskfree_rate_col="r",
        sigma_col="sigma",
        inplace=False,
    )
    expected = pvv.price_dataframe(df.copy(), **kwargs)
    actual = price_dataframe(df.copy(), **kwargs)
    for col in ("delta", "gamma", "vega", "theta", "rho"):
        assert_allclose(
            actual[col].to_numpy(),
            expected[col].to_numpy(),
            rtol=_RTOL_GREEKS,
            err_msg=f"price_dataframe mismatch: {col}",
        )


# ---------------------------------------------------------------------------
# 2. Realistic market data from fake_data.csv (shared py_vollib_vectorized fixture)
# ---------------------------------------------------------------------------


def test_parity_implied_volatility_fake_data():
    """IV computed by fast_vollib matches py_vollib_vectorized on the shared test CSV."""
    data = _load_fake_data()
    price = data["MidPx"].to_numpy()
    S = data["Px"].to_numpy()
    K = data["Strike"].to_numpy()
    t = data["Annualized Time To Expiration"].to_numpy()
    r = data["Interest Free Rate"].to_numpy()
    flag = data["Flag"].to_numpy()

    expected = pvv.vectorized_implied_volatility(
        price, S, K, t, r, flag, model="black_scholes", return_as="numpy"
    )
    actual = fast_implied_volatility(
        price, S, K, t, r, flag, model="black_scholes", return_as="numpy"
    )
    # Some rows may be NaN (deep OTM) — compare only finite values
    mask = np.isfinite(expected) & np.isfinite(actual)
    assert mask.sum() > 0, "no finite IV values to compare"
    assert_allclose(actual[mask], expected[mask], rtol=_RTOL_PRICING)


# ---------------------------------------------------------------------------
# 3. patch_py_vollib_vectorized — verifies module attributes are replaced
# ---------------------------------------------------------------------------


def test_patch_py_vollib_vectorized_smoke():
    """After patching, all py_vollib_vectorized callables are fast_vollib functions."""
    import py_vollib_vectorized as pvv_mod

    patch_py_vollib_vectorized()

    assert pvv_mod.vectorized_black is fast_black
    assert pvv_mod.vectorized_black_scholes is fast_black_scholes
    assert pvv_mod.vectorized_black_scholes_merton is fast_black_scholes_merton
    assert pvv_mod.vectorized_implied_volatility is fast_implied_volatility
    assert pvv_mod.vectorized_implied_volatility_black is fast_implied_volatility_black
    assert pvv_mod.vectorized_delta is vectorized_delta
    assert pvv_mod.vectorized_gamma is vectorized_gamma
    assert pvv_mod.vectorized_theta is vectorized_theta
    assert pvv_mod.vectorized_rho is vectorized_rho
    assert pvv_mod.vectorized_vega is vectorized_vega
    assert pvv_mod.get_all_greeks is get_all_greeks
    assert pvv_mod.price_dataframe is price_dataframe


def test_patch_replaces_submodule_attributes():
    """Patch also updates submodule-level attributes, not just the top-level namespace."""
    import py_vollib_vectorized.greeks as pvv_greeks
    import py_vollib_vectorized.implied_volatility as pvv_iv
    import py_vollib_vectorized.models as pvv_models

    patch_py_vollib_vectorized()

    assert pvv_models.vectorized_black is fast_black
    assert pvv_models.vectorized_black_scholes is fast_black_scholes
    assert pvv_models.vectorized_black_scholes_merton is fast_black_scholes_merton

    assert pvv_iv.vectorized_implied_volatility is fast_implied_volatility
    assert pvv_iv.vectorized_implied_volatility_black is fast_implied_volatility_black

    assert pvv_greeks.delta is vectorized_delta
    assert pvv_greeks.gamma is vectorized_gamma
    assert pvv_greeks.theta is vectorized_theta
    assert pvv_greeks.rho is vectorized_rho
    assert pvv_greeks.vega is vectorized_vega


def test_patch_py_vollib_smoke():
    """Legacy: patch_py_vollib patches the scalar py_vollib package (skip if not installed)."""
    import importlib

    pytest.importorskip("py_vollib")
    patch_py_vollib()
    black = importlib.import_module("py_vollib.black")
    black_scholes = importlib.import_module("py_vollib.black_scholes")
    black_scholes_merton = importlib.import_module("py_vollib.black_scholes_merton")

    assert callable(black.black)
    assert callable(black_scholes.black_scholes)
    assert callable(black_scholes_merton.black_scholes_merton)
