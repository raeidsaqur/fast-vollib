"""Numba backend parity tests.

Verifies that the numba backend produces results numerically equivalent to the
numpy reference backend (within atol=1e-6) for all three pricing models, all
five Greeks, and the implied-volatility solver.

All tests are skipped automatically when numba is not installed, so this file
can live in the standard test suite without breaking CI environments that lack
the optional ``numba`` extra.
"""

from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest

from fast_vollib import (
    fast_black,
    fast_black_scholes,
    fast_black_scholes_merton,
    fast_implied_volatility,
    fast_implied_volatility_black,
    get_all_greeks,
    price_dataframe,
)
from fast_vollib.backends import available_backends
import fast_vollib.config as config_module

# ---------------------------------------------------------------------------
# Suite-level skip when numba is not installed
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.skipif(
    "numba" not in available_backends(),
    reason="numba not installed (pip install 'fast-vollib[numba]')",
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_backend_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config_module, "_BACKEND_OVERRIDE", None)
    monkeypatch.delenv("FAST_VOLLIB_BACKEND", raising=False)


def _bs_inputs() -> tuple[np.ndarray, ...]:
    """Standard Black-Scholes inputs — call, put, call."""
    flag = np.array(["c", "p", "c"])
    s = np.array([100.0, 100.0, 95.0])
    k = np.array([90.0, 110.0, 100.0])
    t = np.array([0.25, 0.25, 0.5])
    r = np.array([0.01, 0.01, 0.03])
    sigma = np.array([0.20, 0.20, 0.35])
    return flag, s, k, t, r, sigma


def _black_inputs() -> tuple[np.ndarray, ...]:
    """Forward-price inputs for the Black-76 model."""
    flag = np.array(["c", "p", "c"])
    f = np.array([100.0, 100.0, 95.0])
    k = np.array([90.0, 110.0, 100.0])
    t = np.array([0.25, 0.25, 0.5])
    r = np.array([0.01, 0.01, 0.03])
    sigma = np.array([0.20, 0.20, 0.35])
    return flag, f, k, t, r, sigma


def _bsm_inputs() -> tuple[np.ndarray, ...]:
    """Spot + dividend inputs for Black-Scholes-Merton."""
    flag = np.array(["c", "p", "c"])
    s = np.array([100.0, 100.0, 95.0])
    k = np.array([90.0, 110.0, 100.0])
    t = np.array([0.25, 0.25, 0.5])
    r = np.array([0.05, 0.05, 0.03])
    sigma = np.array([0.20, 0.20, 0.35])
    q = np.array([0.02, 0.02, 0.01])
    return flag, s, k, t, r, sigma, q


def _bs_frame() -> pd.DataFrame:
    flag, s, k, t, r, sigma = _bs_inputs()
    return pd.DataFrame({"flag": flag, "S": s, "K": k, "t": t, "r": r, "sigma": sigma})


def _bsm_frame() -> pd.DataFrame:
    flag, s, k, t, r, sigma, q = _bsm_inputs()
    return pd.DataFrame({"flag": flag, "S": s, "K": k, "t": t, "r": r, "sigma": sigma, "q": q})


# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------


def test_numba_in_available_backends() -> None:
    assert "numba" in available_backends()


# ---------------------------------------------------------------------------
# Black-Scholes pricing
# ---------------------------------------------------------------------------


def test_numba_bs_pricing_matches_numpy() -> None:
    flag, s, k, t, r, sigma = _bs_inputs()
    ref = fast_black_scholes(flag, s, k, t, r, sigma, backend="numpy", return_as="numpy")
    out = fast_black_scholes(flag, s, k, t, r, sigma, backend="numba", return_as="numpy")
    assert_allclose(out, ref, atol=1e-6)


def test_numba_bs_pricing_output_shape() -> None:
    flag, s, k, t, r, sigma = _bs_inputs()
    out = fast_black_scholes(flag, s, k, t, r, sigma, backend="numba", return_as="numpy")
    assert out.shape == (3,)
    assert out.dtype == np.float64


# ---------------------------------------------------------------------------
# Black-76 pricing
# ---------------------------------------------------------------------------


def test_numba_black_pricing_matches_numpy() -> None:
    flag, f, k, t, r, sigma = _black_inputs()
    ref = fast_black(flag, f, k, t, r, sigma, backend="numpy", return_as="numpy")
    out = fast_black(flag, f, k, t, r, sigma, backend="numba", return_as="numpy")
    assert_allclose(out, ref, atol=1e-6)


# ---------------------------------------------------------------------------
# Black-Scholes-Merton pricing
# ---------------------------------------------------------------------------


def test_numba_bsm_pricing_matches_numpy() -> None:
    flag, s, k, t, r, sigma, q = _bsm_inputs()
    ref = fast_black_scholes_merton(flag, s, k, t, r, sigma, q, backend="numpy", return_as="numpy")
    out = fast_black_scholes_merton(flag, s, k, t, r, sigma, q, backend="numba", return_as="numpy")
    assert_allclose(out, ref, atol=1e-6)


# ---------------------------------------------------------------------------
# Black-Scholes Greeks
# ---------------------------------------------------------------------------


def test_numba_bs_greeks_match_numpy() -> None:
    flag, s, k, t, r, sigma = _bs_inputs()
    ref = get_all_greeks(flag, s, k, t, r, sigma, backend="numpy", return_as="dict")
    out = get_all_greeks(flag, s, k, t, r, sigma, backend="numba", return_as="dict")
    for greek in ("delta", "gamma", "theta", "rho", "vega"):
        assert_allclose(out[greek], ref[greek], atol=1e-6, err_msg=f"{greek} mismatch")


def test_numba_bs_greeks_keys() -> None:
    flag, s, k, t, r, sigma = _bs_inputs()
    out = get_all_greeks(flag, s, k, t, r, sigma, backend="numba", return_as="dict")
    assert set(out) == {"delta", "gamma", "theta", "rho", "vega"}


# ---------------------------------------------------------------------------
# Black-76 Greeks
# ---------------------------------------------------------------------------


def test_numba_black_greeks_match_numpy() -> None:
    flag, f, k, t, r, sigma = _black_inputs()
    ref = get_all_greeks(flag, f, k, t, r, sigma, model="black", backend="numpy", return_as="dict")
    out = get_all_greeks(flag, f, k, t, r, sigma, model="black", backend="numba", return_as="dict")
    for greek in ("delta", "gamma", "theta", "rho", "vega"):
        assert_allclose(out[greek], ref[greek], atol=1e-6, err_msg=f"{greek} mismatch")


# ---------------------------------------------------------------------------
# Black-Scholes-Merton Greeks
# ---------------------------------------------------------------------------


def test_numba_bsm_greeks_match_numpy() -> None:
    flag, s, k, t, r, sigma, q = _bsm_inputs()
    ref = get_all_greeks(
        flag,
        s,
        k,
        t,
        r,
        sigma,
        q=q,
        model="black_scholes_merton",
        backend="numpy",
        return_as="dict",
    )
    out = get_all_greeks(
        flag,
        s,
        k,
        t,
        r,
        sigma,
        q=q,
        model="black_scholes_merton",
        backend="numba",
        return_as="dict",
    )
    for greek in ("delta", "gamma", "theta", "rho", "vega"):
        assert_allclose(out[greek], ref[greek], atol=1e-6, err_msg=f"{greek} mismatch")


# ---------------------------------------------------------------------------
# Implied volatility — Black-Scholes
# ---------------------------------------------------------------------------


def test_numba_bs_iv_matches_numpy() -> None:
    flag, s, k, t, r, sigma = _bs_inputs()
    prices = fast_black_scholes(flag, s, k, t, r, sigma, backend="numpy", return_as="numpy")
    ref = fast_implied_volatility(prices, s, k, t, r, flag, backend="numpy", return_as="numpy")
    out = fast_implied_volatility(prices, s, k, t, r, flag, backend="numba", return_as="numpy")
    assert_allclose(out, ref, atol=1e-6)


def test_numba_bs_iv_recovers_input_sigma() -> None:
    flag, s, k, t, r, sigma = _bs_inputs()
    prices = fast_black_scholes(flag, s, k, t, r, sigma, backend="numba", return_as="numpy")
    iv = fast_implied_volatility(prices, s, k, t, r, flag, backend="numba", return_as="numpy")
    assert_allclose(iv, sigma, atol=1e-5)


# ---------------------------------------------------------------------------
# Implied volatility — Black-76
# ---------------------------------------------------------------------------


def test_numba_black_iv_matches_numpy() -> None:
    flag, f, k, t, r, sigma = _black_inputs()
    prices = fast_black(flag, f, k, t, r, sigma, backend="numpy", return_as="numpy")
    ref = fast_implied_volatility_black(
        prices, f, k, r, t, flag, backend="numpy", return_as="numpy"
    )
    out = fast_implied_volatility_black(
        prices, f, k, r, t, flag, backend="numba", return_as="numpy"
    )
    assert_allclose(out, ref, atol=1e-6)


# ---------------------------------------------------------------------------
# Implied volatility — Black-Scholes-Merton
# ---------------------------------------------------------------------------


def test_numba_bsm_iv_matches_numpy() -> None:
    flag, s, k, t, r, sigma, q = _bsm_inputs()
    prices = fast_black_scholes_merton(
        flag, s, k, t, r, sigma, q, backend="numpy", return_as="numpy"
    )
    ref = fast_implied_volatility(
        prices,
        s,
        k,
        t,
        r,
        flag,
        q=q,
        model="black_scholes_merton",
        backend="numpy",
        return_as="numpy",
    )
    out = fast_implied_volatility(
        prices,
        s,
        k,
        t,
        r,
        flag,
        q=q,
        model="black_scholes_merton",
        backend="numba",
        return_as="numpy",
    )
    assert_allclose(out, ref, atol=1e-6)


# ---------------------------------------------------------------------------
# price_dataframe integration
# ---------------------------------------------------------------------------


def test_numba_bs_price_dataframe_matches_numpy() -> None:
    df = _bs_frame()
    ref = price_dataframe(
        df,
        flag_col="flag",
        underlying_price_col="S",
        strike_col="K",
        annualized_tte_col="t",
        riskfree_rate_col="r",
        sigma_col="sigma",
        backend="numpy",
        inplace=False,
    )
    out = price_dataframe(
        df,
        flag_col="flag",
        underlying_price_col="S",
        strike_col="K",
        annualized_tte_col="t",
        riskfree_rate_col="r",
        sigma_col="sigma",
        backend="numba",
        inplace=False,
    )
    for col in ("Price", "delta", "gamma", "theta", "rho", "vega"):
        assert_allclose(out[col].to_numpy(), ref[col].to_numpy(), atol=1e-6, err_msg=col)


def test_numba_bsm_price_dataframe_matches_numpy() -> None:
    df = _bsm_frame()
    ref = price_dataframe(
        df,
        flag_col="flag",
        underlying_price_col="S",
        strike_col="K",
        annualized_tte_col="t",
        riskfree_rate_col="r",
        sigma_col="sigma",
        dividend_col="q",
        model="black_scholes_merton",
        backend="numpy",
        inplace=False,
    )
    out = price_dataframe(
        df,
        flag_col="flag",
        underlying_price_col="S",
        strike_col="K",
        annualized_tte_col="t",
        riskfree_rate_col="r",
        sigma_col="sigma",
        dividend_col="q",
        model="black_scholes_merton",
        backend="numba",
        inplace=False,
    )
    for col in ("Price", "delta", "gamma", "theta", "rho", "vega"):
        assert_allclose(out[col].to_numpy(), ref[col].to_numpy(), atol=1e-6, err_msg=col)


# ---------------------------------------------------------------------------
# Backend selection helpers
# ---------------------------------------------------------------------------


def test_numba_backend_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAST_VOLLIB_BACKEND", "numba")
    from fast_vollib.config import get_backend

    assert get_backend() == "numba"


def test_numba_set_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    from fast_vollib import set_backend
    from fast_vollib.config import get_backend

    set_backend("numba")
    assert get_backend() == "numba"
    set_backend("auto")


# ---------------------------------------------------------------------------
# Return-format smoke tests
# ---------------------------------------------------------------------------


def test_numba_bs_pricing_return_as_dataframe() -> None:
    flag, s, k, t, r, sigma = _bs_inputs()
    out = fast_black_scholes(flag, s, k, t, r, sigma, backend="numba", return_as="dataframe")
    assert hasattr(out, "columns")
    assert "Price" in out.columns
    assert len(out) == 3


def test_numba_bs_pricing_return_as_series() -> None:
    import pandas as pd

    flag, s, k, t, r, sigma = _bs_inputs()
    out = fast_black_scholes(flag, s, k, t, r, sigma, backend="numba", return_as="series")
    assert isinstance(out, pd.Series)
    assert len(out) == 3


def test_numba_greeks_return_as_dataframe() -> None:
    flag, s, k, t, r, sigma = _bs_inputs()
    out = get_all_greeks(flag, s, k, t, r, sigma, backend="numba", return_as="dataframe")
    assert set(out.columns) >= {"delta", "gamma", "theta", "rho", "vega"}


# ---------------------------------------------------------------------------
# Edge cases — expired options and zero-price OTM
# ---------------------------------------------------------------------------


def test_numba_iv_expired_option_returns_zero() -> None:
    """Options with t=0 (expired) should return 0, not NaN."""
    flag = np.array(["c"])
    s = np.array([100.0])
    k = np.array([100.0])
    t = np.array([0.0])  # expired
    r = np.array([0.05])
    price = np.array([0.0])
    out = fast_implied_volatility(price, s, k, t, r, flag, backend="numba", return_as="numpy")
    assert out[0] == 0.0


def test_numba_iv_zero_price_otm_returns_nan() -> None:
    """Deep OTM options with price=0 have undetermined IV → NaN."""
    flag = np.array(["c"])
    s = np.array([10.0])
    k = np.array([1000.0])
    t = np.array([0.01])
    r = np.array([0.0])
    price = np.array([0.0])
    out = fast_implied_volatility(price, s, k, t, r, flag, backend="numba", return_as="numpy")
    assert np.isnan(out[0])
