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
from fast_vollib.config import get_backend


def _inputs() -> tuple[np.ndarray, ...]:
    flag = np.array(["c", "p", "c"])
    s = np.array([100.0, 100.0, 95.0])
    k = np.array([90.0, 110.0, 100.0])
    t = np.array([0.25, 0.25, 0.5])
    r = np.array([0.01, 0.01, 0.03])
    sigma = np.array([0.2, 0.2, 0.35])
    return flag, s, k, t, r, sigma


@pytest.fixture(autouse=True)
def _reset_backend_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config_module, "_BACKEND_OVERRIDE", None)
    monkeypatch.delenv("FAST_VOLLIB_BACKEND", raising=False)


def _price_inputs_frame() -> pd.DataFrame:
    flag, s, k, t, r, sigma = _inputs()
    return pd.DataFrame(
        {
            "flag": flag,
            "S": s,
            "K": k,
            "t": t,
            "r": r,
            "sigma": sigma,
        }
    )


def test_numpy_backend_parity_baseline() -> None:
    flag, s, k, t, r, sigma = _inputs()
    values = fast_black_scholes(
        flag,
        s,
        k,
        t,
        r,
        sigma,
        backend="numpy",
        return_as="numpy",
    )
    assert values.shape == (3,)


def test_backend_auto_uses_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAST_VOLLIB_BACKEND", "numpy")
    assert get_backend() == "numpy"


def test_backend_auto_prefers_torch_over_jax(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config_module, "_torch_cuda_available", lambda: True)
    monkeypatch.setattr(config_module, "_jax_available", lambda: True)
    assert get_backend("auto") == "torch"


def test_backend_auto_matches_numpy_when_resolved_to_numpy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config_module, "_torch_cuda_available", lambda: False)
    monkeypatch.setattr(config_module, "_jax_available", lambda: False)
    flag, s, k, t, r, sigma = _inputs()
    base = fast_black_scholes(
        flag,
        s,
        k,
        t,
        r,
        sigma,
        backend="numpy",
        return_as="numpy",
    )
    auto = fast_black_scholes(
        flag,
        s,
        k,
        t,
        r,
        sigma,
        backend="auto",
        return_as="numpy",
    )
    assert_allclose(base, auto, atol=1e-6)


@pytest.mark.skipif("torch" not in available_backends(), reason="torch not installed")
def test_torch_pricing_backend_matches_numpy() -> None:
    flag, s, k, t, r, sigma = _inputs()
    base = fast_black_scholes(flag, s, k, t, r, sigma, backend="numpy", return_as="numpy")
    trial = fast_black_scholes(flag, s, k, t, r, sigma, backend="torch", return_as="numpy")
    assert_allclose(base, trial, atol=1e-6)


@pytest.mark.skipif("jax" not in available_backends(), reason="jax not installed")
def test_jax_pricing_backend_matches_numpy() -> None:
    flag, s, k, t, r, sigma = _inputs()
    base = fast_black_scholes(flag, s, k, t, r, sigma, backend="numpy", return_as="numpy")
    trial = fast_black_scholes(flag, s, k, t, r, sigma, backend="jax", return_as="numpy")
    assert_allclose(base, trial, atol=1e-6)


@pytest.mark.skipif("torch" not in available_backends(), reason="torch not installed")
def test_torch_iv_backend_matches_numpy() -> None:
    flag, s, k, t, r, sigma = _inputs()
    prices = fast_black_scholes(flag, s, k, t, r, sigma, backend="numpy", return_as="numpy")
    base = fast_implied_volatility(prices, s, k, t, r, flag, backend="numpy", return_as="numpy")
    trial = fast_implied_volatility(prices, s, k, t, r, flag, backend="torch", return_as="numpy")
    assert_allclose(base, trial, atol=1e-6)


@pytest.mark.skipif("jax" not in available_backends(), reason="jax not installed")
def test_jax_iv_backend_matches_numpy() -> None:
    flag, s, k, t, r, sigma = _inputs()
    prices = fast_black_scholes(flag, s, k, t, r, sigma, backend="numpy", return_as="numpy")
    base = fast_implied_volatility(prices, s, k, t, r, flag, backend="numpy", return_as="numpy")
    trial = fast_implied_volatility(prices, s, k, t, r, flag, backend="jax", return_as="numpy")
    assert_allclose(base, trial, atol=1e-6)


@pytest.mark.skipif("torch" not in available_backends(), reason="torch not installed")
def test_torch_greeks_backend_matches_numpy() -> None:
    flag, s, k, t, r, sigma = _inputs()
    base = get_all_greeks(flag, s, k, t, r, sigma, backend="numpy", return_as="dict")
    trial = get_all_greeks(flag, s, k, t, r, sigma, backend="torch", return_as="dict")
    for name in ("delta", "gamma", "theta", "rho", "vega"):
        assert_allclose(base[name], trial[name], atol=1e-6)


@pytest.mark.skipif("jax" not in available_backends(), reason="jax not installed")
def test_jax_greeks_backend_matches_numpy() -> None:
    flag, s, k, t, r, sigma = _inputs()
    base = get_all_greeks(flag, s, k, t, r, sigma, backend="numpy", return_as="dict")
    trial = get_all_greeks(flag, s, k, t, r, sigma, backend="jax", return_as="dict")
    for name in ("delta", "gamma", "theta", "rho", "vega"):
        assert_allclose(base[name], trial[name], atol=1e-6)


@pytest.mark.skipif("torch" not in available_backends(), reason="torch not installed")
def test_torch_price_dataframe_matches_numpy() -> None:
    df = _price_inputs_frame()
    base = price_dataframe(
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
    trial = price_dataframe(
        df,
        flag_col="flag",
        underlying_price_col="S",
        strike_col="K",
        annualized_tte_col="t",
        riskfree_rate_col="r",
        sigma_col="sigma",
        backend="torch",
        inplace=False,
    )
    for column in ("Price", "delta", "gamma", "theta", "rho", "vega"):
        assert_allclose(base[column].to_numpy(), trial[column].to_numpy(), atol=1e-6)


@pytest.mark.skipif("jax" not in available_backends(), reason="jax not installed")
def test_jax_price_dataframe_matches_numpy() -> None:
    df = _price_inputs_frame()
    base = price_dataframe(
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
    trial = price_dataframe(
        df,
        flag_col="flag",
        underlying_price_col="S",
        strike_col="K",
        annualized_tte_col="t",
        riskfree_rate_col="r",
        sigma_col="sigma",
        backend="jax",
        inplace=False,
    )
    for column in ("Price", "delta", "gamma", "theta", "rho", "vega"):
        assert_allclose(base[column].to_numpy(), trial[column].to_numpy(), atol=1e-6)


@pytest.mark.skipif("torch" not in available_backends(), reason="torch not installed")
def test_torch_backend_accepts_tensor_inputs_and_returns_native() -> None:
    import torch

    flag, s, k, t, r, sigma = _inputs()
    native = fast_black_scholes(
        flag,
        torch.as_tensor(s, dtype=torch.float64),
        torch.as_tensor(k, dtype=torch.float64),
        torch.as_tensor(t, dtype=torch.float64),
        torch.as_tensor(r, dtype=torch.float64),
        torch.as_tensor(sigma, dtype=torch.float64),
        backend="torch",
        return_native=True,
    )
    assert isinstance(native, torch.Tensor)
    expected_device = "cuda" if torch.cuda.is_available() else "cpu"
    assert native.device.type == expected_device


@pytest.mark.skipif("jax" not in available_backends(), reason="jax not installed")
def test_jax_backend_accepts_array_inputs_and_returns_native() -> None:
    import jax
    import jax.numpy as jnp

    flag, s, k, t, r, sigma = _inputs()
    native = fast_black_scholes(
        flag,
        jnp.asarray(s, dtype=jnp.float64),
        jnp.asarray(k, dtype=jnp.float64),
        jnp.asarray(t, dtype=jnp.float64),
        jnp.asarray(r, dtype=jnp.float64),
        jnp.asarray(sigma, dtype=jnp.float64),
        backend="jax",
        return_native=True,
    )
    assert isinstance(native, jax.Array)
    assert len(native.devices()) >= 1


@pytest.mark.skipif("torch" not in available_backends(), reason="torch not installed")
def test_torch_greeks_return_native_returns_tensor_dict() -> None:
    import torch

    flag, s, k, t, r, sigma = _inputs()
    native = get_all_greeks(
        flag,
        torch.as_tensor(s, dtype=torch.float64),
        torch.as_tensor(k, dtype=torch.float64),
        torch.as_tensor(t, dtype=torch.float64),
        torch.as_tensor(r, dtype=torch.float64),
        torch.as_tensor(sigma, dtype=torch.float64),
        backend="torch",
        return_native=True,
    )
    assert isinstance(native, dict)
    assert set(native) == {"delta", "gamma", "theta", "rho", "vega"}
    assert all(isinstance(value, torch.Tensor) for value in native.values())


@pytest.mark.skipif("jax" not in available_backends(), reason="jax not installed")
def test_jax_greeks_return_native_returns_array_dict() -> None:
    import jax
    import jax.numpy as jnp

    flag, s, k, t, r, sigma = _inputs()
    native = get_all_greeks(
        flag,
        jnp.asarray(s, dtype=jnp.float64),
        jnp.asarray(k, dtype=jnp.float64),
        jnp.asarray(t, dtype=jnp.float64),
        jnp.asarray(r, dtype=jnp.float64),
        jnp.asarray(sigma, dtype=jnp.float64),
        backend="jax",
        return_native=True,
    )
    assert isinstance(native, dict)
    assert set(native) == {"delta", "gamma", "theta", "rho", "vega"}
    assert all(isinstance(value, jax.Array) for value in native.values())


# ---------------------------------------------------------------------------
# Black-76 model — backend parity (torch + jax vs numpy)
# ---------------------------------------------------------------------------


def _black_inputs() -> tuple[np.ndarray, ...]:
    """Forward-price inputs for Black-76 model."""
    flag = np.array(["c", "p", "c"])
    f = np.array([100.0, 100.0, 95.0])  # forward price
    k = np.array([90.0, 110.0, 100.0])
    t = np.array([0.25, 0.25, 0.5])
    r = np.array([0.01, 0.01, 0.03])
    sigma = np.array([0.20, 0.20, 0.35])
    return flag, f, k, t, r, sigma


@pytest.mark.skipif("torch" not in available_backends(), reason="torch not installed")
def test_torch_black_pricing_matches_numpy() -> None:
    flag, f, k, t, r, sigma = _black_inputs()
    base = fast_black(flag, f, k, t, r, sigma, backend="numpy", return_as="numpy")
    trial = fast_black(flag, f, k, t, r, sigma, backend="torch", return_as="numpy")
    assert_allclose(base, trial, atol=1e-6)


@pytest.mark.skipif("jax" not in available_backends(), reason="jax not installed")
def test_jax_black_pricing_matches_numpy() -> None:
    flag, f, k, t, r, sigma = _black_inputs()
    base = fast_black(flag, f, k, t, r, sigma, backend="numpy", return_as="numpy")
    trial = fast_black(flag, f, k, t, r, sigma, backend="jax", return_as="numpy")
    assert_allclose(base, trial, atol=1e-6)


@pytest.mark.skipif("torch" not in available_backends(), reason="torch not installed")
def test_torch_black_iv_matches_numpy() -> None:
    flag, f, k, t, r, sigma = _black_inputs()
    prices = fast_black(flag, f, k, t, r, sigma, backend="numpy", return_as="numpy")
    # fast_implied_volatility_black(price, F, K, r, t, flag) — note r before t
    base = fast_implied_volatility_black(
        prices, f, k, r, t, flag, backend="numpy", return_as="numpy"
    )
    trial = fast_implied_volatility_black(
        prices, f, k, r, t, flag, backend="torch", return_as="numpy"
    )
    assert_allclose(base, trial, atol=1e-6)


@pytest.mark.skipif("jax" not in available_backends(), reason="jax not installed")
def test_jax_black_iv_matches_numpy() -> None:
    flag, f, k, t, r, sigma = _black_inputs()
    prices = fast_black(flag, f, k, t, r, sigma, backend="numpy", return_as="numpy")
    base = fast_implied_volatility_black(
        prices, f, k, r, t, flag, backend="numpy", return_as="numpy"
    )
    trial = fast_implied_volatility_black(
        prices, f, k, r, t, flag, backend="jax", return_as="numpy"
    )
    assert_allclose(base, trial, atol=1e-6)


@pytest.mark.skipif("torch" not in available_backends(), reason="torch not installed")
def test_torch_black_greeks_match_numpy() -> None:
    flag, f, k, t, r, sigma = _black_inputs()
    base = get_all_greeks(flag, f, k, t, r, sigma, model="black", backend="numpy", return_as="dict")
    trial = get_all_greeks(
        flag, f, k, t, r, sigma, model="black", backend="torch", return_as="dict"
    )
    for name in ("delta", "gamma", "theta", "rho", "vega"):
        assert_allclose(base[name], trial[name], atol=1e-6)


@pytest.mark.skipif("jax" not in available_backends(), reason="jax not installed")
def test_jax_black_greeks_match_numpy() -> None:
    flag, f, k, t, r, sigma = _black_inputs()
    base = get_all_greeks(flag, f, k, t, r, sigma, model="black", backend="numpy", return_as="dict")
    trial = get_all_greeks(flag, f, k, t, r, sigma, model="black", backend="jax", return_as="dict")
    for name in ("delta", "gamma", "theta", "rho", "vega"):
        assert_allclose(base[name], trial[name], atol=1e-6)


# ---------------------------------------------------------------------------
# Black-Scholes-Merton model — backend parity (torch + jax vs numpy)
# ---------------------------------------------------------------------------


def _bsm_inputs() -> tuple[np.ndarray, ...]:
    """Spot-price + dividend inputs for Black-Scholes-Merton model."""
    flag = np.array(["c", "p", "c"])
    s = np.array([100.0, 100.0, 95.0])
    k = np.array([90.0, 110.0, 100.0])
    t = np.array([0.25, 0.25, 0.5])
    r = np.array([0.05, 0.05, 0.03])
    sigma = np.array([0.20, 0.20, 0.35])
    q = np.array([0.02, 0.02, 0.01])  # continuous dividend yield
    return flag, s, k, t, r, sigma, q


def _bsm_inputs_frame() -> pd.DataFrame:
    flag, s, k, t, r, sigma, q = _bsm_inputs()
    return pd.DataFrame(
        {
            "flag": flag,
            "S": s,
            "K": k,
            "t": t,
            "r": r,
            "sigma": sigma,
            "q": q,
        }
    )


@pytest.mark.skipif("torch" not in available_backends(), reason="torch not installed")
def test_torch_bsm_pricing_matches_numpy() -> None:
    flag, s, k, t, r, sigma, q = _bsm_inputs()
    base = fast_black_scholes_merton(flag, s, k, t, r, sigma, q, backend="numpy", return_as="numpy")
    trial = fast_black_scholes_merton(
        flag, s, k, t, r, sigma, q, backend="torch", return_as="numpy"
    )
    assert_allclose(base, trial, atol=1e-6)


@pytest.mark.skipif("jax" not in available_backends(), reason="jax not installed")
def test_jax_bsm_pricing_matches_numpy() -> None:
    flag, s, k, t, r, sigma, q = _bsm_inputs()
    base = fast_black_scholes_merton(flag, s, k, t, r, sigma, q, backend="numpy", return_as="numpy")
    trial = fast_black_scholes_merton(flag, s, k, t, r, sigma, q, backend="jax", return_as="numpy")
    assert_allclose(base, trial, atol=1e-6)


@pytest.mark.skipif("torch" not in available_backends(), reason="torch not installed")
def test_torch_bsm_iv_matches_numpy() -> None:
    flag, s, k, t, r, sigma, q = _bsm_inputs()
    prices = fast_black_scholes_merton(
        flag, s, k, t, r, sigma, q, backend="numpy", return_as="numpy"
    )
    base = fast_implied_volatility(
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
    trial = fast_implied_volatility(
        prices,
        s,
        k,
        t,
        r,
        flag,
        q=q,
        model="black_scholes_merton",
        backend="torch",
        return_as="numpy",
    )
    assert_allclose(base, trial, atol=1e-6)


@pytest.mark.skipif("jax" not in available_backends(), reason="jax not installed")
def test_jax_bsm_iv_matches_numpy() -> None:
    flag, s, k, t, r, sigma, q = _bsm_inputs()
    prices = fast_black_scholes_merton(
        flag, s, k, t, r, sigma, q, backend="numpy", return_as="numpy"
    )
    base = fast_implied_volatility(
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
    trial = fast_implied_volatility(
        prices,
        s,
        k,
        t,
        r,
        flag,
        q=q,
        model="black_scholes_merton",
        backend="jax",
        return_as="numpy",
    )
    assert_allclose(base, trial, atol=1e-6)


@pytest.mark.skipif("torch" not in available_backends(), reason="torch not installed")
def test_torch_bsm_greeks_match_numpy() -> None:
    flag, s, k, t, r, sigma, q = _bsm_inputs()
    base = get_all_greeks(
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
    trial = get_all_greeks(
        flag,
        s,
        k,
        t,
        r,
        sigma,
        q=q,
        model="black_scholes_merton",
        backend="torch",
        return_as="dict",
    )
    for name in ("delta", "gamma", "theta", "rho", "vega"):
        assert_allclose(base[name], trial[name], atol=1e-6)


@pytest.mark.skipif("jax" not in available_backends(), reason="jax not installed")
def test_jax_bsm_greeks_match_numpy() -> None:
    flag, s, k, t, r, sigma, q = _bsm_inputs()
    base = get_all_greeks(
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
    trial = get_all_greeks(
        flag,
        s,
        k,
        t,
        r,
        sigma,
        q=q,
        model="black_scholes_merton",
        backend="jax",
        return_as="dict",
    )
    for name in ("delta", "gamma", "theta", "rho", "vega"):
        assert_allclose(base[name], trial[name], atol=1e-6)


@pytest.mark.skipif("torch" not in available_backends(), reason="torch not installed")
def test_torch_bsm_price_dataframe_matches_numpy() -> None:
    df = _bsm_inputs_frame()
    base = price_dataframe(
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
    trial = price_dataframe(
        df,
        flag_col="flag",
        underlying_price_col="S",
        strike_col="K",
        annualized_tte_col="t",
        riskfree_rate_col="r",
        sigma_col="sigma",
        dividend_col="q",
        model="black_scholes_merton",
        backend="torch",
        inplace=False,
    )
    for column in ("Price", "delta", "gamma", "theta", "rho", "vega"):
        assert_allclose(base[column].to_numpy(), trial[column].to_numpy(), atol=1e-6)


@pytest.mark.skipif("jax" not in available_backends(), reason="jax not installed")
def test_jax_bsm_price_dataframe_matches_numpy() -> None:
    df = _bsm_inputs_frame()
    base = price_dataframe(
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
    trial = price_dataframe(
        df,
        flag_col="flag",
        underlying_price_col="S",
        strike_col="K",
        annualized_tte_col="t",
        riskfree_rate_col="r",
        sigma_col="sigma",
        dividend_col="q",
        model="black_scholes_merton",
        backend="jax",
        inplace=False,
    )
    for column in ("Price", "delta", "gamma", "theta", "rho", "vega"):
        assert_allclose(base[column].to_numpy(), trial[column].to_numpy(), atol=1e-6)
