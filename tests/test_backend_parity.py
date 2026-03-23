from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

import fast_vollib.config as config_module
from fast_vollib import (
    get_all_greeks,
    price_dataframe,
    vectorized_black_scholes,
    vectorized_implied_volatility,
)
from fast_vollib.backends import available_backends
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
    monkeypatch.delenv("FASTIV_BACKEND", raising=False)


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
    values = vectorized_black_scholes(
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


def test_backend_auto_supports_legacy_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FASTIV_BACKEND", "numpy")
    assert get_backend() == "numpy"


def test_backend_auto_prefers_torch_over_jax(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config_module, "_torch_cuda_available", lambda: True)
    monkeypatch.setattr(config_module, "_jax_available", lambda: True)
    assert get_backend("auto") == "torch"


def test_backend_auto_matches_numpy_when_resolved_to_numpy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config_module, "_torch_cuda_available", lambda: False)
    monkeypatch.setattr(config_module, "_jax_available", lambda: False)
    flag, s, k, t, r, sigma = _inputs()
    base = vectorized_black_scholes(
        flag,
        s,
        k,
        t,
        r,
        sigma,
        backend="numpy",
        return_as="numpy",
    )
    auto = vectorized_black_scholes(
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
    base = vectorized_black_scholes(flag, s, k, t, r, sigma, backend="numpy", return_as="numpy")
    trial = vectorized_black_scholes(flag, s, k, t, r, sigma, backend="torch", return_as="numpy")
    assert_allclose(base, trial, atol=1e-6)


@pytest.mark.skipif("jax" not in available_backends(), reason="jax not installed")
def test_jax_pricing_backend_matches_numpy() -> None:
    flag, s, k, t, r, sigma = _inputs()
    base = vectorized_black_scholes(flag, s, k, t, r, sigma, backend="numpy", return_as="numpy")
    trial = vectorized_black_scholes(flag, s, k, t, r, sigma, backend="jax", return_as="numpy")
    assert_allclose(base, trial, atol=1e-6)


@pytest.mark.skipif("torch" not in available_backends(), reason="torch not installed")
def test_torch_iv_backend_matches_numpy() -> None:
    flag, s, k, t, r, sigma = _inputs()
    prices = vectorized_black_scholes(flag, s, k, t, r, sigma, backend="numpy", return_as="numpy")
    base = vectorized_implied_volatility(prices, s, k, t, r, flag, backend="numpy", return_as="numpy")
    trial = vectorized_implied_volatility(prices, s, k, t, r, flag, backend="torch", return_as="numpy")
    assert_allclose(base, trial, atol=1e-6)


@pytest.mark.skipif("jax" not in available_backends(), reason="jax not installed")
def test_jax_iv_backend_matches_numpy() -> None:
    flag, s, k, t, r, sigma = _inputs()
    prices = vectorized_black_scholes(flag, s, k, t, r, sigma, backend="numpy", return_as="numpy")
    base = vectorized_implied_volatility(prices, s, k, t, r, flag, backend="numpy", return_as="numpy")
    trial = vectorized_implied_volatility(prices, s, k, t, r, flag, backend="jax", return_as="numpy")
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
    native = vectorized_black_scholes(
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
    native = vectorized_black_scholes(
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
