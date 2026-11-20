# Author: Raeid Saqur
# Date: 2026-11-19
"""Explicit implied-volatility solver selection on every public surface.

Omitted solver keeps each surface's default (functional and DataFrame: Halley; typed: Jäckel),
an explicit solver is honoured, an unknown one or an unsupported (solver, backend) pair raises
``UnsupportedSolverError``, and nothing is substituted. The terminal function that ran is
asserted by instrumenting the backend modules.
"""

from __future__ import annotations

import importlib
from typing import Any
import warnings

import numpy as np
import pandas as pd
import pytest

from fast_vollib import backends
from fast_vollib.api import price_dataframe
from fast_vollib.implied_volatility import fast_implied_volatility, fast_implied_volatility_black
from fast_vollib.instruments import (
    EuropeanOptionBatch,
    UnsupportedSolverError,
    VanillaMarketInputs,
    implied_volatility_instrument,
)
from fast_vollib.models import fast_black, fast_black_scholes, fast_black_scholes_merton

BACKENDS = [b for b in ("numpy", "numba", "torch", "jax") if b in backends.available_backends()]
JACKEL_BACKENDS = [b for b in ("numpy", "torch", "jax") if b in BACKENDS]

FLAG = np.array(["c", "p", "c", "p"])
S = np.array([100.0, 100.0, 105.0, 95.0])
K = np.array([90.0, 100.0, 120.0, 80.0])
T = np.array([0.25, 0.5, 1.0, 2.0])
R = np.array([0.02, -0.01, 0.05, 0.03])
Q = np.array([0.01, 0.0, 0.03, 0.02])
SIGMA = np.array([0.2, 0.35, 0.15, 0.5])


class _Spy:
    """Record which terminal IV functions run (Halley backend modules, Jäckel wrappers)."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.calls: list[str] = []
        for name in ("numpy", "numba", "torch", "jax"):
            self._wrap(monkeypatch, backends.get_module(name), f"halley.{name}")
        for name in ("numpy", "torch", "jax"):
            try:
                module = importlib.import_module(f"fast_vollib.jackel.{name}_backend")
            except ImportError:  # pragma: no cover - optional dependency
                continue
            self._wrap(monkeypatch, module, f"jackel.{name}")

    def _wrap(self, monkeypatch: pytest.MonkeyPatch, module: Any, label: str) -> None:
        original = module.implied_volatility

        def spy(*args: Any, **kwargs: Any) -> Any:
            self.calls.append(label)
            return original(*args, **kwargs)

        monkeypatch.setattr(module, "implied_volatility", spy)


def _price(model: str) -> np.ndarray:
    if model == "black":
        return fast_black(FLAG, S, K, T, R, SIGMA, return_as="numpy", backend="numpy")
    if model == "black_scholes":
        return fast_black_scholes(FLAG, S, K, T, R, SIGMA, return_as="numpy", backend="numpy")
    return fast_black_scholes_merton(FLAG, S, K, T, R, SIGMA, Q, return_as="numpy", backend="numpy")


def _functional(model: str, price: Any, **kw: Any) -> Any:
    if model == "black":
        return fast_implied_volatility_black(price, S, K, R, T, FLAG, **kw)
    q = Q if model == "black_scholes_merton" else None
    return fast_implied_volatility(price, S, K, T, R, FLAG, q, model=model, **kw)


# --- functional surface -----------------------------------------------------------------------


@pytest.mark.parametrize("model", ["black", "black_scholes", "black_scholes_merton"])
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("solver", [None, "halley", "jackel"])
def test_functional_routes_to_the_requested_solver(
    monkeypatch: pytest.MonkeyPatch, model: str, backend: str, solver: str | None
) -> None:
    spy = _Spy(monkeypatch)
    price = _price(model)
    if solver == "jackel" and backend not in JACKEL_BACKENDS:
        with pytest.raises(UnsupportedSolverError, match="no 'numba' implementation"):
            _functional(model, price, solver=solver, backend=backend, return_as="numpy")
        assert spy.calls == []
        return
    iv = _functional(model, price, solver=solver, backend=backend, return_as="numpy")
    expected = f"{'jackel' if solver == 'jackel' else 'halley'}.{backend}"
    assert spy.calls == [expected]
    # Black-76 discounting and q != 0 (Black-Scholes-Merton) are honoured on every route.
    np.testing.assert_allclose(iv, SIGMA, rtol=1e-8)


def test_functional_rejects_an_unknown_solver() -> None:
    with pytest.raises(UnsupportedSolverError, match="Unknown implied-volatility solver"):
        fast_implied_volatility(_price("black_scholes"), S, K, T, R, FLAG, solver="newton")


def test_functional_still_ignores_unrelated_compatibility_kwargs() -> None:
    iv = fast_implied_volatility(
        _price("black_scholes"), S, K, T, R, FLAG, return_as="numpy", backend="numpy", foo=1
    )
    np.testing.assert_allclose(iv, SIGMA, rtol=1e-8)


@pytest.mark.parametrize("solver", ["halley", "jackel"])
def test_functional_scalar_and_broadcast_inputs(solver: str) -> None:
    price = fast_black_scholes("c", 100.0, 100.0, 1.0, 0.03, 0.2, return_as="numpy")
    iv = fast_implied_volatility(
        price, 100.0, 100.0, 1.0, 0.03, "c", solver=solver, return_as="numpy"
    )
    np.testing.assert_allclose(iv, [0.2], rtol=1e-8)
    strikes = np.array([90.0, 100.0, 110.0])
    prices = fast_black_scholes("c", 100.0, strikes, 1.0, 0.03, 0.2, return_as="numpy")
    out = fast_implied_volatility(prices, 100.0, strikes, 1.0, 0.03, "c", solver=solver)
    assert isinstance(out, pd.DataFrame) and list(out.columns) == ["IV"] and len(out) == 3
    np.testing.assert_allclose(out["IV"].to_numpy(), 0.2, rtol=1e-8)


@pytest.mark.parametrize("solver", ["halley", "jackel"])
def test_functional_mixed_hard_and_ordinary_rows_and_on_error_modes(solver: str) -> None:
    good = fast_black_scholes("c", 100.0, 100.0, 1.0, 0.0, 0.2, return_as="numpy")[0]
    price = np.array([good, 0.0, 40.0, good, -1.0])  # ok, zero, below intrinsic, ok, invalid
    strike = np.array([100.0, 150.0, 50.0, 100.0, 100.0])
    args = (price, 100.0, strike, 1.0, 0.0, "c")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        iv = fast_implied_volatility(*args, solver=solver, return_as="numpy", on_error="warn")
    library = [w for w in caught if "Implied volatility is undefined" in str(w.message)]
    assert len(library) == 1  # one aggregated signal per call
    np.testing.assert_allclose(iv[[0, 3]], 0.2, rtol=1e-8)
    assert np.isnan(iv[[1, 2, 4]]).all()
    with pytest.raises(ValueError, match="below intrinsic value"):
        fast_implied_volatility(*args, solver=solver, on_error="raise")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        quiet = fast_implied_volatility(*args, solver=solver, return_as="numpy", on_error="ignore")
    assert not [w for w in caught if "Implied volatility" in str(w.message)]
    np.testing.assert_array_equal(np.isnan(quiet), np.isnan(iv))


@pytest.mark.parametrize("backend", [b for b in JACKEL_BACKENDS if b in ("torch", "jax")])
@pytest.mark.parametrize("solver", ["halley", "jackel"])
def test_functional_return_native(backend: str, solver: str) -> None:
    out = fast_implied_volatility(
        _price("black_scholes"),
        S,
        K,
        T,
        R,
        FLAG,
        solver=solver,
        backend=backend,
        return_native=True,
    )
    module = importlib.import_module(backend)
    assert isinstance(out, module.Tensor if backend == "torch" else module.Array)
    np.testing.assert_allclose(
        np.asarray(backends.get_module(backend).from_native(out)), SIGMA, rtol=1e-8
    )


# --- DataFrame surface ------------------------------------------------------------------------


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "flag": FLAG,
            "S": S,
            "K": K,
            "t": T,
            "r": R,
            "q": Q,
            "price": _price("black_scholes_merton"),
        },
        index=pd.Index(["a", "b", "c", "d"], name="id"),
    )


@pytest.mark.parametrize("solver", [None, "halley", "jackel"])
def test_dataframe_iv_branch_uses_the_requested_solver(
    monkeypatch: pytest.MonkeyPatch, solver: str | None
) -> None:
    spy = _Spy(monkeypatch)
    df = _frame()
    kw = dict(
        flag_col="flag",
        underlying_price_col="S",
        strike_col="K",
        annualized_tte_col="t",
        riskfree_rate_col="r",
        dividend_col="q",
        price_col="price",
        model="black_scholes_merton",
        backend="numpy",
    )
    out = price_dataframe(df, solver=solver, **kw)
    assert spy.calls == [f"{'jackel' if solver == 'jackel' else 'halley'}.numpy"]
    assert list(out.index) == list(df.index) and "IV" in out.columns
    np.testing.assert_allclose(out["IV"].to_numpy(), SIGMA, rtol=1e-8)
    inplace = df.copy()
    assert price_dataframe(inplace, solver=solver, inplace=True, **kw) is None
    np.testing.assert_allclose(inplace["IV"].to_numpy(), SIGMA, rtol=1e-8)


def test_dataframe_solver_does_not_touch_the_pricing_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spy = _Spy(monkeypatch)
    df = _frame().assign(sigma=SIGMA)
    price_dataframe(
        df,
        flag_col="flag",
        underlying_price_col="S",
        strike_col="K",
        annualized_tte_col="t",
        riskfree_rate_col="r",
        sigma_col="sigma",
        solver="jackel",
        backend="numpy",
    )
    assert spy.calls == []


# --- typed surface ------------------------------------------------------------------------------


@pytest.mark.parametrize("solver", [None, "halley", "jackel"])
def test_typed_default_stays_jackel(monkeypatch: pytest.MonkeyPatch, solver: str | None) -> None:
    spy = _Spy(monkeypatch)
    batch = EuropeanOptionBatch.from_arrays(
        option_type=list(FLAG), strike=K, maturity=T, underlier="X", notional=1.0
    )
    market = VanillaMarketInputs(
        underlying=S, rate=R, price=_price("black_scholes_merton"), dividend_yield=Q
    )
    kw: dict[str, Any] = {} if solver is None else {"solver": solver}
    iv = implied_volatility_instrument(
        batch, market, model="black_scholes_merton", backend="numpy", **kw
    )
    assert spy.calls == [f"{'halley' if solver == 'halley' else 'jackel'}.numpy"]
    np.testing.assert_allclose(np.asarray(iv), SIGMA, rtol=1e-8)
