"""Public NumPy results checked against SymPy-derived numerical oracles."""

from __future__ import annotations

import numpy as np
import pytest
import sympy as sp

from fast_vollib.api import get_all_greeks
from fast_vollib.models import fast_black_scholes_merton

from ._formulas import (
    CALL_PRICE,
    DIVIDEND_YIELD,
    MATURITY,
    PUT_PRICE,
    RATE,
    SPOT,
    STRIKE,
    VOLATILITY,
)

_ORACLE_ARGUMENTS = (SPOT, STRIKE, MATURITY, RATE, VOLATILITY, DIVIDEND_YIELD)
_SPOT_VALUES = np.array([80.0, 100.0, 120.0])
_STRIKE_VALUES = np.array([100.0, 100.0, 100.0])
_MATURITY_VALUES = np.array([0.25, 1.0, 2.0])
_RATE_VALUES = np.array([-0.01, 0.02, 0.08])
_VOLATILITY_VALUES = np.array([0.15, 0.30, 0.55])
_DIVIDEND_VALUES = np.array([0.00, 0.01, 0.04])
_NUMERICAL_INPUTS = (
    _SPOT_VALUES,
    _STRIKE_VALUES,
    _MATURITY_VALUES,
    _RATE_VALUES,
    _VOLATILITY_VALUES,
    _DIVIDEND_VALUES,
)


def _evaluate(expression: sp.Expr) -> np.ndarray:
    oracle = sp.lambdify(
        _ORACLE_ARGUMENTS,
        expression,
        modules="scipy",
        docstring_limit=0,
    )
    return np.asarray(oracle(*_NUMERICAL_INPUTS), dtype=np.float64)


@pytest.mark.parametrize(
    ("flag", "expression"),
    [("c", CALL_PRICE), ("p", PUT_PRICE)],
    ids=["call", "put"],
)
def test_bsm_prices_match_symbolic_oracle(flag: str, expression: sp.Expr) -> None:
    actual = fast_black_scholes_merton(
        flag,
        _SPOT_VALUES,
        _STRIKE_VALUES,
        _MATURITY_VALUES,
        _RATE_VALUES,
        _VOLATILITY_VALUES,
        _DIVIDEND_VALUES,
        return_as="numpy",
        backend="numpy",
    )
    np.testing.assert_allclose(actual, _evaluate(expression), rtol=2e-14, atol=2e-14)


def test_bsm_greeks_match_symbolic_derivatives() -> None:
    expected = {
        "delta": sp.diff(CALL_PRICE, SPOT),
        "gamma": sp.diff(CALL_PRICE, SPOT, 2),
        "theta": -sp.diff(CALL_PRICE, MATURITY) / 365,
        "rho": sp.diff(CALL_PRICE, RATE) / 100,
        "vega": sp.diff(CALL_PRICE, VOLATILITY) / 100,
    }
    actual = get_all_greeks(
        "c",
        _SPOT_VALUES,
        _STRIKE_VALUES,
        _MATURITY_VALUES,
        _RATE_VALUES,
        _VOLATILITY_VALUES,
        q=_DIVIDEND_VALUES,
        model="black_scholes_merton",
        return_as="dict",
        backend="numpy",
    )

    for name, expression in expected.items():
        np.testing.assert_allclose(actual[name], _evaluate(expression), rtol=2e-14, atol=2e-14)
