"""Implicit-function derivatives for implied volatility."""

from __future__ import annotations

import sympy as sp

from ._formulas import (
    CALL_PRICE,
    D1,
    D2,
    DISCOUNTED_SPOT,
    DIVIDEND_YIELD,
    MATURITY,
    OBSERVED_PRICE,
    RATE,
    SPOT,
    STRIKE,
    VEGA,
    VOLATILITY,
    assert_symbolically_equal,
    normal_cdf,
)

_RESIDUAL = CALL_PRICE - OBSERVED_PRICE


def _implicit_derivative(parameter: sp.Symbol) -> sp.Expr:
    return -sp.diff(_RESIDUAL, parameter) / sp.diff(_RESIDUAL, VOLATILITY)


def test_iv_price_derivative_is_inverse_vega() -> None:
    assert_symbolically_equal(_implicit_derivative(OBSERVED_PRICE), 1 / VEGA)


def test_iv_spot_derivative_is_negative_delta_over_vega() -> None:
    delta = sp.exp(-DIVIDEND_YIELD * MATURITY) * normal_cdf(D1)
    assert_symbolically_equal(_implicit_derivative(SPOT), -delta / VEGA)


def test_iv_strike_derivative() -> None:
    expected = sp.exp(-RATE * MATURITY) * normal_cdf(D2) / VEGA
    assert_symbolically_equal(_implicit_derivative(STRIKE), expected)


def test_iv_rate_derivative_is_negative_rho_over_vega() -> None:
    rho = STRIKE * MATURITY * sp.exp(-RATE * MATURITY) * normal_cdf(D2)
    assert_symbolically_equal(_implicit_derivative(RATE), -rho / VEGA)


def test_iv_dividend_derivative() -> None:
    expected = MATURITY * DISCOUNTED_SPOT * normal_cdf(D1) / VEGA
    assert_symbolically_equal(_implicit_derivative(DIVIDEND_YIELD), expected)
