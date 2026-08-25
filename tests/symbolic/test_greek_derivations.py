"""Greeks obtained by differentiating an independent symbolic price."""

from __future__ import annotations

import sympy as sp

from ._formulas import (
    CALL_PRICE,
    D1,
    D2,
    DISCOUNTED_SPOT,
    DIVIDEND_YIELD,
    MATURITY,
    NORMAL_DENSITY_D1,
    RATE,
    SPOT,
    STRIKE,
    VEGA,
    VOLATILITY,
    assert_symbolically_equal,
    normal_cdf,
)


def test_call_delta_derivation() -> None:
    expected = sp.exp(-DIVIDEND_YIELD * MATURITY) * normal_cdf(D1)
    assert_symbolically_equal(sp.diff(CALL_PRICE, SPOT), expected)


def test_call_gamma_derivation() -> None:
    expected = (
        sp.exp(-DIVIDEND_YIELD * MATURITY)
        * NORMAL_DENSITY_D1
        / (SPOT * VOLATILITY * sp.sqrt(MATURITY))
    )
    assert_symbolically_equal(sp.diff(CALL_PRICE, SPOT, 2), expected)


def test_call_vega_derivation() -> None:
    assert_symbolically_equal(sp.diff(CALL_PRICE, VOLATILITY), VEGA)


def test_call_rho_derivation() -> None:
    expected = STRIKE * MATURITY * sp.exp(-RATE * MATURITY) * normal_cdf(D2)
    assert_symbolically_equal(sp.diff(CALL_PRICE, RATE), expected)


def test_call_dividend_sensitivity_derivation() -> None:
    expected = -MATURITY * DISCOUNTED_SPOT * normal_cdf(D1)
    assert_symbolically_equal(sp.diff(CALL_PRICE, DIVIDEND_YIELD), expected)


def test_call_strike_derivative() -> None:
    expected = -sp.exp(-RATE * MATURITY) * normal_cdf(D2)
    assert_symbolically_equal(sp.diff(CALL_PRICE, STRIKE), expected)
