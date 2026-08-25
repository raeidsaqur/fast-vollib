"""Exact identities shared by the Black-family pricing models."""

from __future__ import annotations

import sympy as sp

from ._formulas import (
    CALL_PRICE,
    DISCOUNTED_SPOT,
    DISCOUNTED_STRIKE,
    DIVIDEND_YIELD,
    MATURITY,
    PUT_PRICE,
    RATE,
    SPOT,
    STRIKE,
    VOLATILITY,
    assert_symbolically_equal,
    normal_cdf,
)


def test_normal_cdf_complement_identity() -> None:
    x = sp.Symbol("x", real=True)
    assert_symbolically_equal(normal_cdf(x) + normal_cdf(-x), sp.Integer(1))


def test_black_scholes_merton_put_call_parity() -> None:
    assert_symbolically_equal(
        CALL_PRICE - PUT_PRICE,
        DISCOUNTED_SPOT - DISCOUNTED_STRIKE,
    )


def test_black_76_is_the_equal_carry_specialization_of_bsm() -> None:
    forward = sp.Symbol("F", positive=True)
    sqrt_t = sp.sqrt(MATURITY)
    d1_black = (sp.log(forward / STRIKE) + sp.Rational(1, 2) * VOLATILITY**2 * MATURITY) / (
        VOLATILITY * sqrt_t
    )
    d2_black = d1_black - VOLATILITY * sqrt_t
    black_call = sp.exp(-RATE * MATURITY) * (
        forward * normal_cdf(d1_black) - STRIKE * normal_cdf(d2_black)
    )

    specialized_bsm = CALL_PRICE.xreplace(
        {
            SPOT: forward,
            DIVIDEND_YIELD: RATE,
        }
    )
    assert_symbolically_equal(specialized_bsm, black_call)


def test_black_76_put_call_parity_is_discounted() -> None:
    forward = sp.Symbol("F", positive=True)
    specialized_call = CALL_PRICE.xreplace({SPOT: forward, DIVIDEND_YIELD: RATE})
    specialized_put = PUT_PRICE.xreplace({SPOT: forward, DIVIDEND_YIELD: RATE})

    assert_symbolically_equal(
        specialized_call - specialized_put,
        sp.exp(-RATE * MATURITY) * (forward - STRIKE),
    )
