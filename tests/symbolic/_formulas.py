"""Independent symbolic definitions of the Black-Scholes-Merton equations."""

from __future__ import annotations

import sympy as sp

SPOT, STRIKE, MATURITY, VOLATILITY, OBSERVED_PRICE = sp.symbols(
    "S K T sigma P",
    positive=True,
)
RATE, DIVIDEND_YIELD = sp.symbols("r q", real=True)

_HALF = sp.Rational(1, 2)


def normal_cdf(value: sp.Expr) -> sp.Expr:
    """The standard-normal cumulative distribution in exact symbolic form."""
    return (1 + sp.erf(value / sp.sqrt(2))) / 2


D1 = (sp.log(SPOT / STRIKE) + (RATE - DIVIDEND_YIELD + _HALF * VOLATILITY**2) * MATURITY) / (
    VOLATILITY * sp.sqrt(MATURITY)
)
D2 = D1 - VOLATILITY * sp.sqrt(MATURITY)

DISCOUNTED_SPOT = SPOT * sp.exp(-DIVIDEND_YIELD * MATURITY)
DISCOUNTED_STRIKE = STRIKE * sp.exp(-RATE * MATURITY)
NORMAL_DENSITY_D1 = sp.exp(-(D1**2) / 2) / sp.sqrt(2 * sp.pi)

CALL_PRICE = DISCOUNTED_SPOT * normal_cdf(D1) - DISCOUNTED_STRIKE * normal_cdf(D2)
PUT_PRICE = DISCOUNTED_STRIKE * normal_cdf(-D2) - DISCOUNTED_SPOT * normal_cdf(-D1)
VEGA = DISCOUNTED_SPOT * NORMAL_DENSITY_D1 * sp.sqrt(MATURITY)


def assert_symbolically_equal(actual: sp.Expr, expected: sp.Expr) -> None:
    """Assert exact equality after symbolic simplification."""
    assert sp.simplify(actual - expected) == 0
