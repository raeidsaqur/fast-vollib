"""
Jäckel numpy backend — machine-precision implied volatility.

Provides an `implied_volatility()` function with the same signature as
`fast_vollib.backends.numpy_backend.implied_volatility`, but using the
Jäckel "Let's Be Rational" (2016) solver instead of Halley × 8.

Accuracy:  max relative error ~ 2e-11 (machine precision)
Speed:     ~108ms / 100k options on CPU (vs ~4ms for the Halley backend)
Use case:  research, stress-testing, oracle comparison, autoresearch iterations.

The main `backends/` module is NOT modified — this module is purely additive.
"""

from __future__ import annotations

import numpy as np

from ..types import ModelLiteral
from ..utils.validation import handle_error
from .jackel_iv import jackel_iv_black as _jackel_iv_black


def implied_volatility(
    model: ModelLiteral,
    price: np.ndarray,
    s: np.ndarray,
    k: np.ndarray,
    t: np.ndarray,
    r: np.ndarray,
    flag: np.ndarray,
    q: np.ndarray | None = None,
    on_error: str = "warn",
) -> np.ndarray:
    """Machine-precision IV using Jäckel "Let's Be Rational" (2016).

    Parameters
    ----------
    model   : "black" | "black_scholes" | "black_scholes_merton"
    price   : option market price (discounted)
    s       : spot (or forward for Black-76)
    k       : strike
    t       : time to expiry (years)
    r       : risk-free rate
    flag    : "c" for call, "p" for put (array of str)
    q       : dividend yield (Black-Scholes-Merton only)
    on_error: "warn" | "raise" | "ignore"

    Returns
    -------
    sigma : annualised implied volatility (NaN for degenerate inputs)
    """
    price_a = np.asarray(price, dtype=float)
    s_a = np.asarray(s, dtype=float)
    k_a = np.asarray(k, dtype=float)
    t_a = np.asarray(t, dtype=float)
    r_a = np.asarray(r, dtype=float)

    # Black-76: q=r → carry=disc, d1 uses only σ² drift term
    q_a = (
        r_a
        if (model == "black" and q is None)
        else (np.zeros_like(r_a) if q is None else np.asarray(q, dtype=float))
    )

    is_call = flag == "c"
    valid = t_a > 0

    # Below-intrinsic check
    disc_spot = s_a * np.exp(-q_a * t_a)
    disc_strike = k_a * np.exp(-r_a * t_a)
    intrinsic = np.where(
        is_call,
        np.maximum(disc_spot - disc_strike, 0.0),
        np.maximum(disc_strike - disc_spot, 0.0),
    )
    below_intrinsic = price_a < intrinsic - 1e-10
    if np.any(below_intrinsic):
        handle_error("Option price is below intrinsic value.", on_error)

    # Convert to undiscounted Black-76 space (Jäckel's normalised domain)
    disc_factor = np.exp(r_a * t_a)
    undiscounted_price = price_a * disc_factor

    if model == "black":
        F_fwd = s_a  # input is already forward
    elif model == "black_scholes":
        F_fwd = s_a * disc_factor  # F = S·exp(r·T), q=0
    else:  # black_scholes_merton
        F_fwd = s_a * np.exp((r_a - q_a) * t_a)  # F = S·exp((r−q)·T)

    # Jäckel Householder(3) × 2 — machine precision, no fallback needed
    sigma = _jackel_iv_black(undiscounted_price, F_fwd, k_a, t_a, is_call)

    result = np.where(valid, sigma, 0.0)
    result = np.where(below_intrinsic, np.nan, result)
    return result
