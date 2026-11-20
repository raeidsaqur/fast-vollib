"""
Jäckel numpy backend — machine-precision implied volatility.

Provides an `implied_volatility()` function with the same signature as
`fast_vollib.backends.numpy_backend.implied_volatility`, but using the
Jäckel "Let's Be Rational" (2016) solver instead of Halley × 8.

Accuracy:  max relative error ~ 2e-11 (machine precision)
Speed:     ~108ms / 100k options on CPU (vs ~4ms for the Halley backend)
Use case:  research, stress-testing, oracle comparison, autoresearch iterations.

Rows are classified by the shared IV policy (see `_policy`); interior rows go to the solver.
"""

from __future__ import annotations

import numpy as np

from ..types import ModelLiteral
from . import _policy
from .jackel_iv import jackel_iv_normalized as _jackel_iv_normalized


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
    """Machine-precision IV using Jäckel "Let's Be Rational" (2016) — NumPy.

    Rows are classified by the shared policy (``jackel._policy``); interior rows are solved on the host (Numba kernels when available).

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
    on_error: "warn" | "raise" | "ignore" (one aggregated signal per call)

    Returns
    -------
    sigma : annualised implied volatility (float64, input shape); NaN on refused rows
    """
    prep = _policy.prepare(model, price, s, k, t, r, flag, q)
    sigma = np.empty(0)
    if prep.solve.size:
        sigma = _jackel_iv_normalized(prep.beta, prep.x) / np.sqrt(prep.t)
    return _policy.finish(prep, sigma, on_error)
