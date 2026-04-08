"""
Jäckel JAX backend — machine-precision implied volatility.

Experiment I-6: Port Jäckel Householder(3)×2 to JAX using
`jax.scipy.special.erfcx` inside `jax.lax.fori_loop(0, 2, householder_step, sigma)`.
JIT with `@jax.jit`.

Current state (stub):
    Delegates model→forward conversion to numpy, then calls the
    numpy `jackel_iv_black` for the IV solve.  Functionally correct
    but provides no JIT/XLA acceleration.

Target (I-6):
    Port `normalised_black_call`, `_jackel_initial_guess`, and the
    Householder loop entirely to JAX arrays, wrap in `@jax.jit`.
    Compare throughput vs `backends/jax_backend.py` Halley×8.

Accuracy:  max relative error ~ 2e-11 (inherited from numpy backend)
Speed:     stub — same as numpy (~108ms/100k).  Target: < 10ms with JIT.
"""

from __future__ import annotations

import numpy as np

from ..types import ModelLiteral
from ..utils.validation import handle_error
from .jackel_iv import jackel_iv_black as _jackel_iv_black_np

# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------


def is_available() -> bool:
    try:
        import jax  # noqa: F401
    except ImportError:
        return False
    return True


def _ensure_x64() -> None:
    """Enable JAX double precision. Must be called before any JAX computation."""
    import jax

    jax.config.update("jax_enable_x64", True)


def to_native(values: np.ndarray):
    import jax.numpy as jnp

    return jnp.asarray(values)


def from_native(values) -> np.ndarray:
    return np.asarray(values)


# ---------------------------------------------------------------------------
# TODO I-6: native JAX Jäckel core
# ---------------------------------------------------------------------------
# Port each of these from jackel_iv.py using JAX ops:
#
#   def _normalised_black_call_j(x, s):
#       """JAX version using jax.scipy.special.erfcx."""
#       import jax.numpy as jnp
#       import jax.scipy.special as jss
#       h = x / jnp.where(s > 0, s, 1e-300)
#       t = 0.5 * s
#       scale = 0.5 * jnp.exp(-0.5 * (h*h + t*t))
#       b = scale * (
#           jss.erfcx(-ONE_OVER_SQRT2 * (h + t))
#           - jss.erfcx(-ONE_OVER_SQRT2 * (h - t))
#       )
#       return jnp.maximum(b, 0.0)
#
#   @jax.jit
#   def _jackel_iv_jax(beta, x):
#       """2-iter Householder(3) using lax.fori_loop — XLA-compilable."""
#       import jax
#       import jax.numpy as jnp
#       s0 = _jackel_initial_guess_j(beta, x)
#       def householder_step(_, s):
#           b  = _normalised_black_call_j(x, s)
#           bp = _normalised_vega_j(x, s)
#           ...
#           return s + ds
#       return jax.lax.fori_loop(0, 2, householder_step, s0)


# ---------------------------------------------------------------------------
# Implied volatility — stub (numpy fallback)
# ---------------------------------------------------------------------------


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
    """Machine-precision IV using Jäckel "Let's Be Rational" (2016) — JAX stub.

    Currently delegates to the numpy Jäckel backend.
    Experiment I-6 will replace this with native JAX ops + @jax.jit.

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

    q_a = (
        r_a
        if (model == "black" and q is None)
        else (np.zeros_like(r_a) if q is None else np.asarray(q, dtype=float))
    )

    is_call = flag == "c"
    valid = t_a > 0

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

    disc_factor = np.exp(r_a * t_a)
    undiscounted_price = price_a * disc_factor

    if model == "black":
        F_fwd = s_a
    elif model == "black_scholes":
        F_fwd = s_a * disc_factor
    else:
        F_fwd = s_a * np.exp((r_a - q_a) * t_a)

    # TODO I-6: replace with native JAX call:
    #   _ensure_x64()
    #   sigma = np.asarray(_jackel_iv_jax(
    #       jnp.asarray(undiscounted_price), jnp.asarray(F_fwd),
    #       jnp.asarray(k_a), jnp.asarray(t_a), jnp.asarray(is_call),
    #   ))
    sigma = _jackel_iv_black_np(undiscounted_price, F_fwd, k_a, t_a, is_call)

    result = np.where(valid, sigma, 0.0)
    result = np.where(below_intrinsic, np.nan, result)
    return result
