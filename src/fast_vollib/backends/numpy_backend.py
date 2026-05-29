from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.special import log_ndtr, ndtr

from ..types import ModelLiteral, OnErrorLiteral
from ..utils.validation import handle_error

if TYPE_CHECKING:
    from .._typing import FlagArray, Float1D, OptionalFloat1D  # noqa: F401

# Use scipy.special.ndtr for accurate extreme-tail CDF (matches erfc method).
# Compute PDF directly via numpy — avoids scipy.stats overhead (~5x faster).
_norm_cdf = ndtr
_SQRT2PI_INV = (2.0 * 3.141592653589793) ** -0.5


def _norm_pdf(x: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * x * x) * _SQRT2PI_INV


def _intrinsic_vec(
    flag: np.ndarray,
    s: np.ndarray,
    k: np.ndarray,
    t: np.ndarray,
    r: np.ndarray,
    q: np.ndarray,
    model: ModelLiteral,
) -> np.ndarray:
    """Vectorized intrinsic value computation."""
    disc = np.exp(-r * t)
    carry = np.exp(-q * t)
    if model == "black":
        call_iv = disc * np.maximum(0.0, s - k)
        put_iv = disc * np.maximum(0.0, k - s)
    elif model == "black_scholes":
        call_iv = np.maximum(0.0, s - k * disc)
        put_iv = np.maximum(0.0, k * disc - s)
    else:  # black_scholes_merton
        call_iv = np.maximum(0.0, s * carry - k * disc)
        put_iv = np.maximum(0.0, k * disc - s * carry)
    return np.where(flag == "c", call_iv, put_iv)


def _d1_d2(
    s: np.ndarray, k: np.ndarray, t: np.ndarray, r: np.ndarray, sigma: np.ndarray, q: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    sqrt_t = np.sqrt(np.maximum(t, 1e-32))
    vol_term = np.maximum(sigma * sqrt_t, 1e-32)
    d1 = (
        np.log(np.maximum(s, 1e-32) / np.maximum(k, 1e-32)) + (r - q + 0.5 * sigma**2) * t
    ) / vol_term
    d2 = d1 - sigma * sqrt_t
    return d1, d2


def _bsm_price_full(
    flag: np.ndarray,
    s: np.ndarray,
    k: np.ndarray,
    t: np.ndarray,
    r: np.ndarray,
    sigma: np.ndarray,
    q: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute BSM price, raw vega, d1, and d2 in one pass (avoids double d1/d2 calls)."""
    d1, d2 = _d1_d2(s, k, t, r, sigma, q)
    discounted_spot = s * np.exp(-q * t)
    discounted_strike = k * np.exp(-r * t)
    sqrt_t = np.sqrt(np.maximum(t, 1e-32))

    # Compute only N(d1), N(d2); derive complements via N(-x) = 1 - N(x)
    # — eliminates 2 of 4 ndtr calls per element (saves ~16 calls across 8 Halley iters)
    cdf_d1 = _norm_cdf(d1)
    cdf_d2 = _norm_cdf(d2)
    call = discounted_spot * cdf_d1 - discounted_strike * cdf_d2
    put = discounted_strike * (1.0 - cdf_d2) - discounted_spot * (1.0 - cdf_d1)
    price = np.where(flag == "c", call, put)
    # raw vega (not scaled by 0.01)
    vega = discounted_spot * _norm_pdf(d1) * sqrt_t
    return price, vega, d1, d2


def _bsm_price(
    flag: np.ndarray,
    s: np.ndarray,
    k: np.ndarray,
    t: np.ndarray,
    r: np.ndarray,
    sigma: np.ndarray,
    q: np.ndarray,
) -> np.ndarray:
    price, _, _, _ = _bsm_price_full(flag, s, k, t, r, sigma, q)
    return price


def price_black(
    flag: FlagArray, f: Float1D, k: Float1D, t: Float1D, r: Float1D, sigma: Float1D
) -> Float1D:
    # Black-76: q=r makes carry=disc so d1 = [ln(F/K)+0.5σ²T]/(σ√T) (correct)
    return _bsm_price(flag, f, k, t, r, sigma, r)


def price_black_scholes(
    flag: FlagArray, s: Float1D, k: Float1D, t: Float1D, r: Float1D, sigma: Float1D
) -> Float1D:
    q = np.zeros_like(r)
    return _bsm_price(flag, s, k, t, r, sigma, q)


def price_black_scholes_merton(
    flag: FlagArray,
    s: Float1D,
    k: Float1D,
    t: Float1D,
    r: Float1D,
    sigma: Float1D,
    q: Float1D,
) -> Float1D:
    return _bsm_price(flag, s, k, t, r, sigma, q)


def _vega_raw(
    s: np.ndarray, k: np.ndarray, t: np.ndarray, r: np.ndarray, sigma: np.ndarray, q: np.ndarray
) -> np.ndarray:
    d1, _ = _d1_d2(s, k, t, r, sigma, q)
    return s * np.exp(-q * t) * _norm_pdf(d1) * np.sqrt(np.maximum(t, 1e-32))


def greeks(
    model: ModelLiteral,
    flag: FlagArray,
    s: Float1D,
    k: Float1D,
    t: Float1D,
    r: Float1D,
    sigma: Float1D,
    q: OptionalFloat1D = None,
) -> dict[str, Float1D]:
    # Black-76: q=r so carry=disc and d1 uses only the σ² drift term (no r-q)
    qv = r if (model == "black" and q is None) else (np.zeros_like(r) if q is None else q)
    d1, d2 = _d1_d2(s, k, t, r, sigma, qv)
    carry = np.exp(-qv * t)
    disc = np.exp(-r * t)
    pdf = _norm_pdf(d1)
    sqrt_t = np.sqrt(np.maximum(t, 1e-32))

    # Pre-compute all four CDF values once (N(-x) = 1 - N(x) avoids 5 extra ndtr calls)
    cdf_d1 = _norm_cdf(d1)
    cdf_d2 = _norm_cdf(d2)
    cdf_nd1 = 1.0 - cdf_d1
    cdf_nd2 = 1.0 - cdf_d2

    is_call = flag == "c"
    delta = np.where(is_call, carry * cdf_d1, carry * (cdf_d1 - 1.0))
    gamma = carry * pdf / (np.maximum(s, 1e-32) * np.maximum(sigma, 1e-32) * sqrt_t)
    # vega scaled by 0.01 (1% move convention)
    vega = s * carry * pdf * sqrt_t * 0.01
    theta_call = (
        -(s * carry * pdf * sigma) / (2.0 * sqrt_t)
        - r * k * disc * cdf_d2
        + qv * s * carry * cdf_d1
    ) / 365.0
    theta_put = (
        -(s * carry * pdf * sigma) / (2.0 * sqrt_t)
        + r * k * disc * cdf_nd2
        - qv * s * carry * cdf_nd1
    ) / 365.0
    rho_call = k * t * disc * cdf_d2 * 0.01
    rho_put = -k * t * disc * cdf_nd2 * 0.01
    rho = np.where(is_call, rho_call, rho_put)
    theta = np.where(is_call, theta_call, theta_put)

    if model == "black":
        delta = disc * np.where(is_call, cdf_d1, cdf_d1 - 1.0)
        gamma = disc * pdf / (np.maximum(s, 1e-32) * np.maximum(sigma, 1e-32) * sqrt_t)

    return {
        "delta": delta,
        "gamma": gamma,
        "theta": theta,
        "rho": rho,
        "vega": vega,
    }


# ---------------------------------------------------------------------------
# Implied volatility — Halley's method (3rd-order) + bisection fallback
# ---------------------------------------------------------------------------


def _iv_initial_guess(price: np.ndarray, s: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Brenner-Subrahmanyam ATM approximation as Halley seed.

    Floor is 0.30: for OTM options the BS approximation underestimates sigma
    severely, landing iteration in a near-zero-vega region where it stalls.
    A floor of 0.30 eliminates bisection fallback for typical equity smile data
    while still converging to any true sigma.
    """
    sqrt_t = np.sqrt(np.maximum(t, 1e-8))
    approx = price / np.maximum(s * sqrt_t, 1e-12) * np.sqrt(2.0 * np.pi)
    return np.clip(approx, 0.30, 5.0)


_HALLEY_ITERS = 8  # 8 Halley steps ≡ ~12 Newton steps in accuracy
_BISECT_ITERS = 30  # 10/(2^30)≈9e-9 < _IV_LO → sufficient accuracy
_IV_LO = 1e-8
_IV_HI = 10.0


def _halley_step_precomputed(
    sigma: np.ndarray,
    price: np.ndarray,
    is_call: np.ndarray,
    discounted_spot: np.ndarray,
    discounted_strike: np.ndarray,
    sqrt_t: np.ndarray,
    log_sk: np.ndarray,
    carry_drift_t: np.ndarray,
    t: np.ndarray,
) -> np.ndarray:
    """Single Halley step using pre-computed loop-invariant quantities.

    Avoids recomputing exp(-q*t), exp(-r*t), sqrt(t), log(s/k), and (r-q)*t
    on every iteration — those are hoisted once before the Halley loop.
    """
    vol_term = np.maximum(sigma * sqrt_t, 1e-32)
    d1 = (log_sk + carry_drift_t + 0.5 * sigma * sigma * t) / vol_term
    d2 = d1 - sigma * sqrt_t
    cdf_d1 = _norm_cdf(d1)
    cdf_d2 = _norm_cdf(d2)
    call = discounted_spot * cdf_d1 - discounted_strike * cdf_d2
    put = discounted_strike * (1.0 - cdf_d2) - discounted_spot * (1.0 - cdf_d1)
    px = np.where(is_call, call, put)
    vega = discounted_spot * _norm_pdf(d1) * sqrt_t
    residual = px - price
    safe_vega = np.where(vega > 1e-14, vega, np.inf)
    newton_step = residual / safe_vega
    safe_sigma = np.where(sigma > 1e-8, sigma, np.inf)
    halley_denom = 1.0 - newton_step * d1 * d2 / safe_sigma
    halley_denom = np.where(
        np.abs(halley_denom) > 0.05, halley_denom, np.sign(halley_denom + 1e-15) * 0.05
    )
    return np.clip(sigma - newton_step / halley_denom, _IV_LO, _IV_HI)


def _price_for_model_full(
    model: ModelLiteral,
    flag: np.ndarray,
    s: np.ndarray,
    k: np.ndarray,
    t: np.ndarray,
    r: np.ndarray,
    sigma: np.ndarray,
    q: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (price, raw_vega, d1, d2) for the chosen model in one pass."""
    if model == "black":
        return _bsm_price_full(flag, s, k, t, r, sigma, r)  # q=r → Black-76
    if model == "black_scholes":
        return _bsm_price_full(flag, s, k, t, r, sigma, np.zeros_like(r))
    return _bsm_price_full(flag, s, k, t, r, sigma, q)


def _price_for_model(
    model: ModelLiteral,
    flag: np.ndarray,
    s: np.ndarray,
    k: np.ndarray,
    t: np.ndarray,
    r: np.ndarray,
    sigma: np.ndarray,
    q: np.ndarray,
) -> np.ndarray:
    px, _, _, _ = _price_for_model_full(model, flag, s, k, t, r, sigma, q)
    return px


def implied_volatility(
    model: ModelLiteral,
    price: Float1D,
    s: Float1D,
    k: Float1D,
    t: Float1D,
    r: Float1D,
    flag: FlagArray,
    q: OptionalFloat1D = None,
    on_error: OnErrorLiteral = "warn",
) -> Float1D:
    # Black-76: q=r so carry=disc and d1 = [ln(F/K)+0.5σ²T]/(σ√T) (correct)
    qv = r if (model == "black" and q is None) else (np.zeros_like(r) if q is None else q)

    # --- intrinsic check ---
    intrinsic = _intrinsic_vec(flag, s, k, t, r, qv, model)
    below_intrinsic = price < intrinsic - 1e-10
    if np.any(below_intrinsic):
        handle_error("Option price is below intrinsic value.", on_error)

    valid = t > 0
    is_call = flag == "c"

    # Pre-compute loop-invariant quantities — hoisted out of 8 Halley iterations
    # (saves 5 array ops × 8 iters = 40 redundant passes over the data)
    if model == "black":
        discounted_spot = s * np.exp(-qv * t)
        discounted_strike = k * np.exp(-r * t)
        carry_drift_t = 0.5 * t  # (r - r) * t + 0.5*sigma^2*t → only σ² term left; (r-q)=0
        # actually carry_drift_t captures (r-q)*t; for black q=r so carry_drift_t=0
        carry_drift_t = np.zeros_like(t)
    else:
        carry_val = qv if model == "black_scholes_merton" else np.zeros_like(r)
        discounted_spot = s * np.exp(-carry_val * t)
        discounted_strike = k * np.exp(-r * t)
        carry_drift_t = (r - carry_val) * t
    sqrt_t = np.sqrt(np.maximum(t, 1e-32))
    log_sk = np.log(np.maximum(s, 1e-32) / np.maximum(k, 1e-32))

    # --- Halley's method (using pre-computed invariants) ---
    sigma = _iv_initial_guess(price, s, t)
    for _ in range(_HALLEY_ITERS):
        sigma = _halley_step_precomputed(
            sigma,
            price,
            is_call,
            discounted_spot,
            discounted_strike,
            sqrt_t,
            log_sk,
            carry_drift_t,
            t,
        )

    # --- Bisection fallback for poorly converged points ---
    px_final = _price_for_model(model, flag, s, k, t, r, sigma, qv)
    underflow_stuck = (px_final == 0.0) & (price > 0.0)
    not_converged = (np.abs(px_final - price) > 1e-6) | underflow_stuck

    if np.any(not_converged):
        sigma_lo = np.where(not_converged, _IV_LO, sigma)
        sigma_hi = np.where(not_converged, _IV_HI, sigma)
        for _ in range(_BISECT_ITERS):
            sigma_mid = 0.5 * (sigma_lo + sigma_hi)
            px_mid = _price_for_model(model, flag, s, k, t, r, sigma_mid, qv)
            mid_res = px_mid - price
            sigma_lo = np.where(not_converged & (mid_res < 0), sigma_mid, sigma_lo)
            sigma_hi = np.where(not_converged & (mid_res >= 0), sigma_mid, sigma_hi)
        sigma = np.where(not_converged, 0.5 * (sigma_lo + sigma_hi), sigma)

    result = np.where(valid, sigma, 0.0)
    # Zero-price OTM options: sigma is undetermined (any σ gives price≈0) → NaN
    zero_price = (price <= 0.0) & valid
    result = np.where(below_intrinsic | zero_price, np.nan, result)
    return result
