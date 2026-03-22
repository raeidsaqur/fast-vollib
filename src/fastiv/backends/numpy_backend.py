from __future__ import annotations

import numpy as np
from scipy.stats import norm

from ..types import ModelLiteral
from ..utils.validation import handle_error


def _intrinsic_vec(flag: np.ndarray, s: np.ndarray, k: np.ndarray, t: np.ndarray, r: np.ndarray, q: np.ndarray, model: ModelLiteral) -> np.ndarray:
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


def _d1_d2(s: np.ndarray, k: np.ndarray, t: np.ndarray, r: np.ndarray, sigma: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sqrt_t = np.sqrt(np.maximum(t, 1e-32))
    vol_term = np.maximum(sigma * sqrt_t, 1e-32)
    d1 = (np.log(np.maximum(s, 1e-32) / np.maximum(k, 1e-32)) + (r - q + 0.5 * sigma**2) * t) / vol_term
    d2 = d1 - sigma * sqrt_t
    return d1, d2


def _bsm_price(flag: np.ndarray, s: np.ndarray, k: np.ndarray, t: np.ndarray, r: np.ndarray, sigma: np.ndarray, q: np.ndarray) -> np.ndarray:
    d1, d2 = _d1_d2(s, k, t, r, sigma, q)
    discounted_spot = s * np.exp(-q * t)
    discounted_strike = k * np.exp(-r * t)
    call = discounted_spot * norm.cdf(d1) - discounted_strike * norm.cdf(d2)
    put = discounted_strike * norm.cdf(-d2) - discounted_spot * norm.cdf(-d1)
    return np.where(flag == "c", call, put)


def price_black(flag: np.ndarray, f: np.ndarray, k: np.ndarray, t: np.ndarray, r: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    q = np.zeros_like(r)
    prices = _bsm_price(flag, f, k, t, r, sigma, q)
    return prices


def price_black_scholes(flag: np.ndarray, s: np.ndarray, k: np.ndarray, t: np.ndarray, r: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    q = np.zeros_like(r)
    return _bsm_price(flag, s, k, t, r, sigma, q)


def price_black_scholes_merton(flag: np.ndarray, s: np.ndarray, k: np.ndarray, t: np.ndarray, r: np.ndarray, sigma: np.ndarray, q: np.ndarray) -> np.ndarray:
    return _bsm_price(flag, s, k, t, r, sigma, q)


def _vega_raw(s: np.ndarray, k: np.ndarray, t: np.ndarray, r: np.ndarray, sigma: np.ndarray, q: np.ndarray) -> np.ndarray:
    d1, _ = _d1_d2(s, k, t, r, sigma, q)
    return s * np.exp(-q * t) * norm.pdf(d1) * np.sqrt(np.maximum(t, 1e-32))


def greeks(model: ModelLiteral, flag: np.ndarray, s: np.ndarray, k: np.ndarray, t: np.ndarray, r: np.ndarray, sigma: np.ndarray, q: np.ndarray | None = None) -> dict[str, np.ndarray]:
    qv = np.zeros_like(r) if q is None else q
    d1, d2 = _d1_d2(s, k, t, r, sigma, qv)
    carry = np.exp(-qv * t)
    disc = np.exp(-r * t)
    pdf = norm.pdf(d1)
    sign = np.where(flag == "c", 1.0, -1.0)

    delta = np.where(flag == "c", carry * norm.cdf(d1), carry * (norm.cdf(d1) - 1.0))
    gamma = carry * pdf / (np.maximum(s, 1e-32) * np.maximum(sigma, 1e-32) * np.sqrt(np.maximum(t, 1e-32)))
    vega = _vega_raw(s, k, t, r, sigma, qv) * 0.01
    theta_call = (
        -(s * carry * pdf * sigma) / (2.0 * np.sqrt(np.maximum(t, 1e-32)))
        - r * k * disc * norm.cdf(d2)
        + qv * s * carry * norm.cdf(d1)
    ) / 365.0
    theta_put = (
        -(s * carry * pdf * sigma) / (2.0 * np.sqrt(np.maximum(t, 1e-32)))
        + r * k * disc * norm.cdf(-d2)
        - qv * s * carry * norm.cdf(-d1)
    ) / 365.0
    rho_call = k * t * disc * norm.cdf(d2) * 0.01
    rho_put = -k * t * disc * norm.cdf(-d2) * 0.01
    rho = np.where(flag == "c", rho_call, rho_put)
    theta = np.where(flag == "c", theta_call, theta_put)

    if model == "black":
        delta = disc * np.where(flag == "c", norm.cdf(d1), norm.cdf(d1) - 1.0)
        gamma = disc * pdf / (np.maximum(s, 1e-32) * np.maximum(sigma, 1e-32) * np.sqrt(np.maximum(t, 1e-32)))

    return {
        "delta": delta,
        "gamma": gamma,
        "theta": theta,
        "rho": rho,
        "vega": vega,
    }


def _price_for_model(model: ModelLiteral, flag: np.ndarray, s: np.ndarray, k: np.ndarray, t: np.ndarray, r: np.ndarray, sigma: np.ndarray, q: np.ndarray) -> np.ndarray:
    if model == "black":
        return price_black(flag, s, k, t, r, sigma)
    if model == "black_scholes":
        return price_black_scholes(flag, s, k, t, r, sigma)
    return price_black_scholes_merton(flag, s, k, t, r, sigma, q)


def _iv_initial_guess(price: np.ndarray, s: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Brenner-Subrahmanyam ATM approximation as Newton seed."""
    sqrt_t = np.sqrt(np.maximum(t, 1e-8))
    approx = price / np.maximum(s * sqrt_t, 1e-12) * np.sqrt(2.0 * np.pi)
    return np.clip(approx, 0.01, 5.0)


_NEWTON_ITERS = 20
_BISECT_ITERS = 50
_IV_LO = 1e-8
_IV_HI = 10.0


def implied_volatility(model: ModelLiteral, price: np.ndarray, s: np.ndarray, k: np.ndarray, t: np.ndarray, r: np.ndarray, flag: np.ndarray, q: np.ndarray | None = None, on_error: str = "warn") -> np.ndarray:
    qv = np.zeros_like(r) if q is None else q

    # --- intrinsic check (vectorized) ---
    intrinsic = _intrinsic_vec(flag, s, k, t, r, qv, model)
    below_intrinsic = price < intrinsic - 1e-10
    if np.any(below_intrinsic):
        handle_error("Option price is below intrinsic value.", on_error)

    valid = t > 0

    # --- Newton-Raphson ---
    sigma = _iv_initial_guess(price, s, t)
    for _ in range(_NEWTON_ITERS):
        px = _price_for_model(model, flag, s, k, t, r, sigma, qv)
        residual = px - price
        v = _vega_raw(s, k, t, r, sigma, qv)
        # Guard against near-zero vega (deep ITM/OTM); skip step where vega is tiny
        safe_vega = np.where(v > 1e-14, v, np.inf)
        sigma = np.clip(sigma - residual / safe_vega, _IV_LO, _IV_HI)

    # --- Bisection fallback for poorly converged points ---
    px_final = _price_for_model(model, flag, s, k, t, r, sigma, qv)
    # Also catch underflow-stuck Newton: price_BS==0 but target price is nonzero.
    # This happens for deep OTM options where many σ values give identical float64 prices.
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
    result = np.where(below_intrinsic, np.nan, result)
    return result
