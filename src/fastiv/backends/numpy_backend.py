from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

from ..types import ModelLiteral
from ..utils.validation import handle_error


def _intrinsic(flag: str, s: float, k: float, t: float, r: float, q: float, model: ModelLiteral) -> float:
    if model == "black":
        return math.exp(-r * t) * max(0.0, s - k) if flag == "c" else math.exp(-r * t) * max(0.0, k - s)
    forward = s * math.exp((r - q) * t)
    call_intrinsic = max(0.0, s * math.exp(-q * t) - k * math.exp(-r * t))
    put_intrinsic = max(0.0, k * math.exp(-r * t) - s * math.exp(-q * t))
    if model == "black_scholes":
        call_intrinsic = max(0.0, s - k * math.exp(-r * t))
        put_intrinsic = max(0.0, k * math.exp(-r * t) - s)
    return call_intrinsic if flag == "c" else put_intrinsic


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


def implied_volatility(model: ModelLiteral, price: np.ndarray, s: np.ndarray, k: np.ndarray, t: np.ndarray, r: np.ndarray, flag: np.ndarray, q: np.ndarray | None = None, on_error: str = "warn") -> np.ndarray:
    qv = np.zeros_like(r) if q is None else q

    def solve_one(px: float, ss: float, kk: float, tt: float, rr: float, ff: str, qq: float) -> float:
        if tt <= 0:
            return 0.0
        intrinsic = _intrinsic(ff, ss, kk, tt, rr, qq, model)
        if px < intrinsic - 1e-10:
            handle_error("Option price is below intrinsic value.", on_error)
            return math.nan

        def objective(sig: float) -> float:
            if model == "black":
                val = price_black(np.asarray([ff]), np.asarray([ss]), np.asarray([kk]), np.asarray([tt]), np.asarray([rr]), np.asarray([sig]))[0]
            elif model == "black_scholes":
                val = price_black_scholes(np.asarray([ff]), np.asarray([ss]), np.asarray([kk]), np.asarray([tt]), np.asarray([rr]), np.asarray([sig]))[0]
            else:
                val = price_black_scholes_merton(np.asarray([ff]), np.asarray([ss]), np.asarray([kk]), np.asarray([tt]), np.asarray([rr]), np.asarray([sig]), np.asarray([qq]))[0]
            return val - px

        try:
            return brentq(objective, 1e-12, 10.0, maxiter=200)
        except ValueError:
            handle_error("Implied volatility root was not bracketed.", on_error)
            return math.nan

    vectorized = np.vectorize(solve_one, otypes=[float])
    return vectorized(price, s, k, t, r, flag, qv)
