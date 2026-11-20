from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.special import log_ndtr, ndtr

from ..types import ModelLiteral, OnErrorLiteral
from . import _iv_common as _ivc

if TYPE_CHECKING:
    from .._typing import FlagArray, Float1D, OptionalFloat1D  # noqa: F401

# Use scipy.special.ndtr for accurate extreme-tail CDF (matches erfc method).
# Compute PDF directly via numpy — avoids scipy.stats overhead (~5x faster).
_norm_cdf = ndtr
_SQRT2PI_INV = (2.0 * 3.141592653589793) ** -0.5


def _norm_pdf(x: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * x * x) * _SQRT2PI_INV


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
# Implied volatility — safeguarded Halley on the out-of-the-money price
# (problem statement, constants and row policy: backends/_iv_common.py)
# ---------------------------------------------------------------------------


def _iv_log_objective(a, s, l_fo, l_ko, upper, tiny):
    """Return (ln w, |w'|/w, d1, d2) with w = c(s) (lower branch) or u(s) = Fo - c(s) (upper).

    c = Fo N(d1) - Ko N(d2) is increasing in s; u = Fo N(-d1) + Ko N(d2) is decreasing and has no
    cancellation. Rows flagged ``tiny`` (deep tail or sub-normal target) use log_ndtr throughout.
    """
    d1 = a / s + 0.5 * s
    d2 = d1 - s
    with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
        fo = np.exp(l_fo)
        ko = np.exp(l_ko)
        n1 = _norm_cdf(np.where(upper, -d1, d1))
        n2 = _norm_cdf(d2)
        w = np.where(upper, fo * n1 + ko * n2, fo * n1 - ko * n2)
        lw = np.where(w > 0, np.log(w), -np.inf)  # w <= 0: c underflowed (sub-normal cancellation)
        if tiny.any():
            ti = np.flatnonzero(tiny)
            up = upper[ti]
            la = l_fo[ti] + log_ndtr(np.where(up, -d1[ti], d1[ti]))
            lb = l_ko[ti] + log_ndtr(d2[ti])
            lw[ti] = np.where(up, np.logaddexp(la, lb), la + np.log1p(-np.exp(lb - la)))
        dlw = np.exp(l_fo - 0.5 * d1 * d1 - _ivc.LOG_SQRT2PI - lw)
    return lw, dlw, d1, d2


def _iv_step(s, lo, hi, a, l_target, l_fo, l_ko, upper, tiny):
    """One safeguarded Halley step on G(s) = +-(ln w(s) - ln target), increasing in s.

    The sign of G at every iterate tightens the bracket [lo, hi]; a step that leaves the bracket
    (or is not finite) is replaced by bisection in log s, or by expansion while hi is unbounded.
    """
    lw, dlw, d1, d2 = _iv_log_objective(a, s, l_fo, l_ko, upper, tiny)
    sgn = np.where(upper, -1.0, 1.0)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        g = sgn * (lw - l_target)
        below = g < 0  # the root lies above s (includes ln c = -inf on underflow)
        lo = np.where(below, s, lo)
        hi = np.where(below, hi, s)
        delta = g / dlw  # Newton step on G (G' = dlw > 0)
        den = 1.0 - 0.5 * delta * (d1 * d2 / s - sgn * dlw)  # Halley: G''/G' = d1 d2 / s - sgn dlw
        den = np.where(np.abs(den) < 0.5, np.where(den < 0, -0.5, 0.5), den)
        step = delta / den
        cand = s - step
        small = np.abs(step) <= _ivc.ACCEPT_RTOL * s
        ok = np.isfinite(cand) & (((cand > lo) & (cand < hi)) | small)
        fallback = np.where(np.isinf(hi), 4.0 * s, np.where(lo > 0, np.sqrt(lo * hi), 0.25 * s))
        s_new = np.minimum(np.where(ok, cand, fallback), _ivc.S_MAX)
    return s_new, lo, hi, ok & small


def _iv_solve_otm(tv, gap, fo, ko, a, s0):
    """Solve c(s) = tv (or u(s) = gap) for the total volatility s; returns (s, converged)."""
    upper = gap < tv
    with np.errstate(divide="ignore"):
        l_target = np.log(np.where(upper, gap, tv))
    l_fo = np.log(fo)
    l_ko = np.log(ko)
    tiny = (l_target - l_fo) < _ivc.TINY_LOG
    s = s0
    lo = np.zeros_like(s)
    hi = np.full_like(s, np.inf)
    conv = np.zeros(s.shape, dtype=bool)
    for _ in range(_ivc.MAIN_ITERS):
        s, lo, hi, conv = _iv_step(s, lo, hi, a, l_target, l_fo, l_ko, upper, tiny)
    # Bracket phase: only the rows that have not converged, until the step or the bracket meets
    # the relative target (or the bracket endpoints are adjacent floats).
    todo = np.flatnonzero(~conv)
    for _ in range(_ivc.MAX_ITERS):
        if todo.size == 0:
            break
        sn, ln, hn, cv = _iv_step(
            s[todo],
            lo[todo],
            hi[todo],
            a[todo],
            l_target[todo],
            l_fo[todo],
            l_ko[todo],
            upper[todo],
            tiny[todo],
        )
        s[todo], lo[todo], hi[todo] = sn, ln, hn
        bounded = np.isfinite(hn) & (ln > 0)
        done = cv | (
            bounded & ((hn <= ln * (1.0 + _ivc.BRACKET_RTOL)) | (np.nextafter(ln, np.inf) >= hn))
        )
        conv[todo] = done
        todo = todo[~done]
    return s, conv


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
    shape, price, s, k, t, r, flag, q = _ivc.flatten_inputs(price, s, k, t, r, flag, q)
    qv = q if (q is not None and model == "black_scholes_merton") else np.zeros_like(r)
    is_call = flag == "c"
    rows = _ivc.classify_np(model, price, s, k, t, r, qv, is_call)
    code = np.asarray(rows["code"], dtype=np.int8)
    result = np.full(price.shape, np.nan)
    result[rows["zero_vol"]] = 0.0
    idx = np.flatnonzero(rows["interior"])
    if idx.size:
        sqrt_t = np.sqrt(t[idx])
        tv, fo = rows["tv"][idx], rows["Fo"][idx]
        sigma0 = np.clip(tv / (fo * sqrt_t) * _ivc.SQRT2PI, _ivc.SEED_SIGMA_LO, _ivc.SEED_SIGMA_HI)
        sol, conv = _iv_solve_otm(
            tv, rows["gap"][idx], fo, rows["Ko"][idx], rows["a"][idx], sigma0 * sqrt_t
        )
        result[idx] = np.where(conv, sol / sqrt_t, np.nan)
        code[idx[~conv]] = _ivc.NOT_CONVERGED
    _ivc.report_row_errors(_ivc.count_codes(code), price.size, on_error)
    return result.reshape(shape)
