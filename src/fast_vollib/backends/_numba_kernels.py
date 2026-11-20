# Author: Raeid Saqur
# Date: 2026-11-19
"""Numba kernels of the CPU backend, at module scope.

Every kernel is a plain module-level function, so ``cache=True`` can serve it from the on-disk
cache in later processes. (Defined inside a builder function, the kernels were closures over
each other and were recompiled, and re-cached, in every process.) This module imports numba and
is imported lazily by ``numba_backend._build_kernels``.
"""

from __future__ import annotations

import math

import numba
import numpy as np

from . import _iv_common as _ivc

# Constants: numba freezes module globals at compile time.
SQRT2 = math.sqrt(2.0)
SQRT2PI_INV = 1.0 / math.sqrt(2.0 * math.pi)

# ------------------------------------------------------------------
# Scalar primitives
# ------------------------------------------------------------------


@numba.njit(cache=True, fastmath=False)
def _norm_cdf(x: float) -> float:
    """Normal CDF via erfc: N(x) = 0.5·erfc(−x/√2).

    Uses ``math.erfc`` so extreme-tail accuracy matches scipy.special.ndtr.
    ``fastmath=False`` is required: fast-math may rearrange the subtraction
    in a way that degrades accuracy for |x| > 8.
    """
    return 0.5 * math.erfc(-x / SQRT2)


@numba.njit(cache=True, fastmath=False)
def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) * SQRT2PI_INV


@numba.njit(cache=True)
def _d1_d2(s: float, k: float, t: float, r: float, sigma: float, q: float):
    sqrt_t = math.sqrt(max(t, 1e-32))
    vol_t = max(sigma * sqrt_t, 1e-32)
    d1 = (math.log(max(s, 1e-32) / max(k, 1e-32)) + (r - q + 0.5 * sigma * sigma) * t) / vol_t
    d2 = d1 - sigma * sqrt_t
    return d1, d2


@numba.njit(cache=True)
def _bsm_price(
    is_call: bool,
    s: float,
    k: float,
    t: float,
    r: float,
    sigma: float,
    q: float,
) -> float:
    d1, d2 = _d1_d2(s, k, t, r, sigma, q)
    ds = s * math.exp(-q * t)
    dk = k * math.exp(-r * t)
    nd1 = _norm_cdf(d1)
    nd2 = _norm_cdf(d2)
    call = ds * nd1 - dk * nd2
    put = dk * (1.0 - nd2) - ds * (1.0 - nd1)
    return call if is_call else put


# ------------------------------------------------------------------
# Vectorised pricing kernel
# ------------------------------------------------------------------


@numba.njit(parallel=True, cache=True)
def _price_kernel(
    is_call: np.ndarray,
    s: np.ndarray,
    k: np.ndarray,
    t: np.ndarray,
    r: np.ndarray,
    sigma: np.ndarray,
    q: np.ndarray,
) -> np.ndarray:
    n = s.shape[0]
    out = np.empty(n, np.float64)
    for i in numba.prange(n):  # type: ignore[attr-defined]
        out[i] = _bsm_price(is_call[i], s[i], k[i], t[i], r[i], sigma[i], q[i])
    return out


# ------------------------------------------------------------------
# Vectorised Greeks kernel (all 5 Greeks in one pass)
#
# ``q`` is pre-adjusted by the Python wrapper so the same kernel handles
# all three models:
#   Black-76:      q = r  → carry = disc (delta/gamma use carry, which equals disc)
#   Black-Scholes: q = 0
#   BSM:           q = dividend yield
# ------------------------------------------------------------------


@numba.njit(parallel=True, cache=True)
def _greeks_kernel(
    is_call: np.ndarray,
    s: np.ndarray,
    k: np.ndarray,
    t: np.ndarray,
    r: np.ndarray,
    sigma: np.ndarray,
    q: np.ndarray,
):
    n = s.shape[0]
    d_ = np.empty(n, np.float64)
    g_ = np.empty(n, np.float64)
    th_ = np.empty(n, np.float64)
    rh_ = np.empty(n, np.float64)
    v_ = np.empty(n, np.float64)

    for i in numba.prange(n):  # type: ignore[attr-defined]
        ic = is_call[i]
        d1, d2 = _d1_d2(s[i], k[i], t[i], r[i], sigma[i], q[i])
        carry = math.exp(-q[i] * t[i])
        disc = math.exp(-r[i] * t[i])
        pdf = _norm_pdf(d1)
        sq_t = math.sqrt(max(t[i], 1e-32))
        ss = max(s[i], 1e-32)
        ssig = max(sigma[i], 1e-32)
        nd1 = _norm_cdf(d1)
        nd2 = _norm_cdf(d2)
        nd1_ = 1.0 - nd1
        nd2_ = 1.0 - nd2

        # For Black-76 q=r → carry=disc, so these are numerically identical
        # to the numpy backend's explicit disc-based Black-76 formulas.
        d_[i] = carry * (nd1 if ic else nd1 - 1.0)
        g_[i] = carry * pdf / (ss * ssig * sq_t)
        v_[i] = s[i] * carry * pdf * sq_t * 0.01

        tc = (
            -(s[i] * carry * pdf * sigma[i]) / (2.0 * sq_t)
            - r[i] * k[i] * disc * nd2
            + q[i] * s[i] * carry * nd1
        ) / 365.0
        tp = (
            -(s[i] * carry * pdf * sigma[i]) / (2.0 * sq_t)
            + r[i] * k[i] * disc * nd2_
            - q[i] * s[i] * carry * nd1_
        ) / 365.0
        th_[i] = tc if ic else tp

        rc = k[i] * t[i] * disc * nd2 * 0.01
        rp = -k[i] * t[i] * disc * nd2_ * 0.01
        rh_[i] = rc if ic else rp

    return d_, g_, th_, rh_, v_


# ------------------------------------------------------------------
# IV solver kernel: per-element classification + safeguarded Halley on the
# out-of-the-money price (problem statement and policy: _iv_common.py).
# Writes sigma and the row code; one Python->numba dispatch per batch.
# ------------------------------------------------------------------

MAIN_ITERS = _ivc.MAIN_ITERS
TOTAL_ITERS = _ivc.MAIN_ITERS + _ivc.MAX_ITERS
ACCEPT_RTOL = _ivc.ACCEPT_RTOL
BRACKET_RTOL = _ivc.BRACKET_RTOL
SEED_LO = _ivc.SEED_SIGMA_LO
SEED_HI = _ivc.SEED_SIGMA_HI
TINY_LOG = _ivc.TINY_LOG
S_MAX = _ivc.S_MAX
SQRT2PI = _ivc.SQRT2PI
LOG_SQRT2PI = _ivc.LOG_SQRT2PI
SPLIT = 134217729.0
C_BELOW = _ivc.BELOW_INTRINSIC
C_ZERO = _ivc.ZERO_PRICE
C_ABOVE = _ivc.ABOVE_UPPER
C_INVALID = _ivc.INVALID_INPUT
C_NOCONV = _ivc.NOT_CONVERGED


@numba.njit(cache=True, fastmath=False)
def _two_sum(a: float, b: float):
    s = a + b
    bb = s - a
    return s, (a - (s - bb)) + (b - bb)


@numba.njit(cache=True, fastmath=False)
def _two_prod(a: float, b: float):
    p = a * b
    t = SPLIT * a
    ah = t - (t - a)
    al = a - ah
    t = SPLIT * b
    bh = t - (t - b)
    bl = b - bh
    return p, ((ah * bh - p) + ah * bl + al * bh) + al * bl


@numba.njit(cache=True, fastmath=False)
def _scaled(x: float, em1: float):
    p, e = _two_prod(x, em1)
    hi, lo = _two_sum(x, p)
    return hi, lo + e


@numba.njit(cache=True, fastmath=False)
def _dd_sub(ah: float, al: float, bh: float, bl: float):
    s, e = _two_sum(ah, -bh)
    return _two_sum(s, e + (al - bl))


LN2_A = _ivc.LN2_A
LN2_B = _ivc.LN2_B
LN2_C = _ivc.LN2_C
INV_FACT = _ivc.INV_FACT
N_FACT = len(_ivc.INV_FACT)
SQUARINGS = _ivc.EXPM1_SQUARINGS
DD_MAX = _ivc.EXPM1_DD_MAX
ERR_FACTOR = _ivc.BOUND_ERR_FACTOR


@numba.njit(cache=True, fastmath=False)
def _fast_two_sum(a: float, b: float):
    s = a + b
    return s, b - (s - a)


@numba.njit(cache=True, fastmath=False)
def _dd_add(ah: float, al: float, bh: float, bl: float):
    s, e = _two_sum(ah, bh)
    return _fast_two_sum(s, e + (al + bl))


@numba.njit(cache=True, fastmath=False)
def _dd_mul(ah: float, al: float, bh: float, bl: float):
    p, e = _two_prod(ah, bh)
    return _fast_two_sum(p, e + (ah * bl + al * bh))


@numba.njit(cache=True, fastmath=False)
def _dd_expm1(y_h: float, y_l: float):
    """Double-double expm1; see _iv_common.dd_expm1 (same steps, scalar form)."""
    kk = float(np.rint(y_h / LN2_A))
    t1 = y_h - kk * LN2_A
    p, pe = _two_prod(kk, LN2_B)
    z_h, z_l = _two_sum(t1, -p)
    z_h, z_l = _fast_two_sum(z_h, z_l + ((y_l - pe) - kk * LN2_C))
    sc = 2.0**-SQUARINGS
    w_h = z_h * sc
    w_l = z_l * sc
    a_h = INV_FACT[N_FACT - 1][0]
    a_l = INV_FACT[N_FACT - 1][1]
    for j in range(N_FACT - 2, -1, -1):
        a_h, a_l = _dd_mul(a_h, a_l, w_h, w_l)
        a_h, a_l = _dd_add(a_h, a_l, INV_FACT[j][0], INV_FACT[j][1])
    e_h, e_l = _dd_mul(a_h, a_l, w_h, w_l)
    for _ in range(SQUARINGS):
        sq_h, sq_l = _dd_mul(e_h, e_l, e_h, e_l)
        e_h, e_l = _dd_add(2.0 * e_h, 2.0 * e_l, sq_h, sq_l)
    two_k = math.ldexp(1.0, int(kk))
    c_h, c_l = _two_sum(two_k, -1.0)
    return _dd_add(e_h * two_k, e_l * two_k, c_h, c_l)


@numba.njit(cache=True, fastmath=False)
def _scaled_dd(x: float, em_h: float, em_l: float):
    p, e = _two_prod(x, em_h)
    hi, lo = _two_sum(x, p)
    return _fast_two_sum(hi, lo + (e + x * em_l))


@numba.njit(cache=True, fastmath=False)
def _expm1_neg(a: float, b: float, precise: bool):
    """expm1(-a b) as hi + lo (double-double when precise and |a b| <= DD_MAX)."""
    ph, pl = _two_prod(a, b)
    if precise and abs(ph) <= DD_MAX:
        return _dd_expm1(-ph, -pl)
    return math.expm1(-ph), 0.0


@numba.njit(cache=True, fastmath=False)
def _bounds(model: int, s: float, k: float, t: float, r: float, q: float, precise: bool):
    """Discounted forward / strike as hi + lo and the error bound of the double-precision
    version (zero when precise); see _iv_common.discounted_bounds."""
    er_h, er_l = _expm1_neg(r, t, precise)
    rt = abs(r * t)
    if precise:
        dk_h, dk_l = _scaled_dd(k, er_h, er_l)
    else:
        dk_h, dk_l = _scaled(k, er_h)
    err = ERR_FACTOR * k * (rt + abs(er_h))
    if model == 0:
        if precise:
            ds_h, ds_l = _scaled_dd(s, er_h, er_l)
        else:
            ds_h, ds_l = _scaled(s, er_h)
        err += ERR_FACTOR * s * (rt + abs(er_h))
    elif model == 1:
        ds_h, ds_l = s, 0.0
    else:
        eq_h, eq_l = _expm1_neg(q, t, precise)
        if precise:
            ds_h, ds_l = _scaled_dd(s, eq_h, eq_l)
        else:
            ds_h, ds_l = _scaled(s, eq_h)
        err += ERR_FACTOR * s * (abs(q * t) + abs(eq_h))
    if precise:
        err = 0.0
    return ds_h, ds_l, dk_h, dk_l, err


@numba.njit(cache=True, fastmath=False)
def _log_ndtr(x: float) -> float:
    if x > 0.0:
        return math.log1p(-0.5 * math.erfc(x / SQRT2))
    if x > -20.0:
        return math.log(0.5 * math.erfc(-x / SQRT2))
    z = 1.0 / (x * x)  # asymptotic series of the Mills ratio
    term = 1.0
    acc = 1.0
    for j in range(1, 14):
        term *= -(2.0 * j - 1.0) * z
        acc += term
    return -0.5 * x * x - math.log(-x) - LOG_SQRT2PI + math.log(acc)


@numba.njit(cache=True, fastmath=False)
def _log_objective(
    a: float, s: float, fo: float, ko: float, l_fo: float, l_ko: float, upper: bool, tiny: bool
):
    d1 = a / s + 0.5 * s
    d2 = d1 - s
    if tiny:
        la = l_fo + _log_ndtr(-d1 if upper else d1)
        lb = l_ko + _log_ndtr(d2)
        if upper:
            m = max(la, lb)
            lw = m + math.log1p(math.exp(-abs(la - lb))) if m > -math.inf else -math.inf
        else:
            lw = la + math.log1p(-math.exp(lb - la)) if lb < la else -math.inf
    else:
        if upper:
            w = fo * _norm_cdf(-d1) + ko * _norm_cdf(d2)
        else:
            w = fo * _norm_cdf(d1) - ko * _norm_cdf(d2)
        lw = math.log(w) if w > 0.0 else -math.inf
    dlw = math.exp(l_fo - 0.5 * d1 * d1 - LOG_SQRT2PI - lw)
    return lw, dlw, d1, d2


@numba.njit(parallel=True, cache=True, fastmath=False)
def _iv_kernel(
    is_call: np.ndarray,
    price: np.ndarray,
    s: np.ndarray,
    k: np.ndarray,
    t: np.ndarray,
    r: np.ndarray,
    q: np.ndarray,
    model: int,
):
    n = s.shape[0]
    out = np.empty(n, np.float64)
    code = np.zeros(n, np.int8)
    for i in numba.prange(n):  # type: ignore[attr-defined]
        p = price[i]
        ti = t[i]
        ic = is_call[i]
        # --- discounted forward / strike in double-double, log-moneyness ---
        ds_h, ds_l, dk_h, dk_l, err = _bounds(model, s[i], k[i], ti, r[i], q[i], False)
        if model == 0:  # black
            x = math.log(s[i] / k[i])
        elif model == 1:  # black_scholes
            x = math.log(s[i] / k[i]) + r[i] * ti
        else:  # black_scholes_merton
            x = math.log(s[i] / k[i]) + (r[i] - q[i]) * ti
        if (
            not (
                math.isfinite(ds_h)
                and math.isfinite(dk_h)
                and math.isfinite(p)
                and math.isfinite(ti)
            )
            or s[i] <= 0.0
            or k[i] <= 0.0
            or ti < 0.0
            or p < 0.0
        ):
            out[i] = math.nan
            code[i] = C_INVALID
            continue
        if ti == 0.0:
            out[i] = 0.0
            continue
        # Classify against the double-precision bounds; rows within their error bound are
        # classified again with double-double bounds (second pass).
        for attempt in range(2):
            m_h, m_l = _dd_sub(ds_h, ds_l, dk_h, dk_l)
            if ic:
                itm = (m_h > 0.0) or (m_h == 0.0 and m_l > 0.0)
                i_h, i_l = (m_h, m_l) if itm else (0.0, 0.0)
                u_h, u_l = ds_h, ds_l
            else:
                itm = (m_h < 0.0) or (m_h == 0.0 and m_l < 0.0)
                i_h, i_l = (-m_h, -m_l) if itm else (0.0, 0.0)
                u_h, u_l = dk_h, dk_l
            tv_h, tv_l = _dd_sub(p, 0.0, i_h, i_l)
            tv = tv_h + tv_l
            g_h, g_l = _dd_sub(u_h, u_l, p, 0.0)
            gap = g_h + g_l
            if (
                attempt == 0
                and err > 0.0
                and (abs(tv) <= err or abs(gap) <= err or abs(m_h) <= err)
            ):
                ds_h, ds_l, dk_h, dk_l, err = _bounds(model, s[i], k[i], ti, r[i], q[i], True)
                continue
            break
        if p == 0.0 and not itm:
            out[i] = math.nan
            code[i] = C_ZERO
            continue
        if tv_h < 0.0 or (tv_h == 0.0 and tv_l < 0.0):
            out[i] = math.nan
            code[i] = C_BELOW
            continue
        if tv == 0.0:
            out[i] = 0.0
            continue
        if g_h < 0.0 or (g_h == 0.0 and g_l <= 0.0):
            out[i] = math.nan
            code[i] = C_ABOVE
            continue
        # --- solve the out-of-the-money problem for s = sigma sqrt(T) ---
        fo = min(ds_h, dk_h)
        ko = max(ds_h, dk_h)
        a = -abs(x)
        sqrt_t = math.sqrt(ti)
        upper = gap < tv
        l_target = math.log(gap if upper else tv)
        l_fo = math.log(fo)
        l_ko = math.log(ko)
        tiny = (l_target - l_fo) < TINY_LOG
        sgn = -1.0 if upper else 1.0
        sv = min(max(tv / (fo * sqrt_t) * SQRT2PI, SEED_LO), SEED_HI) * sqrt_t
        lo = 0.0
        hi = math.inf
        conv = False
        for it in range(TOTAL_ITERS):
            lw, dlw, d1, d2 = _log_objective(a, sv, fo, ko, l_fo, l_ko, upper, tiny)
            g = sgn * (lw - l_target)
            if g < 0.0:
                lo = sv
            else:
                hi = sv
            delta = g / dlw
            den = 1.0 - 0.5 * delta * (d1 * d2 / sv - sgn * dlw)
            if abs(den) < 0.5:
                den = -0.5 if den < 0.0 else 0.5
            step = delta / den
            cand = sv - step
            small = abs(step) <= ACCEPT_RTOL * sv
            if math.isfinite(cand) and ((cand > lo and cand < hi) or small):
                sv = cand
            elif math.isinf(hi):
                sv = 4.0 * sv
            elif lo > 0.0:
                sv = math.sqrt(lo * hi)
            else:
                sv = 0.25 * sv
            sv = min(sv, S_MAX)
            if small and math.isfinite(cand):
                conv = True
                break
            if it >= MAIN_ITERS - 1 and math.isfinite(hi) and lo > 0.0:
                if hi <= lo * (1.0 + BRACKET_RTOL) or np.nextafter(lo, math.inf) >= hi:
                    conv = True
                    break
        if conv:
            out[i] = sv / sqrt_t
        else:
            out[i] = math.nan
            code[i] = C_NOCONV
    return out, code
