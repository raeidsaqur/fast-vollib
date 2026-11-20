"""Triton CUDA kernels for fused BSM pricing, IV solving, and Greeks.

Each kernel reads inputs once from HBM, keeps all intermediate values
in registers, and writes the final result(s) once — eliminating the
repeated HBM read/write cycles that torch.compile's unrolled loops incur
across iteration boundaries.

Normal CDF uses the identity  N(x) = 0.5 * (1 + erf(x / sqrt(2)))  via
tl.math.erf, which maps to CUDA libm __erf (< 1 ULP for float64).

All kernels are element-wise and fully independent across lanes, so the
grid is simply ceil(n / BLOCK_SIZE) with 1D blocks.
"""

from __future__ import annotations

import numpy as np
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from . import _iv_common as _ivc

# ---------------------------------------------------------------------------
# Constants (tl.constexpr so they can be referenced inside @triton.jit kernels)
# ---------------------------------------------------------------------------
_INV_SQRT2 = tl.constexpr(0.7071067811865476)  # 1 / sqrt(2)
_INV_SQRT2PI = tl.constexpr(0.3989422804014327)  # 1 / sqrt(2π)
_SQRT2PI_C = tl.constexpr(_ivc.SQRT2PI)
_LOG_SQRT2PI_C = tl.constexpr(_ivc.LOG_SQRT2PI)


# ---------------------------------------------------------------------------
# Helper: N(x) = 0.5*(1+erf(x/√2))  —  inline in kernels for register fusion
# ---------------------------------------------------------------------------


@triton.jit
def _norm_cdf_tl(x):
    return 0.5 * (1.0 + tl.math.erf(x * _INV_SQRT2))


@triton.jit
def _norm_pdf_tl(x):
    return tl.exp(-0.5 * x * x) * _INV_SQRT2PI


# ---------------------------------------------------------------------------
# 1.  BSM pricing kernel  (single-pass: read 6 inputs, write 1 output)
# ---------------------------------------------------------------------------


@triton.jit
def bsm_price_kernel(
    s_ptr,
    k_ptr,
    t_ptr,
    r_ptr,
    sigma_ptr,
    q_ptr,
    is_call_ptr,
    out_ptr,
    n,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n

    s = tl.load(s_ptr + offs, mask=mask)
    k = tl.load(k_ptr + offs, mask=mask)
    t = tl.load(t_ptr + offs, mask=mask)
    r = tl.load(r_ptr + offs, mask=mask)
    sigma = tl.load(sigma_ptr + offs, mask=mask)
    q = tl.load(q_ptr + offs, mask=mask)
    is_call = tl.load(is_call_ptr + offs, mask=mask).to(tl.int1)

    sqrt_t = tl.sqrt(tl.maximum(t, 1e-32))
    vol_term = tl.maximum(sigma * sqrt_t, 1e-32)
    log_sk = tl.log(tl.maximum(s, 1e-32) / tl.maximum(k, 1e-32))
    d1 = (log_sk + (r - q + 0.5 * sigma * sigma) * t) / vol_term
    d2 = d1 - sigma * sqrt_t

    disc_s = s * tl.exp(-q * t)
    disc_k = k * tl.exp(-r * t)
    cdf_d1 = _norm_cdf_tl(d1)
    cdf_d2 = _norm_cdf_tl(d2)
    call = disc_s * cdf_d1 - disc_k * cdf_d2
    put = disc_k * (1.0 - cdf_d2) - disc_s * (1.0 - cdf_d1)
    price = tl.where(is_call, call, put)

    tl.store(out_ptr + offs, price, mask=mask)


# ---------------------------------------------------------------------------
# 2.  BSM Greeks kernel  (single-pass: read 7 inputs, write 5 outputs)
# ---------------------------------------------------------------------------


@triton.jit
def bsm_greeks_kernel(
    s_ptr,
    k_ptr,
    t_ptr,
    r_ptr,
    sigma_ptr,
    q_ptr,
    is_call_ptr,
    delta_ptr,
    gamma_ptr,
    theta_ptr,
    rho_ptr,
    vega_ptr,
    n,
    is_black: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n

    s = tl.load(s_ptr + offs, mask=mask)
    k = tl.load(k_ptr + offs, mask=mask)
    t = tl.load(t_ptr + offs, mask=mask)
    r = tl.load(r_ptr + offs, mask=mask)
    sigma = tl.load(sigma_ptr + offs, mask=mask)
    q = tl.load(q_ptr + offs, mask=mask)
    is_call = tl.load(is_call_ptr + offs, mask=mask).to(tl.int1)

    sqrt_t = tl.sqrt(tl.maximum(t, 1e-32))
    vol_term = tl.maximum(sigma * sqrt_t, 1e-32)
    log_sk = tl.log(tl.maximum(s, 1e-32) / tl.maximum(k, 1e-32))
    d1 = (log_sk + (r - q + 0.5 * sigma * sigma) * t) / vol_term
    d2 = d1 - sigma * sqrt_t

    disc = tl.exp(-r * t)
    carry = tl.exp(-q * t)
    pdf = _norm_pdf_tl(d1)
    cdf_d1 = _norm_cdf_tl(d1)
    cdf_d2 = _norm_cdf_tl(d2)
    cdf_nd1 = 1.0 - cdf_d1
    cdf_nd2 = 1.0 - cdf_d2

    safe_s = tl.maximum(s, 1e-32)
    safe_sig = tl.maximum(sigma, 1e-32)

    if is_black:
        delta = disc * tl.where(is_call, cdf_d1, cdf_d1 - 1.0)
        gamma = disc * pdf / (safe_s * safe_sig * sqrt_t)
    else:
        delta = tl.where(is_call, carry * cdf_d1, carry * (cdf_d1 - 1.0))
        gamma = carry * pdf / (safe_s * safe_sig * sqrt_t)

    vega = s * carry * pdf * sqrt_t * 0.01
    theta_call = (
        -(s * carry * pdf * sigma) / (2.0 * sqrt_t) - r * k * disc * cdf_d2 + q * s * carry * cdf_d1
    ) / 365.0
    theta_put = (
        -(s * carry * pdf * sigma) / (2.0 * sqrt_t)
        + r * k * disc * cdf_nd2
        - q * s * carry * cdf_nd1
    ) / 365.0
    theta = tl.where(is_call, theta_call, theta_put)

    rho_call = k * t * disc * cdf_d2 * 0.01
    rho_put = -k * t * disc * cdf_nd2 * 0.01
    rho = tl.where(is_call, rho_call, rho_put)

    tl.store(delta_ptr + offs, delta, mask=mask)
    tl.store(gamma_ptr + offs, gamma, mask=mask)
    tl.store(theta_ptr + offs, theta, mask=mask)
    tl.store(rho_ptr + offs, rho, mask=mask)
    tl.store(vega_ptr + offs, vega, mask=mask)


# ---------------------------------------------------------------------------
# 3.  IV kernel: row classification + safeguarded Halley on the out-of-the-money
#     price, fused per element (problem statement and policy: _iv_common.py).
#     Writes sigma and the row code; lanes leave the loop as they converge and
#     the block stops when none is left.
# ---------------------------------------------------------------------------


@triton.jit
def _ndtr_c_tl(x):
    # N(x) = 0.5 erfc(-x / sqrt 2): relative accuracy in the lower tail
    return 0.5 * libdevice.erfc(-x * _INV_SQRT2)


@triton.jit
def _log_ndtr_tl(x):
    neg = x < 0.0
    xn = tl.where(neg, x, 0.0)
    xp = tl.where(neg, 0.0, x)
    lower = tl.log(0.5 * libdevice.erfcx(-xn * _INV_SQRT2)) - 0.5 * xn * xn
    upper = libdevice.log1p(-0.5 * libdevice.erfc(xp * _INV_SQRT2))
    return tl.where(neg, lower, upper)


@triton.jit
def _two_sum_tl(a, b):
    s = a + b
    bb = s - a
    return s, (a - (s - bb)) + (b - bb)


@triton.jit
def _scaled_tl(x, em1):
    # x * (1 + em1) = hi + lo; the product error is taken exactly with an FMA
    p = x * em1
    e = libdevice.fma(x, em1, -p)
    hi, lo = _two_sum_tl(x, p)
    return hi, lo + e


@triton.jit
def _dd_sub_tl(ah, al, bh, bl):
    s, e = _two_sum_tl(ah, -bh)
    return _two_sum_tl(s, e + (al - bl))


# Double-double helpers for the bounds refinement (see _iv_common.dd_expm1); the exact product
# uses an FMA, so floating-point contraction cannot break it.
_LN2_A = tl.constexpr(_ivc.LN2_A)
_LN2_B = tl.constexpr(_ivc.LN2_B)
_LN2_C = tl.constexpr(_ivc.LN2_C)
_F1_H = tl.constexpr(1.0)
_F1_L = tl.constexpr(0.0)
_F2_H = tl.constexpr(0.5)
_F2_L = tl.constexpr(0.0)
_F3_H = tl.constexpr(0.16666666666666666)
_F3_L = tl.constexpr(9.25185853854297e-18)
_F4_H = tl.constexpr(0.041666666666666664)
_F4_L = tl.constexpr(2.3129646346357427e-18)
_F5_H = tl.constexpr(0.008333333333333333)
_F5_L = tl.constexpr(1.1564823173178714e-19)
_F6_H = tl.constexpr(0.001388888888888889)
_F6_L = tl.constexpr(-5.300543954373577e-20)
_F7_H = tl.constexpr(0.0001984126984126984)
_F7_L = tl.constexpr(1.7209558293420705e-22)
_F8_H = tl.constexpr(2.48015873015873e-05)
_F8_L = tl.constexpr(2.1511947866775882e-23)
_F9_H = tl.constexpr(2.7557319223985893e-06)
_F9_L = tl.constexpr(-1.858393274046472e-22)
_DD_MAX = tl.constexpr(_ivc.EXPM1_DD_MAX)
_ERR_FACTOR = tl.constexpr(_ivc.BOUND_ERR_FACTOR)
_EXPM1_SCALE = tl.constexpr(2.0**-_ivc.EXPM1_SQUARINGS)
_EXPM1_SQUARINGS = tl.constexpr(_ivc.EXPM1_SQUARINGS)


@triton.jit
def _two_prod_tl(a, b):
    p = a * b
    return p, libdevice.fma(a, b, -p)


@triton.jit
def _fast_two_sum_tl(a, b):
    s = a + b
    return s, b - (s - a)


@triton.jit
def _dd_add_tl(ah, al, bh, bl):
    s, e = _two_sum_tl(ah, bh)
    return _fast_two_sum_tl(s, e + (al + bl))


@triton.jit
def _dd_mul_tl(ah, al, bh, bl):
    p, e = _two_prod_tl(ah, bh)
    return _fast_two_sum_tl(p, e + (ah * bl + al * bh))


@triton.jit
def _horner_step_tl(a_h, a_l, w_h, w_l, c_h, c_l):
    a_h, a_l = _dd_mul_tl(a_h, a_l, w_h, w_l)
    return _dd_add_tl(a_h, a_l, c_h, c_l)


@triton.jit
def _dd_expm1_tl(y_h, y_l):
    kk = libdevice.rint(y_h / _LN2_A)
    t1 = y_h - kk * _LN2_A
    p, pe = _two_prod_tl(kk, kk * 0.0 + _LN2_B)  # fp64 operand for the libdevice FMA
    z_h, z_l = _two_sum_tl(t1, -p)
    z_h, z_l = _fast_two_sum_tl(z_h, z_l + ((y_l - pe) - kk * _LN2_C))
    w_h = z_h * _EXPM1_SCALE
    w_l = z_l * _EXPM1_SCALE
    zero = w_h * 0.0
    a_h = zero + _F9_H
    a_l = zero + _F9_L
    a_h, a_l = _horner_step_tl(a_h, a_l, w_h, w_l, _F8_H, _F8_L)
    a_h, a_l = _horner_step_tl(a_h, a_l, w_h, w_l, _F7_H, _F7_L)
    a_h, a_l = _horner_step_tl(a_h, a_l, w_h, w_l, _F6_H, _F6_L)
    a_h, a_l = _horner_step_tl(a_h, a_l, w_h, w_l, _F5_H, _F5_L)
    a_h, a_l = _horner_step_tl(a_h, a_l, w_h, w_l, _F4_H, _F4_L)
    a_h, a_l = _horner_step_tl(a_h, a_l, w_h, w_l, _F3_H, _F3_L)
    a_h, a_l = _horner_step_tl(a_h, a_l, w_h, w_l, _F2_H, _F2_L)
    a_h, a_l = _horner_step_tl(a_h, a_l, w_h, w_l, _F1_H, _F1_L)
    e_h, e_l = _dd_mul_tl(a_h, a_l, w_h, w_l)
    for _ in tl.static_range(_EXPM1_SQUARINGS):
        sq_h, sq_l = _dd_mul_tl(e_h, e_l, e_h, e_l)
        e_h, e_l = _dd_add_tl(2.0 * e_h, 2.0 * e_l, sq_h, sq_l)
    two_k = ((kk.to(tl.int64) + 1023) << 52).to(tl.float64, bitcast=True)
    c_h, c_l = _two_sum_tl(two_k, zero - 1.0)
    return _dd_add_tl(e_h * two_k, e_l * two_k, c_h, c_l)


@triton.jit
def _expm1_neg_dd_tl(a, b):
    """expm1(-a b) in double-double (falls back to double precision for |a b| > _DD_MAX)."""
    ph, pl = _two_prod_tl(a, b)
    lim = tl.abs(ph) <= _DD_MAX
    zero = ph * 0.0
    e_h, e_l = _dd_expm1_tl(tl.where(lim, -ph, zero), tl.where(lim, -pl, zero))
    return tl.where(lim, e_h, libdevice.expm1(-ph)), tl.where(lim, e_l, zero)


@triton.jit
def _scaled_dd_tl(x, em_h, em_l):
    p, e = _two_prod_tl(x, em_h)
    hi, lo = _two_sum_tl(x, p)
    return _fast_two_sum_tl(hi, lo + (e + x * em_l))


@triton.jit
def _classify_tl(p, ic, ds_h, ds_l, dk_h, dk_l):
    """Moneyness, intrinsic/time value and distance to the upper bound, in double-double."""
    zero = p * 0.0
    m_h, m_l = _dd_sub_tl(ds_h, ds_l, dk_h, dk_l)
    call_itm = (m_h > 0.0) | ((m_h == 0.0) & (m_l > 0.0))
    put_itm = (m_h < 0.0) | ((m_h == 0.0) & (m_l < 0.0))
    itm = tl.where(ic, call_itm, put_itm)
    i_h = tl.where(itm, tl.where(ic, m_h, -m_h), zero)
    i_l = tl.where(itm, tl.where(ic, m_l, -m_l), zero)
    tv_h, tv_l = _dd_sub_tl(p, zero, i_h, i_l)
    u_h = tl.where(ic, ds_h, dk_h)
    u_l = tl.where(ic, ds_l, dk_l)
    g_h, g_l = _dd_sub_tl(u_h, u_l, p, zero)
    return itm, m_h, tv_h, tv_l, g_h, g_l


@triton.jit
def iv_otm_kernel(
    price_ptr,
    s_ptr,
    k_ptr,
    t_ptr,
    r_ptr,
    q_ptr,
    is_call_ptr,
    sigma_ptr,  # output
    code_ptr,  # output: int8 row code
    n,
    MODEL: tl.constexpr,  # 0 black, 1 black_scholes, 2 black_scholes_merton
    MAIN_ITERS: tl.constexpr,
    TOTAL_ITERS: tl.constexpr,
    ACCEPT_RTOL: tl.constexpr,
    BRACKET_RTOL: tl.constexpr,
    SEED_LO: tl.constexpr,
    SEED_HI: tl.constexpr,
    TINY_LOG: tl.constexpr,
    S_MAX: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    p = tl.load(price_ptr + offs, mask=mask, other=1.0)
    s = tl.load(s_ptr + offs, mask=mask, other=1.0)
    k = tl.load(k_ptr + offs, mask=mask, other=1.0)
    t = tl.load(t_ptr + offs, mask=mask, other=1.0)
    r = tl.load(r_ptr + offs, mask=mask, other=0.0)
    q = tl.load(q_ptr + offs, mask=mask, other=0.0)
    ic = tl.load(is_call_ptr + offs, mask=mask, other=0) != 0
    zero = p * 0.0
    inf = zero + float("inf")

    # --- discounted forward / strike in double-double, log-moneyness -------
    er = libdevice.expm1(-r * t)
    dk_h, dk_l = _scaled_tl(k, er)
    rt = tl.abs(r * t)
    err = _ERR_FACTOR * k * (rt + tl.abs(er))
    if MODEL == 0:
        ds_h, ds_l = _scaled_tl(s, er)
        err = err + _ERR_FACTOR * s * (rt + tl.abs(er))
        x = tl.log(s / k)
    elif MODEL == 1:
        ds_h = s
        ds_l = zero
        x = tl.log(s / k) + r * t
    else:
        eq = libdevice.expm1(-q * t)
        ds_h, ds_l = _scaled_tl(s, eq)
        err = err + _ERR_FACTOR * s * (tl.abs(q * t) + tl.abs(eq))
        x = tl.log(s / k) + (r - q) * t

    finite = (tl.abs(ds_h) < inf) & (tl.abs(dk_h) < inf) & (tl.abs(p) < inf) & (tl.abs(t) < inf)
    invalid = (finite == 0) | (s <= 0.0) | (k <= 0.0) | (t < 0.0) | (p < 0.0)
    expiry = (invalid == 0) & (t == 0.0)
    live = (invalid == 0) & (expiry == 0)

    itm, m_h, tv_h, tv_l, g_h, g_l = _classify_tl(p, ic, ds_h, ds_l, dk_h, dk_l)
    tv = tv_h + tv_l
    gap = g_h + g_l
    # Rows within the error bound of the double-precision bounds: classify again with
    # double-double bounds (only in blocks that contain such a row).
    near = live & mask & (err > 0.0)
    near = near & ((tl.abs(tv) <= err) | (tl.abs(gap) <= err) | (tl.abs(m_h) <= err))
    if tl.sum(near.to(tl.int32), axis=0) > 0:
        er_h, er_l = _expm1_neg_dd_tl(r, t)
        pk_h, pk_l = _scaled_dd_tl(k, er_h, er_l)
        if MODEL == 0:
            ps_h, ps_l = _scaled_dd_tl(s, er_h, er_l)
        elif MODEL == 1:
            ps_h = s
            ps_l = zero
        else:
            eq_h, eq_l = _expm1_neg_dd_tl(q, t)
            ps_h, ps_l = _scaled_dd_tl(s, eq_h, eq_l)
        ds_h = tl.where(near, ps_h, ds_h)
        ds_l = tl.where(near, ps_l, ds_l)
        dk_h = tl.where(near, pk_h, dk_h)
        dk_l = tl.where(near, pk_l, dk_l)
        itm, m_h, tv_h, tv_l, g_h, g_l = _classify_tl(p, ic, ds_h, ds_l, dk_h, dk_l)
        tv = tv_h + tv_l
        gap = g_h + g_l
    below_lb = (tv_h < 0.0) | ((tv_h == 0.0) & (tv_l < 0.0))
    above_ub = (g_h < 0.0) | ((g_h == 0.0) & (g_l <= 0.0))

    zero_price = live & (p == 0.0) & (itm == 0)
    below = live & (zero_price == 0) & below_lb
    at_lb = live & (zero_price == 0) & (below_lb == 0) & (tv == 0.0)
    above = live & (zero_price == 0) & (below == 0) & above_ub
    interior = live & (zero_price == 0) & (below == 0) & (at_lb == 0) & (above == 0) & mask

    code = tl.where(invalid, 4, 0)
    code = tl.where(zero_price, 2, code)
    code = tl.where(below, 1, code)
    code = tl.where(above, 3, code)

    # --- solve the out-of-the-money problem for s = sigma sqrt(T) ----------
    one = zero + 1.0
    fo = tl.where(interior, tl.minimum(ds_h, dk_h), one)
    ko = tl.where(interior, tl.maximum(ds_h, dk_h), one)
    tv_s = tl.where(interior, tv, one)
    gap_s = tl.where(interior, gap, one)
    a = tl.where(interior, -tl.abs(x), zero)
    sqrt_t = tl.sqrt(tl.where(interior, t, one))
    upper = gap_s < tv_s
    l_target = tl.log(tl.where(upper, gap_s, tv_s))
    l_fo = tl.log(fo)
    l_ko = tl.log(ko)
    tiny = (l_target - l_fo) < TINY_LOG
    sgn = tl.where(upper, -1.0, 1.0)
    sv = tl.minimum(tl.maximum(tv_s / (fo * sqrt_t) * _SQRT2PI_C, SEED_LO), SEED_HI) * sqrt_t
    lo = zero
    hi = inf
    active = interior
    conv = interior & (interior == 0)
    it = 0
    n_act = tl.sum(active.to(tl.int32), axis=0)
    while n_act > 0:
        d1 = a / sv + 0.5 * sv
        d2 = d1 - sv
        n1 = _ndtr_c_tl(tl.where(upper, -d1, d1))
        n2 = _ndtr_c_tl(d2)
        w = tl.where(upper, fo * n1 + ko * n2, fo * n1 - ko * n2)
        lw = tl.where(w > 0.0, tl.log(tl.where(w > 0.0, w, one)), -inf)
        if tl.sum((tiny & active).to(tl.int32), axis=0) > 0:
            la = l_fo + _log_ndtr_tl(tl.where(upper, -d1, d1))
            lb = l_ko + _log_ndtr_tl(d2)
            mx = tl.maximum(la, lb)
            lw_up = mx + libdevice.log1p(tl.exp(-tl.abs(la - lb)))
            lw_lo = tl.where(lb < la, la + libdevice.log1p(-tl.exp(lb - la)), -inf)
            lw = tl.where(tiny, tl.where(upper, lw_up, lw_lo), lw)
        dlw = tl.exp(l_fo - 0.5 * d1 * d1 - _LOG_SQRT2PI_C - lw)
        g = sgn * (lw - l_target)
        is_below = g < 0.0
        lo_n = tl.where(is_below, sv, lo)
        hi_n = tl.where(is_below, hi, sv)
        delta = g / dlw
        den = 1.0 - 0.5 * delta * (d1 * d2 / sv - sgn * dlw)
        den = tl.where(tl.abs(den) < 0.5, tl.where(den < 0.0, -0.5, 0.5), den)
        step = delta / den
        cand = sv - step
        small = tl.abs(step) <= ACCEPT_RTOL * sv
        cand_ok = (cand == cand) & (tl.abs(cand) < inf)
        ok = cand_ok & (((cand > lo_n) & (cand < hi_n)) | small)
        fb = tl.where(hi_n == inf, 4.0 * sv, tl.where(lo_n > 0.0, tl.sqrt(lo_n * hi_n), 0.25 * sv))
        sv_n = tl.minimum(tl.where(ok, cand, fb), S_MAX)
        bounded = (hi_n < inf) & (lo_n > 0.0)
        tight = (hi_n <= lo_n * (1.0 + BRACKET_RTOL)) | (libdevice.nextafter(lo_n, inf) >= hi_n)
        done = (small & cand_ok) | (bounded & tight & (it >= MAIN_ITERS - 1))
        sv = tl.where(active, sv_n, sv)
        lo = tl.where(active, lo_n, lo)
        hi = tl.where(active, hi_n, hi)
        conv = conv | (active & done)
        active = active & (done == 0)
        it += 1
        n_act = tl.where(it < TOTAL_ITERS, tl.sum(active.to(tl.int32), axis=0), 0)

    sigma = tl.where(interior & conv, sv / sqrt_t, float("nan"))
    sigma = tl.where((expiry | at_lb) & mask, 0.0, sigma)
    code = tl.where(interior & (conv == 0), 5, code)
    tl.store(sigma_ptr + offs, sigma, mask=mask)
    tl.store(code_ptr + offs, code.to(tl.int8), mask=mask)


# ---------------------------------------------------------------------------
# Python wrappers
# ---------------------------------------------------------------------------

# Auto-tune BLOCK_SIZE for the H100: 128 / 256 / 512 for float64 element-wise kernels
_CONFIGS = [
    triton.Config({"BLOCK_SIZE": bs}, num_warps=nw) for bs in (128, 256, 512) for nw in (4, 8)
]


def _grid(n, bs):
    return ((n + bs - 1) // bs,)


# Lazy-compiled autotune wrappers (compiled on first call, cached thereafter)
_compiled_price: "object | None" = None
_compiled_greeks: "dict[str, object]" = {}


def _get_compiled_price():
    global _compiled_price
    if _compiled_price is None:
        _compiled_price = triton.autotune(configs=_CONFIGS, key=["n"])(bsm_price_kernel)
    return _compiled_price


def _get_compiled_greeks(model: str):
    if model not in _compiled_greeks:
        is_black = model == "black"
        kernel = triton.autotune(configs=_CONFIGS, key=["n"])(
            lambda *a, **kw: bsm_greeks_kernel(*a, is_black=is_black, **kw)
        )
        _compiled_greeks[model] = kernel
    return _compiled_greeks[model]


# --------------- public wrappers ------------------------------------------


def bsm_price_triton(
    is_call: torch.Tensor,  # bool GPU tensor
    s: torch.Tensor,
    k: torch.Tensor,
    t: torch.Tensor,
    r: torch.Tensor,
    sigma: torch.Tensor,
    q: torch.Tensor,
) -> torch.Tensor:
    n = s.numel()
    out = torch.empty(n, dtype=torch.float64, device=s.device)
    # is_call stored as int8 for Triton compatibility
    is_call_i8 = is_call.to(torch.int8)
    bs = 256  # use fixed size; autotune would add first-call overhead
    _grid_fn = _grid(n, bs)
    bsm_price_kernel[_grid_fn](
        s,
        k,
        t,
        r,
        sigma,
        q,
        is_call_i8,
        out,
        n,
        BLOCK_SIZE=bs,
    )
    return out


def bsm_greeks_triton(
    model: str,
    is_call: torch.Tensor,
    s: torch.Tensor,
    k: torch.Tensor,
    t: torch.Tensor,
    r: torch.Tensor,
    sigma: torch.Tensor,
    q: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n = s.numel()
    delta = torch.empty(n, dtype=torch.float64, device=s.device)
    gamma = torch.empty(n, dtype=torch.float64, device=s.device)
    theta = torch.empty(n, dtype=torch.float64, device=s.device)
    rho = torch.empty(n, dtype=torch.float64, device=s.device)
    vega = torch.empty(n, dtype=torch.float64, device=s.device)
    is_call_i8 = is_call.to(torch.int8)
    is_black = model == "black"
    bs = 256
    bsm_greeks_kernel[_grid(n, bs)](
        s,
        k,
        t,
        r,
        sigma,
        q,
        is_call_i8,
        delta,
        gamma,
        theta,
        rho,
        vega,
        n,
        is_black=is_black,
        BLOCK_SIZE=bs,
    )
    return delta, gamma, theta, rho, vega


def iv_otm_triton(
    model: str,
    price: torch.Tensor,
    s: torch.Tensor,
    k: torch.Tensor,
    t: torch.Tensor,
    r: torch.Tensor,
    q: torch.Tensor,
    is_call: torch.Tensor,  # bool GPU tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Classify and solve every row in one kernel. Returns (sigma, int8 row code)."""
    n = s.numel()
    sigma = torch.empty(n, dtype=torch.float64, device=s.device)
    code = torch.empty(n, dtype=torch.int8, device=s.device)
    bs = 128
    iv_otm_kernel[_grid(n, bs)](
        price,
        s,
        k,
        t,
        r,
        q,
        is_call.to(torch.int8),
        sigma,
        code,
        n,
        MODEL={"black": 0, "black_scholes": 1}.get(model, 2),
        MAIN_ITERS=_ivc.MAIN_ITERS,
        TOTAL_ITERS=_ivc.MAIN_ITERS + _ivc.MAX_ITERS,
        ACCEPT_RTOL=_ivc.ACCEPT_RTOL,
        BRACKET_RTOL=_ivc.BRACKET_RTOL,
        SEED_LO=_ivc.SEED_SIGMA_LO,
        SEED_HI=_ivc.SEED_SIGMA_HI,
        TINY_LOG=_ivc.TINY_LOG,
        S_MAX=_ivc.S_MAX,
        BLOCK_SIZE=bs,
        # The error-free transformations (two-sum after a product, double-double bounds) need
        # every product rounded on its own: LLVM must not contract a*b + c into an FMA.
        enable_fp_fusion=False,
    )
    return sigma, code
