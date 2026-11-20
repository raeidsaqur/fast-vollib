# Author: Raeid Saqur
# Date: 2026-11-19
"""Shared implied-volatility policy: solver constants, row classification and error reporting.

Every Halley backend (NumPy, Numba, PyTorch, Triton, JAX) solves the same problem the same way;
the execution-specific kernels stay in their own modules and take their constants from here.

Problem statement
-----------------
All three models are reduced to discounted Black-76 with

* discounted forward  ``ds = S e^{-qT}`` (Black-76: ``F e^{-rT}``; Black-Scholes: ``S``),
* discounted strike   ``dk = K e^{-rT}``,
* log-moneyness       ``x = ln(ds / dk)``.

By put-call parity the time value ``tv = price - intrinsic`` is the price of the out-of-the-money
option, an out-of-the-money call on ``Fo = min(ds, dk)`` struck at ``Ko = max(ds, dk)``::

    c(s) = Fo N(-|x|/s + s/2) - Ko N(-|x|/s - s/2),     0 < c < Fo,   s = sigma sqrt(T).

The solver works on ``ln c(s) = ln tv``, or on the complementary ``ln u(s) = ln(Fo - tv)`` with
``u = Fo N(|x|/s - s/2) + Ko N(-|x|/s - s/2)`` (a sum without cancellation) when the quote is
closer to the upper bound than to the lower one. Both are monotone in ``s``.

Classification (the row-level policy)
-------------------------------------
``ds`` and ``dk`` are formed in double-double precision (``expm1`` plus an exact product), so the
lower and upper bounds are compared with the quote without the rounding of a single product.

======================  ===========================================  =========================
row                     condition                                    result
======================  ===========================================  =========================
invalid input           S <= 0, K <= 0, T < 0, price < 0             NaN + signal
expiry                  T == 0                                       0.0 (documented)
zero price              price == 0 and no intrinsic value            NaN + signal
below intrinsic         price < intrinsic                            NaN + signal
at the lower bound      price == intrinsic (> 0)                     0.0
at/above upper bound    price >= ds (call) or dk (put)               NaN + signal
interior                otherwise                                    solved
not converged           solver did not meet its target               NaN + signal
======================  ===========================================  =========================

The signal follows ``on_error``: one aggregated warning (``"warn"``) or ``ValueError``
(``"raise"``) per call, listing the count of each row condition; nothing under ``"ignore"``.
"""

from __future__ import annotations

import math
from typing import Any

from ..utils.validation import handle_error

# ---------------------------------------------------------------------------
# Solver constants (Triton kernels receive these as tl.constexpr)
# ---------------------------------------------------------------------------

#: Halley steps applied to every interior row before the convergence test.
MAIN_ITERS = 6
#: Further safeguarded steps allowed for rows that have not converged (bracket phase).
MAX_ITERS = 100
#: A row is converged when the last Halley step is at most this fraction of s.
ACCEPT_RTOL = 1e-10
#: ... or when its sign-checked bracket satisfies hi <= lo * (1 + BRACKET_RTOL).
BRACKET_RTOL = 2.0**-40
#: Seed: v0.2.4's Brenner-Subrahmanyam-type guess in sigma units, clipped to this range.
SEED_SIGMA_LO = 0.30
SEED_SIGMA_HI = 5.0
#: Rows with ln(target / Fo) below this are evaluated with log_ndtr (deep tail, sub-normal prices).
TINY_LOG = -650.0
#: Upper limit on the total volatility s = sigma sqrt(T) while expanding the bracket.
S_MAX = 1e3

SQRT2PI = math.sqrt(2.0 * math.pi)
LOG_SQRT2PI = math.log(SQRT2PI)

# ---------------------------------------------------------------------------
# Row codes
# ---------------------------------------------------------------------------

OK = 0
BELOW_INTRINSIC = 1
ZERO_PRICE = 2
ABOVE_UPPER = 3
INVALID_INPUT = 4
NOT_CONVERGED = 5
#: Jäckel solver only: normalised out-of-the-money price below the smallest normal double.
OUTSIDE_DOMAIN = 6
N_CODES = 7

_MESSAGES = {
    BELOW_INTRINSIC: "option price below intrinsic value",
    ZERO_PRICE: "zero option price (volatility undetermined)",
    ABOVE_UPPER: "option price at or above its upper bound",
    INVALID_INPUT: "invalid input (non-positive underlying or strike, negative time or price)",
    NOT_CONVERGED: 'solver did not converge (with solver="jackel", try solver="halley")',
    OUTSIDE_DOMAIN: (
        "normalised out-of-the-money price below the smallest normal double "
        '(outside the domain of the Jäckel solver; try solver="halley")'
    ),
}

#: Jäckel's normalised price beta = tv / sqrt(D F D K) must be at least this (DBL_MIN): below it
#: beta itself is sub-normal and the solver's lower-branch objective loses its precision.
JAECKEL_BETA_MIN = 2.2250738585072014e-308


def report_row_errors(counts: dict[int, int], n_rows: int, on_error: str) -> None:
    """Emit one aggregated signal for all NaN rows of a call (no-op when there are none)."""
    parts = [f"{counts[c]} {_MESSAGES[c]}" for c in sorted(_MESSAGES) if counts.get(c, 0)]
    if not parts:
        return
    total = sum(counts.get(c, 0) for c in _MESSAGES)
    handle_error(
        f"Implied volatility is undefined for {total} of {n_rows} rows; NaN returned "
        f"({'; '.join(parts)}).",
        on_error,
    )


def count_codes(codes: Any) -> dict[int, int]:
    """Count the non-OK codes of a NumPy integer array (cheap when most rows are OK)."""
    import numpy as np

    flat = np.asarray(codes).reshape(-1)
    flagged = flat[np.flatnonzero(flat)]
    if flagged.size == 0:
        return {}
    freq = np.bincount(flagged.astype(np.intp), minlength=N_CODES)
    return {c: int(n) for c, n in enumerate(freq) if c != OK and n}


def flatten_inputs(price: Any, s: Any, k: Any, t: Any, r: Any, flag: Any, q: Any = None) -> tuple:
    """Broadcast the IV inputs to one shape and return (shape, flat float64 arrays..., flat flag, q).

    The solvers work on 1-D arrays; the caller reshapes the result to ``shape``, so 0-d and
    N-d inputs keep the elementwise contract of v0.2.4.
    """
    import numpy as np

    arrays = [np.asarray(v) for v in (price, s, k, t, r, flag)]
    if q is not None:
        arrays.append(np.asarray(q))
    shape = np.broadcast_shapes(*(a.shape for a in arrays))
    flat = [np.ascontiguousarray(np.broadcast_to(a, shape)).reshape(-1) for a in arrays]
    nums = [a.astype(np.float64, copy=False) for a in flat[:5]]
    q_flat = flat[6].astype(np.float64, copy=False) if q is not None else None
    return (shape, *nums, flat[5], q_flat)


# ---------------------------------------------------------------------------
# Error-free transformations (pure arithmetic: NumPy, PyTorch and JAX arrays alike)
# ---------------------------------------------------------------------------

_SPLIT = 134217729.0  # 2**27 + 1


def two_sum(a: Any, b: Any) -> tuple[Any, Any]:
    """a + b = s + e exactly (Knuth)."""
    s = a + b
    bb = s - a
    e = (a - (s - bb)) + (b - bb)
    return s, e


def two_prod(a: Any, b: Any) -> tuple[Any, Any]:
    """a * b = p + e exactly (Dekker/Veltkamp; no fused multiply-add needed)."""
    p = a * b
    t = _SPLIT * a
    ah = t - (t - a)
    al = a - ah
    t = _SPLIT * b
    bh = t - (t - b)
    bl = b - bh
    e = ((ah * bh - p) + ah * bl + al * bh) + al * bl
    return p, e


def fast_two_sum(a: Any, b: Any) -> tuple[Any, Any]:
    """a + b = s + e exactly when |a| >= |b|."""
    s = a + b
    return s, b - (s - a)


def dd_add(ah: Any, al: Any, bh: Any, bl: Any) -> tuple[Any, Any]:
    s, e = two_sum(ah, bh)
    return fast_two_sum(s, e + (al + bl))


def dd_mul(ah: Any, al: Any, bh: Any, bl: Any) -> tuple[Any, Any]:
    p, e = two_prod(ah, bh)
    return fast_two_sum(p, e + (ah * bl + al * bh))


def dd_sub(ah: Any, al: Any, bh: Any, bl: Any) -> tuple[Any, Any]:
    """(ah + al) - (bh + bl) as hi + lo."""
    s, e = two_sum(ah, -bh)
    return two_sum(s, e + (al - bl))


def scaled(x: Any, em1: Any) -> tuple[Any, Any]:
    """x * (1 + em1) as hi + lo, with em1 = expm1(.) (exact apart from em1's own rounding)."""
    p, e = two_prod(x, em1)
    hi, lo = two_sum(x, p)
    return hi, lo + e


def scaled_dd(x: Any, em_h: Any, em_l: Any) -> tuple[Any, Any]:
    """x * (1 + em_h + em_l) as hi + lo, for a double-double em = expm1(.)."""
    p, e = two_prod(x, em_h)
    hi, lo = two_sum(x, p)
    return fast_two_sum(hi, lo + (e + x * em_l))


# Double-double expm1 -------------------------------------------------------
# ln 2 = LN2_A + LN2_B + LN2_C (|residual| < 6e-43); LN2_A has 32 significant bits, so k * LN2_A is
# exact for |k| < 2**21. Inverse factorials 1/j! as hi + lo.
LN2_A = float.fromhex("0x1.62e42fee00000p-1")
LN2_B = float.fromhex("0x1.a39ef35793c76p-33")
LN2_C = float.fromhex("0x1.cc01f97b57a08p-87")
INV_FACT = (
    (float.fromhex("0x1.0000000000000p+0"), 0.0),
    (float.fromhex("0x1.0000000000000p-1"), 0.0),
    (float.fromhex("0x1.5555555555555p-3"), float.fromhex("0x1.5555555555555p-57")),
    (float.fromhex("0x1.5555555555555p-5"), float.fromhex("0x1.5555555555555p-59")),
    (float.fromhex("0x1.1111111111111p-7"), float.fromhex("0x1.1111111111111p-63")),
    (float.fromhex("0x1.6c16c16c16c17p-10"), float.fromhex("-0x1.f49f49f49f49fp-65")),
    (float.fromhex("0x1.a01a01a01a01ap-13"), float.fromhex("0x1.a01a01a01a01ap-73")),
    (float.fromhex("0x1.a01a01a01a01ap-16"), float.fromhex("0x1.a01a01a01a01ap-76")),
    (float.fromhex("0x1.71de3a556c734p-19"), float.fromhex("-0x1.c154f8ddc6c00p-73")),
)
#: z = y - k ln 2 is scaled by 2**-EXPM1_SQUARINGS before the series and squared back.
EXPM1_SQUARINGS = 10
#: The double-double path is used for |y| <= this (|k| <= 1010, no overflow of 2**k).
EXPM1_DD_MAX = 700.0


def _round(xp: Any, v: Any) -> Any:
    return xp.round(v)


def _pow2(xp: Any, k: Any) -> Any:
    """2**k for an integer-valued float64 array k in [-1022, 1023], built from its IEEE bits.

    Exact on every backend (``ldexp``/``pow`` are not exact for all k on some GPU libraries).
    """
    name = getattr(xp, "__name__", "")
    if name == "numpy":
        return ((k.astype(xp.int64) + 1023) << 52).view(xp.float64)
    if name == "torch":
        return ((k.to(xp.int64) + 1023) << 52).view(xp.float64)
    import jax

    bits = (k.astype(xp.int64) + 1023) << 52
    return jax.lax.bitcast_convert_type(bits, xp.float64)


def dd_expm1(xp: Any, y_h: Any, y_l: Any) -> tuple[Any, Any]:
    """expm1(y_h + y_l) in double-double precision (relative error ~1e-29), for |y| <= 700.

    Range reduction y = k ln 2 + z with ln 2 in three parts, |z| <= ln 2 / 2; the series for
    expm1(z / 2**EXPM1_SQUARINGS) to the 9th power; EXPM1_SQUARINGS doublings
    em <- em (2 + em); finally expm1(y) = 2**k em + (2**k - 1). Arithmetic only, so the same code
    runs on NumPy, PyTorch and JAX arrays.
    """
    k = _round(xp, y_h / LN2_A)
    t1 = y_h - k * LN2_A  # exact: k LN2_A is exact and within a factor 2 of y_h
    p, pe = two_prod(k, LN2_B)
    z_h, z_l = two_sum(t1, -p)
    z_h, z_l = fast_two_sum(z_h, z_l + ((y_l - pe) - k * LN2_C))
    sc = 2.0**-EXPM1_SQUARINGS
    w_h, w_l = z_h * sc, z_l * sc
    a_h, a_l = INV_FACT[-1][0] + 0.0 * w_h, INV_FACT[-1][1] + 0.0 * w_h
    for c_h, c_l in INV_FACT[-2::-1]:
        a_h, a_l = dd_mul(a_h, a_l, w_h, w_l)
        a_h, a_l = dd_add(a_h, a_l, c_h + 0.0 * w_h, c_l + 0.0 * w_h)
    e_h, e_l = dd_mul(a_h, a_l, w_h, w_l)
    for _ in range(EXPM1_SQUARINGS):
        sq_h, sq_l = dd_mul(e_h, e_l, e_h, e_l)
        e_h, e_l = dd_add(2.0 * e_h, 2.0 * e_l, sq_h, sq_l)
    two_k = _pow2(xp, k)
    c_h, c_l = two_sum(two_k, -1.0 + 0.0 * k)  # 2**k - 1 exactly (not representable for k > 53)
    s_h, s_l = e_h * two_k, e_l * two_k
    return dd_add(s_h, s_l, c_h, c_l)


# ---------------------------------------------------------------------------
# Backend-generic bounds and classification
# ---------------------------------------------------------------------------

#: Relative factor of the error bound of the double-precision bounds (see discounted_bounds).
BOUND_ERR_FACTOR = 2.0**-44


def discounted_bounds(
    xp: Any, model: str, s: Any, k: Any, t: Any, r: Any, q: Any, precise: bool = False
) -> tuple[Any, Any, Any, Any, Any, Any]:
    """Discounted forward ds = D F and strike dk = D K as hi + lo, log-moneyness x, error bound.

    ``precise=False``: D - 1 = expm1(-r t) in double precision. The products and sums around it are
    exact, so the error comes from rounding r t (half an ulp) and from expm1 (at most a few ulps
    in libm, CUDA libdevice and XLA). The returned bound
    ``err = BOUND_ERR_FACTOR * (K (|r t| + |expm1(-r t)|) + S-term)`` is at least ~256 times
    those errors; it is zero when r t = 0 (and q t = 0), where the bounds are exact.
    ``precise=True``: D - 1 in double-double (dd_expm1, from the exact product r t); err = 0.
    """
    zero = xp.zeros_like(s)
    rt_h, rt_l = two_prod(r, t)
    if precise:
        lim = xp.abs(rt_h) <= EXPM1_DD_MAX
        er_h, er_l = dd_expm1(xp, xp.where(lim, -rt_h, zero), xp.where(lim, -rt_l, zero))
        er_h = xp.where(lim, er_h, xp.expm1(-rt_h))
        er_l = xp.where(lim, er_l, zero)
        dk_h, dk_l = scaled_dd(k, er_h, er_l)
    else:
        er_h = xp.expm1(-rt_h)
        dk_h, dk_l = scaled(k, er_h)
    err = BOUND_ERR_FACTOR * k * (xp.abs(rt_h) + xp.abs(er_h))
    if model == "black":
        ds_h, ds_l = scaled_dd(s, er_h, er_l) if precise else scaled(s, er_h)
        err = err + BOUND_ERR_FACTOR * s * (xp.abs(rt_h) + xp.abs(er_h))
        x = xp.log(s / k)
    elif model == "black_scholes":
        ds_h, ds_l = s, zero
        x = xp.log(s / k) + r * t
    else:
        qt_h, qt_l = two_prod(q, t)
        if precise:
            lim = xp.abs(qt_h) <= EXPM1_DD_MAX
            eq_h, eq_l = dd_expm1(xp, xp.where(lim, -qt_h, zero), xp.where(lim, -qt_l, zero))
            eq_h = xp.where(lim, eq_h, xp.expm1(-qt_h))
            eq_l = xp.where(lim, eq_l, zero)
            ds_h, ds_l = scaled_dd(s, eq_h, eq_l)
        else:
            eq_h = xp.expm1(-qt_h)
            ds_h, ds_l = scaled(s, eq_h)
        err = err + BOUND_ERR_FACTOR * s * (xp.abs(qt_h) + xp.abs(eq_h))
        x = xp.log(s / k) + (r - q) * t
    if precise:
        err = zero
    return ds_h, ds_l, dk_h, dk_l, x, err


def classify(
    xp: Any,
    model: str,
    price: Any,
    s: Any,
    k: Any,
    t: Any,
    r: Any,
    q: Any,
    is_call: Any,
    precise: bool = False,
) -> dict:
    """Classify every row and build the out-of-the-money problem for the interior rows.

    ``xp`` is the array namespace (``numpy``, ``torch`` or ``jax.numpy``); all arrays share one
    shape and dtype. Returns a dict with ``code`` (row code, OK for interior and zero-vol rows),
    ``interior`` and ``zero_vol`` masks, the solver inputs ``tv``, ``gap``, ``Fo``, ``Ko``,
    ``a = -|x|`` (meaningful on interior rows only), and ``near``: rows whose time value, distance
    to the upper bound or moneyness is within the error bound of double-precision bounds. Those
    rows must be classified again with ``precise=True``.
    """
    zero = xp.zeros_like(price)
    ds_h, ds_l, dk_h, dk_l, x, err = discounted_bounds(xp, model, s, k, t, r, q, precise)

    finite = xp.isfinite(ds_h) & xp.isfinite(dk_h) & xp.isfinite(price) & xp.isfinite(t)
    invalid = ~finite | (s <= 0) | (k <= 0) | (t < 0) | (price < 0)
    expiry = ~invalid & (t == 0)
    live = ~invalid & ~expiry

    # moneyness sign and intrinsic value, in double-double
    m_h, m_l = dd_sub(ds_h, ds_l, dk_h, dk_l)  # ds - dk
    call_itm = (m_h > 0) | ((m_h == 0) & (m_l > 0))
    put_itm = (m_h < 0) | ((m_h == 0) & (m_l < 0))
    itm = xp.where(is_call, call_itm, put_itm)
    i_h = xp.where(itm, xp.where(is_call, m_h, -m_h), zero)
    i_l = xp.where(itm, xp.where(is_call, m_l, -m_l), zero)
    tv_h, tv_l = dd_sub(price, zero, i_h, i_l)
    tv = tv_h + tv_l
    below_lb = (tv_h < 0) | ((tv_h == 0) & (tv_l < 0))
    # distance to the upper bound (ds for a call, dk for a put)
    u_h = xp.where(is_call, ds_h, dk_h)
    u_l = xp.where(is_call, ds_l, dk_l)
    g_h, g_l = dd_sub(u_h, u_l, price, zero)
    gap = g_h + g_l
    at_or_above_ub = (g_h < 0) | ((g_h == 0) & (g_l <= 0))

    zero_price = live & (price == 0) & ~itm
    below = live & ~zero_price & below_lb
    at_lb = live & ~zero_price & ~below_lb & (tv == 0)
    above = live & ~zero_price & ~below & at_or_above_ub
    interior = live & ~zero_price & ~below & ~at_lb & ~above
    near = live & (err > 0) & ((xp.abs(tv) <= err) | (xp.abs(gap) <= err) | (xp.abs(m_h) <= err))

    code = xp.where(invalid, INVALID_INPUT, OK)
    code = xp.where(zero_price, ZERO_PRICE, code)
    code = xp.where(below, BELOW_INTRINSIC, code)
    code = xp.where(above, ABOVE_UPPER, code)
    return {
        "code": code,
        "interior": interior,
        "zero_vol": expiry | at_lb,
        "tv": tv,
        "gap": gap,
        "Fo": xp.minimum(ds_h, dk_h),
        "Ko": xp.maximum(ds_h, dk_h),
        "a": -xp.abs(x),
        "near": near,
    }


def classify_np(
    model: str, price: Any, s: Any, k: Any, t: Any, r: Any, q: Any, is_call: Any
) -> dict:
    """NumPy classification with the double-double refinement of near-bound rows."""
    import numpy as np

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        rows = classify(np, model, price, s, k, t, r, q, is_call)
        near = np.flatnonzero(rows["near"])
        if near.size:  # within the error bound of the double-precision bounds
            sub = classify(
                np,
                model,
                price[near],
                s[near],
                k[near],
                t[near],
                r[near],
                q[near],
                is_call[near],
                precise=True,
            )
            for key, value in sub.items():
                rows[key][near] = value
    rows["code"] = np.asarray(rows["code"], dtype=np.int8)
    return rows
