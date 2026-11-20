# Author: Raeid Saqur
# Date: 2026-11-19
"""Jäckel's rational-cubic initial guess for Zones 1 and 4, backend-neutral.

The host solver (``jackel_iv._jackel_initial_guess``) seeds the deep lower zone (beta < b_l) and
the upper zone (beta > b_h) from the lower/upper maps and their rational-cubic interpolation.
The native torch and JAX ports seeded every zone with the cubic Hermite guess in (b, sigma),
which is clamped to s_l in Zone 1 and to s_h in Zone 4; there the two or three Householder steps
do not reach the root (e.g. F=100, K=100 e^-0.1, sigma=0.0182: 0.0182183 instead of 0.0182000).

This module evaluates the same formulas as the host code on every row (masks, no boolean
indexing), so it runs inside torch and JAX programs. ``ops`` supplies the few functions whose
spelling differs between array libraries.
"""

from __future__ import annotations

import sys
from typing import Any

_DBL_EPSILON = sys.float_info.epsilon
_DBL_MIN = sys.float_info.min
_DBL_MAX = sys.float_info.max
_SQRT_DBL_EPSILON = _DBL_EPSILON**0.5
_RC_MIN = -(1.0 - _SQRT_DBL_EPSILON)
_RC_MAX = 2.0 / (_DBL_EPSILON * _DBL_EPSILON)
_ONE_OVER_SQRT2PI = 0.39894228040143267
_SQRTPI_OVER2 = 1.2533141373155003
_SQRT3 = 1.7320508075688772
_SQRT_ONE_OVER3 = 0.5773502691896258
_TWO_PI_OVER_SQRT27 = 1.2091995761561453
_PI_OVER6 = 0.5235987755982989
_TWO_PI = 6.283185307179586476925286766559
_TINY = sys.float_info.min


class TorchOps:
    def __init__(self) -> None:
        import torch

        self.m = torch

    def where(self, c: Any, a: Any, b: Any) -> Any:
        ref = a if isinstance(a, self.m.Tensor) else b
        a = a if isinstance(a, self.m.Tensor) else self.m.full_like(ref, float(a))
        b = b if isinstance(b, self.m.Tensor) else self.m.full_like(ref, float(b))
        return self.m.where(c, a, b)

    def maximum(self, a: Any, b: Any) -> Any:
        if not isinstance(b, self.m.Tensor):
            return self.m.clamp(a, min=float(b))
        if not isinstance(a, self.m.Tensor):
            return self.m.clamp(b, min=float(a))
        return self.m.maximum(a, b)

    def clip(self, a: Any, lo: float, hi: float) -> Any:
        return self.m.clamp(a, lo, hi)

    def abs(self, a: Any) -> Any:
        return self.m.abs(a)

    def exp(self, a: Any) -> Any:
        return self.m.exp(a)

    def cbrt_nonneg(self, a: Any) -> Any:
        return self.m.pow(a, 1.0 / 3.0)

    def ndtr(self, a: Any) -> Any:
        return self.m.special.ndtr(a)

    def ndtri(self, a: Any) -> Any:
        return self.m.special.ndtri(a)

    def full_like(self, a: Any, v: float) -> Any:
        return self.m.full_like(a, v)


class JaxOps:
    def __init__(self) -> None:
        import jax.numpy as jnp
        import jax.scipy.special as jss

        self.m = jnp
        self.s = jss

    def where(self, c: Any, a: Any, b: Any) -> Any:
        return self.m.where(c, a, b)

    def maximum(self, a: Any, b: Any) -> Any:
        return self.m.maximum(a, b)

    def clip(self, a: Any, lo: float, hi: float) -> Any:
        return self.m.clip(a, lo, hi)

    def abs(self, a: Any) -> Any:
        return self.m.abs(a)

    def exp(self, a: Any) -> Any:
        return self.m.exp(a)

    def cbrt_nonneg(self, a: Any) -> Any:
        return self.m.cbrt(a)

    def ndtr(self, a: Any) -> Any:
        return self.s.ndtr(a)

    def ndtri(self, a: Any) -> Any:
        return self.s.ndtri(a)

    def full_like(self, a: Any, v: float) -> Any:
        return self.m.full_like(a, v)


def _safe(ops: Any, v: Any, cond: Any) -> Any:
    return ops.where(cond, v, ops.full_like(v, _TINY))


def _rc_min_control(ops: Any, d_l: Any, d_r: Any, slope: Any, prefer_shape: bool) -> Any:
    monotonic = (d_l * slope >= 0.0) & (d_r * slope >= 0.0)
    convex = (d_l <= slope) & (slope <= d_r)
    concave = (d_l >= slope) & (slope >= d_r)
    none = ~monotonic & ~convex & ~concave
    neg = ops.full_like(slope, -_DBL_MAX)
    slope_safe = _safe(ops, slope, ops.abs(slope) > 0.0)
    r1 = ops.where(monotonic & (ops.abs(slope) > 0.0), (d_r + d_l) / slope_safe, neg)
    if prefer_shape:
        r1 = ops.where(monotonic & (ops.abs(slope) <= 0.0), ops.full_like(slope, _RC_MAX), r1)
    drmd = d_r - d_l
    drms = d_r - slope
    smdi = slope - d_l
    denom_ok = (ops.abs(smdi) > 0.0) & (ops.abs(drms) > 0.0)
    r2_val = ops.maximum(
        ops.abs(drmd / _safe(ops, drms, drms != 0.0)), ops.abs(drmd / _safe(ops, smdi, smdi != 0.0))
    )
    r2 = ops.where((convex | concave) & denom_ok, r2_val, neg)
    if prefer_shape:
        big = ops.full_like(slope, _RC_MAX)
        r2 = ops.where((convex | concave) & ~denom_ok, big, r2)
        r2 = ops.where(monotonic & ~(convex | concave), big, r2)
    result = ops.maximum(ops.maximum(r1, r2), _RC_MIN)
    return ops.where(none, ops.full_like(slope, _RC_MIN), result)


def _rc_fit(ops: Any, x_l, x_r, y_l, y_r, d_l, d_r, d2, prefer_shape: bool, right: bool) -> Any:
    h = x_r - x_l
    h_safe = _safe(ops, h, h != 0.0)
    slope = (y_r - y_l) / h_safe
    num = 0.5 * h * d2 + (d_r - d_l)
    den = (d_r - slope) if right else (slope - d_l)
    den_safe = _safe(ops, den, ops.abs(den) >= _DBL_MIN)
    r_raw = ops.where(
        ops.abs(num) < _DBL_MIN,
        ops.full_like(num, 0.0),
        ops.where(
            ops.abs(den) < _DBL_MIN,
            ops.where(num > 0.0, ops.full_like(num, _RC_MAX), ops.full_like(num, _RC_MIN)),
            num / den_safe,
        ),
    )
    return ops.maximum(r_raw, _rc_min_control(ops, d_l, d_r, slope, prefer_shape))


def _rational_cubic(ops: Any, x, x_l, x_r, y_l, y_r, d_l, d_r, r) -> Any:
    h = x_r - x_l
    h_safe = _safe(ops, h, ops.abs(h) > 0.0)
    t = (x - x_l) / h_safe
    omt = 1.0 - t
    t2, omt2 = t * t, omt * omt
    num = (
        y_r * t2 * t
        + (r * y_r - h_safe * d_r) * t2 * omt
        + (r * y_l + h_safe * d_l) * t * omt2
        + y_l * omt2 * omt
    )
    den = 1.0 + (r - 3.0) * t * omt
    rational = num / _safe(ops, den, den != 0.0)
    linear = y_r * t + y_l * omt
    return ops.where(
        ops.abs(h) <= 0.0, 0.5 * (y_l + y_r), ops.where(r >= _RC_MAX, linear, rational)
    )


def _f_lower_map(ops: Any, x: Any, s: Any) -> tuple[Any, Any, Any]:
    ax = ops.abs(x)
    safe_s = _safe(ops, s, s > 0.0)
    z = _SQRT_ONE_OVER3 * ax / safe_s
    y = z * z
    s2 = s * s
    phi_cdf = ops.ndtr(-z)
    phi = _ONE_OVER_SQRT2PI * ops.exp(-0.5 * z * z)
    safe_phi = _safe(ops, phi, phi > 0.0)
    fpp = (
        _PI_OVER6
        * y
        / (s2 * safe_s)
        * phi_cdf
        * (8 * _SQRT3 * s * ax + (3 * s2 * (s2 - 8) - 8 * x * x) * phi_cdf / safe_phi)
        * ops.exp(2 * y + 0.25 * s2)
    )
    phi2 = phi_cdf * phi_cdf
    fp_full = _TWO_PI * y * phi2 * ops.exp(y + 0.125 * s * s)
    f_full = _TWO_PI_OVER_SQRT27 * ax * phi2 * phi_cdf
    s_below = s < _DBL_MIN
    x_below = ax < _DBL_MIN
    fp = ops.where(s_below, ops.full_like(fp_full, 1.0), fp_full)
    f = ops.where(s_below | x_below, ops.full_like(f_full, 0.0), f_full)
    return f, fp, fpp


def _f_upper_map(ops: Any, x: Any, s: Any) -> tuple[Any, Any, Any]:
    ax = ops.abs(x)
    safe_s = _safe(ops, s, s > 0.0)
    f = ops.ndtr(-0.5 * s)
    w = (x / safe_s) ** 2
    fp_full = -0.5 * ops.exp(0.5 * w)
    fpp_full = _SQRTPI_OVER2 * ops.exp(w + 0.125 * s * s) * w / safe_s
    x_below = ax < _DBL_MIN
    fp = ops.where(x_below, ops.full_like(fp_full, -0.5), fp_full)
    fpp = ops.where(x_below, ops.full_like(fpp_full, 0.0), fpp_full)
    return f, fp, fpp


def _inverse_f_lower_map(ops: Any, x: Any, f: Any) -> Any:
    ax = ops.abs(x)
    ax_safe = _safe(ops, ax, ax > 0.0)
    cbrt = ops.cbrt_nonneg(ops.maximum(f / (_TWO_PI_OVER_SQRT27 * ax_safe), 0.0))
    z = ops.ndtri(ops.clip(cbrt, _DBL_MIN, 1.0 - _DBL_EPSILON))
    denom = _safe(ops, _SQRT3 * z, ops.abs(z) > 0.0)
    return ops.where(f < _DBL_MIN, ops.full_like(f, 0.0), ops.abs(x / denom))


def _inverse_f_upper_map(ops: Any, f: Any) -> Any:
    return -2.0 * ops.ndtri(ops.clip(f, _DBL_MIN, 1.0 - _DBL_EPSILON))


def zone_1_4_guess(ops: Any, beta, x, b_max, s_l, b_l, s_h, b_h) -> tuple[Any, Any]:
    """Return (guess, mask): Jäckel's Zone 1 / Zone 4 initial guess and the rows it applies to.

    Same formulas as ``jackel_iv._jackel_initial_guess`` (Zones 1 and 4), on every row.
    """
    z1 = beta < b_l
    z4 = beta > b_h
    zero = ops.full_like(beta, 0.0)
    one = ops.full_like(beta, 1.0)

    # Zone 1: lower map, rational cubic from (0, 0) to (b_l, f_l), inverse map
    f_l, df_l, d2f_l = _f_lower_map(ops, x, s_l)
    r_ll = _rc_fit(ops, zero, b_l, zero, f_l, one, df_l, d2f_l, True, right=True)
    f_z1 = _rational_cubic(ops, beta, zero, b_l, zero, f_l, one, df_l, r_ll)
    b_l_safe = _safe(ops, b_l, b_l > 0.0)
    t1 = beta / b_l_safe
    f_z1 = ops.where(f_z1 > 0.0, f_z1, (f_l * t1 + b_l * (1.0 - t1)) * t1)
    s_z1 = _inverse_f_lower_map(ops, x, f_z1)

    # Zone 4: upper map, rational cubic from (b_h, f_h) to (b_max, 0), inverse map
    f_h, df_h, d2f_h = _f_upper_map(ops, x, s_h)
    half = ops.full_like(beta, -0.5)
    r_hh = _rc_fit(ops, b_h, b_max, f_h, zero, df_h, half, d2f_h, True, right=False)
    f_z4 = _rational_cubic(ops, beta, b_h, b_max, f_h, zero, df_h, half, r_hh)
    span = b_max - b_h
    span_safe = _safe(ops, span, span > 0.0)
    t4 = (beta - b_h) / span_safe
    f_z4 = ops.where(f_z4 > 0.0, f_z4, (f_h * (1.0 - t4) + 0.5 * span_safe * t4) * (1.0 - t4))
    s_z4 = _inverse_f_upper_map(ops, f_z4)

    return ops.where(z1, s_z1, ops.where(z4, s_z4, zero)), z1 | z4
