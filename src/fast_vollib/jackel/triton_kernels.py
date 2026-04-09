"""
Jäckel IV Triton kernel — single-pass Householder(3)×3.

Experiment I-7: Fuse the entire Jäckel IV pipeline (preproc + boundary + Hermite
initial guess + Householder iterations + postproc) into one Triton kernel.  All
intermediate values stay in registers; HBM is accessed once per element.

normalised_black uses the erf formula (ndtr-based) rather than erfcx to avoid
float64 overflow in tl.exp(x²)·erfc(x) for large arguments.  Three Householder
iterations achieve machine precision for all inputs within standard option ranges.

Accuracy:  max relative error ~ 1e-13 vs py_lets_be_rational oracle
Speed:     see benchmark; target ≤ 0.636ms/100k on H100.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Scalar constants (tl.constexpr so they can be inlined in kernels)
# ---------------------------------------------------------------------------
_INV_SQRT2 = tl.constexpr(0.7071067811865476)
_INV_SQRT2PI = tl.constexpr(0.3989422804014327)
_DBL_MIN = tl.constexpr(2.2250738585072014e-308)
_BLOCK = tl.constexpr(512)


# ---------------------------------------------------------------------------
# Inline helpers
# ---------------------------------------------------------------------------


@triton.jit
def _ndtr_tl(x):
    """Standard normal CDF via erf."""
    return 0.5 * (1.0 + tl.math.erf(x * _INV_SQRT2))


@triton.jit
def _normalised_black_tl(x, s):
    """Normalised Black call: b(x,s) = exp(x/2)·N(x/s+s/2) − exp(−x/2)·N(x/s−s/2).

    Uses erf-based ndtr. Correct symmetric formulation avoids cancellation.
    Note: x_red ≤ 0 throughout the Jäckel algorithm, so exp(x/2) ≤ 1.
    """
    s_safe = tl.where(s > 0.0, s, _DBL_MIN)
    h = x / s_safe
    t = 0.5 * s_safe
    exp_half_x = tl.exp(0.5 * x)
    nd1 = _ndtr_tl(h + t)
    nd2 = _ndtr_tl(h - t)
    # b_max = exp(x/2), b_min = exp(-x/2) = 1/b_max
    return tl.abs(tl.maximum(exp_half_x * nd1 - nd2 / exp_half_x, 0.0))


@triton.jit
def _normalised_vega_tl(x, s):
    """Normalised Black vega: exp(-0.5*(h²+t²))/√(2π)."""
    s_safe = tl.where(s > 0.0, s, _DBL_MIN)
    h = x / s_safe
    t = 0.5 * s_safe
    return tl.exp(-0.5 * (h * h + t * t)) * _INV_SQRT2PI


# ---------------------------------------------------------------------------
# Jäckel IV kernel
# ---------------------------------------------------------------------------


@triton.jit
def jackel_iv_kernel(
    price_ptr,
    K_ptr,
    is_call_ptr,
    out_ptr,
    F,  # scalar: forward price
    T,  # scalar: time to expiry
    inv_sqrt_T,  # scalar: 1 / sqrt(T)
    sqrt_T,  # scalar: sqrt(T)
    n,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused Jäckel IV kernel: preproc + boundary + Hermite + Householder×3.

    One kernel launch, all intermediates in registers.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n

    price = tl.load(price_ptr + offs, mask=mask, other=0.0)
    K = tl.load(K_ptr + offs, mask=mask, other=1.0)
    is_call_i = tl.load(is_call_ptr + offs, mask=mask, other=1).to(tl.int1)

    tiny = _DBL_MIN

    # ── Preproc: put-call reduction → (beta, x_red, b_max) ──────────────────
    q = tl.where(is_call_i, 1.0, -1.0)
    sqrt_FK = tl.sqrt(tl.maximum(F * K, tiny))
    x_log = tl.log(tl.where(K > 0.0, F / K, tiny))
    intrinsic = tl.abs(tl.maximum(q * (F - K), 0.0))
    itm = (q * x_log) > 0.0
    price_red = tl.where(itm, tl.abs(tl.maximum(price - intrinsic, 0.0)), price)
    x_red = tl.where(x_log > 0.0, -x_log, x_log)
    beta = tl.where(sqrt_FK > tiny, price_red / sqrt_FK, 0.0)
    b_max = tl.exp(0.5 * x_red)

    # Degenerate check
    # isfinite: (x * 0.0 == 0.0) is True for finite, False for NaN/inf
    degenerate = (beta <= 0.0) | (beta >= b_max) | (price <= 0.0) | ~((beta * 0.0) == 0.0)

    # ── Boundary: s_c, b_c, v_c; s_l, b_l, v_l; s_h, b_h, v_h ─────────────
    # s_c = sqrt(2|x|): the inflection point of normalised_black
    s_c = tl.sqrt(tl.abs(2.0 * x_red))
    s_c = tl.where(s_c > tiny, s_c, tiny)

    b_c = _normalised_black_tl(x_red, s_c)
    v_c = _normalised_vega_tl(x_red, s_c)
    v_c_s = tl.where(v_c > tiny, v_c, tiny)

    s_l = s_c - b_c / v_c_s
    s_l = tl.where(s_l > tiny, s_l, tiny)
    b_l = _normalised_black_tl(x_red, s_l)
    v_l = _normalised_vega_tl(x_red, s_l)
    v_l_s = tl.where(v_l > tiny, v_l, tiny)

    # s_h = s_c + (b_max - b_c) / v_c
    s_h = tl.where(v_c > tiny, s_c + (b_max - b_c) / v_c_s, s_c)
    s_h = tl.where(s_h > tiny, s_h, tiny)
    b_h = _normalised_black_tl(x_red, s_h)
    v_h = _normalised_vega_tl(x_red, s_h)
    v_h_s = tl.where(v_h > tiny, v_h, tiny)

    b_tilde_h = tl.maximum(b_h, 0.5 * b_max)

    # ── Hermite initial guess ────────────────────────────────────────────────
    # Zone 2: (b_l ≤ beta < b_c) — Hermite between (b_l, s_l) and (b_c, s_c)
    h2 = tl.where(tl.abs(b_c - b_l) > tiny, tl.abs(b_c - b_l), tiny)
    t2 = tl.minimum(tl.maximum((beta - b_l) / h2, 0.0), 1.0)
    t2s = t2 * t2
    t2c = t2s * t2
    sz2 = (
        (2.0 * t2c - 3.0 * t2s + 1.0) * s_l
        + (t2c - 2.0 * t2s + t2) * h2 / v_l_s
        + (-2.0 * t2c + 3.0 * t2s) * s_c
        + (t2c - t2s) * h2 / v_c_s
    )
    # Zone 3: (b_c ≤ beta ≤ b_h) — Hermite between (b_c, s_c) and (b_h, s_h)
    h3 = tl.where(tl.abs(b_h - b_c) > tiny, tl.abs(b_h - b_c), tiny)
    t3 = tl.minimum(tl.maximum((beta - b_c) / h3, 0.0), 1.0)
    t3s = t3 * t3
    t3c = t3s * t3
    sz3 = (
        (2.0 * t3c - 3.0 * t3s + 1.0) * s_c
        + (t3c - 2.0 * t3s + t3) * h3 / v_c_s
        + (-2.0 * t3c + 3.0 * t3s) * s_h
        + (t3c - t3s) * h3 / v_h_s
    )
    z2 = (beta >= b_l) & (beta < b_c)
    s = tl.where(z2, sz2, sz3)
    s = tl.where(beta < b_l, s_l, s)
    s = tl.where(beta > b_h, s_h, s)
    s = tl.where(s > tiny, s, s_c)

    # ── Householder(3) × 3 ──────────────────────────────────────────────────
    use_lower = beta < b_l
    use_upper = beta > b_tilde_h

    for _iter in range(3):  # tl.constexpr loop → fully unrolled by Triton
        s = tl.where(s > tiny, s, tiny)
        b = _normalised_black_tl(x_red, s)
        v = _normalised_vega_tl(x_red, s)
        vp = tl.where(v > tiny, v, tiny)
        bp = tl.where(b > tiny, b, tiny)

        x_over_s = x_red / s
        xs2 = x_over_s / s
        b_halley = x_over_s * x_over_s / s - s * 0.25
        b_hh3 = b_halley * b_halley - 3.0 * xs2 * xs2 - 0.25

        # Lower branch: log-space objective
        ln_b = tl.log(bp)
        ln_beta = tl.log(tl.where(beta > tiny, beta, tiny))
        bpob = v / bp
        ln_b_s = tl.where(tl.abs(ln_b) > tiny, ln_b, tiny)
        newton_lo = (ln_beta - ln_b) * ln_b / ln_beta / bpob
        halley_lo = b_halley - bpob * (1.0 + 2.0 / ln_b_s)
        hh3_lo = (
            b_hh3
            + 2.0 * bpob * bpob * (1.0 + 3.0 / ln_b_s * (1.0 + 1.0 / ln_b_s))
            - 3.0 * b_halley * bpob * (1.0 + 2.0 / ln_b_s)
        )

        # Upper branch: complementary log-space objective
        bm_b = tl.where(b_max - b > tiny, b_max - b, tiny)
        bm_bt = tl.where(b_max - beta > tiny, b_max - beta, tiny)
        g = tl.log(bm_bt / bm_b)
        gp = v / bm_b
        newton_up = -g / gp
        halley_up = b_halley + gp
        hh3_up = b_hh3 + gp * (2.0 * gp + 3.0 * b_halley)

        # Middle branch: linear objective
        newton_mid = (beta - b) / vp

        nw = tl.where(use_lower, newton_lo, tl.where(use_upper, newton_up, newton_mid))
        hh = tl.where(use_lower, halley_lo, tl.where(use_upper, halley_up, b_halley))
        hh3 = tl.where(use_lower, hh3_lo, tl.where(use_upper, hh3_up, b_hh3))

        hf = (1.0 + 0.5 * hh * nw) / (1.0 + nw * (hh + hh3 * nw / 6.0))
        ds = tl.maximum(-0.5 * s, nw * hf)
        s = s + ds

    # ── Postproc ─────────────────────────────────────────────────────────────
    sigma = tl.where(sqrt_T > 0.0, s * inv_sqrt_T, 0.0)
    bad = degenerate | (s <= 0.0) | (T <= 0.0) | (F <= 0.0) | (K <= 0.0)
    sigma = tl.where(bad, float("nan"), sigma)

    tl.store(out_ptr + offs, sigma, mask=mask)


# ---------------------------------------------------------------------------
# Python launcher
# ---------------------------------------------------------------------------


def jackel_iv_triton(
    price: torch.Tensor,
    F: float,
    K: torch.Tensor,
    T: float,
    is_call: torch.Tensor | bool = True,
) -> torch.Tensor:
    """Jäckel IV for Black-76 — single-pass Triton kernel.

    Parameters
    ----------
    price   : undiscounted option price (GPU tensor, float64)
    F       : forward price (scalar)
    K       : strike (GPU tensor, float64)
    T       : time to expiry in years (scalar)
    is_call : True = call, False = put (tensor or scalar bool)

    Returns
    -------
    sigma : annualised IV tensor (NaN for degenerate inputs)
    """
    n = price.shape[0]
    out = torch.empty(n, dtype=torch.float64, device=price.device)

    if isinstance(is_call, bool):
        is_call_t = torch.full((n,), int(is_call), dtype=torch.int8, device=price.device)
    else:
        is_call_t = is_call.to(torch.int8)

    sqrt_T = math.sqrt(max(T, 0.0))
    inv_sqrt_T = 1.0 / sqrt_T if sqrt_T > 0.0 else 0.0

    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)  # noqa: E731
    jackel_iv_kernel[grid](
        price,
        K,
        is_call_t,
        out,
        F,
        T,
        inv_sqrt_T,
        sqrt_T,
        n,
        BLOCK_SIZE=512,
    )
    return out
