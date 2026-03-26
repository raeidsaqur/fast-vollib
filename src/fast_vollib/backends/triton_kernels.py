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

# ---------------------------------------------------------------------------
# Constants (tl.constexpr so they can be referenced inside @triton.jit kernels)
# ---------------------------------------------------------------------------
_INV_SQRT2 = tl.constexpr(0.7071067811865476)  # 1 / sqrt(2)
_INV_SQRT2PI = tl.constexpr(0.3989422804014327)  # 1 / sqrt(2π)
_SQRT2PI = 2.5066282746310002  # Python-side only
_IV_LO_C = tl.constexpr(1e-8)
_IV_HI_C = tl.constexpr(10.0)
# Python-side copies for use outside kernels
_IV_LO = 1e-8
_IV_HI = 10.0
_HALLEY_ITERS = tl.constexpr(8)
_BISECT_ITERS = tl.constexpr(30)


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
# 3a. IV Halley kernel  (8 Halley iters; loop-invariants hoisted to registers)
# ---------------------------------------------------------------------------


@triton.jit
def bsm_iv_halley_kernel(
    price_ptr,
    s_ptr,
    k_ptr,
    t_ptr,
    r_ptr,
    q_ptr,
    is_call_ptr,
    sigma_ptr,  # input: initial guess;  output: Halley result
    below_intrinsic_ptr,  # output: int8 flag  (1 = price below intrinsic)
    not_converged_ptr,  # output: int8 flag  (1 = needs bisection)
    n,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n

    price = tl.load(price_ptr + offs, mask=mask)
    s = tl.load(s_ptr + offs, mask=mask)
    k = tl.load(k_ptr + offs, mask=mask)
    t = tl.load(t_ptr + offs, mask=mask)
    r = tl.load(r_ptr + offs, mask=mask)
    q = tl.load(q_ptr + offs, mask=mask)
    is_call = tl.load(is_call_ptr + offs, mask=mask).to(tl.int1)
    sigma = tl.load(sigma_ptr + offs, mask=mask)

    # Loop-invariant quantities (hoisted out of all 8 Halley iterations)
    disc_s = s * tl.exp(-q * t)
    disc_k = k * tl.exp(-r * t)
    sqrt_t = tl.sqrt(tl.maximum(t, 1e-32))
    log_sk = tl.log(tl.maximum(s, 1e-32) / tl.maximum(k, 1e-32))
    carry_dt = (r - q) * t  # (r-q)*t part of d1 numerator

    # Below-intrinsic check (store result for post-processing in Python)
    intrinsic_call = tl.maximum(disc_s - disc_k, 0.0)
    intrinsic_put = tl.maximum(disc_k - disc_s, 0.0)
    intrinsic = tl.where(is_call, intrinsic_call, intrinsic_put)
    below_int = (price < intrinsic - 1e-10).to(tl.int8)
    tl.store(below_intrinsic_ptr + offs, below_int, mask=mask)

    # 8 Halley iterations — all in registers, no HBM round-trips
    for _ in tl.static_range(8):
        vol_term = tl.maximum(sigma * sqrt_t, 1e-32)
        d1 = (log_sk + carry_dt + 0.5 * sigma * sigma * t) / vol_term
        d2 = d1 - sigma * sqrt_t
        cdf_d1 = _norm_cdf_tl(d1)
        cdf_d2 = _norm_cdf_tl(d2)
        call_p = disc_s * cdf_d1 - disc_k * cdf_d2
        put_p = disc_k * (1.0 - cdf_d2) - disc_s * (1.0 - cdf_d1)
        px = tl.where(is_call, call_p, put_p)
        pdf_d1 = _norm_pdf_tl(d1)
        vega = disc_s * pdf_d1 * sqrt_t
        residual = px - price
        safe_vega = tl.where(vega > 1e-14, vega, float("inf"))
        newton_step = residual / safe_vega
        safe_sigma = tl.where(sigma > 1e-8, sigma, float("inf"))
        h_denom = 1.0 - newton_step * d1 * d2 / safe_sigma
        # Clamp denominator to avoid near-zero divide
        h_denom = tl.where(
            tl.abs(h_denom) > 0.05,
            h_denom,
            tl.where(h_denom + 1e-15 >= 0.0, 1.0, -1.0) * 0.05,
        )
        sigma = tl.maximum(tl.minimum(sigma - newton_step / h_denom, _IV_HI_C), _IV_LO_C)

    # Convergence check — fused here to avoid a separate bsm_price kernel call
    vol_term_f = tl.maximum(sigma * sqrt_t, 1e-32)
    d1_f = (log_sk + carry_dt + 0.5 * sigma * sigma * t) / vol_term_f
    d2_f = d1_f - sigma * sqrt_t
    cdf1_f = _norm_cdf_tl(d1_f)
    cdf2_f = _norm_cdf_tl(d2_f)
    call_f = disc_s * cdf1_f - disc_k * cdf2_f
    put_f = disc_k * (1.0 - cdf2_f) - disc_s * (1.0 - cdf1_f)
    px_f = tl.where(is_call, call_f, put_f)
    not_conv = ((tl.abs(px_f - price) > 1e-6) | ((px_f == 0.0) & (price > 0.0))).to(tl.int8)

    tl.store(sigma_ptr + offs, sigma, mask=mask)
    tl.store(not_converged_ptr + offs, not_conv, mask=mask)


# ---------------------------------------------------------------------------
# 3b. IV Bisection kernel  (30-iter fallback; only updates non-converged lanes)
# ---------------------------------------------------------------------------


@triton.jit
def bsm_iv_bisect_kernel(
    price_ptr,
    s_ptr,
    k_ptr,
    t_ptr,
    r_ptr,
    q_ptr,
    is_call_ptr,
    not_converged_ptr,  # int8 mask: 1 = needs bisection
    sigma_ptr,  # in/out
    n,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n

    price = tl.load(price_ptr + offs, mask=mask)
    s = tl.load(s_ptr + offs, mask=mask)
    k = tl.load(k_ptr + offs, mask=mask)
    t = tl.load(t_ptr + offs, mask=mask)
    r = tl.load(r_ptr + offs, mask=mask)
    q = tl.load(q_ptr + offs, mask=mask)
    is_call = tl.load(is_call_ptr + offs, mask=mask).to(tl.int1)
    nc = tl.load(not_converged_ptr + offs, mask=mask).to(tl.int1)
    sigma = tl.load(sigma_ptr + offs, mask=mask)

    # Loop-invariant quantities
    disc_s = s * tl.exp(-q * t)
    disc_k = k * tl.exp(-r * t)
    sqrt_t = tl.sqrt(tl.maximum(t, 1e-32))
    log_sk = tl.log(tl.maximum(s, 1e-32) / tl.maximum(k, 1e-32))
    carry_dt = (r - q) * t

    sigma_lo = tl.where(nc, _IV_LO_C, sigma)
    sigma_hi = tl.where(nc, _IV_HI_C, sigma)

    for _ in tl.static_range(30):
        sigma_mid = 0.5 * (sigma_lo + sigma_hi)
        vol_term = tl.maximum(sigma_mid * sqrt_t, 1e-32)
        d1 = (log_sk + carry_dt + 0.5 * sigma_mid * sigma_mid * t) / vol_term
        d2 = d1 - sigma_mid * sqrt_t
        cdf_d1 = _norm_cdf_tl(d1)
        cdf_d2 = _norm_cdf_tl(d2)
        call_p = disc_s * cdf_d1 - disc_k * cdf_d2
        put_p = disc_k * (1.0 - cdf_d2) - disc_s * (1.0 - cdf_d1)
        px_mid = tl.where(is_call, call_p, put_p)
        mid_res = px_mid - price
        sigma_lo = tl.where(nc & (mid_res < 0.0), sigma_mid, sigma_lo)
        sigma_hi = tl.where(nc & (mid_res >= 0.0), sigma_mid, sigma_hi)

    sigma = tl.where(nc, 0.5 * (sigma_lo + sigma_hi), sigma)
    tl.store(sigma_ptr + offs, sigma, mask=mask)


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
_compiled_iv_halley: "object | None" = None
_compiled_iv_bisect: "object | None" = None


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


def _get_compiled_iv_halley():
    global _compiled_iv_halley
    if _compiled_iv_halley is None:
        _compiled_iv_halley = triton.autotune(configs=_CONFIGS, key=["n"])(bsm_iv_halley_kernel)
    return _compiled_iv_halley


def _get_compiled_iv_bisect():
    global _compiled_iv_bisect
    if _compiled_iv_bisect is None:
        _compiled_iv_bisect = triton.autotune(configs=_CONFIGS, key=["n"])(bsm_iv_bisect_kernel)
    return _compiled_iv_bisect


# --------------- initial guess (fast, stays on GPU) -----------------------


def _initial_guess_gpu(price: torch.Tensor, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    sqrt_t = torch.sqrt(torch.clamp(t, min=1e-8))
    approx = price / torch.clamp(s * sqrt_t, min=1e-12) * _SQRT2PI
    return torch.clamp(approx, 0.30, 5.0)


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


def bsm_iv_triton(
    price: torch.Tensor,
    s: torch.Tensor,
    k: torch.Tensor,
    t: torch.Tensor,
    r: torch.Tensor,
    q: torch.Tensor,
    is_call: torch.Tensor,  # bool GPU tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run Halley IV solver.  Returns (sigma, below_intrinsic_mask, not_converged_mask).

    Fuses the convergence check into the Halley kernel to save one extra
    bsm_price kernel call in the caller's convergence check step.
    """
    n = s.numel()
    sigma = _initial_guess_gpu(price, s, t)
    below_int = torch.empty(n, dtype=torch.int8, device=s.device)
    not_conv = torch.empty(n, dtype=torch.int8, device=s.device)
    is_call_i8 = is_call.to(torch.int8)
    bs = 256
    bsm_iv_halley_kernel[_grid(n, bs)](
        price,
        s,
        k,
        t,
        r,
        q,
        is_call_i8,
        sigma,
        below_int,
        not_conv,
        n,
        BLOCK_SIZE=bs,
    )
    return sigma, below_int.bool(), not_conv.bool()


def bsm_iv_bisect_triton(
    price: torch.Tensor,
    s: torch.Tensor,
    k: torch.Tensor,
    t: torch.Tensor,
    r: torch.Tensor,
    q: torch.Tensor,
    is_call: torch.Tensor,
    not_converged: torch.Tensor,  # bool GPU tensor
    sigma: torch.Tensor,  # in/out
) -> torch.Tensor:
    """Run bisection fallback in-place on not_converged lanes."""
    n = s.numel()
    is_call_i8 = is_call.to(torch.int8)
    nc_i8 = not_converged.to(torch.int8)
    bs = 256
    bsm_iv_bisect_kernel[_grid(n, bs)](
        price,
        s,
        k,
        t,
        r,
        q,
        is_call_i8,
        nc_i8,
        sigma,
        n,
        BLOCK_SIZE=bs,
    )
    return sigma
