from __future__ import annotations

import warnings

import numpy as np

from ..types import ModelLiteral


def _make_compiled(fn):
    """Wrap torch.compile with an eager fallback if the inductor fails.

    On CPU without setuptools (or other inductor prerequisites), torch.compile
    raises InductorError on first invocation.  Rather than crashing, we warn
    once and fall back to the uncompiled function for the lifetime of the
    process.
    """
    import torch

    compiled = torch.compile(fn, dynamic=True)
    _use_eager: list[bool] = [False]

    def _call(*args, **kwargs):
        if _use_eager[0]:
            return fn(*args, **kwargs)
        try:
            return compiled(*args, **kwargs)
        except Exception as exc:
            # torch._inductor.exc.InductorError is the expected failure type,
            # but we catch broadly so the fallback works for any compile error.
            if not _use_eager[0]:
                _use_eager[0] = True
                warnings.warn(
                    f"torch.compile failed ({exc!r}); switching to eager mode. "
                    "Install 'setuptools' to enable compiled acceleration on CPU.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            return fn(*args, **kwargs)

    return _call


# ---------------------------------------------------------------------------
# Triton kernel availability flag (set True when CUDA + Triton are present)
# ---------------------------------------------------------------------------
_TRITON_AVAILABLE: bool | None = None  # None = not yet checked


def _check_triton() -> bool:
    global _TRITON_AVAILABLE
    if _TRITON_AVAILABLE is None:
        try:
            import torch
            import triton  # noqa: F401

            _TRITON_AVAILABLE = torch.cuda.is_available()
        except ImportError:
            _TRITON_AVAILABLE = False
    return _TRITON_AVAILABLE


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------


def is_available() -> bool:
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def _device():
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Pinned memory buffer pool for faster H2D transfers (~1.75x vs pageable)
# ---------------------------------------------------------------------------

_pinned_pool: "dict[str, object]" = {}  # key → pinned CPU tensor


def _to_tensor_pinned(arr: np.ndarray, key: str, device) -> "torch.Tensor":
    """Copy arr through a persistent pinned-memory buffer to GPU.

    Pinned (page-locked) memory enables faster PCIe DMA than pageable
    allocations.  We keep one buffer per (key, capacity) and reuse it across
    calls, growing it on demand.

    CPU→CPU copy uses numpy directly into the pinned buffer view to avoid
    intermediate torch tensor allocation overhead.
    """
    import torch

    flat = arr.ravel()
    n = flat.size
    buf = _pinned_pool.get(key)
    if buf is None or buf.numel() < n:
        _pinned_pool[key] = torch.empty(n, dtype=torch.float64, pin_memory=True)
        buf = _pinned_pool[key]
    dst = buf[:n]
    # Direct numpy memcpy into pinned buffer — avoids torch dispatch overhead
    np.copyto(dst.numpy(), flat.astype(np.float64, copy=False))
    # Pinned → GPU: fast PCIe DMA path
    return dst.cuda(non_blocking=False)


def _to_tensor(arr: np.ndarray, device):
    import torch

    return torch.as_tensor(arr, dtype=torch.float64, device=device)


def to_native(values: np.ndarray):
    import torch

    return torch.as_tensor(values, dtype=torch.float64, device=_device())


def from_native(values) -> np.ndarray:
    import torch

    if isinstance(values, torch.Tensor):
        return values.detach().cpu().numpy()
    return np.asarray(values)


# ---------------------------------------------------------------------------
# Normal distribution helpers
# ---------------------------------------------------------------------------

_SQRT2 = 2.0**0.5
_SQRT2PI = (2.0 * 3.141592653589793) ** 0.5


def _normal_cdf(x):
    """Accurate normal CDF for all x, including extreme tails.

    torch.distributions.Normal.cdf uses erf which loses precision for |x|>~8
    (the tail value underflows to 0.0 or 1.0).  erfc preserves ~16 significant
    digits down to the float64 underflow floor (~5e-324).
    """
    import torch

    return 0.5 * torch.special.erfc(-x / _SQRT2)


def _normal_pdf(x):
    import torch

    return torch.exp(-0.5 * x * x) / _SQRT2PI


# ---------------------------------------------------------------------------
# Core pricing (all ops on torch tensors)
# ---------------------------------------------------------------------------


def _d1_d2(s, k, t, r, sigma, q):
    import torch

    sqrt_t = torch.sqrt(torch.clamp(t, min=1e-32))
    vol_term = torch.clamp(sigma * sqrt_t, min=1e-32)
    d1 = (
        torch.log(torch.clamp(s, min=1e-32) / torch.clamp(k, min=1e-32))
        + (r - q + 0.5 * sigma**2) * t
    ) / vol_term
    d2 = d1 - sigma * sqrt_t
    return d1, d2


def _bsm_price_t(is_call, s, k, t, r, sigma, q):
    import torch

    d1, d2 = _d1_d2(s, k, t, r, sigma, q)
    discounted_spot = s * torch.exp(-q * t)
    discounted_strike = k * torch.exp(-r * t)
    call = discounted_spot * _normal_cdf(d1) - discounted_strike * _normal_cdf(d2)
    put = discounted_strike * _normal_cdf(-d2) - discounted_spot * _normal_cdf(-d1)
    if isinstance(is_call, np.ndarray):
        is_call = torch.as_tensor(is_call, device=s.device)
    return torch.where(is_call, call, put)


def _vega_raw_t(s, k, t, r, sigma, q):
    import torch

    d1, _ = _d1_d2(s, k, t, r, sigma, q)
    return s * torch.exp(-q * t) * _normal_pdf(d1) * torch.sqrt(torch.clamp(t, min=1e-32))


def _h2d(arr: np.ndarray, key: str, dev) -> "torch.Tensor":
    """Transfer numpy array to GPU, using pinned memory on CUDA for speed."""
    if dev.type == "cuda":
        return _to_tensor_pinned(arr, key, dev)
    return _to_tensor(arr, dev)


def _price_with_triton_or_torch(is_call_np, s_t, k_t, t_t, r_t, sigma_t, q_t) -> np.ndarray:
    import torch

    is_call = torch.as_tensor(is_call_np, device=s_t.device)
    if _check_triton():
        from . import triton_kernels as tk

        out = tk.bsm_price_triton(is_call, s_t, k_t, t_t, r_t, sigma_t, q_t)
    else:
        out = _bsm_price_t(is_call, s_t, k_t, t_t, r_t, sigma_t, q_t)
    return out.cpu().numpy()


def price_black(
    flag: np.ndarray, f: np.ndarray, k: np.ndarray, t: np.ndarray, r: np.ndarray, sigma: np.ndarray
) -> np.ndarray:
    import torch

    dev = _device()
    ft = _h2d(f, "s", dev)
    kt = _h2d(k, "k", dev)
    tt = _h2d(t, "t", dev)
    rt = _h2d(r, "r", dev)
    st = _h2d(sigma, "sig", dev)
    qt = rt.clone()
    return _price_with_triton_or_torch(flag == "c", ft, kt, tt, rt, st, qt)


def price_black_scholes(
    flag: np.ndarray, s: np.ndarray, k: np.ndarray, t: np.ndarray, r: np.ndarray, sigma: np.ndarray
) -> np.ndarray:
    import torch

    dev = _device()
    st = _h2d(s, "s", dev)
    kt = _h2d(k, "k", dev)
    tt = _h2d(t, "t", dev)
    rt = _h2d(r, "r", dev)
    sigt = _h2d(sigma, "sig", dev)
    qt = torch.zeros_like(rt)
    return _price_with_triton_or_torch(flag == "c", st, kt, tt, rt, sigt, qt)


def price_black_scholes_merton(
    flag: np.ndarray,
    s: np.ndarray,
    k: np.ndarray,
    t: np.ndarray,
    r: np.ndarray,
    sigma: np.ndarray,
    q: np.ndarray,
) -> np.ndarray:
    dev = _device()
    st = _h2d(s, "s", dev)
    kt = _h2d(k, "k", dev)
    tt = _h2d(t, "t", dev)
    rt = _h2d(r, "r", dev)
    sigt = _h2d(sigma, "sig", dev)
    qt = _h2d(q, "q", dev)
    return _price_with_triton_or_torch(flag == "c", st, kt, tt, rt, sigt, qt)


# ---------------------------------------------------------------------------
# Greeks
# ---------------------------------------------------------------------------

# One compiled core per model string (compile-time constant branch avoids graph breaks)
_compiled_greeks_cores: "dict[str, object]" = {}


def _get_compiled_greeks_core(model: str):
    if model not in _compiled_greeks_cores:
        import torch

        _is_black = model == "black"

        def _greeks_core(is_call, st, kt, tt, rt, sigt, qv):
            d1, d2 = _d1_d2(st, kt, tt, rt, sigt, qv)
            carry = torch.exp(-qv * tt)
            disc = torch.exp(-rt * tt)
            pdf = _normal_pdf(d1)
            sqrt_t = torch.sqrt(torch.clamp(tt, min=1e-32))
            safe_s = torch.clamp(st, min=1e-32)
            safe_sig = torch.clamp(sigt, min=1e-32)
            cdf_d1 = _normal_cdf(d1)
            cdf_d2 = _normal_cdf(d2)

            if _is_black:
                delta = disc * torch.where(is_call, cdf_d1, cdf_d1 - 1.0)
                gamma = disc * pdf / (safe_s * safe_sig * sqrt_t)
            else:
                delta = torch.where(is_call, carry * cdf_d1, carry * (cdf_d1 - 1.0))
                gamma = carry * pdf / (safe_s * safe_sig * sqrt_t)

            # N(-x) = 1-N(x) — avoids extra erfc calls for -d1, -d2
            cdf_nd1 = 1.0 - cdf_d1
            cdf_nd2 = 1.0 - cdf_d2

            vega = st * carry * pdf * sqrt_t * 0.01
            theta_call = (
                -(st * carry * pdf * sigt) / (2.0 * sqrt_t)
                - rt * kt * disc * cdf_d2
                + qv * st * carry * cdf_d1
            ) / 365.0
            theta_put = (
                -(st * carry * pdf * sigt) / (2.0 * sqrt_t)
                + rt * kt * disc * cdf_nd2
                - qv * st * carry * cdf_nd1
            ) / 365.0
            rho_call = kt * tt * disc * cdf_d2 * 0.01
            rho_put = -kt * tt * disc * cdf_nd2 * 0.01
            rho = torch.where(is_call, rho_call, rho_put)
            theta = torch.where(is_call, theta_call, theta_put)
            return delta, gamma, theta, rho, vega

        _compiled_greeks_cores[model] = _make_compiled(_greeks_core)
    return _compiled_greeks_cores[model]


def greeks(
    model: ModelLiteral,
    flag: np.ndarray,
    s: np.ndarray,
    k: np.ndarray,
    t: np.ndarray,
    r: np.ndarray,
    sigma: np.ndarray,
    q: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    import torch

    dev = _device()
    st = _h2d(s, "s", dev)
    kt = _h2d(k, "k", dev)
    tt = _h2d(t, "t", dev)
    rt = _h2d(r, "r", dev)
    sigt = _h2d(sigma, "sig", dev)
    if model == "black" and q is None:
        qv = rt.clone()
    else:
        qv = torch.zeros_like(rt) if q is None else _h2d(q, "q", dev)
    is_call = torch.as_tensor(flag == "c", device=dev)

    if _check_triton():
        from . import triton_kernels as tk

        delta, gamma, theta, rho, vega = tk.bsm_greeks_triton(
            model, is_call, st, kt, tt, rt, sigt, qv
        )
    else:
        delta, gamma, theta, rho, vega = _get_compiled_greeks_core(model)(
            is_call, st, kt, tt, rt, sigt, qv
        )

    return {
        name: arr.cpu().numpy()
        for name, arr in {
            "delta": delta,
            "gamma": gamma,
            "theta": theta,
            "rho": rho,
            "vega": vega,
        }.items()
    }


# ---------------------------------------------------------------------------
# Implied volatility (vectorized Newton-Raphson on GPU tensors)
# ---------------------------------------------------------------------------

_HALLEY_ITERS = 8
_BISECT_ITERS = 30  # 10/(2^30)≈9e-9, narrower than _IV_LO → sufficient accuracy
_IV_LO = 1e-8
_IV_HI = 10.0

# ---------------------------------------------------------------------------
# torch.compile — fuses the entire Halley loop into a single CUDA kernel set,
# eliminating ~96 individual kernel launches per IV solve batch.
# ---------------------------------------------------------------------------

_compiled_halley: "object | None" = None
_compiled_bisect: "object | None" = None


def _get_compiled_halley():
    """Lazy-init torch.compile wrapper for the Halley IV loop."""
    global _compiled_halley
    if _compiled_halley is not None:
        return _compiled_halley
    import torch

    def _halley_loop(sigma, pt, st, kt, tt, rt, qv, is_call):
        for _ in range(_HALLEY_ITERS):
            d1, d2 = _d1_d2(st, kt, tt, rt, sigma, qv)
            discounted_spot = st * torch.exp(-qv * tt)
            discounted_strike = kt * torch.exp(-rt * tt)
            sqrt_t_inner = torch.sqrt(torch.clamp(tt, min=1e-32))
            cdf_d1 = _normal_cdf(d1)
            cdf_d2 = _normal_cdf(d2)
            call = discounted_spot * cdf_d1 - discounted_strike * cdf_d2
            put = discounted_strike * (1.0 - cdf_d2) - discounted_spot * (1.0 - cdf_d1)
            px = torch.where(is_call, call, put)
            vega = discounted_spot * _normal_pdf(d1) * sqrt_t_inner
            residual = px - pt
            safe_vega = torch.where(vega > 1e-14, vega, torch.full_like(vega, torch.inf))
            newton_step = residual / safe_vega
            safe_sigma = torch.where(sigma > 1e-8, sigma, torch.full_like(sigma, torch.inf))
            halley_denom = 1.0 - newton_step * d1 * d2 / safe_sigma
            halley_denom = torch.where(
                halley_denom.abs() > 0.05,
                halley_denom,
                torch.sign(halley_denom + 1e-15) * 0.05,
            )
            sigma = torch.clamp(sigma - newton_step / halley_denom, _IV_LO, _IV_HI)
        return sigma

    _compiled_halley = _make_compiled(_halley_loop)
    return _compiled_halley


def _get_compiled_bisect():
    """Lazy-init torch.compile wrapper for the bisection fallback loop."""
    global _compiled_bisect
    if _compiled_bisect is not None:
        return _compiled_bisect
    import torch

    def _bisect_loop(sigma, pt, st, kt, tt, rt, qv, is_call, not_converged):
        # Hoist all loop-invariant terms outside the 30-iteration body
        discounted_spot = st * torch.exp(-qv * tt)
        discounted_strike = kt * torch.exp(-rt * tt)
        sqrt_t = torch.sqrt(torch.clamp(tt, min=1e-32))
        log_fk = torch.log(torch.clamp(st, min=1e-32) / torch.clamp(kt, min=1e-32))
        carry_drift = (rt - qv) * tt  # the r-q part of the d1 numerator
        sigma_lo = torch.where(not_converged, torch.full_like(sigma, _IV_LO), sigma)
        sigma_hi = torch.where(not_converged, torch.full_like(sigma, _IV_HI), sigma)
        for _ in range(_BISECT_ITERS):
            sigma_mid = 0.5 * (sigma_lo + sigma_hi)
            vol_term = torch.clamp(sigma_mid * sqrt_t, min=1e-32)
            d1 = (log_fk + carry_drift + 0.5 * sigma_mid**2 * tt) / vol_term
            d2 = d1 - sigma_mid * sqrt_t
            cdf_d1 = _normal_cdf(d1)
            cdf_d2 = _normal_cdf(d2)
            call = discounted_spot * cdf_d1 - discounted_strike * cdf_d2
            put = discounted_strike * (1.0 - cdf_d2) - discounted_spot * (1.0 - cdf_d1)
            px_mid = torch.where(is_call, call, put)
            mid_res = px_mid - pt
            sigma_lo = torch.where(not_converged & (mid_res < 0), sigma_mid, sigma_lo)
            sigma_hi = torch.where(not_converged & (mid_res >= 0), sigma_mid, sigma_hi)
        return torch.where(not_converged, 0.5 * (sigma_lo + sigma_hi), sigma)

    _compiled_bisect = _make_compiled(_bisect_loop)
    return _compiled_bisect


def _price_vega_d1d2_t(is_call, s, k, t, r, sigma, q):
    """Return (price, raw_vega, d1, d2) in a single pass.

    Accepts a pre-computed boolean tensor `is_call` to avoid repeated flag
    string comparison on every Halley iteration.
    """
    import torch

    d1, d2 = _d1_d2(s, k, t, r, sigma, q)
    discounted_spot = s * torch.exp(-q * t)
    discounted_strike = k * torch.exp(-r * t)
    sqrt_t = torch.sqrt(torch.clamp(t, min=1e-32))
    call = discounted_spot * _normal_cdf(d1) - discounted_strike * _normal_cdf(d2)
    put = discounted_strike * _normal_cdf(-d2) - discounted_spot * _normal_cdf(-d1)
    price = torch.where(is_call, call, put)
    vega = discounted_spot * _normal_pdf(d1) * sqrt_t
    return price, vega, d1, d2


def _price_for_model_t(is_call, s, k, t, r, sigma, q):
    import torch

    qt = q if q is not None else torch.zeros_like(r)
    px, _, _, _ = _price_vega_d1d2_t(is_call, s, k, t, r, sigma, qt)
    return px


def _initial_guess_t(price, s, t):
    import torch

    sqrt_t = torch.sqrt(torch.clamp(t, min=1e-8))
    approx = price / torch.clamp(s * sqrt_t, min=1e-12) * _SQRT2PI
    return torch.clamp(approx, 0.30, 5.0)


def implied_volatility(
    model: ModelLiteral,
    price: np.ndarray,
    s: np.ndarray,
    k: np.ndarray,
    t: np.ndarray,
    r: np.ndarray,
    flag: np.ndarray,
    q: np.ndarray | None = None,
    on_error: str = "warn",
) -> np.ndarray:
    import torch

    from ..utils.validation import handle_error

    dev = _device()
    pt = _h2d(price, "price", dev)
    st = _h2d(s, "s", dev)
    kt = _h2d(k, "k", dev)
    tt = _h2d(t, "t", dev)
    rt = _h2d(r, "r", dev)
    # Black-76: q=r so d1 uses only σ² drift (carry=disc eliminates r-q from d1)
    if model == "black" and q is None:
        qv = rt.clone()
    else:
        qv = torch.zeros_like(rt) if q is None else _h2d(q, "q", dev)

    # Pre-compute boolean flag once — avoids re-running `flag == "c"` each Halley iter
    is_call = torch.as_tensor(flag == "c", device=dev)

    valid = tt > 0

    # Below-intrinsic check — returns NaN for impossible prices (same as numpy backend)
    discounted_spot = st * torch.exp(-qv * tt)
    discounted_strike = kt * torch.exp(-rt * tt)
    intrinsic = torch.where(
        is_call,
        torch.clamp(discounted_spot - discounted_strike, min=0.0),
        torch.clamp(discounted_strike - discounted_spot, min=0.0),
    )
    below_intrinsic = pt < intrinsic - 1e-10

    zero_price = (pt <= 0.0) & valid  # undetermined sigma for zero-price OTM options
    if bool(below_intrinsic.any().item()):
        handle_error(
            f"Option price is below intrinsic value. {int(below_intrinsic.sum().item())}", on_error
        )

    if _check_triton():
        from . import triton_kernels as tk

        # Triton path: Halley loop + below-intrinsic + convergence check in one kernel;
        # hoists 5 loop-invariant ops out of 8 Halley iters, and fuses the convergence
        # check to save one separate bsm_price kernel call.
        sigma, below_intrinsic_mask, not_converged = tk.bsm_iv_triton(
            pt, st, kt, tt, rt, qv, is_call
        )

        if not_converged.any():
            sigma = tk.bsm_iv_bisect_triton(pt, st, kt, tt, rt, qv, is_call, not_converged, sigma)

        result = torch.where(valid, sigma, torch.zeros_like(sigma))
        result = torch.where(
            below_intrinsic_mask | zero_price, torch.full_like(result, float("nan")), result
        )
    else:
        sigma = _initial_guess_t(pt, st, tt)
        # Halley's method (3rd order) — torch.compile fuses 8 iters
        sigma = _get_compiled_halley()(sigma, pt, st, kt, tt, rt, qv, is_call)

        px_final = _price_for_model_t(is_call, st, kt, tt, rt, sigma, qv)
        underflow_stuck = (px_final == 0.0) & (pt > 0.0)
        not_converged = (torch.abs(px_final - pt) > 1e-6) | underflow_stuck
        if not_converged.any():
            sigma = _get_compiled_bisect()(sigma, pt, st, kt, tt, rt, qv, is_call, not_converged)

        result = torch.where(valid, sigma, torch.zeros_like(sigma))
        result = torch.where(
            below_intrinsic | zero_price, torch.full_like(result, float("nan")), result
        )

    return result.cpu().numpy()
