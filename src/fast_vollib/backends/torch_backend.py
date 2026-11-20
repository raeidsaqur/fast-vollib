from __future__ import annotations

from typing import TYPE_CHECKING
import warnings

import numpy as np

from ..types import ModelLiteral, OnErrorLiteral
from . import _iv_common as _ivc

if TYPE_CHECKING:
    from .._typing import FlagArray, Float1D, OptionalFloat1D  # noqa: F401


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
    flag: FlagArray, f: Float1D, k: Float1D, t: Float1D, r: Float1D, sigma: Float1D
) -> Float1D:
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
    flag: FlagArray, s: Float1D, k: Float1D, t: Float1D, r: Float1D, sigma: Float1D
) -> Float1D:
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
    flag: FlagArray,
    s: Float1D,
    k: Float1D,
    t: Float1D,
    r: Float1D,
    sigma: Float1D,
    q: Float1D,
) -> Float1D:
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
    flag: FlagArray,
    s: Float1D,
    k: Float1D,
    t: Float1D,
    r: Float1D,
    sigma: Float1D,
    q: OptionalFloat1D = None,
) -> dict[str, Float1D]:
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


def _price_vega_d1d2_t(is_call, s, k, t, r, sigma, q):
    """Return (price, raw_vega, d1, d2) in a single pass (used by the Jaeckel autograd path)."""
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


# ---------------------------------------------------------------------------
# Implied volatility — safeguarded Halley on the out-of-the-money price
# (problem statement, constants and row policy: backends/_iv_common.py).
# CUDA + Triton: one fused kernel (triton_kernels.iv_otm_triton).
# Otherwise: the same algorithm in torch tensor ops on the active device.
# ---------------------------------------------------------------------------


def _iv_log_objective_t(a, s, l_fo, l_ko, upper, tiny):
    """(ln w, |w'|/w, d1, d2), w = c(s) or u(s) = Fo - c(s); see numpy_backend._iv_log_objective."""
    import torch

    d1 = a / s + 0.5 * s
    d2 = d1 - s
    fo = torch.exp(l_fo)
    ko = torch.exp(l_ko)
    n1 = _normal_cdf(torch.where(upper, -d1, d1))
    n2 = _normal_cdf(d2)
    w = torch.where(upper, fo * n1 + ko * n2, fo * n1 - ko * n2)
    # w <= 0: c underflowed (sub-normal cancellation) -> ln c = -inf, i.e. the root lies above s
    lw = torch.where(w > 0, torch.log(w), torch.full_like(w, float("-inf")))
    if bool(tiny.any()):
        la = l_fo + torch.special.log_ndtr(torch.where(upper, -d1, d1))
        lb = l_ko + torch.special.log_ndtr(d2)
        lw_t = torch.where(upper, torch.logaddexp(la, lb), la + torch.log1p(-torch.exp(lb - la)))
        lw = torch.where(tiny, lw_t, lw)
    dlw = torch.exp(l_fo - 0.5 * d1 * d1 - _ivc.LOG_SQRT2PI - lw)
    return lw, dlw, d1, d2


def _iv_step_t(s, lo, hi, a, l_target, l_fo, l_ko, upper, tiny):
    """One safeguarded Halley step; see numpy_backend._iv_step."""
    import torch

    lw, dlw, d1, d2 = _iv_log_objective_t(a, s, l_fo, l_ko, upper, tiny)
    sgn = torch.where(upper, -1.0, 1.0).to(s.dtype)
    g = sgn * (lw - l_target)
    below = g < 0
    lo = torch.where(below, s, lo)
    hi = torch.where(below, hi, s)
    delta = g / dlw
    den = 1.0 - 0.5 * delta * (d1 * d2 / s - sgn * dlw)
    den = torch.where(den.abs() < 0.5, torch.where(den < 0, -0.5, 0.5).to(s.dtype), den)
    step = delta / den
    cand = s - step
    small = step.abs() <= _ivc.ACCEPT_RTOL * s
    ok = torch.isfinite(cand) & (((cand > lo) & (cand < hi)) | small)
    fallback = torch.where(
        torch.isinf(hi), 4.0 * s, torch.where(lo > 0, torch.sqrt(lo * hi), 0.25 * s)
    )
    s_new = torch.clamp(torch.where(ok, cand, fallback), max=_ivc.S_MAX)
    return s_new, lo, hi, ok & small


def _iv_solve_otm_t(tv, gap, fo, ko, a, s0):
    import torch

    upper = gap < tv
    l_target = torch.log(torch.where(upper, gap, tv))
    l_fo = torch.log(fo)
    l_ko = torch.log(ko)
    tiny = (l_target - l_fo) < _ivc.TINY_LOG
    s = s0
    lo = torch.zeros_like(s)
    hi = torch.full_like(s, float("inf"))
    conv = torch.zeros_like(s, dtype=torch.bool)
    for _ in range(_ivc.MAIN_ITERS):
        s, lo, hi, conv = _iv_step_t(s, lo, hi, a, l_target, l_fo, l_ko, upper, tiny)
    todo = torch.nonzero(~conv).flatten()
    for _ in range(_ivc.MAX_ITERS):
        if todo.numel() == 0:
            break
        sn, ln, hn, cv = _iv_step_t(
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
        bounded = torch.isfinite(hn) & (ln > 0)
        tight = (hn <= ln * (1.0 + _ivc.BRACKET_RTOL)) | (
            torch.nextafter(ln, torch.full_like(ln, float("inf"))) >= hn
        )
        done = cv | (bounded & tight)
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
    import torch

    shape, price, s, k, t, r, flag, q = _ivc.flatten_inputs(price, s, k, t, r, flag, q)
    dev = _device()
    pt = _h2d(price, "price", dev)
    st = _h2d(s, "s", dev)
    kt = _h2d(k, "k", dev)
    tt = _h2d(t, "t", dev)
    rt = _h2d(r, "r", dev)
    if q is not None and model == "black_scholes_merton":
        qv = _h2d(q, "q", dev)
    else:
        qv = torch.zeros_like(rt)
    is_call = torch.as_tensor(flag == "c", device=dev)

    if _check_triton():
        from . import triton_kernels as tk

        sigma, code = tk.iv_otm_triton(model, pt, st, kt, tt, rt, qv, is_call)
    else:
        rows = _ivc.classify(torch, model, pt, st, kt, tt, rt, qv, is_call)
        near = torch.nonzero(rows["near"]).flatten()
        if near.numel():  # within the error bound of the double-precision bounds
            sub = _ivc.classify(
                torch,
                model,
                pt[near],
                st[near],
                kt[near],
                tt[near],
                rt[near],
                qv[near],
                is_call[near],
                precise=True,
            )
            for key, value in sub.items():
                rows[key][near] = value.to(rows[key].dtype)
        code = rows["code"].to(torch.int8)
        sigma = torch.full_like(pt, float("nan"))
        sigma = torch.where(rows["zero_vol"], torch.zeros_like(sigma), sigma)
        idx = torch.nonzero(rows["interior"]).flatten()
        if idx.numel():
            sqrt_t = torch.sqrt(tt[idx])
            tv, fo = rows["tv"][idx], rows["Fo"][idx]
            sigma0 = torch.clamp(
                tv / (fo * sqrt_t) * _ivc.SQRT2PI, _ivc.SEED_SIGMA_LO, _ivc.SEED_SIGMA_HI
            )
            sol, conv = _iv_solve_otm_t(
                tv, rows["gap"][idx], fo, rows["Ko"][idx], rows["a"][idx], sigma0 * sqrt_t
            )
            sigma[idx] = torch.where(conv, sol / sqrt_t, torch.full_like(sol, float("nan")))
            code[idx[~conv]] = _ivc.NOT_CONVERGED
    counts = torch.bincount(code.to(torch.int64).flatten(), minlength=_ivc.N_CODES).cpu().tolist()
    _ivc.report_row_errors(
        {c: n for c, n in enumerate(counts) if c != _ivc.OK}, int(pt.numel()), on_error
    )
    return sigma.cpu().numpy().reshape(shape)
