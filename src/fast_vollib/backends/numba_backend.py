"""Numba JIT-compiled CPU backend for fast-vollib.

Vectorised kernels are compiled to native code via ``@numba.njit(parallel=True)``
with element-wise ``prange`` loops.  The scalar Halley and bisection steps run
entirely inside the compiled code, so the solver incurs only a single Python
call per batch regardless of batch size.

Kernels are compiled on first call and reused for the lifetime of the process
(``cache=True`` writes compiled artefacts to ``__pycache__`` for subsequent
runs).

Design conventions that match the rest of fast-vollib
------------------------------------------------------
* All public functions accept **numpy arrays** (broadcast-ready, float64) and
  return ``numpy.ndarray``.
* ``is_available()`` / ``to_native()`` / ``from_native()`` follow the same
  interface as the NumPy, PyTorch, and JAX backends.
* The ``q`` parameter is pre-adjusted by the public wrappers before being
  handed to the numba kernels:

    - Black-76:          ``q = r``   (carry = disc, d1 has no r-q drift term)
    - Black-Scholes:     ``q = 0``
    - Black-Scholes-Merton: ``q = dividend yield``

  A single set of kernels therefore handles all three models without branching.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from ..types import ModelLiteral, OnErrorLiteral
from ..utils.validation import handle_error

if TYPE_CHECKING:
    from .._typing import FlagArray, Float1D, OptionalFloat1D  # noqa: F401

# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------


def is_available() -> bool:
    try:
        import numba  # noqa: F401
    except ImportError:
        return False
    return True


def to_native(values: np.ndarray) -> np.ndarray:
    """Numba operates on NumPy arrays; the native type is already ndarray."""
    return values


def from_native(values: np.ndarray) -> np.ndarray:
    return np.asarray(values)


# ---------------------------------------------------------------------------
# Module-level constants (captured as literals by numba at JIT time)
# ---------------------------------------------------------------------------

_SQRT2 = math.sqrt(2.0)
_SQRT2PI_INV = 1.0 / math.sqrt(2.0 * math.pi)

_HALLEY_ITERS = 8  # 8 Halley steps ≈ 12 Newton steps in accuracy
_BISECT_ITERS = 30  # 10/(2^30) ≈ 9e-9 < _IV_LO → sufficient accuracy
_IV_LO = 1e-8
_IV_HI = 10.0


# ---------------------------------------------------------------------------
# Lazy kernel compilation
# ---------------------------------------------------------------------------

_kernels: dict[str, object] = {}


def _get_kernels() -> dict[str, object]:
    """Return the compiled kernel bundle, compiling it on the very first call."""
    if _kernels:
        return _kernels
    _build_kernels()
    return _kernels


def _build_kernels() -> None:  # noqa: C901
    """JIT-compile all numba kernels and store them in ``_kernels``."""
    import numba

    # Local aliases so they are captured as compile-time constants by numba.
    SQRT2 = _SQRT2
    SQRT2PI_INV = _SQRT2PI_INV
    IV_LO = _IV_LO
    IV_HI = _IV_HI

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

    @numba.njit(cache=True)
    def _halley_step(
        sigma: float,
        price: float,
        is_call: bool,
        ds: float,
        dk: float,
        sqrt_t: float,
        log_sk: float,
        carry_drift_t: float,
        t: float,
    ) -> float:
        """Single Halley step using pre-computed loop-invariant quantities.

        Hoisting ``ds``, ``dk``, ``sqrt_t``, ``log_sk``, and ``carry_drift_t``
        outside the per-element iteration eliminates 5 redundant array ops per
        Halley step.
        """
        vol_t = max(sigma * sqrt_t, 1e-32)
        d1 = (log_sk + carry_drift_t + 0.5 * sigma * sigma * t) / vol_t
        d2 = d1 - sigma * sqrt_t
        nd1 = _norm_cdf(d1)
        nd2 = _norm_cdf(d2)
        call = ds * nd1 - dk * nd2
        put = dk * (1.0 - nd2) - ds * (1.0 - nd1)
        px = call if is_call else put
        raw_vega = ds * _norm_pdf(d1) * sqrt_t
        residual = px - price
        safe_vega = raw_vega if raw_vega > 1e-14 else math.inf
        newton_step = residual / safe_vega
        safe_sigma = sigma if sigma > 1e-8 else math.inf
        hd = 1.0 - newton_step * d1 * d2 / safe_sigma
        if abs(hd) <= 0.05:
            hd = math.copysign(0.05, hd + 1e-15)
        return min(max(sigma - newton_step / hd, IV_LO), IV_HI)

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
    # IV solver kernel — Halley × 8 + bisection fallback, all per-element
    #
    # The entire Halley loop and bisection fallback execute inside the
    # parallel prange, so only one Python→numba dispatch occurs per batch.
    # ``q`` is pre-adjusted by the Python wrapper (see above).
    # ``valid`` marks elements with t > 0; expired options get 0.
    # NaN masking (below-intrinsic, zero-price) is applied in the wrapper.
    # ------------------------------------------------------------------

    @numba.njit(parallel=True, cache=True)
    def _iv_kernel(
        is_call: np.ndarray,
        price: np.ndarray,
        s: np.ndarray,
        k: np.ndarray,
        t: np.ndarray,
        r: np.ndarray,
        q: np.ndarray,
        valid: np.ndarray,
    ) -> np.ndarray:
        n = s.shape[0]
        out = np.empty(n, np.float64)
        SQRT2PI = math.sqrt(2.0 * math.pi)

        for i in numba.prange(n):  # type: ignore[attr-defined]
            if not valid[i]:
                out[i] = 0.0
                continue

            ic = is_call[i]
            ds = s[i] * math.exp(-q[i] * t[i])
            dk = k[i] * math.exp(-r[i] * t[i])
            sq_t = math.sqrt(max(t[i], 1e-32))
            lsk = math.log(max(s[i], 1e-32) / max(k[i], 1e-32))
            cdt = (r[i] - q[i]) * t[i]  # carry-drift term: (r-q)·t

            # Initial guess — Brenner-Subrahmanyam ATM approximation, floor 0.30
            sq_t_init = math.sqrt(max(t[i], 1e-8))
            approx = price[i] / max(s[i] * sq_t_init, 1e-12) * SQRT2PI
            sig = min(max(approx, 0.30), 5.0)

            # Halley iterations
            for _ in range(8):
                sig = _halley_step(sig, price[i], ic, ds, dk, sq_t, lsk, cdt, t[i])

            # Convergence check
            vol_t = max(sig * sq_t, 1e-32)
            d1 = (lsk + cdt + 0.5 * sig * sig * t[i]) / vol_t
            d2 = d1 - sig * sq_t
            nd1 = _norm_cdf(d1)
            nd2 = _norm_cdf(d2)
            call = ds * nd1 - dk * nd2
            put = dk * (1.0 - nd2) - ds * (1.0 - nd1)
            px = call if ic else put

            underflow = (px == 0.0) and (price[i] > 0.0)
            if abs(px - price[i]) > 1e-6 or underflow:
                # Bisection fallback — bracket [IV_LO, IV_HI], 30 iterations
                lo = IV_LO
                hi = IV_HI
                for _ in range(30):
                    mid = 0.5 * (lo + hi)
                    vol_m = max(mid * sq_t, 1e-32)
                    d1_m = (lsk + cdt + 0.5 * mid * mid * t[i]) / vol_m
                    d2_m = d1_m - mid * sq_t
                    nd1_m = _norm_cdf(d1_m)
                    nd2_m = _norm_cdf(d2_m)
                    call_m = ds * nd1_m - dk * nd2_m
                    put_m = dk * (1.0 - nd2_m) - ds * (1.0 - nd1_m)
                    px_m = call_m if ic else put_m
                    if px_m - price[i] < 0:
                        lo = mid
                    else:
                        hi = mid
                sig = 0.5 * (lo + hi)

            out[i] = sig

        return out

    _kernels["price"] = _price_kernel
    _kernels["greeks"] = _greeks_kernel
    _kernels["iv"] = _iv_kernel


# ---------------------------------------------------------------------------
# Intrinsic-value helper (NumPy; called once in the Python wrapper)
# ---------------------------------------------------------------------------


def _intrinsic_vec(
    flag: np.ndarray,
    s: np.ndarray,
    k: np.ndarray,
    t: np.ndarray,
    r: np.ndarray,
    q: np.ndarray,
    model: ModelLiteral,
) -> np.ndarray:
    disc = np.exp(-r * t)
    carry = np.exp(-q * t)
    if model == "black":
        call_iv = disc * np.maximum(0.0, s - k)
        put_iv = disc * np.maximum(0.0, k - s)
    elif model == "black_scholes":
        call_iv = np.maximum(0.0, s - k * disc)
        put_iv = np.maximum(0.0, k * disc - s)
    else:  # black_scholes_merton
        call_iv = np.maximum(0.0, s * carry - k * disc)
        put_iv = np.maximum(0.0, k * disc - s * carry)
    return np.where(flag == "c", call_iv, put_iv)


# ---------------------------------------------------------------------------
# Public pricing API
# ---------------------------------------------------------------------------


def price_black(
    flag: FlagArray,
    f: Float1D,
    k: Float1D,
    t: Float1D,
    r: Float1D,
    sigma: Float1D,
) -> Float1D:
    # Black-76: q = r so carry = disc and d1 = [ln(F/K) + 0.5σ²T] / (σ√T)
    is_call = flag == "c"
    return _get_kernels()["price"](is_call, f, k, t, r, sigma, r)


def price_black_scholes(
    flag: FlagArray,
    s: Float1D,
    k: Float1D,
    t: Float1D,
    r: Float1D,
    sigma: Float1D,
) -> Float1D:
    is_call = flag == "c"
    q = np.zeros_like(r)
    return _get_kernels()["price"](is_call, s, k, t, r, sigma, q)


def price_black_scholes_merton(
    flag: FlagArray,
    s: Float1D,
    k: Float1D,
    t: Float1D,
    r: Float1D,
    sigma: Float1D,
    q: Float1D,
) -> Float1D:
    is_call = flag == "c"
    return _get_kernels()["price"](is_call, s, k, t, r, sigma, q)


# ---------------------------------------------------------------------------
# Public Greeks API
# ---------------------------------------------------------------------------


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
    # Black-76: q = r so carry = disc and d1 uses only the σ² drift term
    qv = r if (model == "black" and q is None) else (np.zeros_like(r) if q is None else q)
    is_call = flag == "c"
    delta, gamma, theta, rho, vega = _get_kernels()["greeks"](is_call, s, k, t, r, sigma, qv)
    return {
        "delta": delta,
        "gamma": gamma,
        "theta": theta,
        "rho": rho,
        "vega": vega,
    }


# ---------------------------------------------------------------------------
# Public IV API  —  Halley's method (3rd-order) + bisection fallback
# ---------------------------------------------------------------------------


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
    # Black-76: q = r so carry = disc and d1 = [ln(F/K) + 0.5σ²T] / (σ√T)
    qv = r if (model == "black" and q is None) else (np.zeros_like(r) if q is None else q)

    # Intrinsic-value check (NumPy; once before the kernel)
    intrinsic = _intrinsic_vec(flag, s, k, t, r, qv, model)
    below_intrinsic = price < intrinsic - 1e-10
    if np.any(below_intrinsic):
        handle_error("Option price is below intrinsic value.", on_error)

    valid = t > 0
    is_call = flag == "c"

    # Numba kernel: Halley × 8 + bisection fallback, all per-element in prange
    sigma = _get_kernels()["iv"](is_call, price, s, k, t, r, qv, valid)

    # NaN masking (consistent with NumPy/PyTorch/JAX backends)
    result = np.where(valid, sigma, 0.0)
    # Zero-price OTM options: sigma is undetermined (any σ gives price≈0) → NaN
    zero_price = (price <= 0.0) & valid
    result = np.where(below_intrinsic | zero_price, np.nan, result)
    return result
