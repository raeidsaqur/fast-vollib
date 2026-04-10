"""
Jäckel torch backend — machine-precision implied volatility.

Experiment I-5: Native torch Jäckel Householder(3)×2 implementation using
`torch.special.erfcx` and `torch.special.ndtri`.  The Householder loop is
wrapped with `torch.compile(dynamic=True)` so that erfcx evaluations and
branch dispatches fuse into a single CUDA kernel.

Accuracy:  max relative error ~ 2e-11  (same guarantee as numpy Jäckel backend)
Speed:     see benchmark; target ≤ 0.636ms/100k on H100 GPU.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch

from ..types import ModelLiteral
from ..utils.validation import handle_error

# ---------------------------------------------------------------------------
# Availability
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


def to_native(values: np.ndarray):
    import torch

    return torch.as_tensor(values, dtype=torch.float64, device=_device())


def from_native(values) -> np.ndarray:
    import torch

    if isinstance(values, torch.Tensor):
        return values.detach().cpu().numpy()
    return np.asarray(values)


# ---------------------------------------------------------------------------
# Math constants
# ---------------------------------------------------------------------------

_ONE_OVER_SQRT2 = 1.0 / math.sqrt(2.0)
_ONE_OVER_SQRT2PI = 1.0 / math.sqrt(2.0 * math.pi)
_DBL_MIN = 2.2250738585072014e-308


# ---------------------------------------------------------------------------
# Core math — torch versions of jackel_iv.py helpers
# ---------------------------------------------------------------------------


def _normalised_black_and_vega_t(x, s):
    """Fused normalised Black call + vega in one pass (torch).

    Shares the exp(-0.5*(h²+t²)) factor between b and bp to save one exp call.
    """
    import torch

    tiny = torch.finfo(x.dtype).tiny
    s_safe = s.clamp(min=tiny)
    h = x / s_safe
    t = 0.5 * s_safe
    factor = torch.exp(-0.5 * (h * h + t * t))
    diff = torch.special.erfcx(-_ONE_OVER_SQRT2 * (h + t)) - torch.special.erfcx(
        -_ONE_OVER_SQRT2 * (h - t)
    )
    b = (0.5 * factor * diff).abs().clamp(min=0.0)
    bp = factor * _ONE_OVER_SQRT2PI
    return b, bp


def _boundary_t(x, b_max):
    """Compute all 3 boundary (s, b, v_safe) pairs for Jäckel IV (torch)."""
    import torch

    tiny = torch.finfo(x.dtype).tiny
    s_c = (2.0 * x.abs()).sqrt()
    b_c, v_c = _normalised_black_and_vega_t(x, s_c)
    v_c_safe = v_c.clamp(min=tiny)

    s_l = s_c - b_c / v_c_safe
    b_l, v_l = _normalised_black_and_vega_t(x, s_l)
    v_l_safe = v_l.clamp(min=tiny)

    s_h = torch.where(v_c > _DBL_MIN, s_c + (b_max - b_c) / v_c_safe, s_c)
    b_h, v_h = _normalised_black_and_vega_t(x, s_h)
    v_h_safe = v_h.clamp(min=tiny)

    return s_c, b_c, v_c_safe, s_l, b_l, v_l_safe, s_h, b_h, v_h_safe


def _hermite_guess_t(beta, s_l, b_l, v_l_safe, s_c, b_c, v_c_safe, s_h, b_h, v_h_safe):
    """Cubic Hermite initial guess (torch).

    Zones 2/3: cubic Hermite in (b, σ) space with slopes 1/v.
    Zones 1/4: use boundary value as starting point (Householder converges from it).
    """
    import torch

    tiny = torch.finfo(beta.dtype).tiny

    # Zone 2: interpolate between (b_l, s_l) and (b_c, s_c)
    h2 = (b_c - b_l).abs().clamp(min=tiny)
    t2 = ((beta - b_l) / h2).clamp(0.0, 1.0)
    t2s, t2c = t2 * t2, t2**3
    s_z2 = (
        (2.0 * t2c - 3.0 * t2s + 1.0) * s_l
        + (t2c - 2.0 * t2s + t2) * h2 / v_l_safe
        + (-2.0 * t2c + 3.0 * t2s) * s_c
        + (t2c - t2s) * h2 / v_c_safe
    )

    # Zone 3: interpolate between (b_c, s_c) and (b_h, s_h)
    h3 = (b_h - b_c).abs().clamp(min=tiny)
    t3 = ((beta - b_c) / h3).clamp(0.0, 1.0)
    t3s, t3c = t3 * t3, t3**3
    s_z3 = (
        (2.0 * t3c - 3.0 * t3s + 1.0) * s_c
        + (t3c - 2.0 * t3s + t3) * h3 / v_c_safe
        + (-2.0 * t3c + 3.0 * t3s) * s_h
        + (t3c - t3s) * h3 / v_h_safe
    )

    z2_mask = (beta >= b_l) & (beta < b_c)
    s = torch.where(z2_mask, s_z2, s_z3)

    # Zones 1 and 4: use boundary values as starting point for Householder
    s = torch.where(beta < b_l, s_l, s)
    s = torch.where(beta > b_h, s_h, s)

    return s.clamp(min=tiny)


def _householder_loop_t(s_init, beta, x, use_lower, use_upper, b_max, n_iters: int = 3):
    """Householder(3) × n_iters with 3-branch objective dispatch (torch).

    Hot path — torch.compile fuses erfcx, exp, log, and where into one CUDA kernel.
    """
    import torch

    tiny = torch.finfo(s_init.dtype).tiny
    s = s_init.clone()

    for _ in range(n_iters):
        s_safe = s.clamp(min=tiny)

        # Fused normalised Black + vega (shared exp factor)
        h = x / s_safe
        t = 0.5 * s_safe
        factor = torch.exp(-0.5 * (h * h + t * t))
        diff = torch.special.erfcx(-_ONE_OVER_SQRT2 * (h + t)) - torch.special.erfcx(
            -_ONE_OVER_SQRT2 * (h - t)
        )
        b = (0.5 * factor * diff).abs().clamp(min=0.0)
        bp = factor * _ONE_OVER_SQRT2PI

        bp_safe = bp.clamp(min=tiny)
        b_safe = b.clamp(min=tiny)

        # Common Householder second/third derivative terms
        x_over_s = x / s_safe
        xs2 = x_over_s / s_safe
        b_halley = x_over_s * x_over_s / s_safe - s_safe * 0.25
        b_hh3 = b_halley * b_halley - 3.0 * xs2 * xs2 - 0.25

        # ── Lower branch (log-space objective) ─────────────────────────────
        ln_b = torch.log(b_safe)
        ln_beta = torch.log(beta.clamp(min=tiny))
        bpob = bp / b_safe
        # Guard: ln_b_safe avoids divide-by-zero when ln_b ≈ 0
        ln_b_safe = torch.where(ln_b.abs() > 0.0, ln_b, torch.full_like(ln_b, tiny))
        newton_lo = (ln_beta - ln_b) * ln_b / ln_beta / bpob
        halley_lo = b_halley - bpob * (1.0 + 2.0 / ln_b_safe)
        hh3_lo = (
            b_hh3
            + 2.0 * bpob * bpob * (1.0 + 3.0 / ln_b_safe * (1.0 + 1.0 / ln_b_safe))
            - 3.0 * b_halley * bpob * (1.0 + 2.0 / ln_b_safe)
        )

        # ── Upper branch (complementary log-space objective) ───────────────
        bm_b = (b_max - b).clamp(min=tiny)
        bm_bt = (b_max - beta).clamp(min=tiny)
        g = torch.log(bm_bt / bm_b)
        gp = bp / bm_b
        newton_up = -g / gp
        halley_up = b_halley + gp
        hh3_up = b_hh3 + gp * (2.0 * gp + 3.0 * b_halley)

        # ── Middle branch (linear objective) ──────────────────────────────
        newton_mid = (beta - b) / bp_safe
        halley_mid = b_halley
        hh3_mid = b_hh3

        # Dispatch
        newton = torch.where(use_lower, newton_lo, torch.where(use_upper, newton_up, newton_mid))
        halley = torch.where(use_lower, halley_lo, torch.where(use_upper, halley_up, halley_mid))
        hh3 = torch.where(use_lower, hh3_lo, torch.where(use_upper, hh3_up, hh3_mid))

        # Householder(3) rational correction
        hf = (1.0 + 0.5 * halley * newton) / (1.0 + newton * (halley + hh3 * newton / 6.0))
        ds = torch.maximum(-0.5 * s_safe, newton * hf)
        s = s_safe + ds

    return s


# torch.compile wrapper with eager fallback
import warnings as _warnings


def _make_compiled(fn):
    import torch

    compiled = torch.compile(fn, dynamic=True)
    _use_eager: list[bool] = [False]

    def _call(*args, **kwargs):
        if _use_eager[0]:
            return fn(*args, **kwargs)
        try:
            return compiled(*args, **kwargs)
        except Exception as exc:
            if not _use_eager[0]:
                _use_eager[0] = True
                _warnings.warn(
                    f"torch.compile failed ({exc!r}); switching to eager mode.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            return fn(*args, **kwargs)

    return _call


_compiled_householder_loop = _make_compiled(_householder_loop_t)


# ---------------------------------------------------------------------------
# Public: jackel_iv_black (tensor in, tensor out) — native torch
# ---------------------------------------------------------------------------


def jackel_iv_black_torch(
    price: "torch.Tensor",
    F: float,
    K: "torch.Tensor",
    T: float,
    is_call: "torch.Tensor | bool" = True,
) -> "torch.Tensor":
    """Jäckel IV for Black-76 options — native torch implementation.

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
    import torch  # noqa: F811

    tiny = torch.finfo(price.dtype).tiny

    # Broadcast is_call
    if isinstance(is_call, bool):
        is_call_t = torch.full_like(price, is_call, dtype=torch.bool)
    else:
        is_call_t = is_call

    sqrt_FK = (F * K).sqrt()
    x = torch.log(torch.tensor(F, dtype=price.dtype, device=price.device) / K)
    sqrt_T = math.sqrt(max(T, 0.0))

    q = torch.where(is_call_t, torch.ones_like(price), -torch.ones_like(price))
    intrinsic = (q * (F - K)).clamp(min=0.0).abs()
    itm = (q * x) > 0.0
    price_red = torch.where(itm, (price - intrinsic).clamp(min=0.0).abs(), price)
    x_red = torch.where(x > 0.0, -x, x)
    beta = price_red / sqrt_FK.clamp(min=tiny)
    b_max = (0.5 * x_red).exp()

    # Boundary
    s_c, b_c, v_c_safe, s_l, b_l, v_l_safe, s_h, b_h, v_h_safe = _boundary_t(x_red, b_max)
    b_tilde_h = torch.maximum(b_h, 0.5 * b_max)

    # Initial guess
    s_init = _hermite_guess_t(beta, s_l, b_l, v_l_safe, s_c, b_c, v_c_safe, s_h, b_h, v_h_safe)

    # Householder(3) × 2 — compiled
    use_lower = beta < b_l
    use_upper = beta > b_tilde_h
    sigma_hat = _compiled_householder_loop(s_init, beta, x_red, use_lower, use_upper, b_max)

    # Denormalize
    sigma = sigma_hat / sqrt_T if sqrt_T > 0.0 else torch.zeros_like(sigma_hat)

    # NaN guards
    bad = (price <= 0.0) | (T <= 0.0) | (F <= 0.0) | (K <= 0.0) | (sigma_hat <= 0.0)
    return torch.where(bad, torch.full_like(sigma, float("nan")), sigma)


# ---------------------------------------------------------------------------
# Implied volatility — model pipeline (public API)
# ---------------------------------------------------------------------------


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
    """Machine-precision IV using Jäckel "Let's Be Rational" (2016) — torch backend.

    Uses native torch ops + torch.compile(dynamic=True) on GPU.

    Parameters
    ----------
    model   : "black" | "black_scholes" | "black_scholes_merton"
    price   : option market price (discounted)
    s       : spot (or forward for Black-76)
    k       : strike
    t       : time to expiry (years)
    r       : risk-free rate
    flag    : "c" for call, "p" for put (array of str)
    q       : dividend yield (Black-Scholes-Merton only)
    on_error: "warn" | "raise" | "ignore"

    Returns
    -------
    sigma : annualised implied volatility (NaN for degenerate inputs)
    """
    import torch

    price_a = np.asarray(price, dtype=float)
    s_a = np.asarray(s, dtype=float)
    k_a = np.asarray(k, dtype=float)
    t_a = np.asarray(t, dtype=float)
    r_a = np.asarray(r, dtype=float)

    q_a = (
        r_a
        if (model == "black" and q is None)
        else (np.zeros_like(r_a) if q is None else np.asarray(q, dtype=float))
    )

    is_call = flag == "c"
    valid = t_a > 0

    disc_spot = s_a * np.exp(-q_a * t_a)
    disc_strike = k_a * np.exp(-r_a * t_a)
    intrinsic_np = np.where(
        is_call,
        np.maximum(disc_spot - disc_strike, 0.0),
        np.maximum(disc_strike - disc_spot, 0.0),
    )
    below_intrinsic = price_a < intrinsic_np - 1e-10
    if np.any(below_intrinsic):
        handle_error("Option price is below intrinsic value.", on_error)

    disc_factor = np.exp(r_a * t_a)
    undiscounted_price = price_a * disc_factor

    if model == "black":
        F_fwd = s_a
    elif model == "black_scholes":
        F_fwd = s_a * disc_factor
    else:
        F_fwd = s_a * np.exp((r_a - q_a) * t_a)

    # Handle scalar vs. array T/F (batch them by unique T × F combinations for
    # the common single-expiry case to keep the torch call simple).
    # For generality, process all as arrays by iterating unique (T, F) values.
    # Common fast path: all T same, all F same.
    T_unique = np.unique(t_a[valid]) if np.any(valid) else np.array([])
    F_unique = np.unique(F_fwd[valid]) if np.any(valid) else np.array([])

    dev = _device()
    sigma = np.zeros_like(price_a)

    if len(T_unique) == 1 and len(F_unique) == 1:
        # Single expiry / single forward — use fast path
        T_val = float(T_unique[0])
        F_val = float(F_unique[0])
        mask = valid
        idx = np.where(mask)[0]
        price_t = torch.as_tensor(undiscounted_price[idx], dtype=torch.float64, device=dev)
        k_t = torch.as_tensor(k_a[idx], dtype=torch.float64, device=dev)
        ic_t = torch.as_tensor(is_call[idx], dtype=torch.bool, device=dev)
        sigma_t = jackel_iv_black_torch(price_t, F_val, k_t, T_val, ic_t)
        sigma[idx] = sigma_t.detach().cpu().numpy()
    else:
        # General path: fallback per-row (handles mixed T, F, model types)
        from .jackel_iv import jackel_iv_black as _jackel_iv_black_np

        sigma = _jackel_iv_black_np(undiscounted_price, F_fwd, k_a, t_a, is_call)

    result = np.where(valid, sigma, 0.0)
    result = np.where(below_intrinsic, np.nan, result)
    return result
