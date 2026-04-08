"""
Jäckel torch backend — machine-precision implied volatility.

Experiment I-5: Port Jäckel Householder(3)×2 to torch using
`torch.special.erfcx` and `torch.special.ndtri`, then wrap the
Householder loop with `torch.compile(dynamic=True)` so the erfcx
calls fuse into a single CUDA kernel.

Current state (stub):
    Delegates model→forward conversion to numpy, then calls the
    numpy `jackel_iv_black` for the IV solve.  Functionally correct
    but provides no GPU acceleration.

Target (I-5):
    Port `normalised_black_call`, `_jackel_initial_guess`, and the
    Householder loop entirely to torch tensors, then compile.
    Compare throughput vs `backends/torch_backend.py` Halley×8 on GPU.

Accuracy:  max relative error ~ 2e-11 (inherited from numpy backend)
Speed:     stub — same as numpy (~108ms/100k).  Target: ≤ 0.636ms on GPU.
"""

from __future__ import annotations

import numpy as np

from ..types import ModelLiteral
from ..utils.validation import handle_error
from .jackel_iv import jackel_iv_black as _jackel_iv_black_np

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
# TODO I-5: native torch Jäckel core
# ---------------------------------------------------------------------------
# Port each of these from jackel_iv.py using torch ops:
#
#   def _normalised_black_call_t(x, s):
#       """torch version using torch.special.erfcx."""
#       import torch
#       h = x / torch.clamp(s, min=1e-300)
#       t = 0.5 * s
#       # Region 2 (small s) — erfcx avoids catastrophic cancellation
#       b = 0.5 * torch.exp(-0.5 * (h*h + t*t)) * (
#           torch.special.erfcx(-ONE_OVER_SQRT2 * (h + t))
#           - torch.special.erfcx(-ONE_OVER_SQRT2 * (h - t))
#       )
#       return torch.clamp(b, min=0.0)
#
#   def _jackel_iv_torch(beta, x):
#       """2-iteration Householder(3) in torch — torch.compile-ready."""
#       ...
#
# Then compile:
#   _compiled_jackel_iv = torch.compile(_jackel_iv_torch, dynamic=True)


# ---------------------------------------------------------------------------
# Implied volatility — stub (numpy fallback)
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
    """Machine-precision IV using Jäckel "Let's Be Rational" (2016) — torch stub.

    Currently delegates to the numpy Jäckel backend.
    Experiment I-5 will replace this with native torch ops + torch.compile.

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
    intrinsic = np.where(
        is_call,
        np.maximum(disc_spot - disc_strike, 0.0),
        np.maximum(disc_strike - disc_spot, 0.0),
    )
    below_intrinsic = price_a < intrinsic - 1e-10
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

    # TODO I-5: replace with native torch call:
    #   sigma = _compiled_jackel_iv(undiscounted_price_t, F_fwd_t, k_a_t, t_a_t, is_call_t)
    sigma = _jackel_iv_black_np(undiscounted_price, F_fwd, k_a, t_a, is_call)

    result = np.where(valid, sigma, 0.0)
    result = np.where(below_intrinsic, np.nan, result)
    return result
