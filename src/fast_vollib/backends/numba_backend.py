"""Numba JIT-compiled CPU backend for fast-vollib.

Vectorised kernels are compiled to native code via ``@numba.njit(parallel=True)``
with element-wise ``prange`` loops.  The scalar Halley steps and their bracket safeguard run
entirely inside the compiled code, so the solver incurs only a single Python
call per batch regardless of batch size.

Kernels live at module scope in ``_numba_kernels`` (imported on first use) and are compiled on
first call; ``cache=True`` serves them from numba's on-disk cache in later processes.

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

from typing import TYPE_CHECKING, Any

import numpy as np

from ..types import ModelLiteral, OnErrorLiteral
from . import _iv_common as _ivc

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
# Lazy kernel compilation
# ---------------------------------------------------------------------------

_kernels: dict[str, Any] = {}


def _get_kernels() -> dict[str, Any]:
    """Return the compiled kernel bundle, compiling it on the very first call."""
    if _kernels:
        return _kernels
    _build_kernels()
    return _kernels


def _build_kernels() -> None:
    """Load the numba kernels (compiled on first call; served from the on-disk cache after)."""
    from . import _numba_kernels as kernels

    _kernels["price"] = kernels._price_kernel
    _kernels["greeks"] = kernels._greeks_kernel
    _kernels["iv"] = kernels._iv_kernel


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
# Public IV API  —  safeguarded Halley on the out-of-the-money price
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
    shape, price, s, k, t, r, flag, q = _ivc.flatten_inputs(price, s, k, t, r, flag, q)
    qv = q if (q is not None and model == "black_scholes_merton") else np.zeros_like(r)
    is_call = flag == "c"
    model_id = {"black": 0, "black_scholes": 1}.get(model, 2)
    sigma, code = _get_kernels()["iv"](is_call, price, s, k, t, r, qv, model_id)
    _ivc.report_row_errors(_ivc.count_codes(code), price.size, on_error)
    return sigma.reshape(shape)
