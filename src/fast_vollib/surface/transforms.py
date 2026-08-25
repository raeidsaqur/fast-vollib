"""Coordinate / representation transforms for IV surfaces.

All functions are **namespace-generic**: they take an :class:`ArrayNS` (``xp``)
and operate on whatever backend's arrays it wraps, so the identical code path
serves the numpy report and the torch/jax differentiable penalty.

Internal convention (design §3): forward log-moneyness ``k = log(K / F)`` and
year-fraction maturity ``T``.  Total (implied) variance is ``w = σ² · T``.
The *normalized undiscounted* Black call (per unit forward) is

    c̃/F = N(d₁) − e^k · N(d₂),     d₁ = (−k + w/2)/√w,  d₂ = d₁ − √w = (−k − w/2)/√w

so the undiscounted forward call is ``c̃ = F·(N(d₁) − e^k N(d₂)) = F N(d₁) − K N(d₂)``
and the discounted call is ``C = e^{−rT} c̃``.  These are the coordinates in
which the no-arbitrage conditions of Gatheral–Jacquier (2014) and the
price-space inequalities of Davis–Hobson (2007) are stated.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .._array_api import ArrayNS

# Floor for √w / w to keep d₁,d₂ finite as w → 0 (mirrors the 1e-32 idiom used
# throughout the numpy backend). w = 0 corresponds to T = 0 or σ = 0 where the
# call collapses to its intrinsic value; interior arbitrage analysis assumes w > 0.
_W_FLOOR = 1e-300
_SQRTW_FLOOR = 1e-150


def iv_to_total_variance(iv, T, xp: "ArrayNS"):
    """``w = σ² · T``.  ``iv`` shape ``(Nk, Nt)``, ``T`` broadcastable to it."""
    return iv * iv * T


def total_variance_to_iv(w, T, xp: "ArrayNS"):
    """``σ = √(w / T)``.  Inverse of :func:`iv_to_total_variance`."""
    return xp.sqrt(w / T)


def strikes_from_logmoneyness(k, forward, xp: "ArrayNS"):
    """``K = F · e^k``.  ``k`` shape ``(Nk, Nt)``, ``forward`` broadcastable."""
    return forward * xp.exp(k)


def normalized_black_call(k, w, xp: "ArrayNS"):
    """Undiscounted Black call per unit forward, ``c̃/F = N(d₁) − e^k N(d₂)``.

    Parameters
    ----------
    k, w:
        Log-moneyness and total variance, same shape ``(Nk, Nt)``.

    Returns
    -------
    Normalized call in ``[0, 1)``; multiply by the forward to recover ``c̃``.
    """
    sqrt_w = xp.sqrt(xp.maximum(w, xp.asarray(_W_FLOOR, like=w)))
    sqrt_w = xp.maximum(sqrt_w, xp.asarray(_SQRTW_FLOOR, like=w))
    d1 = (-k + 0.5 * w) / sqrt_w
    d2 = d1 - sqrt_w
    return xp.normcdf(d1) - xp.exp(k) * xp.normcdf(d2)


def undiscounted_call(k, w, forward, xp: "ArrayNS"):
    """Undiscounted forward call ``c̃ = F·(N(d₁) − e^k N(d₂))``."""
    return forward * normalized_black_call(k, w, xp)


def discounted_call(k, w, forward, discount, xp: "ArrayNS"):
    """Discounted call ``C = e^{−rT} · c̃``.  ``discount`` is ``e^{−rT}``."""
    return discount * undiscounted_call(k, w, forward, xp)
