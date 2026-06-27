"""Differentiable arbitrage penalty (design §7).

Because the arbitrage fields route through the namespace-generic kernels, on
the torch / jax backend every violation magnitude is a differentiable function
of the input IV tensor.  :func:`arbitrage_penalty` combines them into a single
scalar suitable as a soft-constraint term in a generator's training loss —
the reusable, tested replacement for the inline penalties that VolGAN /
deep-smoothing / VAE families each re-derive.

Gradients flow to ``iv`` because the penalty never leaves the input array's
namespace (no host round-trip, unlike the inference ``price_*`` backends).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._xp import get_namespace
from .arbitrage import compute_fields

if TYPE_CHECKING:
    from .grid import IVSurface

# Penalty term weights (independent of the report's SAS weights — this is the
# soft training-loss form, not the diagnostic composite).
DEFAULT_PENALTY_WEIGHTS: dict[str, float] = {
    "butterfly": 1.0,
    "calendar": 1.0,
    "vertical": 1.0,
    "bound": 1.0,
}


def arbitrage_penalty(
    iv,
    k,
    T,
    forward,
    r=0.0,
    *,
    weights: dict[str, float] | None = None,
    reduction: str = "mean",
    shared_k: bool = True,
):
    """Differentiable scalar arbitrage penalty — soft form of the §5 components.

    Drop into a generator's training loss; gradients flow back to ``iv``.

    Parameters
    ----------
    iv:
        Implied-vol grid, shape ``(Nk, Nt)`` (torch/jax tensor for autograd;
        numpy also works but yields no gradient).
    k:
        Log-moneyness, shape ``(Nk,)`` or ``(Nk, Nt)``.
    T:
        Maturities, shape ``(Nt,)``.
    forward, r:
        Forward curve and rate(s), scalar or shape ``(Nt,)``.
    weights:
        Per-condition weights; defaults to :data:`DEFAULT_PENALTY_WEIGHTS`.
    reduction:
        ``"mean"`` (default) or ``"sum"`` over each field's violation
        magnitudes.
    shared_k:
        See :func:`~fast_vollib.surface.arbitrage.compute_fields`.

    Returns
    -------
    Scalar penalty in the namespace of ``iv`` (≥ 0; 0 for an arbitrage-free
    surface).  Differentiable w.r.t. ``iv``.
    """
    xp = get_namespace(iv)
    iv2 = iv
    k2d, T2d, forward2d, discount2d = _broadcast(iv2, k, T, forward, r, xp)
    w = iv2 * iv2 * T2d
    fields = compute_fields(k2d, w, forward2d, discount2d, xp, shared_k=shared_k)

    wts = weights or DEFAULT_PENALTY_WEIGHTS
    reduce = _reducer(xp, reduction)
    terms = {
        "butterfly": reduce(fields.bfly_mag),
        "calendar": reduce(fields.cal_depth),
        "vertical": reduce(fields.vert_mag),
        "bound": reduce(fields.bound_mag),
    }
    total = None
    for name, value in terms.items():
        contrib = wts.get(name, 0.0) * value
        total = contrib if total is None else total + contrib
    return total


def penalty_from_surface(
    surf: "IVSurface", *, weights: dict[str, float] | None = None, reduction: str = "mean"
):
    """Convenience wrapper computing :func:`arbitrage_penalty` from an IVSurface."""
    return arbitrage_penalty(
        surf.iv,
        surf.k,
        surf.T,
        surf.forward,
        surf.r,
        weights=weights,
        reduction=reduction,
        shared_k=surf.shared_k,
    )


def _broadcast(iv, k, T, forward, r, xp):
    """Broadcast penalty inputs to ``(Nk, Nt)`` in the iv namespace (grad-safe)."""
    Nk, Nt = iv.shape
    k = xp.asarray(k, like=iv)
    T = xp.asarray(T, like=iv)
    forward = xp.asarray(forward, like=iv)
    r = xp.asarray(r, like=iv)
    # NaN-free (Nk, Nt) broadcasting basis: geometry must not depend on iv
    # (an unquoted NaN node would otherwise contaminate T/forward/k).
    ones = xp.zeros((Nk, Nt), like=iv)
    T2d = (T[None, :] if getattr(T, "ndim", 0) == 1 else T) + ones
    forward2d = (forward[None, :] if getattr(forward, "ndim", 0) == 1 else forward) + ones
    r2d = (r[None, :] if getattr(r, "ndim", 0) == 1 else r) + ones
    k2d = (k[:, None] if getattr(k, "ndim", 1) == 1 else k) + ones
    discount2d = xp.exp(-r2d * T2d)
    return k2d, T2d, forward2d, discount2d


def _reducer(xp, reduction: str):
    if reduction == "sum":
        return lambda x: xp.sum(_nan_to_zero(x, xp))
    if reduction == "mean":
        return lambda x: xp.sum(_nan_to_zero(x, xp)) / _size(x)
    raise ValueError(f"reduction must be 'mean' or 'sum'; got {reduction!r}.")


def _nan_to_zero(x, xp):
    """Replace NaN (unquoted nodes) with 0 so they contribute no penalty/grad."""
    return xp.where(xp.isnan(x), xp.asarray(0.0, like=x), x)


def _size(x) -> int:
    n = 1
    for s in x.shape:
        n *= int(s)
    return max(n, 1)
