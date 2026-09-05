r"""The quadrature shell, with the model held at arm's length.

Every Fourier pricer in this package is the same three things: a half-line
quadrature, one of two inversion formulas, and the bookkeeping that broadcasts
inputs, prices one maturity at a time, and converts a call to a put.  Only the
characteristic function differs between models -- so only the characteristic
function is passed in.

What is *not* here is any notion of a model.  ``fourier_price`` takes a callable
``(u, maturity) -> phi`` and a callable ``maturity -> scale``, and has no way to
ask what produced them.  That is deliberate: a shell that could recognize its
caller would eventually special-case one.

The transform this shell wants
------------------------------
:math:`\phi^T`, the characteristic function of :math:`\ln(S_T / F_T)` **under
the T-forward measure**, so that

.. math:: \phi^T(0) = 1, \qquad \phi^T(-i) = 1.

The second identity is the martingale property of the forward, and it is what
makes the Gatheral share-measure shift :math:`\phi_1(u) = \phi^T(u - i)` need no
normalization.  Both are asserted by test for every model that reaches this
module; a transform failing either produces plausible prices that are wrong,
which is the failure mode this shell cannot detect on its own.

Discounting is a separate, explicit multiplication.  The quadrature prices the
*undiscounted forward* option, and a caller can therefore see which market input
entered where.

Compatibility
-------------
The shared quadrature retains the Heston operation order. Numerical reference
tests cover the ordinary parameter grid with bounded platform rounding, and
model-reduction tests compare the pricing routes in the same environment.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Callable, Protocol, runtime_checkable

import numpy as np

__all__ = [
    "FORMULATIONS",
    "ForwardMeasureTransform",
    "fourier_price",
    "gatheral_call",
    "half_line",
    "lewis_call",
]

#: The two Fourier formulations.  ``'lewis'`` is one integral whose integrand is
#: regular at the origin; ``'gatheral'`` is the two-probability decomposition,
#: whose integrand has a removable singularity there and is kept as an
#: independent cross-check rather than as the working route.
FORMULATIONS = ("lewis", "gatheral")


@runtime_checkable
class ForwardMeasureTransform(Protocol):
    """A model that can supply everything a T-forward inversion needs.

    Three methods, because under stochastic rates the discount factor and the
    forward are *outputs of the model* rather than market inputs a caller can
    hand over separately. Under constant rates they are inputs, and the pricers
    that assume constant rates take them as arguments instead of implementing
    this.

    Notes
    -----
    ``isinstance`` against this protocol checks that the three members exist,
    not that they satisfy the two identities above.
    """

    def discount_factor(self, maturity: float) -> float:
        """``P(0, maturity)``."""
        ...  # pragma: no cover - protocol declaration

    def forward(self, maturity: float, *, spot: float) -> float:
        """The T-forward of the spot."""
        ...  # pragma: no cover - protocol declaration

    def characteristic_function(self, u: Any, *, maturity: float) -> np.ndarray:
        """:math:`\\phi^T(u)`, normalized so ``phi(0) == phi(-i) == 1``."""
        ...  # pragma: no cover - protocol declaration


@lru_cache(maxsize=8)
def _gauss_legendre(n_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre nodes and weights on ``(0, 1)``, cached by node count."""
    from scipy.special import roots_legendre

    nodes, weights = roots_legendre(n_nodes)
    return 0.5 * (nodes + 1.0), 0.5 * weights


def half_line(n_nodes: int, scale: float) -> tuple[np.ndarray, np.ndarray]:
    """Quadrature nodes and weights for ``int_0^inf f(u) du`` via ``u = c x/(1 - x)``.

    The substitution maps the half-line to a unit interval; its Jacobian
    ``c/(1 - x)^2`` is folded into the weights, so a caller integrates by a plain
    weighted sum of ``f(u)``.

    ``c`` matters, and getting it wrong is the whole difficulty of pricing a
    short-dated option this way.  The integrand's width is set by the total
    variance: it decays roughly like ``exp(-w u^2 / 2)`` with ``w`` the total
    variance to maturity, so the range that carries the integral is
    ``u ~ 1/sqrt(w)``.  A fixed ``c`` that resolves a one-year option puts almost
    every node where a one-week option's integrand has already decayed, and the
    price comes back wrong in the second decimal place.  Scaling ``c`` with
    ``1/sqrt(w)`` keeps the node density matched to the integrand at every
    maturity. Including jump variance is a scale heuristic, not a Gaussian
    tail claim: a finite-activity jump characteristic function need not decay
    to zero at high frequency.
    """
    x, w = _gauss_legendre(n_nodes)
    u = scale * x / (1.0 - x)
    return u, scale * w / (1.0 - x) ** 2


def lewis_call(
    F: np.ndarray,
    K: np.ndarray,
    T: float,
    characteristic_function: Callable[..., np.ndarray],
    n_nodes: int,
    scale: float,
) -> np.ndarray:
    """Lewis's single-integral undiscounted forward call.

    ``c = F - sqrt(F K)/pi * int_0^inf Re[e^{-i u k} phi(u - i/2)] / (u^2 + 1/4) du``
    with ``k = log(K/F)``.  The integrand is regular at ``u = 0`` -- the pole of
    the two-probability form is what the ``i/2`` shift removes -- so a fixed node
    set resolves it without special handling near the origin.

    *This returns a price with absolute accuracy, not relative accuracy.*  The
    expression is ``F`` minus a quantity that approaches ``F`` in the wings, so a
    deep option's value is the difference of two nearly equal numbers and comes
    back with an error near ``1e-13 F`` however many nodes are used.  That is
    fine for a price and fatal for the implied volatility inverted from it, which
    is why :class:`~fast_vollib.surface.fitting.heston.HestonIVSurface` declines
    points whose vega is too small to carry that error.

    Rearranging to compute the time value directly does not help, and the reason
    is worth recording: subtracting the ``phi = 1`` identity leaves an integrand
    that tends to ``-1/(u^2 + 1/4)``, which decays algebraically rather than
    exponentially, and Gauss-Legendre on the mapped half-line resolves it far
    worse than it resolves the original.  The cancellation moves from the
    subtraction into the quadrature and gets worse, measurably.
    """
    u, w = half_line(n_nodes, scale)
    phi = characteristic_function(u - 0.5j, maturity=T)
    k = np.log(K / F)
    kernel = phi / (u * u + 0.25)
    integral = (np.exp(-1j * np.outer(k, u)) * kernel[None, :]).real @ w
    return F - np.sqrt(F * K) / np.pi * integral


def gatheral_call(
    F: np.ndarray,
    K: np.ndarray,
    T: float,
    characteristic_function: Callable[..., np.ndarray],
    n_nodes: int,
    scale: float,
) -> np.ndarray:
    """The two-probability undiscounted forward call ``F P_1 - K P_2``.

    ``P_j = 1/2 + (1/pi) int_0^inf Re[e^{-i u k} phi_j(u) / (i u)] du`` with
    ``phi_2 = phi`` and ``phi_1(u) = phi(u - i)`` -- the share-measure shift,
    which needs no normalization because ``phi(-i) = 1`` exactly for a forward
    that is a martingale.
    """
    u, w = half_line(n_nodes, scale)
    phi2 = characteristic_function(u, maturity=T)
    phi1 = characteristic_function(u - 1j, maturity=T)
    k = np.log(K / F)
    rotation = np.exp(-1j * np.outer(k, u))
    p1 = 0.5 + (rotation * (phi1 / (1j * u))[None, :]).real @ w / np.pi
    p2 = 0.5 + (rotation * (phi2 / (1j * u))[None, :]).real @ w / np.pi
    return F * p1 - K * p2


def fourier_price(
    *,
    forward: Any,
    strike: Any,
    maturity: Any,
    characteristic_function: Callable[..., np.ndarray],
    quadrature_scale: Callable[[float], float],
    is_call: Any = True,
    discount: Any = 1.0,
    formulation: str = "lewis",
    n_nodes: int,
) -> np.ndarray:
    """European option prices from a T-forward transform, by Fourier inversion.

    Parameters
    ----------
    forward, strike, maturity : array-like
        Broadcastable. ``maturity`` is in years and strictly positive; every
        element of ``forward`` and ``strike`` is strictly positive.
    characteristic_function : callable
        ``(u, *, maturity) -> phi``, the T-forward transform. Called with
        complex ``u`` at shifted arguments, so it must accept them.
    quadrature_scale : callable
        ``maturity -> c``, the half-line scale. The model owns this because the
        right scale depends on the model's total variance; see :func:`half_line`.
    is_call : array-like, default True
        Puts are priced from the call by parity, which is exact.
    discount : array-like, default 1.0
        ``P(0, T)``, applied after the quadrature so a caller can see which
        market input entered where.
    formulation : str
        One of :data:`FORMULATIONS`.
    n_nodes : int
        Gauss-Legendre node count. Required rather than defaulted: the right
        count depends on the model, and a shell that supplied one would be
        making a numerical claim on behalf of a model it cannot see.

    Returns
    -------
    ndarray
        Discounted option prices, broadcast to the common shape.
    """
    if formulation not in FORMULATIONS:
        raise ValueError(f"formulation must be one of {FORMULATIONS}; got {formulation!r}.")
    if n_nodes < 8:
        raise ValueError(f"n_nodes must be at least 8; got {n_nodes}.")
    F, K, T, call, df = np.broadcast_arrays(
        np.asarray(forward, dtype=np.float64),
        np.asarray(strike, dtype=np.float64),
        np.asarray(maturity, dtype=np.float64),
        np.asarray(is_call, dtype=bool),
        np.asarray(discount, dtype=np.float64),
    )
    if not bool(np.all(F > 0.0)) or not bool(np.all(K > 0.0)):
        raise ValueError("forward and strike must be strictly positive.")
    if not bool(np.all(T > 0.0)):
        raise ValueError("maturity must be strictly positive.")

    shape = F.shape
    call_undiscounted = np.empty(shape, dtype=np.float64)
    flat_F, flat_K, flat_T = F.reshape(-1), K.reshape(-1), T.reshape(-1)
    flat_out = call_undiscounted.reshape(-1)
    inversion = lewis_call if formulation == "lewis" else gatheral_call
    # One quadrature per distinct maturity: the characteristic function depends
    # on T but not on the strike, so a whole smile costs one evaluation of it.
    for value in np.unique(flat_T):
        mask = flat_T == value
        flat_out[mask] = inversion(
            flat_F[mask],
            flat_K[mask],
            float(value),
            characteristic_function,
            n_nodes,
            quadrature_scale(float(value)),
        )
    call_undiscounted = np.maximum(flat_out.reshape(shape), 0.0)
    # Put-call parity on the undiscounted forward: c - p = F - K.
    undiscounted = np.where(call, call_undiscounted, call_undiscounted - (F - K))
    return df * undiscounted
