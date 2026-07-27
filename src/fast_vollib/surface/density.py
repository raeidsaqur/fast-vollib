"""Risk-neutral density and butterfly diagnostics (design §4.2–§4.3).

Two views of the butterfly (convexity) condition, both namespace-generic:

* **Durrleman's g** in total-variance coordinates — interpretable, yields the
  risk-neutral density up to a positive factor, but needs ``w'`` and ``w''``
  (unstable at sparse wings).
* **Breeden–Litzenberger density** in strike space — the primary, robust view,
  computed from the *price-space* convexity stencil so it is numerically
  consistent with the discrete price checks of §4.1 (one estimator, not two).

Both derivative estimators use **non-uniform divided differences** (a local
parabola through three points), so ragged / non-uniform meshes are handled
exactly; they reduce to the textbook central stencils when spacing is uniform.
Derivatives are defined on *interior* nodes only (endpoints are flagged as
low-confidence wings by the caller).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .transforms import undiscounted_call

if TYPE_CHECKING:
    from ._xp import ArrayNS


def parabolic_derivatives(x, y, xp: "ArrayNS"):
    """First and second derivatives of ``y`` w.r.t. ``x`` along axis 0.

    Fits the unique parabola through each consecutive triple
    ``(x_{i-1}, x_i, x_{i+1})`` and differentiates it at ``x_i``.  This is the
    Lagrange / divided-difference stencil for non-uniform grids:

        D1 = (y_i − y_{i-1}) / h1
        D2 = ((y_{i+1} − y_i)/h2 − D1) / (h1 + h2)
        y'(x_i)  = D1 + h1 · D2
        y''(x_i) = 2 · D2

    with ``h1 = x_i − x_{i-1}``, ``h2 = x_{i+1} − x_i``.

    Parameters
    ----------
    x:
        Abscissa, shape ``(Nk,)`` or ``(Nk, Nt)`` (broadcast along axis 0).
    y:
        Ordinate, shape ``(Nk, Nt)``.

    Returns
    -------
    (first, second):
        Interior derivatives, each shape ``(Nk-2, Nt)``.
    """
    x2 = _broadcast_along0(x, y, xp)
    h1 = x2[1:-1] - x2[:-2]
    h2 = x2[2:] - x2[1:-1]
    ym1, y0, yp1 = y[:-2], y[1:-1], y[2:]
    d1 = (y0 - ym1) / h1
    d2 = ((yp1 - y0) / h2 - d1) / (h1 + h2)
    first = d1 + h1 * d2
    second = 2.0 * d2
    return first, second


def durrleman_g(k, w, xp: "ArrayNS"):
    """Durrleman's function ``g(k)`` per maturity slice (Gatheral–Jacquier 2014).

        g = (1 − k·w'/(2w))² − (w'/2)² · (1/4 + 1/w) + w''/2

    ``g(k) ≥ 0`` ⟺ the slice is butterfly-free ⟺ the implied risk-neutral
    density is non-negative.  Derivatives are taken along the moneyness axis.

    Parameters
    ----------
    k:
        Log-moneyness grid, shape ``(Nk,)`` or ``(Nk, Nt)``.
    w:
        Total variance, shape ``(Nk, Nt)``.

    Returns
    -------
    g on interior nodes, shape ``(Nk-2, Nt)``.
    """
    w_prime, w_dprime = parabolic_derivatives(k, w, xp)
    w_in = w[1:-1]
    term1 = (1.0 - k_interior(k, xp) * w_prime / (2.0 * w_in)) ** 2
    term2 = (w_prime * 0.5) ** 2 * (0.25 + 1.0 / w_in)
    return term1 - term2 + 0.5 * w_dprime


def bl_density(k, w, forward, xp: "ArrayNS"):
    """Breeden–Litzenberger risk-neutral density ``f(K) = ∂²c̃/∂K²``.

    Computed as the second divided difference of the *undiscounted* forward
    call with respect to strike, which is exactly the price-space convexity
    stencil (so it is consistent with the discrete butterfly check).  The
    ``e^{rT}`` factor relating the discounted-call second derivative to the
    density cancels against the discount, so the undiscounted form needs no
    rate input.

    Parameters
    ----------
    k, w:
        Log-moneyness and total variance, shape ``(Nk, Nt)``.
    forward:
        Forward curve, broadcastable to ``(Nk, Nt)``.

    Returns
    -------
    (density, strikes_interior):
        Density on interior strikes and the corresponding strikes, each
        shape ``(Nk-2, Nt)``.
    """
    from .transforms import strikes_from_logmoneyness

    strikes = strikes_from_logmoneyness(k, forward, xp)
    c_tilde = undiscounted_call(k, w, forward, xp)
    _, second = parabolic_derivatives(strikes, c_tilde, xp)
    return second, strikes[1:-1]


def k_interior(k, xp: "ArrayNS"):
    """Interior slice of the moneyness grid (drops both endpoints, axis 0)."""
    return _broadcast_along0(k, None, xp, interior=True)


def _broadcast_along0(x, y, xp: "ArrayNS", interior: bool = False):
    """Return ``x`` shaped to broadcast against ``y`` along axis 0.

    ``x`` may be 1-D ``(Nk,)`` (a shared moneyness axis) or already 2-D
    ``(Nk, Nt)`` (per-maturity strikes).  When ``interior`` is set, the result
    is the endpoint-dropped interior ``x[1:-1]``.
    """
    is_1d = getattr(x, "ndim", 1) == 1
    if is_1d:
        x = x[:, None]
    if interior:
        return x[1:-1]
    return x
