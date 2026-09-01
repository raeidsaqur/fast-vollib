"""Heston vanilla pricing by Fourier inversion of the characteristic function.

The Heston log-forward has a characteristic function in closed form, so a
European option price is one numerical integral rather than a simulation.  This
module is that integral, in two independent formulations, and they are here in
both forms deliberately: two derivations that agree to twelve digits is the only
evidence available that either is right, short of a reference nobody can audit.

The characteristic function is written in the **"little trap"** form -- the
branch of the square root chosen so that the complex logarithm's argument stays
in the principal branch for every maturity.  The textbook form is algebraically
identical and numerically wrong past a maturity of a year or two, where the
logarithm wraps and the price develops discontinuities in ``T``.  The two forms
differ by which of ``(beta - d)`` and ``(beta + d)`` is factored out, and the
one used here is the stable one.

*This module is host-side float64.*  Quadrature is Gauss-Legendre on a
variance-scaled ``u = c x/(1 - x)``, which maps the half-line to a unit interval
and resolves the integrand's tail with a fixed, deterministic node set.  There is
no adaptive subdivision, so two runs give bitwise identical prices and the
accuracy is a stated function of the node count rather than a tolerance the
routine chased.  Nodes come from :func:`scipy.special.roots_legendre`, whose
Newton-Ritz algorithm is linear in the node count; NumPy's ``leggauss`` solves a
companion eigenproblem instead and takes tens of seconds at the node counts a
heavy-tailed parameter set needs.

Examples
--------
>>> from fast_vollib.pricing import heston_call_price
>>> price = heston_call_price(
...     forward=100.0, strike=100.0, maturity=1.0,
...     v0=0.04, kappa=2.0, theta=0.04, vol_of_vol=0.3, rho=-0.7,
... )
>>> bool(0.0 < price < 100.0)
True
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np

__all__ = [
    "DEFAULT_QUADRATURE_NODES",
    "FORMULATIONS",
    "heston_call_price",
    "heston_characteristic_function",
    "heston_price",
]

#: Gauss-Legendre nodes used by default.  Under the variance-scaled ``x/(1 - x)``
#: substitution of :func:`_half_line`, Gauss-Legendre converges geometrically,
#: and 768 nodes hold the Lewis route to about 2e-9 of the forward over the range
#: the accompanying tests sweep: maturities from a week to thirty years, strikes
#: from 0.6 to 1.8 of the forward, and Feller ratios from 0.07 to 5.  The worst
#: case is the heavy-tailed one -- vol-of-vol 0.9 against a Feller ratio of 0.07 --
#: and ordinary parameters reach the float64 floor several hundred nodes earlier.
#: This is a measured accuracy, not a tolerance the routine chased: the error
#: falls monotonically in the node count until cancellation dominates near 1e-11.
DEFAULT_QUADRATURE_NODES = 768

#: The two Fourier formulations.  ``'lewis'`` is one integral whose integrand is
#: regular at the origin; ``'gatheral'`` is the two-probability decomposition,
#: whose integrand has a removable singularity there and is kept as an
#: independent cross-check rather than as the working route.
FORMULATIONS = ("lewis", "gatheral")


def heston_characteristic_function(
    u: Any,
    *,
    maturity: float,
    v0: float,
    kappa: float,
    theta: float,
    vol_of_vol: float,
    rho: float,
) -> np.ndarray:
    """``E[exp(i u log(F_T / F_0))]`` under the Heston risk-neutral dynamics.

    Parameters
    ----------
    u:
        Complex (or real) Fourier argument, any shape.
    maturity:
        Time to expiry in years, strictly positive.
    v0, kappa, theta, vol_of_vol, rho:
        Heston parameters; see :class:`~fast_vollib.processes.Heston`.

    Returns
    -------
    The characteristic function at ``u``, complex, shaped like ``u``.

    Notes
    -----
    Three identities hold exactly and are tested rather than assumed:
    ``phi(0) = 1``; ``phi(-u) = conj(phi(u))`` for real ``u``; and
    ``phi(-i) = E[F_T]/F_0 = 1``, which is the statement that the forward is a
    martingale and is the single most sensitive check on the algebra -- a sign
    error anywhere in the exponent breaks it.

    Examples
    --------
    >>> from fast_vollib.pricing import heston_characteristic_function
    >>> value = heston_characteristic_function(
    ...     0.0, maturity=1.0, v0=0.04, kappa=2.0, theta=0.04, vol_of_vol=0.3, rho=-0.7
    ... )
    >>> complex(value)
    (1+0j)
    """
    if maturity <= 0.0:
        raise ValueError(f"maturity must be strictly positive; got {maturity!r}.")
    if vol_of_vol <= 0.0:
        raise ValueError(f"vol_of_vol must be strictly positive; got {vol_of_vol!r}.")
    u = np.asarray(u, dtype=np.complex128)
    xi2 = vol_of_vol * vol_of_vol
    beta = kappa - rho * vol_of_vol * 1j * u
    d = np.sqrt(beta * beta + xi2 * (1j * u + u * u))
    # The stable branch: |g| < 1, so 1 - g e^{-dT} never approaches the negative
    # real axis and the principal logarithm never wraps.
    minus = beta - d
    plus = beta + d
    g = minus / plus
    exponential = np.exp(-d * maturity)
    ratio = (1.0 - g * exponential) / (1.0 - g)
    D = minus / xi2 * (1.0 - exponential) / (1.0 - g * exponential)
    C = kappa * theta / xi2 * (minus * maturity - 2.0 * np.log(ratio))
    return np.exp(C + D * v0)


@lru_cache(maxsize=8)
def _gauss_legendre(n_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre nodes and weights on ``(0, 1)``, cached by node count."""
    from scipy.special import roots_legendre

    nodes, weights = roots_legendre(n_nodes)
    return 0.5 * (nodes + 1.0), 0.5 * weights


def _half_line(n_nodes: int, scale: float) -> tuple[np.ndarray, np.ndarray]:
    """Quadrature nodes and weights for ``int_0^inf f(u) du`` via ``u = c x/(1 - x)``.

    The substitution maps the half-line to a unit interval; its Jacobian
    ``c/(1 - x)^2`` is folded into the weights, so a caller integrates by a plain
    weighted sum of ``f(u)``.

    ``c`` matters, and getting it wrong is the whole difficulty of pricing a
    short-dated option this way.  The integrand's width is set by the total
    variance: it decays roughly like ``exp(-w u^2 / 2)`` with ``w = v T``, so the
    range that carries the integral is ``u ~ 1/sqrt(w)``.  A fixed ``c`` that
    resolves a one-year option puts almost every node where a one-week option's
    integrand has already decayed, and the price comes back wrong in the second
    decimal place.  Scaling ``c`` with ``1/sqrt(w)`` keeps the node density
    matched to the integrand at every maturity.
    """
    x, w = _gauss_legendre(n_nodes)
    u = scale * x / (1.0 - x)
    return u, scale * w / (1.0 - x) ** 2


def _quadrature_scale(maturity: float, parameters: dict[str, float]) -> float:
    """The half-line scale ``c`` matched to this maturity's total variance."""
    reference = max(parameters["v0"], parameters["theta"], 1e-8)
    return 1.0 / np.sqrt(reference * maturity)


def heston_price(
    *,
    forward: Any,
    strike: Any,
    maturity: Any,
    v0: float,
    kappa: float,
    theta: float,
    vol_of_vol: float,
    rho: float,
    is_call: Any = True,
    discount: Any = 1.0,
    formulation: str = "lewis",
    n_nodes: int = DEFAULT_QUADRATURE_NODES,
) -> np.ndarray:
    """European option prices under Heston, by Fourier inversion.

    Parameters
    ----------
    forward, strike, maturity:
        Broadcastable arrays.  ``maturity`` is in years and strictly positive.
        Every element of ``forward`` and ``strike`` is strictly positive.
    v0, kappa, theta, vol_of_vol, rho:
        Heston parameters, scalar.  A term structure of parameters is a
        different model and is not silently accepted here.
    is_call:
        Boolean flag or array.  Puts are priced from the call by parity, which
        is exact and is tested as such.
    discount:
        Discount factor ``e^{-rT}``, broadcastable.  The Fourier integral prices
        the *undiscounted forward* option; discounting is a separate, explicit
        multiplication so a caller can see which market input entered where.
    formulation:
        One of :data:`FORMULATIONS`.
    n_nodes:
        Gauss-Legendre node count.

    Returns
    -------
    Discounted option prices, broadcast to the common shape.

    Notes
    -----
    The two formulations are mathematically identical and numerically
    independent: ``'lewis'`` inverts a single integral with a regular integrand,
    ``'gatheral'`` assembles ``F P_1 - K P_2`` from two integrals whose
    integrands have a removable singularity at the origin.  Agreement between
    them to twelve digits is what this module offers in place of a published
    reference table.

    Examples
    --------
    >>> import numpy as np
    >>> from fast_vollib.pricing import heston_price
    >>> lewis = heston_price(
    ...     forward=100.0, strike=90.0, maturity=1.0, v0=0.04, kappa=2.0,
    ...     theta=0.04, vol_of_vol=0.3, rho=-0.7,
    ... )
    >>> gatheral = heston_price(
    ...     forward=100.0, strike=90.0, maturity=1.0, v0=0.04, kappa=2.0,
    ...     theta=0.04, vol_of_vol=0.3, rho=-0.7, formulation="gatheral",
    ... )
    >>> bool(abs(lewis - gatheral) < 1e-10)
    True
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

    parameters = {
        "v0": v0,
        "kappa": kappa,
        "theta": theta,
        "vol_of_vol": vol_of_vol,
        "rho": rho,
    }
    shape = F.shape
    call_undiscounted = np.empty(shape, dtype=np.float64)
    flat_F, flat_K, flat_T = F.reshape(-1), K.reshape(-1), T.reshape(-1)
    flat_out = call_undiscounted.reshape(-1)
    # One quadrature per distinct maturity: the characteristic function depends
    # on T but not on the strike, so a whole smile costs one evaluation of it.
    for value in np.unique(flat_T):
        mask = flat_T == value
        flat_out[mask] = (
            _lewis_call(flat_F[mask], flat_K[mask], float(value), parameters, n_nodes)
            if formulation == "lewis"
            else _gatheral_call(flat_F[mask], flat_K[mask], float(value), parameters, n_nodes)
        )
    call_undiscounted = np.maximum(flat_out.reshape(shape), 0.0)
    # Put-call parity on the undiscounted forward: c - p = F - K.
    undiscounted = np.where(call, call_undiscounted, call_undiscounted - (F - K))
    return df * undiscounted


def heston_call_price(
    *,
    forward: Any,
    strike: Any,
    maturity: Any,
    v0: float,
    kappa: float,
    theta: float,
    vol_of_vol: float,
    rho: float,
    discount: Any = 1.0,
    formulation: str = "lewis",
    n_nodes: int = DEFAULT_QUADRATURE_NODES,
) -> np.ndarray:
    """Call prices; :func:`heston_price` with ``is_call=True``."""
    return heston_price(
        forward=forward,
        strike=strike,
        maturity=maturity,
        v0=v0,
        kappa=kappa,
        theta=theta,
        vol_of_vol=vol_of_vol,
        rho=rho,
        is_call=True,
        discount=discount,
        formulation=formulation,
        n_nodes=n_nodes,
    )


def _lewis_call(
    F: np.ndarray, K: np.ndarray, T: float, parameters: dict[str, float], n_nodes: int
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
    u, w = _half_line(n_nodes, _quadrature_scale(T, parameters))
    phi = heston_characteristic_function(u - 0.5j, maturity=T, **parameters)
    k = np.log(K / F)
    kernel = phi / (u * u + 0.25)
    integral = (np.exp(-1j * np.outer(k, u)) * kernel[None, :]).real @ w
    return F - np.sqrt(F * K) / np.pi * integral


def _gatheral_call(
    F: np.ndarray, K: np.ndarray, T: float, parameters: dict[str, float], n_nodes: int
) -> np.ndarray:
    """The two-probability undiscounted forward call ``F P_1 - K P_2``.

    ``P_j = 1/2 + (1/pi) int_0^inf Re[e^{-i u k} phi_j(u) / (i u)] du`` with
    ``phi_2 = phi`` and ``phi_1(u) = phi(u - i)`` -- the share-measure shift,
    which needs no normalization because ``phi(-i) = 1`` exactly for a forward
    that is a martingale.
    """
    u, w = _half_line(n_nodes, _quadrature_scale(T, parameters))
    phi2 = heston_characteristic_function(u, maturity=T, **parameters)
    phi1 = heston_characteristic_function(u - 1j, maturity=T, **parameters)
    k = np.log(K / F)
    rotation = np.exp(-1j * np.outer(k, u))
    p1 = 0.5 + (rotation * (phi1 / (1j * u))[None, :]).real @ w / np.pi
    p2 = 0.5 + (rotation * (phi2 / (1j * u))[None, :]).real @ w / np.pi
    return F * p1 - K * p2
