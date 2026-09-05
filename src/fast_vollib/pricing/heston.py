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

from typing import Any

import numpy as np

from ._fourier import FORMULATIONS, fourier_price

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
    # Avoid the removable 0/0 at the martingale argument when beta <= 0.
    martingale = u == -1j
    u = np.where(martingale, 0.0, u)
    xi2 = vol_of_vol * vol_of_vol
    beta = kappa - rho * vol_of_vol * 1j * u
    d = np.sqrt(beta * beta + xi2 * (1j * u + u * u))
    if vol_of_vol < np.finfo(float).eps ** (1 / 6) * abs(kappa):
        # Rationalize (beta-d)/xi^2 before the subtraction loses precision.
        # This evaluates the same stochastic model, not its zero-xi limit.
        # Keep the established arithmetic at ordinary parameter values.
        scaled_minus = -(u * u + 1j * u) / (beta + d)
        g_scaled = scaled_minus / (beta + d)
        g = xi2 * g_scaled
        one_minus_exp = -np.expm1(-d * maturity)
        x_scaled = g_scaled * one_minus_exp / (1.0 - g)
        x = xi2 * x_scaled
        small = np.abs(x) < 1e-4
        # log(1+x)/x has a removable singularity at zero. A short Taylor
        # series also avoids complex log1p implementations that first add 1.
        safe_x = np.where(small, 1.0 + 0j, x)
        quotient = np.where(
            small,
            1.0 + x * (-0.5 + x * (1 / 3 + x * (-0.25 + x * (0.2 - x / 6)))),
            np.log1p(safe_x) / safe_x,
        )
        C = kappa * theta * (scaled_minus * maturity - 2.0 * x_scaled * quotient)
        D = scaled_minus * one_minus_exp / (1.0 - g * np.exp(-d * maturity))
        return np.exp(C + D * v0)
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
    parameters = {
        "v0": v0,
        "kappa": kappa,
        "theta": theta,
        "vol_of_vol": vol_of_vol,
        "rho": rho,
    }

    def transform(u: Any, *, maturity: float) -> np.ndarray:
        return heston_characteristic_function(u, maturity=maturity, **parameters)

    return fourier_price(
        forward=forward,
        strike=strike,
        maturity=maturity,
        characteristic_function=transform,
        quadrature_scale=lambda T: _quadrature_scale(T, parameters),
        is_call=is_call,
        discount=discount,
        formulation=formulation,
        n_nodes=n_nodes,
    )


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
