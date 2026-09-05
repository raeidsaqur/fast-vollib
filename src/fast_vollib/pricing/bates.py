r"""Bates (1996): Heston's diffusion with Merton's jumps, priced by inversion.

The characteristic function factorizes, and that is the whole of the model's
tractability.  Under the T-forward measure the log-forward is a sum of two
independent pieces -- the continuous part Heston already describes, and a
compound Poisson sum of log-normal jumps -- so

.. math::

    \phi^T_{\text{Bates}}(u)
      = e^{-i u \lambda \mu_J T}\; \phi_H(u)\; \phi_J(u),
    \qquad
    \phi_J(u) = \exp\!\big(\lambda T (e^{i u m - u^2\delta^2/2} - 1)\big),

with :math:`\mu_J = e^{m + \delta^2/2} - 1` the mean relative jump.  The leading
factor is the drift compensation, and it is the *same* :math:`\mu_J` the sampler
subtracts -- both read
:func:`fast_vollib._jump_law.mean_relative_jump`, so they cannot drift apart.

Two identities pin the normalization, and both are asserted by test:
:math:`\phi^T(0) = 1` because it is a characteristic function, and
:math:`\phi^T(-i) = 1` because the forward is a martingale.  The second is what
the compensator buys: without it the jumps would add
:math:`e^{\lambda \mu_J T}` of drift and every price would be wrong in a way no
individual term looks wrong.

Reductions
----------
At ``jump_intensity = 0`` the jump factor is ``exp(0)`` and the compensator is
zero, so the transform is Heston's -- and, because the quadrature scale reduces
to Heston's too, the price comes back **bitwise** equal to
:func:`fast_vollib.pricing.heston_price`.  With ``vol_of_vol`` sent to zero and
``v0 = theta`` the diffusion is Black-Scholes and the model is Merton (1976),
which the tests check against Merton's own series rather than against this
routine.

The quadrature scale
--------------------
:func:`fast_vollib.pricing._fourier.half_line` matches its node density to the
integrand's width, which is set by the **total** variance.  Jumps contribute
:math:`\lambda T (m^2 + \delta^2)` to that -- the second moment of the aggregate
jump -- and ignoring it would leave a heavily-jumping short-dated option
resolved by a node set built for its diffusion alone.  With no jumps the term is
exactly zero and the scale is Heston's, which is what makes the reduction
bitwise rather than merely close.

References
----------
Bates, D. S. (1996). Jumps and stochastic volatility: exchange rate processes
implicit in Deutsche Mark options. *Review of Financial Studies* 9(1), 69-107.

Merton, R. C. (1976). Option pricing when underlying stock returns are
discontinuous. *Journal of Financial Economics* 3(1-2), 125-144.

Lewis, A. L. (2000). *Option Valuation under Stochastic Volatility*. Finance
Press.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .._jump_law import mean_relative_jump
from ._fourier import fourier_price
from .heston import DEFAULT_QUADRATURE_NODES, heston_characteristic_function

__all__ = [
    "BATES_QUADRATURE_NODES",
    "bates_characteristic_function",
    "bates_price",
]

#: Gauss-Legendre nodes used by default for a jump model.
#:
#: **Re-measured rather than inherited.**  Heston's 768 was established for a
#: purely diffusive integrand, and a jump characteristic function does not decay
#: the same way: ``phi_J`` contributes ``exp(-lambda T u^2 delta^2 / 2)``, which
#: is Gaussian in ``u`` and helps, while the oscillatory ``exp(i u lambda T m)``
#: factor does not decay at all and has to be resolved.
#:
#: The sweep -- three Heston sets from Feller 0.07 to 5, four jump sets out to
#: intensity 5 with a 45% jump volatility, maturities from a week to thirty
#: years, strikes from 0.6 to 1.8 of the forward, each against ten times the
#: node count -- puts the Lewis route at **3.2e-9 of the forward** here, which is
#: the same order as Heston's own 2e-9 and is why the count is unchanged. The
#: worst case is a fast-reverting diffusion under wild jumps at a two-week
#: maturity, where the jump term dominates the quadrature scale.
#:
#: The Gatheral route is the weaker one and needs about twice as many nodes for
#: jump-heavy parameters: 4.1e-7 of the forward at 768, 4.1e-10 at 1536. It is
#: an independent cross-check rather than the working route, so the default
#: follows Lewis; the cross-check test raises the count instead of loosening its
#: tolerance. Both routes converge to the same value, which is what says the
#: difference is quadrature rather than a disagreement about the model.
#:
#: Keeping 768 also keeps the zero-intensity reduction *bitwise* against
#: ``heston_price``, which a different count would silently give up.
BATES_QUADRATURE_NODES = DEFAULT_QUADRATURE_NODES


def bates_characteristic_function(
    u: Any,
    *,
    maturity: float,
    v0: float,
    kappa: float,
    theta: float,
    vol_of_vol: float,
    rho: float,
    jump_intensity: float,
    mean_log_jump: float,
    jump_volatility: float,
) -> np.ndarray:
    r"""``phi^T(u)``: the T-forward transform of ``ln(S_T / F_T)`` under Bates.

    Parameters
    ----------
    u : array-like
        Real or complex. The inversion routines evaluate at ``u - i/2`` and
        ``u - i``, so complex arguments are ordinary rather than exceptional.
    maturity : float
    v0, kappa, theta, vol_of_vol, rho : float
        Heston parameters, scalar.
    jump_intensity, mean_log_jump, jump_volatility : float
        ``lambda``, ``m``, ``delta``. A zero intensity gives Heston's transform
        exactly.

    Returns
    -------
    ndarray
        Complex, the same shape as ``u``.

    Notes
    -----
    The three factors are multiplied rather than combined into one exponent,
    which costs one complex multiplication and keeps the Heston part *the same
    call* the Heston pricer makes -- so a reduction is an identity in the code
    as well as in the algebra.

    Examples
    --------
    >>> import numpy as np
    >>> from fast_vollib.pricing.bates import bates_characteristic_function
    >>> parameters = dict(
    ...     maturity=1.0, v0=0.04, kappa=2.0, theta=0.04, vol_of_vol=0.3, rho=-0.7,
    ...     jump_intensity=0.5, mean_log_jump=-0.05, jump_volatility=0.2,
    ... )
    >>> bool(abs(bates_characteristic_function(0.0 + 0j, **parameters) - 1.0) < 1e-14)
    True
    >>> bool(abs(bates_characteristic_function(-1j, **parameters) - 1.0) < 1e-12)
    True
    """
    z = np.asarray(u, dtype=np.complex128)
    continuous = heston_characteristic_function(
        z, maturity=maturity, v0=v0, kappa=kappa, theta=theta, vol_of_vol=vol_of_vol, rho=rho
    )
    mu_jump = mean_relative_jump(mean_log_jump, jump_volatility)
    jump = np.exp(
        jump_intensity
        * maturity
        * (np.exp(1j * z * mean_log_jump - 0.5 * z * z * jump_volatility * jump_volatility) - 1.0)
    )
    compensation = np.exp(-1j * z * jump_intensity * mu_jump * maturity)
    return compensation * continuous * jump


def _quadrature_scale(
    maturity: float,
    v0: float,
    theta: float,
    jump_intensity: float,
    mean_log_jump: float,
    jump_volatility: float,
) -> float:
    """The half-line scale matched to this maturity's **total** variance.

    Diffusive variance plus ``lambda T (m^2 + delta^2)``, the second moment of
    the aggregate jump over the period. With no jumps the second term is exactly
    zero and this is bit for bit Heston's own scale, which is what makes the
    reduction bitwise.
    """
    diffusive = max(v0, theta, 1e-8) * maturity
    jumps = jump_intensity * maturity * (mean_log_jump * mean_log_jump + jump_volatility**2)
    return 1.0 / np.sqrt(diffusive + jumps)


def bates_price(
    *,
    forward: Any,
    strike: Any,
    maturity: Any,
    v0: float,
    kappa: float,
    theta: float,
    vol_of_vol: float,
    rho: float,
    jump_intensity: float = 0.0,
    mean_log_jump: float = 0.0,
    jump_volatility: float = 0.0,
    is_call: Any = True,
    discount: Any = 1.0,
    formulation: str = "lewis",
    n_nodes: int = BATES_QUADRATURE_NODES,
) -> np.ndarray:
    """European option prices under Bates, by Fourier inversion.

    Parameters
    ----------
    forward, strike, maturity : array-like
        Broadcastable. ``maturity`` is in years and strictly positive; every
        element of ``forward`` and ``strike`` is strictly positive.
    v0, kappa, theta, vol_of_vol, rho : float
        Heston parameters, scalar.
    jump_intensity, mean_log_jump, jump_volatility : float
        Merton jump parameters, scalar. All three default to zero, which is
        Heston -- and the price is then bitwise equal to
        :func:`fast_vollib.pricing.heston_price`.
    is_call : array-like, default True
        Puts by exact put-call parity on the forward.
    discount : array-like, default 1.0
        ``P(0, T)``. The quadrature prices the undiscounted forward option and
        this is applied afterwards, so a caller can see which input entered
        where.
    formulation : {'lewis', 'gatheral'}
        Two mathematically identical, numerically independent routes.
    n_nodes : int

    Returns
    -------
    ndarray
        Discounted option prices, broadcast to the common shape.

    Examples
    --------
    >>> from fast_vollib.pricing import bates_price, heston_price
    >>> common = dict(
    ...     forward=100.0, strike=90.0, maturity=1.0, v0=0.04, kappa=2.0,
    ...     theta=0.04, vol_of_vol=0.3, rho=-0.7,
    ... )
    >>> jumps = bates_price(
    ...     **common, jump_intensity=0.5, mean_log_jump=-0.05, jump_volatility=0.2
    ... )
    >>> bool(jumps > heston_price(**common))
    True

    With no jumps it *is* the Heston price, to the bit:

    >>> bool(bates_price(**common) == heston_price(**common))
    True
    """
    parameters = {
        "v0": v0,
        "kappa": kappa,
        "theta": theta,
        "vol_of_vol": vol_of_vol,
        "rho": rho,
        "jump_intensity": jump_intensity,
        "mean_log_jump": mean_log_jump,
        "jump_volatility": jump_volatility,
    }

    def transform(u: Any, *, maturity: float) -> np.ndarray:
        return bates_characteristic_function(u, maturity=maturity, **parameters)

    def scale(T: float) -> float:
        return _quadrature_scale(T, v0, theta, jump_intensity, mean_log_jump, jump_volatility)

    return fourier_price(
        forward=forward,
        strike=strike,
        maturity=maturity,
        characteristic_function=transform,
        quadrature_scale=scale,
        is_call=is_call,
        discount=discount,
        formulation=formulation,
        n_nodes=n_nodes,
    )
