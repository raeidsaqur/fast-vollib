r"""Bakshi, Cao and Chen (1997), priced by inversion under the T-forward measure.

Everything the model adds to Bates is one factor, and the reason it is only one
factor is the model's own independence assumption: the rate driver is
independent of :math:`(W^S, W^v, N, J)`, so the **discounted transform**
factorizes exactly.  Writing :math:`\mu_J` for the mean relative jump,

.. math::

    \Phi(u) := E\!\left[e^{-\int_0^T r_u du}\, S_T^{iu}\right]
      = S_0^{iu}\, e^{-iu q T}\,
        \Psi_R(1 - iu)\, \phi^{\text{Bates}}(u),

where :math:`\phi^{\text{Bates}}` is the T-forward Bates transform this library
already ships -- compensator, Heston factor and jump factor together -- and
:math:`\Psi_R` is the CIR integrated-rate transform of
:func:`fast_vollib.rates.cir_integrated_rate_transform` at a *complex*
argument.  The rate enters at :math:`s = 1 - iu` rather than at :math:`s = 1`
because the discount factor and the terminal spot are functions of the same
rate path: :math:`e^{-\int r} S_T^{iu}` carries :math:`\exp(-(1-iu)\int r)`,
not :math:`\exp(-\int r)` times something independent of it.

Two normalizations, two identity sets
-------------------------------------
:math:`\Phi` is *not* a characteristic function: :math:`\Phi(0) = P(0,T)` and
:math:`\Phi(-i) = S_0 e^{-qT}`.  Dividing by the bond and rotating to the
forward gives one that is,

.. math::

    P = \Psi_R(1), \qquad F = \frac{S_0 e^{-qT}}{P}, \qquad
    \phi^T(u) = \frac{\Phi(u)\, F^{-iu}}{P},

with :math:`\phi^T(0) = \phi^T(-i) = 1` -- the two identities
:mod:`fast_vollib.pricing.heston` already asserts, now true of a stochastic-rate
model.  Both sets are tested, because they say different things: the first that
the discounted transform prices a bond and the forward, the second that the
normalized one is a probability transform a Fourier inversion may be handed.

Substituting the definitions collapses the rotation algebraically, and the
collapsed form is what is evaluated:

.. math::

    \phi^T(u) = \exp\!\big(\log\Psi_R(1-iu) - (1-iu)\log\Psi_R(1)\big)\,
                \phi^{\text{Bates}}(u).

That is not a micro-optimization.  Under a deterministic flat rate
:math:`\Psi_R(s) = e^{-srT}` and the leading factor is :math:`e^{0} = 1`
exactly, so the Bates reduction is an *evaluation of the same code path* rather
than a cancellation of two large numbers -- and it comes out bitwise.  Forming
:math:`S_0^{iu} F^{-iu}` numerically instead would leave a rounding residue in
every price.

The forward is an output
------------------------
:func:`bcc97_price` takes a **spot**, not a forward, and returns the discount
factor and forward it derived.  Under stochastic rates those are not market
inputs a caller could supply consistently: they are what the rate model says.
That is exactly the split
:class:`fast_vollib.pricing._fourier.ForwardMeasureTransform` describes, and
this is the library's first implementation of it.

The CIR kernel is imported inside the two functions that use it rather than at
module level, for the same reason
:meth:`fast_vollib.processes.CIRShortRate.discount_curve` defers its import:
``fast_vollib.pricing`` must not pull in ``fast_vollib.rates`` for a caller who
only wanted Black-76.  ``tests/test_package_boundaries.py`` enforces it
statically and ``tests/rates/test_import_isolation.py`` at runtime.

References
----------
Bakshi, G., Cao, C., Chen, Z. (1997). Empirical Performance of Alternative
Option Pricing Models. *Journal of Finance* 52(5), 2003-2049.

Lewis, A. L. (2001). A Simple Option Formula for General Jump-Diffusion and
Other Exponential Levy Processes.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ._fourier import fourier_price
from .bates import BATES_QUADRATURE_NODES, bates_characteristic_function

__all__ = [
    "BCC97_QUADRATURE_NODES",
    "bcc97_characteristic_function",
    "bcc97_discounted_transform",
    "bcc97_forward_measure",
    "bcc97_price",
]

#: Gauss-Legendre nodes for :func:`bcc97_price`.
#:
#: The same count Bates uses, and re-measured rather than assumed to carry over:
#: a CIR rate contributes to the integrand's width, and the sweep in
#: ``tests/test_pricing/test_bcc97.py`` compares this count against ten times as
#: many nodes across the parameter grid the tests use.
BCC97_QUADRATURE_NODES = BATES_QUADRATURE_NODES


def _spot_parameters(
    v0: float,
    kappa: float,
    theta: float,
    vol_of_vol: float,
    rho: float,
    jump_intensity: float,
    mean_log_jump: float,
    jump_volatility: float,
) -> dict[str, float]:
    return {
        "v0": v0,
        "kappa": kappa,
        "theta": theta,
        "vol_of_vol": vol_of_vol,
        "rho": rho,
        "jump_intensity": jump_intensity,
        "mean_log_jump": mean_log_jump,
        "jump_volatility": jump_volatility,
    }


def _log_transform(
    s: Any,
    *,
    maturity: float,
    rate_kappa: float,
    rate_theta: float,
    rate_volatility: float,
    initial_rate: float,
) -> Any:
    """``log Psi_R(s)``, the affine exponent rather than its exponential.

    In log form because a long maturity and a large ``|s|`` send the
    exponential itself past the end of float64 long before the exponent stops
    being an ordinary number -- the same reason
    :func:`fast_vollib.rates.cir_affine_coefficients` returns ``log_A``.
    """
    from ..rates import cir_integrated_rate_coefficients

    log_a, b = cir_integrated_rate_coefficients(
        s,
        kappa=rate_kappa,
        theta=rate_theta,
        volatility=rate_volatility,
        maturity=maturity,
    )
    return log_a - b * initial_rate


def _rate_factor(
    z: np.ndarray,
    *,
    maturity: float,
    rate_kappa: float,
    rate_theta: float,
    rate_volatility: float,
    initial_rate: float,
) -> np.ndarray:
    r"""``Psi_R(1-iu) / Psi_R(1)^{1-iu}``, the whole of what BCC97 adds to Bates.

    Formed by *differencing the affine coefficients* rather than by subtracting
    two assembled exponents, and the difference is not cosmetic.  Writing
    :math:`s = 1 - iu`,

    .. math::

        \log\Psi(s) - s\log\Psi(1)
          = ig(\log A_s - s \log A_1ig) - ig(B_s - s B_1ig) r_0 ,

    and under a deterministic rate the coefficients are exactly linear in
    :math:`s` -- :func:`fast_vollib.rates.cir_integrated_rate_coefficients`
    returns ``s * A`` and ``s * B`` from that branch -- so each bracket is a
    quantity minus *the identical bits*, which is exactly zero.  The factor is
    then ``exp(0) = 1``, and multiplying a Bates transform by it changes
    nothing at all.

    Assembling ``log_a - b*r_0`` first and subtracting would instead leave two
    roundings that do not cancel: measured over a sweep of curves and
    maturities the two groupings disagree in the last bits of one exponent in
    seven, which is enough to move the last digits of every price and to turn
    "the deterministic limit *is* Bates" into "the deterministic limit is very
    nearly Bates".
    """
    from ..rates import cir_integrated_rate_coefficients

    shifted = 1.0 - 1j * z
    coefficients = {
        "kappa": rate_kappa,
        "theta": rate_theta,
        "volatility": rate_volatility,
        "maturity": maturity,
    }
    log_a_s, b_s = cir_integrated_rate_coefficients(shifted, **coefficients)
    log_a_1, b_1 = cir_integrated_rate_coefficients(1.0, **coefficients)
    return np.exp((log_a_s - shifted * log_a_1) - (b_s - shifted * b_1) * initial_rate)


def bcc97_forward_measure(
    *,
    spot: float,
    maturity: float,
    dividend_yield: float = 0.0,
    rate_kappa: float,
    rate_theta: float,
    rate_volatility: float,
    initial_rate: float,
) -> tuple[float, float]:
    r"""``(discount_factor, forward)`` implied by the rate model at ``maturity``.

    ``P(0,T) = Psi_R(1)`` is the CIR zero-coupon bond, and
    ``F = spot * exp(-q T) / P`` is the T-forward that goes with it.  Returned
    together because they are two readings of one model and a caller who mixed
    a bond from here with a forward from elsewhere would be pricing a different
    contract.

    Examples
    --------
    >>> from fast_vollib.pricing.bcc97 import bcc97_forward_measure
    >>> from fast_vollib.rates import cir_discount_factor
    >>> curve = dict(rate_kappa=0.3, rate_theta=0.04, rate_volatility=0.1, initial_rate=0.04)
    >>> discount, forward = bcc97_forward_measure(spot=100.0, maturity=2.0, **curve)
    >>> bool(discount == cir_discount_factor(
    ...     kappa=0.3, theta=0.04, volatility=0.1, initial_rate=0.04, maturity=2.0
    ... ))
    True
    >>> float(round(forward, 10))
    108.2912016535
    """
    log_p = _log_transform(
        1.0,
        maturity=maturity,
        rate_kappa=rate_kappa,
        rate_theta=rate_theta,
        rate_volatility=rate_volatility,
        initial_rate=initial_rate,
    )
    discount = float(np.exp(log_p))
    return discount, float(spot) * float(np.exp(-dividend_yield * maturity)) / discount


def bcc97_characteristic_function(
    u: Any,
    *,
    maturity: float,
    v0: float,
    kappa: float,
    theta: float,
    vol_of_vol: float,
    rho: float,
    jump_intensity: float = 0.0,
    mean_log_jump: float = 0.0,
    jump_volatility: float = 0.0,
    rate_kappa: float,
    rate_theta: float,
    rate_volatility: float,
    initial_rate: float,
) -> np.ndarray:
    r"""``phi^T(u)``: the T-forward transform of ``ln(S_T / F_T)`` under BCC97.

    Parameters
    ----------
    u : array-like
        Real or complex. The inversion routines evaluate at ``u - i/2`` and
        ``u - i``, so complex arguments are ordinary rather than exceptional --
        and :math:`\Psi_R` is reached at ``s = 1 - iz`` for each, giving
        ``1/2 - iu`` and ``-iu``. Both have non-negative real part, so
        ``kappa**2 + 2*volatility**2*s`` keeps a real part of at least
        ``kappa**2 > 0`` and the principal square root never approaches its
        cut.
    maturity : float
    v0, kappa, theta, vol_of_vol, rho : float
        Heston parameters, scalar.
    jump_intensity, mean_log_jump, jump_volatility : float
        Merton jump parameters, scalar.
    rate_kappa, rate_theta, rate_volatility, initial_rate : float
        Risk-neutral CIR parameters and the short rate now. A
        ``rate_volatility`` of exactly zero is the deterministic-rate limit, and
        with ``rate_theta == initial_rate`` it is a flat rate ``r``.

    Returns
    -------
    ndarray
        Complex, the same shape as ``u``.

    Notes
    -----
    The rate factor is ``exp(log Psi_R(1 - iu) - (1 - iu) log Psi_R(1))``, which
    is identically ``1`` when the rate is deterministic *by cancellation in the
    exponent* rather than by division. That is what makes the Bates reduction
    bitwise; see the module docstring.

    Examples
    --------
    >>> import numpy as np
    >>> from fast_vollib.pricing.bcc97 import bcc97_characteristic_function
    >>> parameters = dict(
    ...     maturity=1.0, v0=0.04, kappa=2.0, theta=0.04, vol_of_vol=0.3, rho=-0.7,
    ...     jump_intensity=0.5, mean_log_jump=-0.05, jump_volatility=0.2,
    ...     rate_kappa=0.3, rate_theta=0.04, rate_volatility=0.1, initial_rate=0.03,
    ... )
    >>> bool(abs(bcc97_characteristic_function(0.0 + 0j, **parameters) - 1.0) < 1e-14)
    True
    >>> bool(abs(bcc97_characteristic_function(-1j, **parameters) - 1.0) < 1e-12)
    True
    """
    z = np.asarray(u, dtype=np.complex128)
    rate_factor = _rate_factor(
        z,
        maturity=maturity,
        rate_kappa=rate_kappa,
        rate_theta=rate_theta,
        rate_volatility=rate_volatility,
        initial_rate=initial_rate,
    )
    spot_factor = bates_characteristic_function(
        z,
        maturity=maturity,
        **_spot_parameters(
            v0, kappa, theta, vol_of_vol, rho, jump_intensity, mean_log_jump, jump_volatility
        ),
    )
    return rate_factor * spot_factor


def bcc97_discounted_transform(
    u: Any,
    *,
    spot: float,
    maturity: float,
    v0: float,
    kappa: float,
    theta: float,
    vol_of_vol: float,
    rho: float,
    jump_intensity: float = 0.0,
    mean_log_jump: float = 0.0,
    jump_volatility: float = 0.0,
    rate_kappa: float,
    rate_theta: float,
    rate_volatility: float,
    initial_rate: float,
    dividend_yield: float = 0.0,
) -> np.ndarray:
    r"""``Phi(u) = E[exp(-int_0^T r) * S_T^{iu}]``, the *discounted* transform.

    Written out as the product of §10.1 rather than derived from
    :func:`bcc97_characteristic_function`, because it is the form the two
    normalization identities are stated in and the point of having it is to
    check the normalized one against something that was not built from it:

    ``Phi(0) = P(0,T)`` and ``Phi(-i) = spot * exp(-dividend_yield * maturity)``.

    Neither identity is available from :math:`\phi^T`, which is ``1`` at both
    arguments by construction -- that is precisely what normalizing threw away.

    Examples
    --------
    >>> import numpy as np
    >>> from fast_vollib.pricing.bcc97 import (
    ...     bcc97_discounted_transform, bcc97_forward_measure,
    ... )
    >>> parameters = dict(
    ...     spot=100.0, maturity=2.0, v0=0.04, kappa=2.0, theta=0.04,
    ...     vol_of_vol=0.3, rho=-0.7, jump_intensity=0.5, mean_log_jump=-0.05,
    ...     jump_volatility=0.2, rate_kappa=0.3, rate_theta=0.04,
    ...     rate_volatility=0.1, initial_rate=0.03, dividend_yield=0.01,
    ... )
    >>> discount, forward = bcc97_forward_measure(
    ...     spot=100.0, maturity=2.0, dividend_yield=0.01, rate_kappa=0.3,
    ...     rate_theta=0.04, rate_volatility=0.1, initial_rate=0.03,
    ... )
    >>> bool(abs(bcc97_discounted_transform(0.0 + 0j, **parameters) - discount) < 1e-14)
    True
    >>> bool(abs(bcc97_discounted_transform(-1j, **parameters) - discount * forward) < 1e-12)
    True
    """
    z = np.asarray(u, dtype=np.complex128)
    rate_transform = np.exp(
        _log_transform(
            1.0 - 1j * z,
            maturity=maturity,
            rate_kappa=rate_kappa,
            rate_theta=rate_theta,
            rate_volatility=rate_volatility,
            initial_rate=initial_rate,
        )
    )
    spot_factor = bates_characteristic_function(
        z,
        maturity=maturity,
        **_spot_parameters(
            v0, kappa, theta, vol_of_vol, rho, jump_intensity, mean_log_jump, jump_volatility
        ),
    )
    return (
        np.power(float(spot), 1j * z)
        * np.exp(-1j * z * dividend_yield * maturity)
        * rate_transform
        * spot_factor
    )


def _quadrature_scale(
    maturity: float,
    v0: float,
    theta: float,
    jump_intensity: float,
    mean_log_jump: float,
    jump_volatility: float,
    rate_volatility: float,
    rate_theta: float,
    initial_rate: float,
) -> float:
    r"""The half-line scale matched to this maturity's **total** variance.

    The two terms :func:`fast_vollib.pricing.bates._quadrature_scale` uses, plus
    the rate's own contribution.  Under CIR,
    :math:`\mathrm{Var}(\int_0^T r) \approx \sigma_R^2 \bar r\, T^3/3` for
    :math:`\kappa_R T` small, and the long-maturity behaviour is bounded by it
    -- an over-estimate widens the node spread slightly and costs accuracy far
    more slowly than an under-estimate does, so the small-:math:`\kappa` form is
    used at every maturity rather than the exact expression.

    At ``rate_volatility == 0`` the term is exactly zero and adding it changes
    nothing, so a deterministic-rate configuration takes bit for bit the scale
    Bates would have used.  That is what makes the price reduction bitwise, and
    it is why the term is added to the same sum rather than folded in some
    other way.
    """
    diffusive = max(v0, theta, 1e-8) * maturity
    jumps = jump_intensity * maturity * (mean_log_jump * mean_log_jump + jump_volatility**2)
    rates = (
        rate_volatility * rate_volatility * max(initial_rate, rate_theta, 0.0) * maturity**3 / 3.0
    )
    return 1.0 / np.sqrt(diffusive + jumps + rates)


def bcc97_price(
    *,
    spot: Any,
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
    rate_kappa: float,
    rate_theta: float,
    rate_volatility: float,
    initial_rate: float,
    dividend_yield: float = 0.0,
    is_call: Any = True,
    formulation: str = "lewis",
    n_nodes: int = BCC97_QUADRATURE_NODES,
) -> np.ndarray:
    """European option prices under BCC97, by Fourier inversion.

    Parameters
    ----------
    spot, strike, maturity : array-like
        Broadcastable. ``maturity`` is in years and strictly positive; every
        element of ``spot`` and ``strike`` is strictly positive. A **spot**
        rather than a forward, because under a stochastic rate the forward and
        the discount factor are outputs of the model; see
        :func:`bcc97_forward_measure`.
    v0, kappa, theta, vol_of_vol, rho : float
        Heston parameters, scalar.
    jump_intensity, mean_log_jump, jump_volatility : float
        Merton jump parameters, scalar. All three default to zero, which is
        SVSI -- stochastic volatility, stochastic rates, no jumps.
    rate_kappa, rate_theta, rate_volatility, initial_rate : float
        Risk-neutral CIR parameters and the short rate now, scalar.
        ``rate_volatility=0.0`` with ``rate_theta=initial_rate=r`` is a flat
        deterministic rate, and the price is then bitwise equal to
        :func:`fast_vollib.pricing.bates_price` at
        ``forward = spot * exp((r - dividend_yield) * maturity)`` and
        ``discount = exp(-r * maturity)``.
    dividend_yield : float, default 0.0
        Continuous.
    is_call : array-like, default True
        Puts by exact put-call parity on the forward.
    formulation : {'lewis', 'gatheral'}
        Two mathematically identical, numerically independent routes.
    n_nodes : int

    Returns
    -------
    ndarray
        Discounted option prices, broadcast to the common shape.

    Notes
    -----
    A separate maturity per element is supported and each distinct one costs its
    own bond, forward and quadrature -- the rate model makes the forward depend
    on the maturity, so a smile at one expiry is one evaluation and a surface is
    one per expiry.

    Examples
    --------
    >>> from fast_vollib.pricing import bates_price, bcc97_price
    >>> equity = dict(v0=0.04, kappa=2.0, theta=0.04, vol_of_vol=0.3, rho=-0.7)
    >>> stochastic = bcc97_price(
    ...     spot=100.0, strike=95.0, maturity=1.0, **equity,
    ...     rate_kappa=0.3, rate_theta=0.05, rate_volatility=0.2, initial_rate=0.03,
    ... )
    >>> flat = bcc97_price(
    ...     spot=100.0, strike=95.0, maturity=1.0, **equity,
    ...     rate_kappa=0.3, rate_theta=0.03, rate_volatility=0.0, initial_rate=0.03,
    ... )
    >>> bool(stochastic != flat)
    True

    The deterministic-rate limit *is* the Bates price, to the bit -- priced at
    the forward and discount this model itself reports, which is the only
    comparison that is about the transform rather than about how two routes
    happened to round ``exp(-r T)``:

    >>> from fast_vollib.pricing.bcc97 import bcc97_forward_measure
    >>> discount, forward = bcc97_forward_measure(
    ...     spot=100.0, maturity=1.0,
    ...     rate_kappa=0.3, rate_theta=0.03, rate_volatility=0.0, initial_rate=0.03,
    ... )
    >>> bool(flat == bates_price(
    ...     forward=forward, strike=95.0, maturity=1.0, discount=discount, **equity,
    ... ))
    True
    """
    rate = dict(
        rate_kappa=rate_kappa,
        rate_theta=rate_theta,
        rate_volatility=rate_volatility,
        initial_rate=initial_rate,
    )
    spot_parameters = _spot_parameters(
        v0, kappa, theta, vol_of_vol, rho, jump_intensity, mean_log_jump, jump_volatility
    )

    spot_array, strike_array, maturity_array = np.broadcast_arrays(
        np.asarray(spot, dtype=np.float64),
        np.asarray(strike, dtype=np.float64),
        np.asarray(maturity, dtype=np.float64),
    )
    if not bool(np.all(spot_array > 0.0)):
        raise ValueError("spot must be strictly positive.")
    if not bool(np.all(maturity_array > 0.0)):
        raise ValueError("maturity must be strictly positive.")

    # The forward and the bond are maturity-dependent outputs of the rate model,
    # so they are built per distinct maturity rather than once for the request.
    discount = np.empty(spot_array.shape, dtype=np.float64)
    forward = np.empty(spot_array.shape, dtype=np.float64)
    for value in np.unique(maturity_array):
        mask = maturity_array == value
        factor, _one = bcc97_forward_measure(
            spot=1.0, maturity=float(value), dividend_yield=dividend_yield, **rate
        )
        discount[mask] = factor
        # ``(spot * e^{-qT}) / P``, in that association: the same one
        # ``bcc97_forward_measure`` uses, so a caller who read the forward from
        # there and priced with it gets the identical number rather than one
        # that differs in the last bit.
        forward[mask] = spot_array[mask] * float(np.exp(-dividend_yield * value)) / factor

    def transform(u: Any, *, maturity: float) -> np.ndarray:
        return bcc97_characteristic_function(u, maturity=maturity, **spot_parameters, **rate)

    def scale(T: float) -> float:
        return _quadrature_scale(
            T,
            v0,
            theta,
            jump_intensity,
            mean_log_jump,
            jump_volatility,
            rate_volatility,
            rate_theta,
            initial_rate,
        )

    return fourier_price(
        forward=forward,
        strike=strike_array,
        maturity=maturity_array,
        characteristic_function=transform,
        quadrature_scale=scale,
        is_call=is_call,
        discount=discount,
        formulation=formulation,
        n_nodes=n_nodes,
    )
