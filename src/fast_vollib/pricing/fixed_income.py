"""Present value of a dated schedule: one routine, any curve, any backend.

Fixed-income valuation starts with one line of mathematics --
each payment is worth its amount times the discount factor at *its own* payment
time, and the security is worth their sum:

.. math::

    PV = \\sum_i c_i \\, P(0, t_i).

What makes it worth a module is everything that must *not* happen around it.
The sum is over payments at different dates, so there is no single maturity to
discount at and no terminal state to evaluate; that is precisely why a bond does
not go through :func:`~fast_vollib.instruments.payoff`.  And the curve is an
argument rather than a lookup: nothing here infers a curve from a currency, an
asset class, or an installed default, so the discount factors that produced a
number are always the ones the caller named.

*Array-API native, unlike the rest of this package.*  The Fourier pricers here
are host-side float64 by construction -- a Gauss-Legendre quadrature over a
complex characteristic function does not vectorize onto a device usefully, and
:mod:`fast_vollib.pricing` says so.  This routine is the exception, and it is an
exception because it can be: it does arithmetic in whatever namespace the curve
hands back.  A curve holding a ``requires_grad`` torch tensor therefore yields a
present value that differentiates back to the curve's own parameters, which is
what makes a bond usable inside a calibration loss rather than only in a report.

Any object with a compliant ``discount_factor`` works, with no registration and
no subclassing -- see :class:`fast_vollib.rates.DiscountCurve`.  A caller's own
firm curve is a first-class argument.

Examples
--------
>>> from fast_vollib.instruments import ZeroCouponBond
>>> from fast_vollib.pricing import present_value
>>> from fast_vollib.rates import FlatDiscountCurve
>>> bond = ZeroCouponBond(maturity=2.0, face_value=100.0)
>>> round(present_value(bond, discount_curve=FlatDiscountCurve(rate=0.03)), 10)
94.1764533584

A payment due now is worth its face value under any curve at all:

>>> present_value(ZeroCouponBond(maturity=0.0, face_value=100.0),
...               discount_curve=FlatDiscountCurve(rate=0.25))
100.0
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .._array_api import concrete_float, get_namespace
from ..instruments import UnsupportedInstrumentError
from ..instruments.fixed_income import FixedIncomeSecurity, cashflows

__all__ = ["present_value"]


def _is_shipped_curve(curve: Any) -> bool:
    """Whether ``curve`` is one of the curves this library ships.

    Decided from the type's module rather than by ``isinstance`` against the
    classes, because getting the classes would mean importing
    :mod:`fast_vollib.rates` -- and doing so unconditionally, including for a
    caller who passed a duck-typed curve of their own and never mentioned the
    package.  Reading ``__module__`` costs nothing and imports nothing.

    The distinction it draws is narrow and deliberate: it decides only whether
    the strict-positivity check applies, and positivity is a theorem about the
    shipped curves rather than a rule this routine imposes on everyone.
    """
    return type(curve).__module__.startswith("fast_vollib.rates")


def _validated_factor(factor: Any, *, payment_time: float, require_positive: bool) -> Any:
    """Refuse a discount factor that would poison the sum, before it is added.

    Finiteness is checked for every curve.  Strict positivity is checked only
    for the curves this library ships, where it is a theorem: a user's curve is
    entitled to return whatever their market says, and this routine is not the
    place to overrule it.

    Both checks are skipped when the value cannot be read eagerly -- a JAX
    tracer has no number in it, and a library cannot raise from inside a trace.
    """
    shape = getattr(factor, "shape", ())
    if shape not in ((), (1,)):
        raise ValueError(
            f"A discount curve must return one factor per maturity; at t={payment_time!r} "
            f"it returned an array with shape {shape}. Present value sums one number per "
            f"cashflow, so a term structure returned here would be summed as if the "
            f"payments were separate securities."
        )
    as_float = concrete_float(factor)
    if as_float is None:  # pragma: no cover - traced values carry the precondition
        return factor
    if as_float != as_float or as_float in (float("inf"), float("-inf")):
        raise ValueError(
            f"The discount curve returned a non-finite factor {as_float!r} at t={payment_time!r}."
        )
    if require_positive and as_float <= 0.0:
        raise ValueError(
            f"The discount curve returned a non-positive factor {as_float!r} at "
            f"t={payment_time!r}. Every curve fast-vollib ships is strictly positive "
            f"everywhere, so this is a bug rather than a market condition."
        )
    return factor


def present_value(
    instrument: FixedIncomeSecurity,
    *,
    discount_curve: Any,
    return_native: bool = False,
) -> Any:
    """The value now of every payment ``instrument`` promises.

    Parameters
    ----------
    instrument : FixedIncomeSecurity
        A :class:`~fast_vollib.instruments.ZeroCouponBond` or any other
        security exposing dated cashflows.
    discount_curve : DiscountCurve
        Anything with a ``discount_factor(maturity)`` method. Supplied
        explicitly and never inferred: a currency on the contract is
        descriptive and does not choose a curve.
    return_native : bool, default False
        ``True`` returns the value in the curve's own namespace, so a torch or
        JAX curve yields a value that still carries its autograd graph.
        ``False`` (the default) returns a host value: a Python ``float`` for a
        scalar, a NumPy array otherwise.

    Returns
    -------
    The present value, scaled by ``face_value`` through the cashflow amounts.

    Raises
    ------
    UnsupportedInstrumentError
        If ``instrument`` is not a fixed-income security, or ``discount_curve``
        has no ``discount_factor``.
    ValueError
        If the curve returns a non-finite factor, a non-scalar one, or -- for
        the curves this library ships -- a non-positive one.

    Notes
    -----
    Each cashflow is discounted at **its own** payment time. Discounting a
    whole schedule at the final maturity is a different and wrong number, and
    the shape of this function is what makes that error unavailable.

    Examples
    --------
    >>> from fast_vollib.instruments import ZeroCouponBond
    >>> from fast_vollib.pricing import present_value
    >>> from fast_vollib.rates import CIRDiscountCurve, FlatDiscountCurve
    >>> bond = ZeroCouponBond(maturity=1.0, face_value=1000.0)
    >>> round(present_value(bond, discount_curve=FlatDiscountCurve(rate=0.05)), 8)
    951.2294245

    The face value scales the answer exactly, because it scales the cashflow:

    >>> ten = ZeroCouponBond(maturity=1.0, face_value=10.0)
    >>> one = ZeroCouponBond(maturity=1.0, face_value=1.0)
    >>> curve = CIRDiscountCurve(kappa=0.3, theta=0.04, volatility=0.1, initial_rate=0.04)
    >>> present_value(ten, discount_curve=curve) == 10.0 * present_value(one, discount_curve=curve)
    True

    A curve of the caller's own works with no registration:

    >>> class NeverDiscount:
    ...     def discount_factor(self, maturity):
    ...         return 1.0
    >>> present_value(bond, discount_curve=NeverDiscount())
    1000.0
    """
    if not isinstance(instrument, FixedIncomeSecurity):
        raise UnsupportedInstrumentError(
            f"present_value values fixed-income securities against a discount curve; "
            f"{type(instrument).__name__} is not one. A contract with a terminal payoff "
            f"is priced with price_instrument() or by simulation."
        )
    if not callable(getattr(discount_curve, "discount_factor", None)):
        raise UnsupportedInstrumentError(
            f"discount_curve must provide a discount_factor(maturity) method; "
            f"{type(discount_curve).__name__} does not. See "
            f"fast_vollib.rates.DiscountCurve for the contract, which is structural: "
            f"any object with that method works."
        )

    require_positive = _is_shipped_curve(discount_curve)
    schedule = cashflows(instrument)

    total: Any = None
    for cashflow in schedule:
        factor = _validated_factor(
            discount_curve.discount_factor(cashflow.payment_time),
            payment_time=cashflow.payment_time,
            require_positive=require_positive,
        )
        term = cashflow.amount * factor
        total = term if total is None else total + term

    if return_native:
        return total
    xp = get_namespace(total)
    host = xp.to_numpy(total) if xp.is_native(total) else np.asarray(total)
    return float(host) if host.shape == () else host
