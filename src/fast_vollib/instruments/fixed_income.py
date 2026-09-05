"""The third branch of the contract tree: securities that pay dated cashflows.

:class:`~fast_vollib.instruments.Asset` describes an underlier.
:class:`~fast_vollib.instruments.Derivative` describes a contract whose payment
depends on one.  A bond is neither: it promises a fixed schedule of payments
that no underlier enters, so it hangs directly off
:class:`~fast_vollib.instruments.Instrument`.

*A bond is not routed through* :func:`~fast_vollib.instruments.payoff`, and that
is the load-bearing decision in this module.  ``payoff`` maps *the state at one
horizon* to *one cashflow* -- that signature is the whole reason an engine can
simulate a contract, and it cannot express "one unit at two years and one unit
at five".  Squeezing a bond into it would mean either inventing a fictional
terminal state or summing payments made at different dates as if they were
worth the same, and the second is not an approximation, it is an error.  So
fixed-income securities expose :func:`cashflows` instead, and every existing
entry point refuses them by name and points at
:func:`fast_vollib.pricing.present_value`.

The layering rule from :mod:`fast_vollib.instruments.base` holds unchanged: a
contract carries *terms*, and nothing else.  A bond therefore has payment times
and amounts and no curve, no yield, no accrued interest, and no settlement date.
Which curve discounts it is a valuation input supplied alongside it.

``face_value`` rather than ``Derivative.notional``: ``notional`` is documented as
a contract multiplier whose sign denotes a short position, and that meaning does
not belong on a primary security whose principal is redeemed.

Everything a calendar would decide is deliberately absent.  ``payment_time`` is a
year fraction, never a date, so a contract's meaning does not depend on an
evaluation date it does not carry.  Day counts, business-day rules, ex-coupon
periods, accrued interest, clean versus dirty price, and yield all belong to a
conventions layer above this one.

Examples
--------
>>> from fast_vollib.instruments import ZeroCouponBond, cashflows
>>> bond = ZeroCouponBond(maturity=2.0, face_value=1000.0, currency="usd")
>>> bond.currency
'USD'
>>> cashflows(bond)
(Cashflow(payment_time=2.0, amount=1000.0),)

A payment due today is a legitimate contract, worth its face value:

>>> cashflows(ZeroCouponBond(maturity=0.0))
(Cashflow(payment_time=0.0, amount=1.0),)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable as IterableABC
from dataclasses import dataclass

from ._validate import (
    ensure_finite_float,
    ensure_non_negative,
    ensure_optional_currency,
    ensure_positive,
)
from .base import Instrument
from .enums import InstrumentKind
from .errors import InstrumentValidationError, UnsupportedInstrumentError

__all__ = [
    "Cashflow",
    "FixedIncomeSecurity",
    "FixedRateBond",
    "ZeroCouponBond",
    "cashflows",
]


@dataclass(frozen=True, slots=True, order=True)
class Cashflow:
    """One payment, and when it is made.

    Parameters
    ----------
    payment_time : float
        Year fraction from the valuation date, non-negative and finite. Zero
        means "due now", which is not a degenerate case: the last payment of a
        matured bond is exactly that.
    amount : float
        Finite. May be negative -- a fee, or the paying leg of a schedule --
        and may be zero.

    Notes
    -----
    Ordered by ``payment_time`` first, so a schedule sorts chronologically
    without a key function; ``amount`` breaks ties only so that the ordering is
    total.

    Examples
    --------
    >>> from fast_vollib.instruments import Cashflow
    >>> Cashflow(payment_time=1.0, amount=25.0) < Cashflow(payment_time=2.0, amount=1.0)
    True
    """

    payment_time: float
    amount: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "payment_time", ensure_non_negative(self.payment_time, field="payment_time")
        )
        object.__setattr__(self, "amount", ensure_finite_float(self.amount, field="amount"))


@dataclass(frozen=True, slots=True, kw_only=True)
class FixedIncomeSecurity(Instrument, ABC):
    """A security that pays a schedule of amounts on dates fixed at inception.

    Parameters
    ----------
    face_value : float, default 1.0
        The principal redeemed, strictly positive and finite. A unit face is the
        default so a price reads as a discount factor.
    currency : str, optional
        Upper-cased on construction, matching
        :class:`~fast_vollib.instruments.Asset`. Descriptive: it never selects a
        curve, and discounting a USD bond off a EUR curve is the caller's error
        to make, not one this class silently prevents by picking a curve.
    instrument_id : str, optional
        See :class:`~fast_vollib.instruments.Instrument`.

    Notes
    -----
    :attr:`maturity` is declared here so that ``getattr(instrument, "maturity")``
    means one thing across every instrument family -- it is what
    ``MonteCarloEngine._maturity_of`` and the surface tooling already read off
    options and forwards.

    Abstract: it has no cashflows of its own to describe.
    """

    face_value: float = 1.0
    currency: str | None = None

    def __post_init__(self) -> None:
        Instrument.__post_init__(self)
        object.__setattr__(self, "face_value", ensure_positive(self.face_value, field="face_value"))
        object.__setattr__(
            self, "currency", ensure_optional_currency(self.currency, field="currency")
        )

    @property
    @abstractmethod
    def cashflows(self) -> tuple[Cashflow, ...]:
        """Every payment, in strictly increasing ``payment_time`` order."""

    @property
    @abstractmethod
    def maturity(self) -> float:
        """The time of the final payment."""


@dataclass(frozen=True, slots=True, kw_only=True)
class ZeroCouponBond(FixedIncomeSecurity):
    """One payment of ``face_value`` at ``maturity``, and nothing before it.

    Parameters
    ----------
    maturity : float
        Year fraction, non-negative and finite. Zero is a payment due now, whose
        present value is ``face_value`` under any curve.
    face_value : float, default 1.0
    currency : str, optional
    instrument_id : str, optional

    Notes
    -----
    Carries no coupon rate and no schedule, so there is nothing to keep
    consistent: the whole contract is one date and one amount. Its present value
    under a curve is ``face_value * curve.discount_factor(maturity)``, which is
    what makes it the natural instrument for reading a curve back out.

    Examples
    --------
    >>> from fast_vollib.instruments import ZeroCouponBond
    >>> bond = ZeroCouponBond(maturity=5.0, face_value=100.0)
    >>> bond.cashflows
    (Cashflow(payment_time=5.0, amount=100.0),)
    >>> bond.kind.value
    'zero_coupon_bond'

    A contract is a value: two bonds with the same terms are the same bond.

    >>> ZeroCouponBond(maturity=5.0) == ZeroCouponBond(maturity=5.0)
    True
    """

    maturity: float

    def __post_init__(self) -> None:
        FixedIncomeSecurity.__post_init__(self)
        object.__setattr__(self, "maturity", ensure_non_negative(self.maturity, field="maturity"))

    @property
    def kind(self) -> InstrumentKind:
        return InstrumentKind.ZERO_COUPON_BOND

    @property
    def cashflows(self) -> tuple[Cashflow, ...]:
        """The single redemption payment."""
        return (Cashflow(payment_time=self.maturity, amount=self.face_value),)


@dataclass(frozen=True, slots=True, kw_only=True)
class FixedRateBond(FixedIncomeSecurity):
    """Periodic coupons at a fixed rate, and the principal with the last one.

    Parameters
    ----------
    payment_times : sequence of float
        Year fractions, non-empty, finite, **strictly increasing**, and
        strictly positive. A payment at time zero is excluded here, unlike on
        :class:`ZeroCouponBond`: a coupon accrues over a period, and a period
        that ends at the valuation date began before it.
    accrual_fractions : sequence of float
        The year fraction each coupon accrues over, one per payment, finite and
        strictly positive. Supplied rather than derived, because deriving them
        from the payment times would silently impose a day-count convention --
        the gap between two payment dates is not the accrual factor under
        30/360, ACT/365, or ACT/ACT, and the three disagree.
    coupon_rate : float
        Annual rate as a decimal, finite. **Zero and negative are accepted**:
        a zero-coupon security with a schedule is a legitimate contract, and
        negative coupons occur on some inflation-linked and cross-currency
        structures.
    face_value : float, default 1.0
    currency : str, optional
    instrument_id : str, optional

    Notes
    -----
    Coupon ``i`` is ``face_value * coupon_rate * accrual_fractions[i]``, and the
    final payment adds ``face_value``. One `Cashflow` carries both, because on
    that date one payment is made.

    No ``frequency`` field, and its absence is deliberate. A frequency cannot
    describe a stub, a long first period, or an irregular redemption, so a
    contract carrying one would be unable to represent bonds that exist. The
    canonical terms are the schedule itself; resolving dates, calendars, and day
    counts *into* that schedule belongs to a conventions layer above this one,
    which this library does not implement.

    The two sequences are stored as tuples, so a contract stays hashable and
    cannot be mutated behind a reference a caller kept.

    Examples
    --------
    >>> from fast_vollib.instruments import FixedRateBond
    >>> bond = FixedRateBond(
    ...     payment_times=(0.5, 1.0),
    ...     accrual_fractions=(0.5, 0.5),
    ...     coupon_rate=0.04,
    ...     face_value=100.0,
    ... )
    >>> bond.cashflows
    (Cashflow(payment_time=0.5, amount=2.0), Cashflow(payment_time=1.0, amount=102.0))
    >>> bond.maturity
    1.0
    >>> bond.kind.value
    'fixed_rate_bond'

    A zero coupon rate leaves only the redemption, which is a schedule with a
    lot of zeros rather than a :class:`ZeroCouponBond`:

    >>> FixedRateBond(
    ...     payment_times=(1.0, 2.0), accrual_fractions=(1.0, 1.0), coupon_rate=0.0
    ... ).cashflows
    (Cashflow(payment_time=1.0, amount=0.0), Cashflow(payment_time=2.0, amount=1.0))
    """

    payment_times: tuple[float, ...]
    accrual_fractions: tuple[float, ...]
    coupon_rate: float

    def __post_init__(self) -> None:
        FixedIncomeSecurity.__post_init__(self)
        times = _ensure_schedule(self.payment_times, field="payment_times", positive=True)
        if len(times) > 1 and any(b <= a for a, b in zip(times, times[1:])):
            raise InstrumentValidationError(
                f"payment_times must be strictly increasing; got {list(times)}. Two "
                f"payments at one time are one payment, and an out-of-order schedule is "
                f"a data error rather than a contract this library will sort for you."
            )
        fractions = _ensure_schedule(
            self.accrual_fractions, field="accrual_fractions", positive=True
        )
        if len(fractions) != len(times):
            raise InstrumentValidationError(
                f"accrual_fractions has {len(fractions)} entries but payment_times has "
                f"{len(times)}. Each coupon accrues over exactly one period."
            )
        object.__setattr__(self, "payment_times", times)
        object.__setattr__(self, "accrual_fractions", fractions)
        object.__setattr__(
            self, "coupon_rate", ensure_finite_float(self.coupon_rate, field="coupon_rate")
        )

    @property
    def kind(self) -> InstrumentKind:
        return InstrumentKind.FIXED_RATE_BOND

    @property
    def maturity(self) -> float:
        """The final payment time."""
        return self.payment_times[-1]

    @property
    def cashflows(self) -> tuple[Cashflow, ...]:
        """One coupon per period, with the principal added to the last."""
        coupon = self.face_value * self.coupon_rate
        amounts = [coupon * fraction for fraction in self.accrual_fractions]
        amounts[-1] += self.face_value
        return tuple(
            Cashflow(payment_time=time, amount=amount)
            for time, amount in zip(self.payment_times, amounts)
        )


def _ensure_schedule(value: object, *, field: str, positive: bool) -> tuple[float, ...]:
    """A non-empty tuple of finite floats, from any sequence but a string.

    Strings are excluded explicitly. ``"0.5"`` is a sequence of three
    characters, and iterating it would produce a length-three schedule of
    characters that fail one at a time with a confusing message.
    """
    if isinstance(value, (str, bytes)) or not isinstance(value, IterableABC):
        raise InstrumentValidationError(
            f"{field} must be a sequence of numbers; got {type(value).__name__}."
        )
    entries: tuple[object, ...] = tuple(value)
    if not entries:
        raise InstrumentValidationError(
            f"{field} must not be empty: a bond with no payments is not a contract."
        )
    check = ensure_positive if positive else ensure_finite_float
    return tuple(check(entry, field=f"{field}[{index}]") for index, entry in enumerate(entries))


def cashflows(instrument: FixedIncomeSecurity) -> tuple[Cashflow, ...]:
    """Every payment ``instrument`` promises, in chronological order.

    The counterpart of :func:`~fast_vollib.instruments.payoff` for contracts
    whose payments are dated rather than terminal.  It is a separate dispatcher
    on purpose: ``payoff`` takes a state and returns one number, and no schedule
    fits through that signature without either inventing a state or discarding
    the dates.

    Parameters
    ----------
    instrument : FixedIncomeSecurity

    Returns
    -------
    tuple[Cashflow, ...]
        Strictly increasing in ``payment_time``.

    Raises
    ------
    UnsupportedInstrumentError
        For anything else, naming :func:`fast_vollib.pricing.present_value` as
        the route these contracts take.

    Examples
    --------
    >>> from fast_vollib.instruments import ZeroCouponBond, cashflows
    >>> cashflows(ZeroCouponBond(maturity=1.0, face_value=100.0))
    (Cashflow(payment_time=1.0, amount=100.0),)

    An option has a payoff, not a schedule, and is refused rather than coerced:

    >>> from fast_vollib.instruments import EuropeanOption
    >>> option = EuropeanOption(
    ...     underlier="ACME", option_type="call", strike=100.0, maturity=1.0
    ... )
    >>> from fast_vollib.instruments import UnsupportedInstrumentError
    >>> try:
    ...     cashflows(option)
    ... except UnsupportedInstrumentError as error:
    ...     print(str(error).split(".")[0])
    EuropeanOption has no dated cashflows
    """
    if not isinstance(instrument, FixedIncomeSecurity):
        raise UnsupportedInstrumentError(
            f"{type(instrument).__name__} has no dated cashflows. cashflows() serves "
            f"fixed-income securities, whose payments are dated rather than terminal; "
            f"a contract with a terminal payoff is evaluated with payoff(). Present "
            f"values of fixed-income securities come from "
            f"fast_vollib.pricing.present_value."
        )
    return instrument.cashflows
