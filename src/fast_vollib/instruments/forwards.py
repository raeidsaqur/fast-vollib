"""Linear contracts: :class:`Forward` and :class:`Future`.

Both are agreements to exchange an underlier at maturity for a price fixed at
inception, and both therefore carry the same three terms.  They are kept as
separate types rather than one type with a flag because their cashflow timing
genuinely differs: a forward settles once, at maturity, against the price
agreed at trade date, while a future is margined daily against the exchange's
settlement price, so its economic cashflow is a stream of variation margin.

The payoff evaluator models the *terminal contract payoff* of each -- what the
position is worth at maturity -- and does not model a futures margin schedule.
The distinct types preserve the contracts' different cashflow semantics even
though their terminal-payoff formulas coincide.
"""

from __future__ import annotations

from dataclasses import dataclass

from ._validate import coerce_underlier, ensure_enum, ensure_finite_float, ensure_non_negative
from .base import Derivative, InstrumentRef
from .enums import InstrumentKind, SettlementType

__all__ = ["Forward", "Future"]


@dataclass(frozen=True, slots=True, kw_only=True)
class Forward(Derivative):
    """An over-the-counter agreement to buy the underlier at maturity.

    Parameters
    ----------
    underlier : InstrumentRef or Asset or str
        What the contract is written on.  An asset or a bare identifier is
        normalized to an :class:`~fast_vollib.instruments.InstrumentRef`.
    delivery_price : float
        The price agreed at inception.  Any finite value is admissible,
        negative included: commodity forwards have settled below zero.
    maturity : float
        Delivery date as a year fraction from valuation.  Must be
        non-negative; zero means expiry.
    settlement : SettlementType or str, default ``"cash"``
        Whether the contract settles in cash or by physical delivery.
    notional : float, default 1.0
        Contract multiplier; negative denotes a short position.
    instrument_id : str, optional
        Caller-chosen record identity.

    Examples
    --------
    >>> from fast_vollib.instruments import Forward
    >>> fwd = Forward(underlier="ACME", delivery_price=100.0, maturity=0.5)
    >>> fwd.underliers[0].identifier
    'ACME'
    >>> fwd.kind.value
    'forward'
    """

    underlier: InstrumentRef
    delivery_price: float
    maturity: float
    settlement: SettlementType = SettlementType.CASH

    def __post_init__(self) -> None:
        Derivative.__post_init__(self)
        object.__setattr__(self, "underlier", coerce_underlier(self.underlier))
        object.__setattr__(
            self,
            "delivery_price",
            ensure_finite_float(self.delivery_price, field="delivery_price"),
        )
        object.__setattr__(self, "maturity", ensure_non_negative(self.maturity, field="maturity"))
        object.__setattr__(
            self, "settlement", ensure_enum(self.settlement, SettlementType, field="settlement")
        )

    @property
    def kind(self) -> InstrumentKind:
        return InstrumentKind.FORWARD

    @property
    def underliers(self) -> tuple[InstrumentRef, ...]:
        return (self.underlier,)


@dataclass(frozen=True, slots=True, kw_only=True)
class Future(Derivative):
    """An exchange-traded contract to buy the underlier at maturity.

    Parameters
    ----------
    underlier : InstrumentRef or Asset or str
        What the contract is written on.
    contract_price : float
        The traded futures price.  Named separately from a forward's
        ``delivery_price`` because it is the level the position is marked
        against, not a one-off delivery obligation.  Any finite value.
    maturity : float
        Expiry as a year fraction from valuation; non-negative.
    settlement : SettlementType or str, default ``"cash"``
    notional : float, default 1.0
    instrument_id : str, optional

    Notes
    -----
    The payoff evaluator returns the terminal contract payoff. Daily variation
    margin is not modelled, so a future and an otherwise identical forward
    evaluate to the same terminal cashflow. They remain distinct types because
    their economic cashflow timing differs.

    Examples
    --------
    >>> from fast_vollib.instruments import Future
    >>> Future(underlier="CL", contract_price=75.0, maturity=0.25).kind.value
    'future'
    """

    underlier: InstrumentRef
    contract_price: float
    maturity: float
    settlement: SettlementType = SettlementType.CASH

    def __post_init__(self) -> None:
        Derivative.__post_init__(self)
        object.__setattr__(self, "underlier", coerce_underlier(self.underlier))
        object.__setattr__(
            self, "contract_price", ensure_finite_float(self.contract_price, field="contract_price")
        )
        object.__setattr__(self, "maturity", ensure_non_negative(self.maturity, field="maturity"))
        object.__setattr__(
            self, "settlement", ensure_enum(self.settlement, SettlementType, field="settlement")
        )

    @property
    def kind(self) -> InstrumentKind:
        return InstrumentKind.FUTURE

    @property
    def underliers(self) -> tuple[InstrumentRef, ...]:
        return (self.underlier,)
