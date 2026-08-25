"""European option contracts.

European exercise is the only option style represented by this package.
Unsupported styles are not approximated with a European price;
:class:`~fast_vollib.instruments.UnsupportedInstrumentError` identifies an
instrument type for which no implementation exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ._validate import (
    coerce_underlier,
    ensure_enum,
    ensure_non_negative,
    ensure_positive,
    parse_option_type as _parse_option_type,
)
from .base import Derivative, InstrumentRef
from .enums import ExerciseStyle, InstrumentKind, OptionType, SettlementType

__all__ = ["EuropeanOption"]


@dataclass(frozen=True, slots=True, kw_only=True)
class EuropeanOption(Derivative):
    """A European-exercise option on a single underlier.

    Parameters
    ----------
    underlier : InstrumentRef or Asset or str
        What the option is written on.  An :class:`~fast_vollib.instruments.Asset`
        or a bare identifier string is normalized to an
        :class:`~fast_vollib.instruments.InstrumentRef`, so the stored value is
        always a reference.
    option_type : OptionType or str
        ``"call"`` / ``"put"``, or the kernels' ``"c"`` / ``"p"`` flags in
        either case.  Serialization always emits the long spelling.
    strike : float
        Strike price; must be strictly positive.
    maturity : float
        Time to expiry as a year fraction from valuation.  Must be
        non-negative; zero means expiry.  This is never a date: calendars and
        day counts belong to a layer above the contract.
    settlement : SettlementType or str, default ``"cash"``
    notional : float, default 1.0
        Contract multiplier; negative denotes a short position.  Scales price
        and Greeks, and an observed price is divided by it before implied
        volatility is inverted.
    instrument_id : str, optional

    Notes
    -----
    ``exercise_style`` is a constant property rather than a stored field.
    Storing it would admit ``EuropeanOption(exercise_style="american")``, an
    object whose type and terms disagree.

    Examples
    --------
    >>> from fast_vollib.instruments import EuropeanOption
    >>> call = EuropeanOption(underlier="SPX", option_type="c", strike=5000.0, maturity=0.75)
    >>> call.option_type.value, call.flag
    ('call', 'c')
    >>> call.exercise_style.value
    'european'
    >>> call == EuropeanOption(
    ...     underlier="SPX", option_type="call", strike=5000.0, maturity=0.75
    ... )
    True
    """

    underlier: InstrumentRef
    option_type: OptionType
    strike: float
    maturity: float
    settlement: SettlementType = SettlementType.CASH

    def __post_init__(self) -> None:
        Derivative.__post_init__(self)
        object.__setattr__(self, "underlier", coerce_underlier(self.underlier))
        object.__setattr__(self, "option_type", _parse_option_type(self.option_type))
        object.__setattr__(self, "strike", ensure_positive(self.strike, field="strike"))
        object.__setattr__(self, "maturity", ensure_non_negative(self.maturity, field="maturity"))
        object.__setattr__(
            self, "settlement", ensure_enum(self.settlement, SettlementType, field="settlement")
        )

    @property
    def kind(self) -> InstrumentKind:
        return InstrumentKind.EUROPEAN_OPTION

    @property
    def underliers(self) -> tuple[InstrumentRef, ...]:
        return (self.underlier,)

    @property
    def exercise_style(self) -> ExerciseStyle:
        """Always :attr:`~fast_vollib.instruments.ExerciseStyle.EUROPEAN`."""
        return ExerciseStyle.EUROPEAN

    @property
    def flag(self) -> Literal["c", "p"]:
        """The single-character flag the pricing kernels take.

        Converts the public enum spelling to the flag expected by the
        functional pricing API.
        """
        return "c" if self.option_type is OptionType.CALL else "p"
