"""Contracts whose payoff is more than a difference of two numbers.

Everything here is built on the same validation core as
:mod:`fast_vollib.instruments.options`, holds terms and nothing else, and
serializes through the same versioned codec.  What distinguishes these types is
what their payoff needs: some still read only the underlier's value at
maturity, and others need the whole trajectory.
:func:`fast_vollib.instruments.payoff_requirement` is where that is declared,
and an engine consults it before it simulates rather than after.

Conventions that change what a contract is *worth* are contract fields --
barrier direction, averaging method, strike convention -- never defaults buried
in an evaluator.  Conventions that are fixed by the type are constant
properties rather than stored fields, so a record cannot describe a
cash-settled instrument as physically settled.
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
from .enums import (
    AveragingMethod,
    BarrierType,
    InstrumentKind,
    OptionType,
    SettlementType,
    StrikeConvention,
)
from .errors import InstrumentValidationError

__all__ = [
    "AsianOption",
    "BarrierOption",
    "BinaryOption",
    "LookbackOption",
    "VarianceSwap",
]


def _resolved_strike(
    strike: object, convention: StrikeConvention, *, type_name: str
) -> float | None:
    """A strike under ``FIXED``, or ``None`` under ``FLOATING``.

    The two conventions are different instruments, and a contract carrying both
    a floating convention and a strike would be describing neither. Requiring
    exactly one makes the record self-consistent: a reader never has to guess
    which field the payoff will actually consult.
    """
    if convention is StrikeConvention.FIXED:
        if strike is None:
            raise InstrumentValidationError(
                f"A fixed-strike {type_name} needs a strike; got None. Use "
                f"strike_convention='floating' for a contract struck off its own path."
            )
        return ensure_positive(strike, field="strike")
    if strike is not None:
        raise InstrumentValidationError(
            f"A floating-strike {type_name} takes its strike from the path, so strike "
            f"must be None; got {strike!r}."
        )
    return None


@dataclass(frozen=True, slots=True, kw_only=True)
class BinaryOption(Derivative):
    """A cash-or-nothing digital option on a single underlier.

    Pays a fixed amount if the underlier finishes on the right side of the
    strike, and nothing otherwise.  The payoff is therefore discontinuous at
    the strike, which is a property of the contract rather than of any
    numerical method used on it.

    Parameters
    ----------
    underlier : InstrumentRef or Asset or str
        Normalized to a reference, as everywhere else in the package.
    option_type : OptionType or str
        ``"call"`` pays when the terminal value is *above* the strike,
        ``"put"`` when it is below.  The kernels' ``"c"`` / ``"p"`` flags work
        too.
    strike : float
        Strictly positive.
    maturity : float
        Year fraction; non-negative.  Zero is admissible and means expiry,
        where the payoff is still exactly defined.
    cash_amount : float, default 1.0
        What one contract pays when it finishes in the money.  Strictly
        positive; a short position is expressed with a negative ``notional``,
        not a negative cash amount, so the two cannot cancel into an ambiguity.
    notional : float, default 1.0
        Contract multiplier; negative denotes a short position.
    instrument_id : str, optional

    Notes
    -----
    At the strike exactly, both a call and a put pay zero: the comparison is
    strict on both sides.  Any other rule would make one of the two pay at a
    boundary the market treats as unresolved, and would break the identity that
    a call and a put on the same strike never both pay.

    ``settlement`` is a constant property rather than a field.  A digital
    settles in cash by definition -- there is nothing to deliver -- so storing
    a settlement would admit a record describing a contract that does not
    exist.

    Examples
    --------
    >>> from fast_vollib.instruments import BinaryOption
    >>> digital = BinaryOption(
    ...     underlier="SPX", option_type="c", strike=5000.0, maturity=0.5,
    ...     cash_amount=10.0,
    ... )
    >>> digital.kind.value, digital.settlement.value
    ('binary_option', 'cash')
    >>> digital.flag
    'c'
    """

    underlier: InstrumentRef
    option_type: OptionType
    strike: float
    maturity: float
    cash_amount: float = 1.0

    def __post_init__(self) -> None:
        Derivative.__post_init__(self)
        object.__setattr__(self, "underlier", coerce_underlier(self.underlier))
        object.__setattr__(self, "option_type", _parse_option_type(self.option_type))
        object.__setattr__(self, "strike", ensure_positive(self.strike, field="strike"))
        object.__setattr__(self, "maturity", ensure_non_negative(self.maturity, field="maturity"))
        object.__setattr__(
            self, "cash_amount", ensure_positive(self.cash_amount, field="cash_amount")
        )

    @property
    def kind(self) -> InstrumentKind:
        return InstrumentKind.BINARY_OPTION

    @property
    def underliers(self) -> tuple[InstrumentRef, ...]:
        return (self.underlier,)

    @property
    def settlement(self) -> SettlementType:
        """Always cash: a digital has nothing to deliver."""
        return SettlementType.CASH

    @property
    def flag(self) -> Literal["c", "p"]:
        """The single-character flag the pricing kernels take."""
        return "c" if self.option_type is OptionType.CALL else "p"


@dataclass(frozen=True, slots=True, kw_only=True)
class AsianOption(Derivative):
    """An average-rate or average-strike option on a single underlier.

    The payoff depends on an average of the underlier over the contract's life
    rather than on its value at maturity alone, which is what makes it
    path-dependent: :func:`fast_vollib.instruments.payoff_requirement` reports
    ``PATH``, and evaluating one needs a
    :class:`~fast_vollib.simulation.Scenario` rather than a terminal state.

    Parameters
    ----------
    underlier : InstrumentRef or Asset or str
    option_type : OptionType or str
    strike : float, optional
        Required under a fixed strike convention and forbidden under a floating
        one, where the average itself plays the strike's role.
    averaging_method : AveragingMethod or str
        ``"arithmetic"`` or ``"geometric"``. A term of the contract: the two
        are different instruments with different values.
    strike_convention : StrikeConvention or str
        ``"fixed"`` compares the average against ``strike``; ``"floating"``
        compares the terminal value against the average.
    maturity : float
        Year fraction; strictly positive. A path-dependent payoff needs a path,
        and a contract expiring now has none.
    settlement : SettlementType or str, default ``"cash"``
    notional : float, default 1.0
    instrument_id : str, optional

    Notes
    -----
    The averaging *schedule* is not a contract field here. Fixings are the
    observation times of the scenario the payoff is evaluated on, excluding the
    valuation date, so the schedule is stated once -- as the simulation grid --
    rather than twice in two places that could disagree. A contract with its
    own fixing calendar is a separate type, not a default.

    Examples
    --------
    >>> from fast_vollib.instruments import AsianOption, payoff_requirement
    >>> average_rate = AsianOption(
    ...     underlier="ACME", option_type="call", strike=100.0,
    ...     averaging_method="arithmetic", strike_convention="fixed", maturity=1.0,
    ... )
    >>> payoff_requirement(average_rate).value
    'path'
    >>> average_strike = AsianOption(
    ...     underlier="ACME", option_type="put", averaging_method="geometric",
    ...     strike_convention="floating", maturity=1.0,
    ... )
    >>> average_strike.strike is None
    True
    """

    underlier: InstrumentRef
    option_type: OptionType
    averaging_method: AveragingMethod
    strike_convention: StrikeConvention
    maturity: float
    strike: float | None = None
    settlement: SettlementType = SettlementType.CASH

    def __post_init__(self) -> None:
        Derivative.__post_init__(self)
        object.__setattr__(self, "underlier", coerce_underlier(self.underlier))
        object.__setattr__(self, "option_type", _parse_option_type(self.option_type))
        object.__setattr__(
            self,
            "averaging_method",
            ensure_enum(self.averaging_method, AveragingMethod, field="averaging_method"),
        )
        object.__setattr__(
            self,
            "strike_convention",
            ensure_enum(self.strike_convention, StrikeConvention, field="strike_convention"),
        )
        object.__setattr__(
            self,
            "strike",
            _resolved_strike(self.strike, self.strike_convention, type_name="Asian option"),
        )
        object.__setattr__(self, "maturity", ensure_positive(self.maturity, field="maturity"))
        object.__setattr__(
            self, "settlement", ensure_enum(self.settlement, SettlementType, field="settlement")
        )

    @property
    def kind(self) -> InstrumentKind:
        return InstrumentKind.ASIAN_OPTION

    @property
    def underliers(self) -> tuple[InstrumentRef, ...]:
        return (self.underlier,)

    @property
    def flag(self) -> Literal["c", "p"]:
        """The single-character flag the pricing kernels take."""
        return "c" if self.option_type is OptionType.CALL else "p"


@dataclass(frozen=True, slots=True, kw_only=True)
class BarrierOption(Derivative):
    """A knock-in or knock-out option on a single underlier.

    Pays a European intrinsic value at maturity, but only if the underlier did
    -- or did not -- touch a barrier level along the way.  That makes the
    payoff depend on the whole trajectory, so
    :func:`fast_vollib.instruments.payoff_requirement` reports ``PATH``.

    Parameters
    ----------
    underlier : InstrumentRef or Asset or str
    option_type : OptionType or str
        The intrinsic payoff at maturity, exactly as for a European option.
    strike : float
        Strictly positive.
    barrier : float
        The monitored level; strictly positive. It is *not* compared against
        spot at construction: whether a barrier is above or below the market is
        a market observation, and a contract that validated against one would
        become invalid when the market moved.
    barrier_type : BarrierType or str
        Direction and knock sense together: ``"up_and_in"``, ``"up_and_out"``,
        ``"down_and_in"``, ``"down_and_out"``.
    maturity : float
        Year fraction; strictly positive.
    settlement : SettlementType or str, default ``"cash"``
    notional : float, default 1.0
    instrument_id : str, optional

    Notes
    -----
    Monitoring is **discrete and inclusive**: the barrier counts as touched
    when an observed state reaches it, at any point of the scenario's grid
    including the first and last. A contract monitored continuously is worth
    something different, and evaluating this one on a coarse grid does not
    approximate it -- it prices a differently monitored contract. Nothing here
    applies a Brownian-bridge or other continuity correction.

    There is no rebate. A knocked-out contract pays nothing.

    Examples
    --------
    >>> from fast_vollib.instruments import BarrierOption
    >>> knock_out = BarrierOption(
    ...     underlier="ACME", option_type="call", strike=100.0, barrier=130.0,
    ...     barrier_type="up_and_out", maturity=1.0,
    ... )
    >>> knock_out.barrier_type.is_up, knock_out.barrier_type.knocks_in
    (True, False)
    """

    underlier: InstrumentRef
    option_type: OptionType
    strike: float
    barrier: float
    barrier_type: BarrierType
    maturity: float
    settlement: SettlementType = SettlementType.CASH

    def __post_init__(self) -> None:
        Derivative.__post_init__(self)
        object.__setattr__(self, "underlier", coerce_underlier(self.underlier))
        object.__setattr__(self, "option_type", _parse_option_type(self.option_type))
        object.__setattr__(self, "strike", ensure_positive(self.strike, field="strike"))
        object.__setattr__(self, "barrier", ensure_positive(self.barrier, field="barrier"))
        object.__setattr__(
            self, "barrier_type", ensure_enum(self.barrier_type, BarrierType, field="barrier_type")
        )
        object.__setattr__(self, "maturity", ensure_positive(self.maturity, field="maturity"))
        object.__setattr__(
            self, "settlement", ensure_enum(self.settlement, SettlementType, field="settlement")
        )

    @property
    def kind(self) -> InstrumentKind:
        return InstrumentKind.BARRIER_OPTION

    @property
    def underliers(self) -> tuple[InstrumentRef, ...]:
        return (self.underlier,)

    @property
    def flag(self) -> Literal["c", "p"]:
        """The single-character flag the pricing kernels take."""
        return "c" if self.option_type is OptionType.CALL else "p"


@dataclass(frozen=True, slots=True, kw_only=True)
class LookbackOption(Derivative):
    """An option struck against, or settled at, the extreme of the path.

    Parameters
    ----------
    underlier : InstrumentRef or Asset or str
    option_type : OptionType or str
    strike : float, optional
        Required under a fixed strike convention and forbidden under a floating
        one, where the path's own extreme plays the strike's role.
    strike_convention : StrikeConvention or str
        ``"fixed"`` pays the best level reached against a strike agreed at
        inception; ``"floating"`` pays the terminal value against the best
        level reached.
    maturity : float
        Year fraction; strictly positive.
    settlement : SettlementType or str, default ``"cash"``
    notional : float, default 1.0
    instrument_id : str, optional

    Notes
    -----
    The extremes are taken over the whole scenario, **including** the valuation
    date and maturity. Unlike an average, a running maximum is a property of
    the path as a whole rather than a set of agreed fixings, and excluding
    either end would silently discard an observation the contract monitors.

    Monitoring is discrete, at the scenario's own observation times. A
    continuously monitored lookback is worth more, and no correction for that
    is applied here.

    Examples
    --------
    >>> from fast_vollib.instruments import LookbackOption
    >>> floating = LookbackOption(
    ...     underlier="ACME", option_type="call", strike_convention="floating",
    ...     maturity=1.0,
    ... )
    >>> floating.strike is None
    True
    """

    underlier: InstrumentRef
    option_type: OptionType
    strike_convention: StrikeConvention
    maturity: float
    strike: float | None = None
    settlement: SettlementType = SettlementType.CASH

    def __post_init__(self) -> None:
        Derivative.__post_init__(self)
        object.__setattr__(self, "underlier", coerce_underlier(self.underlier))
        object.__setattr__(self, "option_type", _parse_option_type(self.option_type))
        object.__setattr__(
            self,
            "strike_convention",
            ensure_enum(self.strike_convention, StrikeConvention, field="strike_convention"),
        )
        object.__setattr__(
            self,
            "strike",
            _resolved_strike(self.strike, self.strike_convention, type_name="lookback option"),
        )
        object.__setattr__(self, "maturity", ensure_positive(self.maturity, field="maturity"))
        object.__setattr__(
            self, "settlement", ensure_enum(self.settlement, SettlementType, field="settlement")
        )

    @property
    def kind(self) -> InstrumentKind:
        return InstrumentKind.LOOKBACK_OPTION

    @property
    def underliers(self) -> tuple[InstrumentRef, ...]:
        return (self.underlier,)

    @property
    def flag(self) -> Literal["c", "p"]:
        """The single-character flag the pricing kernels take."""
        return "c" if self.option_type is OptionType.CALL else "p"


@dataclass(frozen=True, slots=True, kw_only=True)
class VarianceSwap(Derivative):
    r"""A swap on realized variance against a variance strike.

    Pays the difference between the variance the underlier actually realized
    over the contract's life and a level agreed at inception, so it is a
    directional position on variance itself rather than on the underlier.

    Parameters
    ----------
    underlier : InstrumentRef or Asset or str
    strike_variance : float
        The agreed level, quoted as a *variance* -- the square of a volatility.
        Non-negative; zero is admissible and makes the contract a pure long
        position in realized variance.
    maturity : float
        Year fraction; strictly positive.
    notional : float, default 1.0
        Variance notional: what one unit of realized-minus-strike variance
        pays. Negative denotes the short side.
    instrument_id : str, optional

    Notes
    -----
    Realized variance is

    .. math::

        \mathrm{RV} = \frac{1}{T}\sum_{i=1}^{n}
            \left(\log \frac{S_i}{S_{i-1}}\right)^2 ,

    the convention traded contracts use. Two things it deliberately is not:
    there is no subtraction of the sample mean, because the contract pays on
    the sum of squared returns rather than on a statistical variance estimate;
    and there is no annualization factor of 252 or 365 anywhere, because
    dividing by the year fraction already annualizes. For daily observations
    :math:`T = n/252` and :math:`1/T` *is* the familiar :math:`252/n`, so
    applying both would annualize twice.

    ``settlement`` is a constant property rather than a field: a variance swap
    settles in cash by definition, since there is nothing to deliver.

    Examples
    --------
    >>> from fast_vollib.instruments import VarianceSwap
    >>> swap = VarianceSwap(underlier="SPX", strike_variance=0.04, maturity=1.0)
    >>> swap.strike_variance, swap.settlement.value
    (0.04, 'cash')
    """

    underlier: InstrumentRef
    strike_variance: float
    maturity: float

    def __post_init__(self) -> None:
        Derivative.__post_init__(self)
        object.__setattr__(self, "underlier", coerce_underlier(self.underlier))
        object.__setattr__(
            self,
            "strike_variance",
            ensure_non_negative(self.strike_variance, field="strike_variance"),
        )
        object.__setattr__(self, "maturity", ensure_positive(self.maturity, field="maturity"))

    @property
    def kind(self) -> InstrumentKind:
        return InstrumentKind.VARIANCE_SWAP

    @property
    def underliers(self) -> tuple[InstrumentRef, ...]:
        return (self.underlier,)

    @property
    def settlement(self) -> SettlementType:
        """Always cash: a variance swap has nothing to deliver."""
        return SettlementType.CASH
