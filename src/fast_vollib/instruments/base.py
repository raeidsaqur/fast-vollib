"""Contract roots: :class:`Instrument`, :class:`Asset`, :class:`Derivative`.

The layering rule for everything in this module: a contract holds *terms*, and
nothing else.  No arrays or tensors, no market observations, no curves, no
devices or dtypes, no RNG state, no calibrated parameters, and no attached
pricing engine.  What is left is a value: frozen, slotted, hashable,
comparable, and serializable, with an identity that does not depend on any
backend being installed.

Two consequences are worth spelling out because they differ from the common
object-graph design:

*Underliers are references, not objects.*  A :class:`Derivative` points at an
:class:`InstrumentRef` -- an identifier plus optional descriptive metadata --
rather than at an :class:`Asset` instance.  A book of a million options then
carries a million small references instead of a million edges into a shared
mutable graph, equality stays structural, and the JSON record of an option is
self-contained.

*Maturity is a year fraction.*  ``maturity: float`` is measured in years and is
never a date or a date-or-float union.  Calendars, day counts, and schedule
resolution belong in a layer above this one; admitting them here would make
every contract's meaning depend on an evaluation date that contracts do not
carry.

Examples
--------
>>> from fast_vollib.instruments import Asset, AssetClass
>>> spx = Asset(identifier="SPX", asset_class=AssetClass.INDEX, currency="usd")
>>> spx.currency
'USD'
>>> spx.ref()
InstrumentRef(identifier='SPX', asset_class=<AssetClass.INDEX: 'index'>, currency='USD')
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from ._validate import (
    ensure_enum,
    ensure_identifier,
    ensure_nonzero,
    ensure_optional_currency,
    ensure_optional_identifier,
)
from .enums import AssetClass, InstrumentKind

__all__ = ["Asset", "Derivative", "Instrument", "InstrumentRef"]


@dataclass(frozen=True, slots=True, kw_only=True)
class InstrumentRef:
    """An immutable reference to an underlier.

    Parameters
    ----------
    identifier : str
        Symbol, ticker, or any stable key for the underlier.  Trimmed on
        construction and required to be non-empty.
    asset_class : AssetClass or str, optional
        Descriptive only.  It never selects a pricing model.
    currency : str, optional
        Upper-cased on construction; kept as a plain string rather than an
        enum.

    Notes
    -----
    A reference deliberately carries no market data.  Resolving it to a spot
    price, a curve, or a dividend schedule is the caller's job, and the result
    is passed alongside the contract as market inputs -- never stored on it.

    Examples
    --------
    >>> from fast_vollib.instruments import InstrumentRef
    >>> ref = InstrumentRef(identifier="  ACME ", currency="usd")
    >>> ref.identifier, ref.currency
    ('ACME', 'USD')
    """

    identifier: str
    asset_class: AssetClass | None = None
    currency: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "identifier", ensure_identifier(self.identifier, field="identifier")
        )
        if self.asset_class is not None:
            object.__setattr__(
                self, "asset_class", ensure_enum(self.asset_class, AssetClass, field="asset_class")
            )
        object.__setattr__(
            self, "currency", ensure_optional_currency(self.currency, field="currency")
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class Instrument(ABC):
    """Base class for a financial contract or market-observable asset.

    Parameters
    ----------
    instrument_id : str, optional
        Caller-chosen identity for this record (an exchange symbol, a position
        key, a row id).  It participates in equality and serialization but has
        no valuation meaning.
    """

    instrument_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "instrument_id",
            ensure_optional_identifier(self.instrument_id, field="instrument_id"),
        )

    @property
    @abstractmethod
    def kind(self) -> InstrumentKind:
        """The discriminator for this instrument type."""


@dataclass(frozen=True, slots=True, kw_only=True)
class Asset(Instrument):
    """An economic underlier: what a derivative is written on.

    Parameters
    ----------
    identifier : str
        Symbol or ticker; trimmed and required to be non-empty.
    asset_class : AssetClass or str
        The economic class.  Descriptive; it never selects a model.
    currency : str, optional
        Upper-cased on construction.
    instrument_id : str, optional
        See :class:`Instrument`.

    Notes
    -----
    An asset is a description, not a state holder.  It gains no ``.spot``, no
    simulated path buffer, and no ``.to(device)``: simulation is a separate
    operation returning explicit values, so re-running it can never change what
    an asset means.
    """

    identifier: str
    asset_class: AssetClass
    currency: str | None = None

    def __post_init__(self) -> None:
        Instrument.__post_init__(self)
        object.__setattr__(
            self, "identifier", ensure_identifier(self.identifier, field="identifier")
        )
        object.__setattr__(
            self, "asset_class", ensure_enum(self.asset_class, AssetClass, field="asset_class")
        )
        object.__setattr__(
            self, "currency", ensure_optional_currency(self.currency, field="currency")
        )

    @property
    def kind(self) -> InstrumentKind:
        return InstrumentKind.ASSET

    def ref(self) -> InstrumentRef:
        """The :class:`InstrumentRef` a derivative should point at.

        Examples
        --------
        >>> from fast_vollib.instruments import Asset, AssetClass
        >>> Asset(identifier="ACME", asset_class="equity").ref().identifier
        'ACME'
        """
        return InstrumentRef(
            identifier=self.identifier,
            asset_class=self.asset_class,
            currency=self.currency,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class Derivative(Instrument, ABC):
    """An instrument whose payoff depends on one or more underliers.

    Parameters
    ----------
    notional : float, default 1.0
        Contract multiplier.  Required to be finite and non-zero: a zero
        notional is economically empty, and it would divide by zero when an
        observed price is reduced to unit terms before implied-volatility
        inversion.  Negative notionals are valid and denote a short position.
    instrument_id : str, optional
        See :class:`Instrument`.
    """

    notional: float = 1.0

    def __post_init__(self) -> None:
        Instrument.__post_init__(self)
        object.__setattr__(self, "notional", ensure_nonzero(self.notional, field="notional"))

    @property
    @abstractmethod
    def underliers(self) -> tuple[InstrumentRef, ...]:
        """The underliers this contract references, in a stable order."""
