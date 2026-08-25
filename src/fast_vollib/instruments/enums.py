"""Closed vocabularies for instrument records and pricing adapters.

Every enum is declared as ``class X(str, Enum)`` rather than
:class:`enum.StrEnum`, which is 3.11+; fast-vollib supports Python 3.10.  The
mixin gives the members the string behaviour serialization and column
normalization rely on while keeping the library's floor version.

Values are stable, lower-case, and snake-cased: they are the wire format for
:mod:`fast_vollib.instruments.serialization` and the keys of the checked-in
JSON Schema, so renaming one is a schema-breaking change.

Each member corresponds to behaviour implemented by the library. Unsupported
instrument types and operations are omitted instead of being advertised as
available capabilities.

Examples
--------
>>> from fast_vollib.instruments import OptionType, PricingModel
>>> OptionType.CALL.value
'call'
>>> PricingModel.BLACK_SCHOLES_MERTON.value
'black_scholes_merton'
"""

from __future__ import annotations

from enum import Enum

__all__ = [
    "AssetClass",
    "ExerciseStyle",
    "IVSolver",
    "InstrumentKind",
    "OptionType",
    "PayoffRequirement",
    "PricingModel",
    "SettlementType",
]


class AssetClass(str, Enum):
    """Economic class of an underlier.

    Purely descriptive metadata.  An asset class never selects a pricing model,
    an engine, a solver, or a backend; it may only be used to *validate* an
    explicitly requested combination.
    """

    EQUITY = "equity"
    INDEX = "index"
    FX = "fx"
    COMMODITY = "commodity"
    RATE = "rate"
    CREDIT = "credit"
    DIGITAL_ASSET = "digital_asset"
    OTHER = "other"


class InstrumentKind(str, Enum):
    """Discriminator for the instrument types in the public registry.

    Doubles as the ``instrument_type`` discriminator in the serialization
    schema and as the key of :func:`fast_vollib.instruments.instrument_types`.
    Grows one member per shipped contract type.
    """

    ASSET = "asset"
    FORWARD = "forward"
    FUTURE = "future"
    EUROPEAN_OPTION = "european_option"


class OptionType(str, Enum):
    """Call or put.

    Contract constructors also accept the ``'c'`` / ``'p'`` spellings the
    pricing kernels use; serialization always emits ``call`` / ``put``.
    """

    CALL = "call"
    PUT = "put"

    @property
    def flag(self) -> str:
        """The single-character kernel flag (``'c'`` or ``'p'``).

        Examples
        --------
        >>> from fast_vollib.instruments import OptionType
        >>> OptionType.PUT.flag
        'p'
        """
        return "c" if self is OptionType.CALL else "p"


class ExerciseStyle(str, Enum):
    """When the holder may exercise.

    Only ``EUROPEAN`` is defined because it is the only exercise style
    represented by the instruments package. American and Bermudan exercise
    styles are not supported.
    """

    EUROPEAN = "european"


class SettlementType(str, Enum):
    """How the contract settles at maturity."""

    CASH = "cash"
    PHYSICAL = "physical"


class PricingModel(str, Enum):
    """The three model conventions fast-vollib's kernels implement.

    The values are exactly the members of
    :data:`fast_vollib.types.ModelLiteral`: one vocabulary, two typing
    surfaces.  A round-trip through ``.value`` is therefore always a valid
    ``model=`` argument to the functional API, and vice versa.
    """

    BLACK = "black"
    BLACK_SCHOLES = "black_scholes"
    BLACK_SCHOLES_MERTON = "black_scholes_merton"


class IVSolver(str, Enum):
    """Implied-volatility inversion algorithm.

    ``HALLEY`` is the library's Halley-with-bisection solver reached through
    :func:`fast_vollib.fast_implied_volatility`.  ``JACKEL`` is the
    machine-precision "Let's Be Rational" solver in
    :mod:`fast_vollib.jackel`, and is the default for the instrument adapters:
    it is the only route with gradient support.

    The solver is part of the capability key, never inferred: a request for one
    solver never falls back to the other.
    """

    HALLEY = "halley"
    JACKEL = "jackel"


class PayoffRequirement(str, Enum):
    """What state a payoff evaluator needs in order to produce a cashflow.

    Declared per instrument type so an engine can reject incompatible data
    before it simulates anything.  ``TERMINAL`` means the terminal underlying
    state alone suffices. No path-dependent payoff requirement is defined.
    """

    TERMINAL = "terminal"
