"""The single canonical description of every instrument type.

Three things need to agree about what an instrument record contains: the
registry (what types exist), the serialization codec (how a record is written
and read back), and the checked-in JSON Schema (what an external consumer is
allowed to send).  Maintaining three lists by hand guarantees they drift, and
the drift is invisible until a record that validates against the schema fails
to decode.

So there is one list, here, and the other three are derived from it.  Adding an
instrument type means adding one entry below; the registry gains a row, the
codec gains a type, and the schema regenerates -- and the consistency test
fails loudly if any consumer is left behind.

Not public API: callers reach this through
:func:`fast_vollib.instruments.instrument_types`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .base import Asset, Instrument
from .enums import (
    AssetClass,
    AveragingMethod,
    BarrierType,
    InstrumentKind,
    OptionType,
    PayoffRequirement,
    SettlementType,
    StrikeConvention,
)
from .exotics import (
    AsianOption,
    BarrierOption,
    BinaryOption,
    LookbackOption,
    VarianceSwap,
)
from .forwards import Forward, Future
from .options import EuropeanOption

__all__ = [
    "INSTRUMENT_REF_FIELDS",
    "SCHEMA_VERSION",
    "TYPE_SPECS",
    "FieldSpec",
    "TypeSpec",
    "spec_for_type",
    "spec_for_type_id",
]

#: Version of the instrument record format.  Bumped only for a breaking change
#: to the shape of a record; the codec refuses any other value outright rather
#: than guessing at a migration.
SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class FieldSpec:
    """One field of one instrument record.

    Attributes
    ----------
    name : str
        Both the dataclass attribute and the JSON key -- they are deliberately
        the same string, so a record reads like the object it came from.
    json_type : {"string", "number", "object"}
        The encoded type.
    required : bool
        False when the contract gives the field a default, in which case a
        record may omit it and the decoder supplies the same default.  The
        encoder always writes it.
    enum_cls : type[enum.Enum], optional
        Present when the value is drawn from a closed vocabulary; supplies the
        schema's ``enum`` list and the decoder's admissible values.
    nullable : bool
        Whether ``None`` is an admissible value.
    minimum, exclusive_minimum : float, optional
        Numeric bounds, mirroring the runtime validation exactly.
    non_zero : bool
        Mirrors the runtime "must not be zero" rule for notionals.
    min_length : int, optional
        Minimum string length; mirrors the non-empty-identifier rule.
    description : str
        Prose carried into the JSON Schema.
    """

    name: str
    json_type: str
    required: bool
    enum_cls: type[Enum] | None = None
    nullable: bool = False
    minimum: float | None = None
    exclusive_minimum: float | None = None
    non_zero: bool = False
    min_length: int | None = None
    description: str = ""


@dataclass(frozen=True, slots=True)
class TypeSpec:
    """One instrument type, as the registry, codec, and schema all see it."""

    type_id: str
    kind: InstrumentKind
    python_type: type[Instrument]
    payoff_requirement: PayoffRequirement | None
    description: str
    fields: tuple[FieldSpec, ...] = field(default_factory=tuple)

    @property
    def field_names(self) -> frozenset[str]:
        return frozenset(f.name for f in self.fields)


# --- shared field definitions ------------------------------------------------

_INSTRUMENT_ID = FieldSpec(
    name="instrument_id",
    json_type="string",
    required=False,
    nullable=True,
    min_length=1,
    description="Caller-chosen identity for the record. No valuation meaning.",
)

_NOTIONAL = FieldSpec(
    name="notional",
    json_type="number",
    required=False,
    non_zero=True,
    description=("Contract multiplier. Non-zero and finite; negative denotes a short position."),
)

_OPTION_TYPE = FieldSpec(
    name="option_type",
    json_type="string",
    required=True,
    enum_cls=OptionType,
    description="Call or put. The short 'c'/'p' flags are not wire format.",
)

_STRIKE = FieldSpec(
    name="strike",
    json_type="number",
    required=True,
    exclusive_minimum=0.0,
    description="Strike price; strictly positive.",
)

_UNDERLIER = FieldSpec(
    name="underlier",
    json_type="object",
    required=True,
    description="Reference to the underlier this contract is written on.",
)

_MATURITY = FieldSpec(
    name="maturity",
    json_type="number",
    required=True,
    minimum=0.0,
    description="Time to maturity as a year fraction from valuation. Never a date.",
)

_OPTIONAL_STRIKE = FieldSpec(
    name="strike",
    json_type="number",
    required=False,
    nullable=True,
    exclusive_minimum=0.0,
    description=(
        "Strike price under a fixed strike convention; null under a floating one, "
        "where a level read off the path plays the strike's role."
    ),
)

_STRIKE_CONVENTION = FieldSpec(
    name="strike_convention",
    json_type="string",
    required=True,
    enum_cls=StrikeConvention,
    description="Whether the strike is agreed at inception or read off the path.",
)

_POSITIVE_MATURITY = FieldSpec(
    name="maturity",
    json_type="number",
    required=True,
    exclusive_minimum=0.0,
    description=(
        "Time to maturity as a year fraction from valuation. Strictly positive: a "
        "path-dependent payoff needs a path."
    ),
)

_SETTLEMENT = FieldSpec(
    name="settlement",
    json_type="string",
    required=False,
    enum_cls=SettlementType,
    description="How the contract settles at maturity.",
)

#: Fields of the nested underlier reference object.
INSTRUMENT_REF_FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec(
        name="identifier",
        json_type="string",
        required=True,
        min_length=1,
        description="Symbol, ticker, or other stable key for the underlier.",
    ),
    FieldSpec(
        name="asset_class",
        json_type="string",
        required=False,
        enum_cls=AssetClass,
        nullable=True,
        description="Economic class of the underlier. Descriptive; selects nothing.",
    ),
    FieldSpec(
        name="currency",
        json_type="string",
        required=False,
        nullable=True,
        min_length=1,
        description="Upper-cased currency code.",
    ),
)


# --- the type table ----------------------------------------------------------

TYPE_SPECS: tuple[TypeSpec, ...] = (
    TypeSpec(
        type_id=InstrumentKind.ASSET.value,
        kind=InstrumentKind.ASSET,
        python_type=Asset,
        payoff_requirement=None,
        description="An economic underlier: what derivatives are written on.",
        fields=(
            _INSTRUMENT_ID,
            FieldSpec(
                name="identifier",
                json_type="string",
                required=True,
                min_length=1,
                description="Symbol or ticker.",
            ),
            FieldSpec(
                name="asset_class",
                json_type="string",
                required=True,
                enum_cls=AssetClass,
                description="Economic class of the asset. Descriptive; selects nothing.",
            ),
            FieldSpec(
                name="currency",
                json_type="string",
                required=False,
                nullable=True,
                min_length=1,
                description="Upper-cased currency code.",
            ),
        ),
    ),
    TypeSpec(
        type_id=InstrumentKind.FORWARD.value,
        kind=InstrumentKind.FORWARD,
        python_type=Forward,
        payoff_requirement=PayoffRequirement.TERMINAL,
        description="Agreement to buy the underlier at maturity for a price fixed now.",
        fields=(
            _INSTRUMENT_ID,
            _UNDERLIER,
            FieldSpec(
                name="delivery_price",
                json_type="number",
                required=True,
                description="Price agreed at inception. Any finite value, negatives included.",
            ),
            _MATURITY,
            _SETTLEMENT,
            _NOTIONAL,
        ),
    ),
    TypeSpec(
        type_id=InstrumentKind.FUTURE.value,
        kind=InstrumentKind.FUTURE,
        python_type=Future,
        payoff_requirement=PayoffRequirement.TERMINAL,
        description="Exchange-traded contract on the underlier, expiring at maturity.",
        fields=(
            _INSTRUMENT_ID,
            _UNDERLIER,
            FieldSpec(
                name="contract_price",
                json_type="number",
                required=True,
                description="Traded futures price. Any finite value, negatives included.",
            ),
            _MATURITY,
            _SETTLEMENT,
            _NOTIONAL,
        ),
    ),
    TypeSpec(
        type_id=InstrumentKind.EUROPEAN_OPTION.value,
        kind=InstrumentKind.EUROPEAN_OPTION,
        python_type=EuropeanOption,
        payoff_requirement=PayoffRequirement.TERMINAL,
        description="Option on a single underlier, exercisable only at maturity.",
        fields=(
            _INSTRUMENT_ID,
            _UNDERLIER,
            _OPTION_TYPE,
            _STRIKE,
            _MATURITY,
            _SETTLEMENT,
            _NOTIONAL,
        ),
    ),
    TypeSpec(
        type_id=InstrumentKind.BINARY_OPTION.value,
        kind=InstrumentKind.BINARY_OPTION,
        python_type=BinaryOption,
        payoff_requirement=PayoffRequirement.TERMINAL,
        description="Cash-or-nothing digital paying a fixed amount if it finishes in the money.",
        fields=(
            _INSTRUMENT_ID,
            _UNDERLIER,
            _OPTION_TYPE,
            _STRIKE,
            _MATURITY,
            FieldSpec(
                name="cash_amount",
                json_type="number",
                required=False,
                exclusive_minimum=0.0,
                description="Fixed amount paid per contract when in the money.",
            ),
            _NOTIONAL,
        ),
    ),
    TypeSpec(
        type_id=InstrumentKind.ASIAN_OPTION.value,
        kind=InstrumentKind.ASIAN_OPTION,
        python_type=AsianOption,
        payoff_requirement=PayoffRequirement.PATH,
        description="Option on an average of the underlier over the contract's life.",
        fields=(
            _INSTRUMENT_ID,
            _UNDERLIER,
            _OPTION_TYPE,
            _OPTIONAL_STRIKE,
            FieldSpec(
                name="averaging_method",
                json_type="string",
                required=True,
                enum_cls=AveragingMethod,
                description="Whether the average is arithmetic or geometric.",
            ),
            _STRIKE_CONVENTION,
            _POSITIVE_MATURITY,
            _SETTLEMENT,
            _NOTIONAL,
        ),
    ),
    TypeSpec(
        type_id=InstrumentKind.BARRIER_OPTION.value,
        kind=InstrumentKind.BARRIER_OPTION,
        python_type=BarrierOption,
        payoff_requirement=PayoffRequirement.PATH,
        description="Option that knocks in or out when the underlier touches a barrier.",
        fields=(
            _INSTRUMENT_ID,
            _UNDERLIER,
            _OPTION_TYPE,
            _STRIKE,
            FieldSpec(
                name="barrier",
                json_type="number",
                required=True,
                exclusive_minimum=0.0,
                description=(
                    "Monitored level; strictly positive. Never compared against spot, "
                    "which is market state rather than a contract term."
                ),
            ),
            FieldSpec(
                name="barrier_type",
                json_type="string",
                required=True,
                enum_cls=BarrierType,
                description="Barrier direction and knock sense.",
            ),
            _POSITIVE_MATURITY,
            _SETTLEMENT,
            _NOTIONAL,
        ),
    ),
    TypeSpec(
        type_id=InstrumentKind.LOOKBACK_OPTION.value,
        kind=InstrumentKind.LOOKBACK_OPTION,
        python_type=LookbackOption,
        payoff_requirement=PayoffRequirement.PATH,
        description="Option settled against the highest or lowest level the underlier reached.",
        fields=(
            _INSTRUMENT_ID,
            _UNDERLIER,
            _OPTION_TYPE,
            _OPTIONAL_STRIKE,
            _STRIKE_CONVENTION,
            _POSITIVE_MATURITY,
            _SETTLEMENT,
            _NOTIONAL,
        ),
    ),
    TypeSpec(
        type_id=InstrumentKind.VARIANCE_SWAP.value,
        kind=InstrumentKind.VARIANCE_SWAP,
        python_type=VarianceSwap,
        payoff_requirement=PayoffRequirement.PATH,
        description="Swap paying realized variance against a level agreed at inception.",
        fields=(
            _INSTRUMENT_ID,
            _UNDERLIER,
            FieldSpec(
                name="strike_variance",
                json_type="number",
                required=True,
                minimum=0.0,
                description="Agreed variance level, the square of a volatility. Non-negative.",
            ),
            _POSITIVE_MATURITY,
            _NOTIONAL,
        ),
    ),
)

_BY_TYPE_ID: dict[str, TypeSpec] = {spec.type_id: spec for spec in TYPE_SPECS}
_BY_PYTHON_TYPE: dict[type[Instrument], TypeSpec] = {spec.python_type: spec for spec in TYPE_SPECS}


def spec_for_type_id(type_id: str) -> TypeSpec | None:
    """The spec registered under ``type_id``, or ``None`` if unknown."""
    return _BY_TYPE_ID.get(type_id)


def spec_for_type(python_type: type) -> TypeSpec | None:
    """The spec for an instrument class, or ``None`` if it is not registered."""
    return _BY_PYTHON_TYPE.get(python_type)


# Guard against a type entry that silently disagrees with the class it names.
for _spec in TYPE_SPECS:
    _dataclass_fields = {f for f in _spec.python_type.__dataclass_fields__}
    if _dataclass_fields != set(_spec.field_names):  # pragma: no cover - import-time guard
        raise RuntimeError(
            f"Instrument metadata for {_spec.type_id!r} lists "
            f"{sorted(_spec.field_names)} but {_spec.python_type.__name__} has "
            f"{sorted(_dataclass_fields)}."
        )
del _spec, _dataclass_fields
