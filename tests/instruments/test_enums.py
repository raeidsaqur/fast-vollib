"""The enum vocabularies are wire format; their values must not drift."""

from __future__ import annotations

from enum import Enum
import typing

import pytest

from fast_vollib import types
from fast_vollib.instruments import (
    AssetClass,
    AveragingMethod,
    BarrierType,
    ExerciseStyle,
    InstrumentKind,
    InstrumentValidationError,
    IVSolver,
    OptionType,
    PayoffRequirement,
    PricingModel,
    SettlementType,
    StrikeConvention,
)
from fast_vollib.instruments._validate import ensure_enum, parse_option_type

ALL_ENUMS = [
    AssetClass,
    AveragingMethod,
    BarrierType,
    ExerciseStyle,
    InstrumentKind,
    IVSolver,
    OptionType,
    PayoffRequirement,
    PricingModel,
    SettlementType,
    StrikeConvention,
]

EXPECTED_VALUES = {
    AssetClass: [
        "equity",
        "index",
        "fx",
        "commodity",
        "rate",
        "credit",
        "digital_asset",
        "other",
    ],
    ExerciseStyle: ["european"],
    AveragingMethod: ["arithmetic", "geometric"],
    BarrierType: ["up_and_in", "up_and_out", "down_and_in", "down_and_out"],
    InstrumentKind: [
        "asset",
        "forward",
        "future",
        "european_option",
        "binary_option",
        "asian_option",
        "barrier_option",
        "lookback_option",
        "variance_swap",
    ],
    IVSolver: ["halley", "jackel"],
    OptionType: ["call", "put"],
    PayoffRequirement: ["terminal", "path"],
    PricingModel: ["black", "black_scholes", "black_scholes_merton"],
    SettlementType: ["cash", "physical"],
    StrikeConvention: ["fixed", "floating"],
}


@pytest.mark.parametrize("enum_cls", ALL_ENUMS, ids=lambda c: c.__name__)
def test_values_are_stable(enum_cls: type[Enum]) -> None:
    assert [m.value for m in enum_cls] == EXPECTED_VALUES[enum_cls]


@pytest.mark.parametrize("enum_cls", ALL_ENUMS, ids=lambda c: c.__name__)
def test_values_are_lower_case_strings(enum_cls: type[Enum]) -> None:
    for member in enum_cls:
        assert isinstance(member.value, str)
        assert member.value == member.value.lower()


@pytest.mark.parametrize("enum_cls", ALL_ENUMS, ids=lambda c: c.__name__)
def test_members_are_strings(enum_cls: type[Enum]) -> None:
    # ``class X(str, Enum)`` rather than StrEnum, which needs Python 3.11.
    for member in enum_cls:
        assert isinstance(member, str)
        assert member == member.value


@pytest.mark.parametrize("enum_cls", ALL_ENUMS, ids=lambda c: c.__name__)
def test_unknown_value_message_lists_the_valid_ones(enum_cls: type[Enum]) -> None:
    with pytest.raises(ValueError) as excinfo:
        enum_cls("definitely-not-a-member")
    assert enum_cls.__name__ in str(excinfo.value)


@pytest.mark.parametrize(
    ("enum_cls", "literal"),
    [
        (PricingModel, types.ModelLiteral),
        (IVSolver, types.IVSolverLiteral),
        (InstrumentKind, types.InstrumentKindLiteral),
        (ExerciseStyle, types.ExerciseLiteral),
    ],
    ids=["model", "solver", "kind", "exercise"],
)
def test_enum_matches_its_literal(enum_cls: type[Enum], literal: object) -> None:
    """One vocabulary, two typing surfaces -- they must agree exactly."""
    assert {m.value for m in enum_cls} == set(typing.get_args(literal))


def test_option_type_flag_bridges_to_kernel_convention() -> None:
    assert OptionType.CALL.flag == "c"
    assert OptionType.PUT.flag == "p"


# --- what an invalid value is told -------------------------------------------

#: Every closed vocabulary reached through a contract term, with the field name
#: the constructor validates it under.
VALIDATED_FIELDS = [
    (AssetClass, "asset_class"),
    (SettlementType, "settlement"),
    (AveragingMethod, "averaging_method"),
    (BarrierType, "barrier_type"),
    (StrikeConvention, "strike_convention"),
]


@pytest.mark.parametrize(
    ("enum_cls", "field"), VALIDATED_FIELDS, ids=lambda v: getattr(v, "__name__", v)
)
def test_validation_error_lists_every_admissible_value(enum_cls: type[Enum], field: str) -> None:
    """A rejection has to name the alternatives, not just the type that failed.

    ``"got 'netted'; expected a SettlementType"`` sends the caller to the
    source. Listing the values answers the question in the message.
    """
    with pytest.raises(InstrumentValidationError) as excinfo:
        ensure_enum("definitely-not-a-member", enum_cls, field=field)
    message = str(excinfo.value)
    assert field in message
    assert "definitely-not-a-member" in message
    for member in enum_cls:
        assert repr(member.value) in message, f"{member.value!r} missing from: {message}"


def test_option_type_error_lists_both_spellings() -> None:
    """The kernels' 'c'/'p' flags are accepted, so the message has to say so."""
    with pytest.raises(InstrumentValidationError) as excinfo:
        parse_option_type("straddle")
    message = str(excinfo.value)
    for spelling in ("'call'", "'put'", "'c'", "'p'"):
        assert spelling in message
    assert "straddle" in message
