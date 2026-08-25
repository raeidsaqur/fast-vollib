"""The enum vocabularies are wire format; their values must not drift."""

from __future__ import annotations

from enum import Enum
import typing

import pytest

from fast_vollib import types
from fast_vollib.instruments import (
    AssetClass,
    ExerciseStyle,
    InstrumentKind,
    IVSolver,
    OptionType,
    PayoffRequirement,
    PricingModel,
    SettlementType,
)

ALL_ENUMS = [
    AssetClass,
    ExerciseStyle,
    InstrumentKind,
    IVSolver,
    OptionType,
    PayoffRequirement,
    PricingModel,
    SettlementType,
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
    InstrumentKind: ["asset", "forward", "future", "european_option"],
    IVSolver: ["halley", "jackel"],
    OptionType: ["call", "put"],
    PayoffRequirement: ["terminal"],
    PricingModel: ["black", "black_scholes", "black_scholes_merton"],
    SettlementType: ["cash", "physical"],
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
