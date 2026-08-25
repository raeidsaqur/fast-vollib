"""Contract-root behaviour: immutability, identity, validation, references."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from fast_vollib.instruments import (
    Asset,
    AssetClass,
    Derivative,
    Instrument,
    InstrumentKind,
    InstrumentRef,
    InstrumentValidationError,
)
from fast_vollib.instruments._validate import (
    ensure_finite_float,
    ensure_non_negative,
    ensure_nonzero,
    ensure_positive,
    parse_option_type,
)
from fast_vollib.instruments.enums import OptionType


def make_asset(**overrides: object) -> Asset:
    kwargs: dict[str, object] = {"identifier": "ACME", "asset_class": AssetClass.EQUITY}
    kwargs.update(overrides)
    return Asset(**kwargs)  # type: ignore[arg-type]


# --- shape of the dataclasses ------------------------------------------------


@pytest.mark.parametrize("cls", [InstrumentRef, Asset], ids=["ref", "asset"])
def test_frozen_slotted_and_keyword_only(cls: type) -> None:
    params = dataclasses.fields(cls)
    assert params  # non-empty
    assert all(f.kw_only for f in params)
    assert dataclasses.is_dataclass(cls)
    assert "__slots__" in cls.__dict__
    assert not hasattr(cls, "__dict__") or "__dict__" not in cls.__dict__


def test_asset_is_immutable_and_closed() -> None:
    asset = make_asset()
    with pytest.raises(dataclasses.FrozenInstanceError):
        asset.identifier = "OTHER"  # type: ignore[misc]
    # Attaching state a contract must never hold (a spot, a path buffer, a
    # device) is impossible because the class is slotted. CPython raises
    # TypeError rather than AttributeError for an unknown name on a
    # ``frozen=True, slots=True`` dataclass, so assert the invariant, not the
    # exception class.
    with pytest.raises((AttributeError, TypeError)):
        asset.spot = 100.0  # type: ignore[attr-defined]


def test_positional_arguments_are_rejected() -> None:
    with pytest.raises(TypeError):
        Asset("ACME", AssetClass.EQUITY)  # type: ignore[misc]


def test_equality_and_hash_are_structural() -> None:
    a, b = make_asset(), make_asset()
    assert a == b
    assert hash(a) == hash(b)
    assert len({a, b}) == 1
    assert a != make_asset(identifier="OTHER")


def test_abstract_roots_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        Instrument()  # type: ignore[abstract]
    with pytest.raises(TypeError):
        Derivative()  # type: ignore[abstract]


def test_repr_is_informative_and_not_the_wire_format() -> None:
    text = repr(make_asset(currency="USD"))
    assert text.startswith("Asset(")
    assert "ACME" in text and "USD" in text
    assert "schema_version" not in text


# --- normalization and validation -------------------------------------------


def test_identifier_is_trimmed() -> None:
    assert make_asset(identifier="  ACME\t").identifier == "ACME"
    assert InstrumentRef(identifier=" X ").identifier == "X"


def test_currency_is_upper_cased() -> None:
    assert make_asset(currency=" usd ").currency == "USD"
    assert make_asset().currency is None


def test_enum_fields_accept_canonical_strings() -> None:
    assert make_asset(asset_class="index").asset_class is AssetClass.INDEX
    assert InstrumentRef(identifier="X", asset_class="fx").asset_class is AssetClass.FX


@pytest.mark.parametrize("bad", ["", "   ", 3, None], ids=["empty", "blank", "int", "none"])
def test_empty_identifiers_are_rejected(bad: object) -> None:
    with pytest.raises(InstrumentValidationError, match="identifier"):
        make_asset(identifier=bad)


def test_unknown_enum_value_message_lists_valid_values() -> None:
    with pytest.raises(InstrumentValidationError) as excinfo:
        make_asset(asset_class="stonks")
    message = str(excinfo.value)
    assert "asset_class" in message
    assert "'equity'" in message and "'other'" in message


def test_instrument_id_is_optional_but_validated() -> None:
    assert make_asset().instrument_id is None
    assert make_asset(instrument_id=" ACME-1 ").instrument_id == "ACME-1"
    with pytest.raises(InstrumentValidationError, match="instrument_id"):
        make_asset(instrument_id="  ")


def test_kind_discriminates() -> None:
    assert make_asset().kind is InstrumentKind.ASSET


def test_ref_round_trips_the_descriptive_fields() -> None:
    asset = make_asset(currency="usd", instrument_id="row-7")
    ref = asset.ref()
    assert ref == InstrumentRef(identifier="ACME", asset_class=AssetClass.EQUITY, currency="USD")
    # The reference carries no record identity -- it points at an underlier,
    # it is not a copy of the asset record.
    assert not hasattr(ref, "instrument_id")


# --- the shared scalar validation core --------------------------------------


@pytest.mark.parametrize("value", [1, 1.5, -2.0, 0, np.float64(3.25), np.int64(4)])
def test_finite_real_scalars_are_accepted(value: object) -> None:
    assert isinstance(ensure_finite_float(value, field="x"), float)


@pytest.mark.parametrize(
    "value",
    [True, np.bool_(False), math.nan, math.inf, -math.inf, 1 + 2j, "1.0", None, np.array([1.0])],
    ids=["bool", "np-bool", "nan", "inf", "-inf", "complex", "str", "none", "array"],
)
def test_non_finite_and_non_real_scalars_are_rejected(value: object) -> None:
    with pytest.raises(InstrumentValidationError, match="x"):
        ensure_finite_float(value, field="x")


def test_range_predicates() -> None:
    assert ensure_positive(1e-12, field="strike") == 1e-12
    with pytest.raises(InstrumentValidationError, match="strictly positive"):
        ensure_positive(0.0, field="strike")
    assert ensure_non_negative(0.0, field="maturity") == 0.0
    with pytest.raises(InstrumentValidationError, match="non-negative"):
        ensure_non_negative(-1e-9, field="maturity")
    assert ensure_nonzero(-1.0, field="notional") == -1.0
    with pytest.raises(InstrumentValidationError, match="non-zero"):
        ensure_nonzero(0.0, field="notional")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("call", OptionType.CALL),
        ("PUT", OptionType.PUT),
        ("c", OptionType.CALL),
        ("P", OptionType.PUT),
        (OptionType.CALL, OptionType.CALL),
    ],
)
def test_option_type_parsing_accepts_kernel_flags(value: object, expected: OptionType) -> None:
    assert parse_option_type(value) is expected


@pytest.mark.parametrize("bad", ["callable", "x", 1, None])
def test_option_type_parsing_rejects_everything_else(bad: object) -> None:
    with pytest.raises(InstrumentValidationError, match="option_type"):
        parse_option_type(bad)


def test_negative_rates_are_not_a_contract_concern() -> None:
    """Rates live in market inputs, never on a contract -- and may be negative."""
    assert ensure_finite_float(-0.005, field="rate") == -0.005
