"""Contract term validation and normalization."""

from __future__ import annotations

import dataclasses

import pytest

from fast_vollib.instruments import (
    Asset,
    AssetClass,
    EuropeanOption,
    ExerciseStyle,
    Forward,
    Future,
    InstrumentKind,
    InstrumentRef,
    InstrumentValidationError,
    OptionType,
    SettlementType,
)

CONTRACT_KWARGS: dict[type, dict[str, object]] = {
    Forward: {"underlier": "ACME", "delivery_price": 100.0, "maturity": 0.5},
    Future: {"underlier": "ACME", "contract_price": 100.0, "maturity": 0.5},
    EuropeanOption: {
        "underlier": "ACME",
        "option_type": "call",
        "strike": 100.0,
        "maturity": 0.5,
    },
}
CONTRACT_TYPES = list(CONTRACT_KWARGS)


def build(cls: type, **overrides: object):
    kwargs = dict(CONTRACT_KWARGS[cls])
    kwargs.update(overrides)
    return cls(**kwargs)


# --- shared derivative behaviour --------------------------------------------


@pytest.mark.parametrize("cls", CONTRACT_TYPES, ids=lambda c: c.__name__)
def test_is_frozen_slotted_kw_only(cls: type) -> None:
    instrument = build(cls)
    assert all(f.kw_only for f in dataclasses.fields(cls))
    with pytest.raises(dataclasses.FrozenInstanceError):
        instrument.maturity = 1.0  # type: ignore[misc]
    with pytest.raises((AttributeError, TypeError)):
        instrument.paths = [1.0]  # type: ignore[attr-defined]


@pytest.mark.parametrize("cls", CONTRACT_TYPES, ids=lambda c: c.__name__)
def test_holds_no_arrays_or_market_state(cls: type) -> None:
    instrument = build(cls)
    for f in dataclasses.fields(cls):
        value = getattr(instrument, f.name)
        assert isinstance(value, (str, float, InstrumentRef, type(None)))


@pytest.mark.parametrize("cls", CONTRACT_TYPES, ids=lambda c: c.__name__)
def test_equality_and_hash(cls: type) -> None:
    assert build(cls) == build(cls)
    assert hash(build(cls)) == hash(build(cls))
    assert build(cls) != build(cls, maturity=0.75)


@pytest.mark.parametrize("cls", CONTRACT_TYPES, ids=lambda c: c.__name__)
def test_kind_is_unique_and_matches_the_type(cls: type) -> None:
    assert isinstance(build(cls).kind, InstrumentKind)


def test_kinds_are_distinct() -> None:
    kinds = {build(cls).kind for cls in CONTRACT_TYPES}
    assert len(kinds) == len(CONTRACT_TYPES)


@pytest.mark.parametrize("cls", CONTRACT_TYPES, ids=lambda c: c.__name__)
def test_underliers_is_a_tuple_of_refs(cls: type) -> None:
    underliers = build(cls).underliers
    assert isinstance(underliers, tuple)
    assert all(isinstance(u, InstrumentRef) for u in underliers)


@pytest.mark.parametrize("cls", CONTRACT_TYPES, ids=lambda c: c.__name__)
def test_underlier_accepts_ref_asset_or_identifier(cls: type) -> None:
    asset = Asset(identifier="ACME", asset_class=AssetClass.EQUITY, currency="USD")
    from_asset = build(cls, underlier=asset)
    from_ref = build(cls, underlier=asset.ref())
    assert from_asset == from_ref
    # An Asset is never stored -- a contract holds a reference, not a graph edge.
    assert isinstance(from_asset.underlier, InstrumentRef)
    assert not isinstance(from_asset.underlier, Asset)
    assert build(cls, underlier="ACME").underlier == InstrumentRef(identifier="ACME")


@pytest.mark.parametrize("cls", CONTRACT_TYPES, ids=lambda c: c.__name__)
@pytest.mark.parametrize("bad", [None, 42, ["ACME"]], ids=["none", "int", "list"])
def test_underlier_rejects_other_types(cls: type, bad: object) -> None:
    with pytest.raises(InstrumentValidationError, match="underlier"):
        build(cls, underlier=bad)


@pytest.mark.parametrize("cls", CONTRACT_TYPES, ids=lambda c: c.__name__)
def test_maturity_is_non_negative(cls: type) -> None:
    assert build(cls, maturity=0).maturity == 0.0
    with pytest.raises(InstrumentValidationError, match="maturity"):
        build(cls, maturity=-1e-9)
    with pytest.raises(InstrumentValidationError, match="maturity"):
        build(cls, maturity=float("nan"))


@pytest.mark.parametrize("cls", CONTRACT_TYPES, ids=lambda c: c.__name__)
def test_notional_must_be_non_zero_but_may_be_negative(cls: type) -> None:
    assert build(cls, notional=-3.0).notional == -3.0
    with pytest.raises(InstrumentValidationError, match="notional"):
        build(cls, notional=0.0)
    with pytest.raises(InstrumentValidationError, match="notional"):
        build(cls, notional=float("inf"))


@pytest.mark.parametrize("cls", CONTRACT_TYPES, ids=lambda c: c.__name__)
def test_settlement_defaults_to_cash_and_parses_strings(cls: type) -> None:
    assert build(cls).settlement is SettlementType.CASH
    assert build(cls, settlement="physical").settlement is SettlementType.PHYSICAL
    with pytest.raises(InstrumentValidationError, match="settlement"):
        build(cls, settlement="netted")


# --- type-specific terms -----------------------------------------------------


def test_strike_must_be_strictly_positive() -> None:
    with pytest.raises(InstrumentValidationError, match="strike"):
        build(EuropeanOption, strike=0.0)
    with pytest.raises(InstrumentValidationError, match="strike"):
        build(EuropeanOption, strike=-1.0)


@pytest.mark.parametrize(
    ("cls", "price_field"),
    [(Forward, "delivery_price"), (Future, "contract_price")],
    ids=["forward", "future"],
)
def test_linear_contract_prices_may_be_negative(cls: type, price_field: str) -> None:
    """Commodity forwards have settled below zero; only finiteness is required."""
    instrument = build(cls, **{price_field: -37.63})
    assert getattr(instrument, price_field) == -37.63
    with pytest.raises(InstrumentValidationError, match=price_field):
        build(cls, **{price_field: float("nan")})


def test_option_type_accepts_both_spellings() -> None:
    assert build(EuropeanOption, option_type="p").option_type is OptionType.PUT
    assert build(EuropeanOption, option_type="PUT").option_type is OptionType.PUT
    assert build(EuropeanOption, option_type=OptionType.CALL).flag == "c"
    with pytest.raises(InstrumentValidationError, match="option_type"):
        build(EuropeanOption, option_type="straddle")


def test_exercise_style_is_a_constant_not_a_field() -> None:
    field_names = {f.name for f in dataclasses.fields(EuropeanOption)}
    assert "exercise_style" not in field_names
    assert build(EuropeanOption).exercise_style is ExerciseStyle.EUROPEAN
    with pytest.raises(TypeError):
        EuropeanOption(  # type: ignore[call-arg]
            underlier="ACME",
            option_type="call",
            strike=1.0,
            maturity=1.0,
            exercise_style="american",
        )


def test_forward_and_future_are_distinct_types() -> None:
    fwd = Forward(underlier="ACME", delivery_price=100.0, maturity=0.5)
    fut = Future(underlier="ACME", contract_price=100.0, maturity=0.5)
    assert fwd != fut
    assert fwd.kind is InstrumentKind.FORWARD
    assert fut.kind is InstrumentKind.FUTURE
