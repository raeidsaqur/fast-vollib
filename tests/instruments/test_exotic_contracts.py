"""Terms, validation, and serialization for the contracts in ``exotics``."""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from fast_vollib.instruments import (
    AsianOption,
    Asset,
    AssetClass,
    BarrierOption,
    BarrierType,
    BinaryOption,
    InstrumentKind,
    InstrumentRef,
    InstrumentValidationError,
    LookbackOption,
    OptionType,
    PayoffRequirement,
    SettlementType,
    VarianceSwap,
    capabilities,
    instrument_from_dict,
    instrument_from_json,
    instrument_to_dict,
    instrument_to_json,
    instrument_types,
    payoff_requirement,
)

EXOTIC_KWARGS: dict[type, dict[str, Any]] = {
    BinaryOption: {
        "underlier": "ACME",
        "option_type": "call",
        "strike": 100.0,
        "maturity": 0.5,
    },
    AsianOption: {
        "underlier": "ACME",
        "option_type": "call",
        "strike": 100.0,
        "averaging_method": "arithmetic",
        "strike_convention": "fixed",
        "maturity": 0.5,
    },
    BarrierOption: {
        "underlier": "ACME",
        "option_type": "call",
        "strike": 100.0,
        "barrier": 130.0,
        "barrier_type": "up_and_out",
        "maturity": 0.5,
    },
    LookbackOption: {
        "underlier": "ACME",
        "option_type": "call",
        "strike": 100.0,
        "strike_convention": "fixed",
        "maturity": 0.5,
    },
    VarianceSwap: {
        "underlier": "ACME",
        "strike_variance": 0.04,
        "maturity": 0.5,
    },
}
EXOTIC_TYPES = list(EXOTIC_KWARGS)


def build(cls: type, **overrides: Any) -> Any:
    kwargs = dict(EXOTIC_KWARGS[cls])
    kwargs.update(overrides)
    return cls(**kwargs)


# --- shared contract behaviour -------------------------------------------------


@pytest.mark.parametrize("cls", EXOTIC_TYPES, ids=lambda c: c.__name__)
def test_is_frozen_slotted_kw_only(cls: type) -> None:
    instrument = build(cls)
    assert all(f.kw_only for f in dataclasses.fields(cls))
    with pytest.raises(dataclasses.FrozenInstanceError):
        instrument.maturity = 1.0
    with pytest.raises((AttributeError, TypeError)):
        instrument.paths = [1.0]


@pytest.mark.parametrize("cls", EXOTIC_TYPES, ids=lambda c: c.__name__)
def test_holds_no_arrays_or_market_state(cls: type) -> None:
    instrument = build(cls)
    for field in dataclasses.fields(cls):
        value = getattr(instrument, field.name)
        assert isinstance(value, (str, float, InstrumentRef, type(None)))


@pytest.mark.parametrize("cls", EXOTIC_TYPES, ids=lambda c: c.__name__)
def test_equality_hash_and_underliers(cls: type) -> None:
    assert build(cls) == build(cls)
    assert hash(build(cls)) == hash(build(cls))
    assert build(cls) != build(cls, maturity=0.75)
    underliers = build(cls).underliers
    assert isinstance(underliers, tuple) and len(underliers) == 1
    assert isinstance(underliers[0], InstrumentRef)


@pytest.mark.parametrize("cls", EXOTIC_TYPES, ids=lambda c: c.__name__)
def test_underlier_accepts_ref_asset_or_identifier(cls: type) -> None:
    asset = Asset(identifier="ACME", asset_class=AssetClass.EQUITY, currency="USD")
    assert build(cls, underlier=asset) == build(cls, underlier=asset.ref())
    assert not isinstance(build(cls, underlier=asset).underlier, Asset)


@pytest.mark.parametrize("cls", EXOTIC_TYPES, ids=lambda c: c.__name__)
def test_notional_must_be_non_zero_but_may_be_negative(cls: type) -> None:
    assert build(cls, notional=-3.0).notional == -3.0
    with pytest.raises(InstrumentValidationError, match="notional"):
        build(cls, notional=0.0)


@pytest.mark.parametrize("cls", EXOTIC_TYPES, ids=lambda c: c.__name__)
def test_registered_in_the_public_registry(cls: type) -> None:
    matches = [i for i in instrument_types().values() if i.python_type is cls]
    assert len(matches) == 1
    assert matches[0].description


@pytest.mark.parametrize("cls", EXOTIC_TYPES, ids=lambda c: c.__name__)
def test_round_trips_through_dict_and_json(cls: type) -> None:
    instrument = build(cls, instrument_id="rec-1", notional=-250.0)
    assert instrument_from_dict(instrument_to_dict(instrument)) == instrument
    assert instrument_from_json(instrument_to_json(instrument)) == instrument


@pytest.mark.parametrize("cls", EXOTIC_TYPES, ids=lambda c: c.__name__)
def test_an_unknown_field_is_refused_rather_than_ignored(cls: type) -> None:
    from fast_vollib.instruments import SerializationError

    record = instrument_to_dict(build(cls))
    record["rebate"] = 1.0
    with pytest.raises(SerializationError, match="rebate"):
        instrument_from_dict(record)


# --- BinaryOption --------------------------------------------------------------


def test_binary_kind_and_flag() -> None:
    digital = build(BinaryOption)
    assert digital.kind is InstrumentKind.BINARY_OPTION
    assert digital.flag == "c"
    assert build(BinaryOption, option_type="p").flag == "p"
    assert build(BinaryOption, option_type="P").option_type is OptionType.PUT


def test_binary_settlement_is_a_constant_property_not_a_field() -> None:
    """A digital has nothing to deliver, so a record cannot say otherwise."""
    field_names = {f.name for f in dataclasses.fields(BinaryOption)}
    assert "settlement" not in field_names
    assert build(BinaryOption).settlement is SettlementType.CASH
    with pytest.raises(TypeError):
        BinaryOption(
            underlier="ACME",
            option_type="call",
            strike=100.0,
            maturity=1.0,
            settlement="physical",
        )
    assert "settlement" not in instrument_to_dict(build(BinaryOption))


def test_binary_strike_must_be_strictly_positive() -> None:
    for bad in (0.0, -1.0):
        with pytest.raises(InstrumentValidationError, match="strike"):
            build(BinaryOption, strike=bad)


def test_binary_cash_amount_must_be_strictly_positive() -> None:
    assert build(BinaryOption, cash_amount=2.5).cash_amount == 2.5
    for bad in (0.0, -1.0, float("nan")):
        with pytest.raises(InstrumentValidationError, match="cash_amount"):
            build(BinaryOption, cash_amount=bad)


def test_a_short_digital_uses_a_negative_notional_not_a_negative_cash_amount() -> None:
    short = build(BinaryOption, notional=-1.0, cash_amount=10.0)
    assert short.notional == -1.0 and short.cash_amount == 10.0


def test_binary_maturity_may_be_zero_to_keep_the_expiry_payoff() -> None:
    assert build(BinaryOption, maturity=0.0).maturity == 0.0
    with pytest.raises(InstrumentValidationError, match="maturity"):
        build(BinaryOption, maturity=-1e-9)


def test_binary_declares_a_terminal_payoff_requirement() -> None:
    assert payoff_requirement(BinaryOption) is PayoffRequirement.TERMINAL
    assert payoff_requirement(build(BinaryOption)) is PayoffRequirement.TERMINAL


def test_binary_capabilities_are_payoff_and_simulation_only() -> None:
    caps = capabilities(BinaryOption)
    assert caps.payoff is True
    assert caps.simulate is True
    assert caps.price == frozenset()
    assert caps.greeks == frozenset()
    assert caps.implied_volatility == {}


# --- AsianOption ---------------------------------------------------------------


def test_asian_kind_flag_and_settlement() -> None:
    contract = build(AsianOption)
    assert contract.kind is InstrumentKind.ASIAN_OPTION
    assert contract.flag == "c"
    assert contract.settlement is SettlementType.CASH
    assert build(AsianOption, settlement="physical").settlement is SettlementType.PHYSICAL


def test_asian_declares_a_path_payoff_requirement() -> None:
    assert payoff_requirement(AsianOption) is PayoffRequirement.PATH
    assert payoff_requirement(build(AsianOption)) is PayoffRequirement.PATH


def test_a_fixed_strike_asian_requires_a_strike() -> None:
    with pytest.raises(InstrumentValidationError, match="needs a strike"):
        build(AsianOption, strike=None)


def test_a_floating_strike_asian_refuses_a_strike() -> None:
    floating = build(AsianOption, strike=None, strike_convention="floating")
    assert floating.strike is None
    with pytest.raises(InstrumentValidationError, match="takes its strike from the path"):
        build(AsianOption, strike_convention="floating", strike=100.0)


def test_asian_strike_must_be_strictly_positive_when_present() -> None:
    for bad in (0.0, -1.0):
        with pytest.raises(InstrumentValidationError, match="strike"):
            build(AsianOption, strike=bad)


def test_asian_maturity_must_be_strictly_positive() -> None:
    """A path-dependent payoff needs a path, and an expiring contract has none."""
    with pytest.raises(InstrumentValidationError, match="maturity must be strictly positive"):
        build(AsianOption, maturity=0.0)


def test_asian_vocabularies_are_validated_and_list_their_values() -> None:
    for field, value in (("averaging_method", "harmonic"), ("strike_convention", "asian")):
        with pytest.raises(InstrumentValidationError) as excinfo:
            build(AsianOption, **{field: value})
        message = str(excinfo.value)
        assert field in message and value in message


def test_asian_capabilities_are_payoff_and_simulation_only() -> None:
    caps = capabilities(AsianOption)
    assert caps.payoff is True
    assert caps.simulate is True
    assert caps.price == frozenset()


def test_a_floating_strike_asian_round_trips_with_a_null_strike() -> None:
    floating = build(AsianOption, strike=None, strike_convention="floating")
    record = instrument_to_dict(floating)
    assert record["strike"] is None
    assert instrument_from_dict(record) == floating


def test_the_averaging_schedule_is_not_a_contract_field() -> None:
    """Fixings come from the scenario, so the schedule is stated once."""
    field_names = {f.name for f in dataclasses.fields(AsianOption)}
    assert "fixings" not in field_names
    assert "fixing_dates" not in field_names
    assert "observation_frequency" not in field_names


# --- BarrierOption -------------------------------------------------------------


def test_barrier_kind_and_terms() -> None:
    contract = build(BarrierOption)
    assert contract.kind is InstrumentKind.BARRIER_OPTION
    assert contract.barrier == 130.0
    assert contract.barrier_type is BarrierType.UP_AND_OUT
    assert payoff_requirement(BarrierOption) is PayoffRequirement.PATH


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan")], ids=["zero", "negative", "nan"])
def test_barrier_must_be_strictly_positive(bad: float) -> None:
    with pytest.raises(InstrumentValidationError, match="barrier"):
        build(BarrierOption, barrier=bad)


def test_construction_never_compares_the_barrier_with_a_spot() -> None:
    """Whether a barrier is above or below the market is market state.

    Refusing an up-barrier below today's spot would make a contract's validity
    depend on an observation contracts do not carry, so a perfectly ordinary
    already-knocked-in option could not be represented at all.
    """
    assert build(BarrierOption, barrier=1e-6, barrier_type="up_and_out").barrier == 1e-6
    assert build(BarrierOption, barrier=1e9, barrier_type="down_and_in").barrier == 1e9


@pytest.mark.parametrize("barrier_type", list(BarrierType), ids=lambda b: b.value)
def test_every_barrier_type_round_trips(barrier_type: BarrierType) -> None:
    contract = build(BarrierOption, barrier_type=barrier_type)
    assert instrument_from_dict(instrument_to_dict(contract)) == contract


def test_barrier_direction_and_knock_sense_are_derived_not_stored() -> None:
    """Four combinations exist; two booleans would admit a fifth that does not."""
    assert BarrierType.UP_AND_IN.is_up and BarrierType.UP_AND_IN.knocks_in
    assert BarrierType.UP_AND_OUT.is_up and not BarrierType.UP_AND_OUT.knocks_in
    assert not BarrierType.DOWN_AND_IN.is_up and BarrierType.DOWN_AND_IN.knocks_in
    assert not BarrierType.DOWN_AND_OUT.is_up and not BarrierType.DOWN_AND_OUT.knocks_in


def test_barrier_has_no_rebate_field() -> None:
    """A knocked-out contract pays nothing; a rebate would be a different type."""
    assert "rebate" not in {f.name for f in dataclasses.fields(BarrierOption)}


def test_barrier_maturity_must_be_strictly_positive() -> None:
    with pytest.raises(InstrumentValidationError, match="maturity must be strictly positive"):
        build(BarrierOption, maturity=0.0)


# --- LookbackOption ------------------------------------------------------------


def test_lookback_kind_and_requirement() -> None:
    assert build(LookbackOption).kind is InstrumentKind.LOOKBACK_OPTION
    assert payoff_requirement(LookbackOption) is PayoffRequirement.PATH


def test_a_fixed_strike_lookback_requires_a_strike() -> None:
    with pytest.raises(InstrumentValidationError, match="needs a strike"):
        build(LookbackOption, strike=None)


def test_a_floating_strike_lookback_refuses_a_strike() -> None:
    floating = build(LookbackOption, strike=None, strike_convention="floating")
    assert floating.strike is None
    assert instrument_from_dict(instrument_to_dict(floating)) == floating
    with pytest.raises(InstrumentValidationError, match="takes its strike from the path"):
        build(LookbackOption, strike_convention="floating", strike=100.0)


def test_lookback_maturity_must_be_strictly_positive() -> None:
    with pytest.raises(InstrumentValidationError, match="maturity must be strictly positive"):
        build(LookbackOption, maturity=0.0)


@pytest.mark.parametrize("cls", [BarrierOption, LookbackOption], ids=["barrier", "lookback"])
def test_capabilities_are_payoff_and_simulation_only(cls: type) -> None:
    caps = capabilities(cls)
    assert caps.payoff is True
    assert caps.simulate is True
    assert caps.price == frozenset()
    assert caps.greeks == frozenset()


# --- VarianceSwap --------------------------------------------------------------


def test_variance_swap_kind_and_requirement() -> None:
    swap = build(VarianceSwap)
    assert swap.kind is InstrumentKind.VARIANCE_SWAP
    assert payoff_requirement(VarianceSwap) is PayoffRequirement.PATH


def test_strike_variance_may_be_zero_but_not_negative() -> None:
    """Zero makes the contract a pure long position in realized variance."""
    assert build(VarianceSwap, strike_variance=0.0).strike_variance == 0.0
    with pytest.raises(InstrumentValidationError, match="strike_variance"):
        build(VarianceSwap, strike_variance=-1e-9)


def test_variance_swap_settlement_is_a_constant_property() -> None:
    assert "settlement" not in {f.name for f in dataclasses.fields(VarianceSwap)}
    assert build(VarianceSwap).settlement is SettlementType.CASH
    assert "settlement" not in instrument_to_dict(build(VarianceSwap))
    with pytest.raises(TypeError):
        VarianceSwap(underlier="ACME", strike_variance=0.04, maturity=1.0, settlement="physical")


def test_variance_swap_maturity_must_be_strictly_positive() -> None:
    with pytest.raises(InstrumentValidationError, match="maturity must be strictly positive"):
        build(VarianceSwap, maturity=0.0)


def test_variance_swap_has_no_option_type() -> None:
    """It is a swap, not an option: there is no side to choose beyond notional."""
    field_names = {f.name for f in dataclasses.fields(VarianceSwap)}
    assert "option_type" not in field_names
    assert "strike" not in field_names


def test_variance_swap_capabilities() -> None:
    caps = capabilities(VarianceSwap)
    assert caps.payoff is True
    assert caps.simulate is True
    assert caps.price == frozenset()


def test_the_variance_record_carries_no_market_words() -> None:
    """'volatility' is a market observation; the contract quotes a variance."""
    import json

    flat = json.dumps(instrument_to_dict(build(VarianceSwap)))
    assert "volatility" not in flat
    assert "strike_variance" in flat
