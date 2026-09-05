"""``FixedRateBond``, and the array field it is the first contract to need.

Two things are being tested and they are worth separating.  The contract is
ordinary: a schedule, a rate, and the arithmetic that turns them into
cashflows.  The *codec* is not -- ``FieldSpec`` learned a fourth JSON type here,
and a new encoded shape is the kind of change that can quietly alter records of
types that have nothing to do with it.  So the schema is checked for having
grown by exactly one member, and the strict decoder is checked for still being
strict on the new shape rather than merely accepting it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fast_vollib.instruments import (
    Cashflow,
    EuropeanOption,
    FixedIncomeSecurity,
    FixedRateBond,
    InstrumentKind,
    InstrumentValidationError,
    SerializationError,
    UnsupportedInstrumentError,
    ZeroCouponBond,
    capabilities,
    cashflows,
    instrument_from_dict,
    instrument_from_json,
    instrument_to_dict,
    instrument_to_json,
    payoff,
)

SEMIANNUAL = {
    "payment_times": (0.5, 1.0, 1.5, 2.0),
    "accrual_fractions": (0.5, 0.5, 0.5, 0.5),
    "coupon_rate": 0.04,
}


def bond(**overrides):
    return FixedRateBond(**{**SEMIANNUAL, **overrides})


# --- the cashflows -------------------------------------------------------------


def test_each_coupon_is_face_times_rate_times_its_own_accrual_fraction() -> None:
    """Hand-computable, and the accrual fraction is per period rather than shared."""
    schedule = bond(
        payment_times=(0.25, 1.0),
        accrual_fractions=(0.25, 0.75),
        coupon_rate=0.08,
        face_value=1000.0,
    ).cashflows
    assert schedule[0] == Cashflow(payment_time=0.25, amount=1000.0 * 0.08 * 0.25)
    # The last carries the principal as well.
    assert schedule[1] == Cashflow(payment_time=1.0, amount=1000.0 * 0.08 * 0.75 + 1000.0)


def test_the_principal_is_paid_once_and_only_with_the_last_coupon() -> None:
    schedule = bond(face_value=100.0).cashflows
    assert len(schedule) == 4
    assert [flow.amount for flow in schedule[:-1]] == [2.0, 2.0, 2.0]
    assert schedule[-1].amount == 102.0


def test_the_schedule_is_chronological_and_matches_the_payment_times() -> None:
    schedule = bond().cashflows
    assert [flow.payment_time for flow in schedule] == list(SEMIANNUAL["payment_times"])
    assert list(schedule) == sorted(schedule)


def test_maturity_is_the_final_payment_time() -> None:
    """The one attribute every instrument family must agree on the meaning of."""
    assert bond().maturity == 2.0
    assert getattr(bond(), "maturity") == bond().payment_times[-1]


def test_a_zero_coupon_rate_leaves_only_the_redemption() -> None:
    """A legitimate contract, and not the same object as a ``ZeroCouponBond``."""
    schedule = bond(coupon_rate=0.0, face_value=100.0).cashflows
    assert [flow.amount for flow in schedule] == [0.0, 0.0, 0.0, 100.0]


def test_a_negative_coupon_rate_is_accepted() -> None:
    """They occur; refusing them would make the contract unable to hold them."""
    schedule = bond(coupon_rate=-0.01, face_value=100.0).cashflows
    assert schedule[0].amount == pytest.approx(-0.5)
    assert schedule[-1].amount == pytest.approx(99.5)


def test_an_irregular_schedule_with_a_stub_is_representable() -> None:
    """Which is the reason there is no ``frequency`` field."""
    stub = bond(
        payment_times=(0.1, 0.6, 1.1),
        accrual_fractions=(0.1, 0.5, 0.5),
        coupon_rate=0.05,
        face_value=100.0,
    )
    assert [round(flow.amount, 10) for flow in stub.cashflows] == [0.5, 2.5, 102.5]


def test_the_dispatcher_returns_the_same_schedule_as_the_property() -> None:
    assert cashflows(bond()) == bond().cashflows


# --- it is a value -------------------------------------------------------------


def test_the_schedule_is_stored_as_tuples_so_the_contract_stays_hashable() -> None:
    """A list field would make the contract unhashable and mutable behind a
    reference the caller kept."""
    made = FixedRateBond(payment_times=[0.5, 1.0], accrual_fractions=[0.5, 0.5], coupon_rate=0.03)
    assert isinstance(made.payment_times, tuple)
    assert isinstance(made.accrual_fractions, tuple)
    assert hash(made) is not None
    assert made == FixedRateBond(
        payment_times=(0.5, 1.0), accrual_fractions=(0.5, 0.5), coupon_rate=0.03
    )


def test_the_contract_is_frozen() -> None:
    with pytest.raises(Exception):
        bond().coupon_rate = 0.1  # type: ignore[misc]


def test_it_is_a_fixed_income_security_and_declares_its_kind() -> None:
    assert isinstance(bond(), FixedIncomeSecurity)
    assert bond().kind is InstrumentKind.FIXED_RATE_BOND
    assert bond().kind.value == "fixed_rate_bond"


# --- refusals ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"payment_times": ()}, "must not be empty"),
        ({"accrual_fractions": ()}, "must not be empty"),
        ({"payment_times": (1.0, 0.5, 1.5, 2.0)}, "strictly increasing"),
        ({"payment_times": (0.5, 0.5, 1.5, 2.0)}, "strictly increasing"),
        ({"payment_times": (0.0, 1.0, 1.5, 2.0)}, "strictly positive"),
        ({"payment_times": (-0.5, 1.0, 1.5, 2.0)}, "strictly positive"),
        ({"accrual_fractions": (0.5, 0.5)}, "Each coupon accrues over exactly one period"),
        ({"accrual_fractions": (0.5, 0.0, 0.5, 0.5)}, "strictly positive"),
        ({"accrual_fractions": (0.5, -0.5, 0.5, 0.5)}, "strictly positive"),
        ({"coupon_rate": float("nan")}, "finite"),
        ({"payment_times": (0.5, float("inf"), 1.5, 2.0)}, "finite"),
        ({"face_value": 0.0}, "face_value"),
    ],
)
def test_an_invalid_schedule_is_refused_at_construction(overrides, message) -> None:
    with pytest.raises(InstrumentValidationError, match=message):
        bond(**overrides)


def test_a_string_schedule_is_refused_rather_than_iterated_as_characters() -> None:
    """``"0.5"`` is three characters, and iterating it would be a silent disaster."""
    with pytest.raises(InstrumentValidationError, match="sequence of numbers"):
        bond(payment_times="0.5")


def test_a_non_sequence_schedule_is_refused() -> None:
    with pytest.raises(InstrumentValidationError, match="sequence of numbers"):
        bond(payment_times=2.0)


def test_a_single_payment_bond_is_allowed() -> None:
    """A one-period bond is a contract; only an *empty* schedule is not."""
    single = bond(payment_times=(1.0,), accrual_fractions=(1.0,), face_value=100.0)
    assert single.cashflows == (Cashflow(payment_time=1.0, amount=104.0),)


# --- the fixed-income branch behaviours it inherits ----------------------------


def test_it_has_no_payoff_and_the_refusal_names_present_value() -> None:
    """Inherited from ``FixedIncomeSecurity``; verified rather than assumed."""
    with pytest.raises(UnsupportedInstrumentError, match="present_value"):
        payoff(bond(), 100.0)


def test_the_monte_carlo_engine_refuses_it_by_name() -> None:
    from fast_vollib.simulation import MonteCarloEngine

    engine = MonteCarloEngine()
    assert engine.supports(FixedRateBond) is False
    with pytest.raises(UnsupportedInstrumentError, match="present_value"):
        engine.price(bond(), None, process=None, n_paths=8, rng=0, n_steps=1)


def test_its_capabilities_match_the_other_fixed_income_type() -> None:
    reported = capabilities(FixedRateBond)
    assert reported == capabilities(ZeroCouponBond)
    assert reported.cashflows and reported.present_value
    assert not reported.payoff and reported.price == frozenset()


def test_present_value_discounts_every_cashflow_at_its_own_time() -> None:
    """The reason bonds are not routed through ``payoff``, stated numerically."""
    import math

    from fast_vollib.pricing import present_value
    from fast_vollib.rates import FlatDiscountCurve

    curve = FlatDiscountCurve(rate=0.03)
    priced = present_value(bond(face_value=100.0), discount_curve=curve)
    expected = sum(
        flow.amount * math.exp(-0.03 * flow.payment_time)
        for flow in bond(face_value=100.0).cashflows
    )
    assert priced == pytest.approx(expected, rel=1e-14)


def test_a_par_bond_prices_at_par_under_its_own_flat_curve() -> None:
    """A closed-form anchor: with continuous compounding a bond whose coupon
    rate equals ``exp(r * dt) - 1`` per period is worth its face value."""
    import math

    from fast_vollib.pricing import present_value
    from fast_vollib.rates import FlatDiscountCurve

    rate, periods, dt = 0.05, 10, 0.5
    par = FixedRateBond(
        payment_times=tuple(dt * (i + 1) for i in range(periods)),
        accrual_fractions=(dt,) * periods,
        coupon_rate=(math.exp(rate * dt) - 1.0) / dt,
        face_value=100.0,
    )
    assert present_value(par, discount_curve=FlatDiscountCurve(rate=rate)) == pytest.approx(
        100.0, rel=1e-12
    )


# --- the array codec -----------------------------------------------------------


def test_the_schedule_encodes_as_a_json_array_and_decodes_back_to_a_tuple() -> None:
    record = instrument_to_dict(bond(face_value=100.0, currency="usd", instrument_id="b-1"))
    assert record["payment_times"] == [0.5, 1.0, 1.5, 2.0]
    assert isinstance(record["payment_times"], list)
    restored = instrument_from_dict(record)
    assert restored == bond(face_value=100.0, currency="usd", instrument_id="b-1")
    assert isinstance(restored.payment_times, tuple)


def test_the_json_round_trip_preserves_equality() -> None:
    original = bond(face_value=250.0, currency="EUR")
    assert instrument_from_json(instrument_to_json(original)) == original


def test_a_record_holding_a_scalar_where_an_array_belongs_is_refused() -> None:
    record = instrument_to_dict(bond())
    record["payment_times"] = 2.0
    with pytest.raises(SerializationError, match="must be a JSON array"):
        instrument_from_dict(record)


def test_a_record_holding_a_string_where_an_array_belongs_is_refused() -> None:
    """A string is a sequence, which is exactly why it has to be named out."""
    record = instrument_to_dict(bond())
    record["payment_times"] = "0.5"
    with pytest.raises(SerializationError, match="must be a JSON array"):
        instrument_from_dict(record)


def test_an_empty_array_is_refused_by_the_decoder_not_only_by_the_contract() -> None:
    record = instrument_to_dict(bond())
    record["payment_times"] = []
    record["accrual_fractions"] = []
    with pytest.raises(SerializationError, match="at least 1 entry"):
        instrument_from_dict(record)


def test_a_non_numeric_element_is_refused_naming_its_position() -> None:
    record = instrument_to_dict(bond())
    record["payment_times"] = [0.5, "1.0", 1.5, 2.0]
    with pytest.raises(SerializationError, match=r"payment_times\[1\]"):
        instrument_from_dict(record)


def test_a_null_element_is_refused() -> None:
    record = instrument_to_dict(bond())
    record["payment_times"] = [0.5, None, 1.5, 2.0]
    with pytest.raises(SerializationError, match="must not be null"):
        instrument_from_dict(record)


def test_arrays_of_disagreeing_length_are_refused_by_the_contract() -> None:
    """The schema cannot state a cross-field rule, so the decoder relies on the
    dataclass -- which means the dataclass has to be reached, not bypassed."""
    record = instrument_to_dict(bond())
    record["accrual_fractions"] = [0.5, 0.5]
    with pytest.raises(InstrumentValidationError, match="one period"):
        instrument_from_dict(record)


def test_an_unknown_field_is_still_refused_on_the_new_type() -> None:
    record = instrument_to_dict(bond())
    record["frequency"] = 2
    with pytest.raises(SerializationError, match="Unknown field"):
        instrument_from_dict(record)


def test_an_option_record_is_unaffected_by_the_new_json_type() -> None:
    """The regression the array change could plausibly have caused."""
    option = EuropeanOption(underlier="SPX", option_type="call", strike=5000.0, maturity=0.75)
    assert instrument_from_dict(instrument_to_dict(option)) == option
    assert "payment_times" not in instrument_to_dict(option)


# --- the schema ----------------------------------------------------------------


def test_the_published_schema_gained_exactly_one_member_at_version_one() -> None:
    """What "the schema version stays 1" means operationally.

    A new ``oneOf`` alternative changes no existing record, so no consumer of
    an existing type needs to know. That claim is only true if every other
    definition is untouched, which is what this checks.
    """
    schema = json.loads(
        (Path(__file__).resolve().parents[2] / "docs/schemas/instrument-v1.schema.json").read_text(
            encoding="utf-8"
        )
    )
    assert "fixed_rate_bond" in schema["$defs"]
    for definition in schema["$defs"].values():
        version = definition.get("properties", {}).get("schema_version")
        if version is not None:
            assert version["const"] == 1

    payment_times = schema["$defs"]["fixed_rate_bond"]["properties"]["payment_times"]
    assert payment_times["type"] == "array"
    assert payment_times["items"] == {"type": "number", "exclusiveMinimum": 0.0}
    assert payment_times["minItems"] == 1
    # The cross-field rule the schema cannot check is stated rather than implied.
    assert "same length as accrual_fractions" in payment_times["description"]


def test_every_array_field_spec_carries_an_item_spec() -> None:
    """The import-time guard, exercised so a future array field cannot skip it."""
    from fast_vollib.instruments._metadata import JSON_TYPES, TYPE_SPECS

    for spec in TYPE_SPECS:
        for field_spec in spec.fields:
            assert field_spec.json_type in JSON_TYPES, (spec.type_id, field_spec.name)
            assert (field_spec.json_type == "array") == (field_spec.items is not None)
