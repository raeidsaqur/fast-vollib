"""The third contract branch: what it is, and what it refuses to be.

Two halves.  The first is ordinary contract testing -- a bond is a frozen,
hashable, serializable value with hand-computable cashflows.

The second is the part that matters more, and it is a *negative* specification.
Adding a contract family that is not evaluated through
:func:`~fast_vollib.instruments.payoff` creates a new way for the library to be
wrong: an entry point that accepts a bond and returns a plausible number
computed from a fictional terminal state.  Every such entry point is therefore
listed here and required to refuse by name, pointing at
``fast_vollib.pricing.present_value``.  A number from any of them would be
worse than an exception, because it would look like an answer.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from fast_vollib.instruments import (
    Asset,
    Cashflow,
    Derivative,
    EuropeanOption,
    EuropeanOptionBatch,
    FixedIncomeSecurity,
    Instrument,
    InstrumentKind,
    InstrumentValidationError,
    UnsupportedInstrumentError,
    VanillaMarketInputs,
    ZeroCouponBond,
    capabilities,
    cashflows,
    greeks_instrument,
    implied_volatility_instrument,
    instrument_from_dict,
    instrument_from_json,
    instrument_to_dict,
    instrument_to_json,
    payoff,
    payoff_requirement,
    price_instrument,
)

BOND = ZeroCouponBond(maturity=2.0, face_value=1000.0, currency="USD")


# --- the taxonomy --------------------------------------------------------------


def test_a_bond_is_an_instrument_but_neither_an_asset_nor_a_derivative() -> None:
    """The third branch is a sibling of the other two, not a child of either.

    Not a pedantic point. ``Asset`` requires a symbol and describes an
    underlier; ``Derivative`` carries a ``notional`` whose sign means a short
    position and points at an underlier a bond does not have.
    """
    assert isinstance(BOND, Instrument)
    assert isinstance(BOND, FixedIncomeSecurity)
    assert not isinstance(BOND, Asset)
    assert not isinstance(BOND, Derivative)


def test_the_abstract_root_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError, match="abstract"):
        FixedIncomeSecurity()  # type: ignore[abstract]


def test_a_bond_carries_no_underlier_and_no_notional() -> None:
    """A bond's payments do not depend on anything, which is the whole point."""
    fields = set(ZeroCouponBond.__dataclass_fields__)
    assert "underlier" not in fields
    assert "notional" not in fields
    assert fields == {"instrument_id", "face_value", "currency", "maturity"}


def test_a_bond_holds_terms_and_nothing_else() -> None:
    """No curve, no yield, no accrued interest, no settlement date."""
    forbidden = {"curve", "discount_curve", "yield_", "accrued", "settlement", "issuer", "spread"}
    assert forbidden.isdisjoint(set(ZeroCouponBond.__dataclass_fields__))


def test_the_kind_discriminator_is_the_wire_identifier() -> None:
    assert BOND.kind is InstrumentKind.ZERO_COUPON_BOND
    assert BOND.kind.value == "zero_coupon_bond"


# --- a contract is a value -----------------------------------------------------


def test_a_bond_is_frozen_hashable_and_compares_structurally() -> None:
    assert ZeroCouponBond(maturity=5.0) == ZeroCouponBond(maturity=5.0)
    assert ZeroCouponBond(maturity=5.0) != ZeroCouponBond(maturity=5.5)
    assert len({ZeroCouponBond(maturity=5.0), ZeroCouponBond(maturity=5.0)}) == 1
    with pytest.raises(dataclasses.FrozenInstanceError):
        BOND.maturity = 3.0  # type: ignore[misc]


def test_the_currency_is_upper_cased_like_every_other_contract() -> None:
    assert ZeroCouponBond(maturity=1.0, currency=" eur ").currency == "EUR"
    assert ZeroCouponBond(maturity=1.0).currency is None


# --- cashflows -----------------------------------------------------------------


def test_the_cashflow_is_the_hand_computed_one() -> None:
    assert cashflows(BOND) == (Cashflow(payment_time=2.0, amount=1000.0),)


def test_a_payment_due_now_is_a_valid_contract() -> None:
    """Zero maturity is not degenerate: it is the last payment of a matured bond."""
    assert cashflows(ZeroCouponBond(maturity=0.0, face_value=7.0)) == (
        Cashflow(payment_time=0.0, amount=7.0),
    )


def test_the_face_value_scales_the_cashflow_exactly() -> None:
    for face in (0.5, 1.0, 100.0, 1e6):
        (flow,) = cashflows(ZeroCouponBond(maturity=1.0, face_value=face))
        assert flow.amount == face


def test_the_property_and_the_dispatcher_agree() -> None:
    assert BOND.cashflows == cashflows(BOND)


def test_a_cashflow_orders_chronologically() -> None:
    flows = [Cashflow(payment_time=t, amount=1.0) for t in (3.0, 1.0, 2.0)]
    assert [f.payment_time for f in sorted(flows)] == [1.0, 2.0, 3.0]


def test_cashflows_refuses_anything_that_is_not_a_fixed_income_security() -> None:
    option = EuropeanOption(underlier="ACME", option_type="call", strike=100.0, maturity=1.0)
    with pytest.raises(UnsupportedInstrumentError, match="present_value"):
        cashflows(option)  # type: ignore[arg-type]


# --- validation ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"maturity": -1.0}, "maturity must be non-negative"),
        ({"maturity": float("nan")}, "maturity must be finite"),
        ({"maturity": float("inf")}, "maturity must be finite"),
        ({"face_value": 0.0}, "face_value must be strictly positive"),
        ({"face_value": -100.0}, "face_value must be strictly positive"),
        ({"face_value": float("inf")}, "face_value must be finite"),
        ({"maturity": True}, "maturity must be a real number, not a bool"),
        ({"maturity": np.array([1.0, 2.0])}, "maturity must be a real scalar number"),
    ],
)
def test_invalid_terms_are_refused(kwargs, message) -> None:
    with pytest.raises(InstrumentValidationError, match=message):
        ZeroCouponBond(**{"maturity": 1.0, **kwargs})


@pytest.mark.parametrize(
    ("payment_time", "amount", "message"),
    [
        (-1.0, 1.0, "payment_time must be non-negative"),
        (float("nan"), 1.0, "payment_time must be finite"),
        (1.0, float("inf"), "amount must be finite"),
    ],
)
def test_an_invalid_cashflow_is_refused(payment_time, amount, message) -> None:
    with pytest.raises(InstrumentValidationError, match=message):
        Cashflow(payment_time=payment_time, amount=amount)


def test_a_negative_cashflow_amount_is_allowed() -> None:
    """A fee, or the paying leg of a schedule. Only the timing is constrained."""
    assert Cashflow(payment_time=1.0, amount=-25.0).amount == -25.0


# --- capabilities and the registry ---------------------------------------------


def test_the_capability_set_reports_the_fixed_income_route_and_nothing_else() -> None:
    caps = capabilities(ZeroCouponBond)
    assert caps.cashflows and caps.present_value
    assert not caps.payoff
    assert not caps.simulate
    assert caps.price == frozenset()
    assert caps.greeks == frozenset()
    assert dict(caps.implied_volatility) == {}


def test_no_other_type_claims_the_fixed_income_capabilities() -> None:
    """The two new booleans default to False and must not have leaked."""
    for cls in (EuropeanOption, Asset):
        caps = capabilities(cls)
        assert not caps.cashflows
        assert not caps.present_value


def test_the_payoff_requirement_is_none_because_there_is_no_payoff() -> None:
    assert payoff_requirement(ZeroCouponBond) is None
    assert payoff_requirement(BOND) is None


# --- serialization -------------------------------------------------------------


def test_a_record_round_trips_exactly() -> None:
    for bond in (
        BOND,
        ZeroCouponBond(maturity=0.0),
        ZeroCouponBond(maturity=30.0, face_value=1e6, instrument_id="row-7"),
    ):
        assert instrument_from_dict(instrument_to_dict(bond)) == bond
        assert instrument_from_json(instrument_to_json(bond)) == bond


def test_the_record_carries_the_schema_version_and_discriminator() -> None:
    record = instrument_to_dict(BOND)
    assert record["schema_version"] == 1
    assert record["instrument_type"] == "zero_coupon_bond"
    assert record["face_value"] == 1000.0
    assert record["maturity"] == 2.0
    assert record["currency"] == "USD"


def test_an_unknown_field_is_rejected() -> None:
    record = instrument_to_dict(BOND)
    record["coupon_rate"] = 0.05
    with pytest.raises(Exception, match="Unknown field"):
        instrument_from_dict(record)


def test_a_defaulted_field_may_be_omitted() -> None:
    record = instrument_to_dict(ZeroCouponBond(maturity=3.0))
    del record["face_value"]
    del record["currency"]
    assert instrument_from_dict(record) == ZeroCouponBond(maturity=3.0)


def test_a_missing_required_field_is_rejected() -> None:
    record = instrument_to_dict(BOND)
    del record["maturity"]
    with pytest.raises(Exception, match="missing required field 'maturity'"):
        instrument_from_dict(record)


# --- fail-closed: the entry points a bond must never pass through ---------------


def _market() -> VanillaMarketInputs:
    return VanillaMarketInputs(underlying=100.0, rate=0.02, volatility=0.2)


def test_payoff_refuses_a_bond_and_names_the_route_that_works() -> None:
    with pytest.raises(UnsupportedInstrumentError) as excinfo:
        payoff(BOND, 100.0)
    message = str(excinfo.value)
    assert "fast_vollib.pricing.present_value" in message
    assert "cashflows" in message


def test_price_instrument_refuses_a_bond_and_names_present_value() -> None:
    with pytest.raises(UnsupportedInstrumentError, match="fast_vollib.pricing.present_value"):
        price_instrument(BOND, _market(), model="black_scholes")


def test_greeks_instrument_refuses_a_bond_and_names_present_value() -> None:
    with pytest.raises(UnsupportedInstrumentError, match="fast_vollib.pricing.present_value"):
        greeks_instrument(BOND, _market(), model="black_scholes")


def test_implied_volatility_instrument_refuses_a_bond_and_names_present_value() -> None:
    with pytest.raises(UnsupportedInstrumentError, match="fast_vollib.pricing.present_value"):
        implied_volatility_instrument(BOND, _market(), model="black_scholes")


def test_the_monte_carlo_engine_refuses_a_bond_and_names_present_value() -> None:
    """Simulating an underlier would not value a contract that has none."""
    from fast_vollib.processes import GBM
    from fast_vollib.simulation import MonteCarloEngine

    with pytest.raises(UnsupportedInstrumentError, match="fast_vollib.pricing.present_value"):
        MonteCarloEngine().price(
            BOND,
            _market(),
            process=GBM.risk_neutral(rate=0.02, volatility=0.2),
            n_paths=16,
            n_steps=2,
            rng=0,
        )


def test_the_monte_carlo_engine_reports_that_it_does_not_support_a_bond() -> None:
    from fast_vollib.simulation import MonteCarloEngine

    engine = MonteCarloEngine()
    assert engine.supports(ZeroCouponBond) is False
    assert engine.supports(BOND) is False


def test_a_batch_cannot_be_built_from_a_bond() -> None:
    """A batch is a columnar container of European options, not of contracts."""
    with pytest.raises((UnsupportedInstrumentError, TypeError, AttributeError, ValueError)):
        EuropeanOptionBatch.from_instruments([BOND])  # type: ignore[list-item]
