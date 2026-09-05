"""The registry is the single discovery surface, and it is read-only."""

from __future__ import annotations

import dataclasses

import pytest

from fast_vollib.instruments import (
    AsianOption,
    Asset,
    BarrierOption,
    BinaryOption,
    CapabilitySet,
    EuropeanOption,
    FixedRateBond,
    Forward,
    Future,
    InstrumentKind,
    IVSolver,
    LookbackOption,
    PayoffRequirement,
    PricingModel,
    UnsupportedInstrumentError,
    VarianceSwap,
    ZeroCouponBond,
    capabilities,
    instrument_type,
    instrument_types,
)
from fast_vollib.instruments.registry import InstrumentTypeInfo, type_id_for

EXPECTED_TYPES = {
    "asset": Asset,
    "forward": Forward,
    "future": Future,
    "european_option": EuropeanOption,
    "binary_option": BinaryOption,
    "asian_option": AsianOption,
    "barrier_option": BarrierOption,
    "lookback_option": LookbackOption,
    "variance_swap": VarianceSwap,
    "zero_coupon_bond": ZeroCouponBond,
    "fixed_rate_bond": FixedRateBond,
}


def test_registry_lists_every_shipped_type() -> None:
    assert (
        dict((type_id, info.python_type) for type_id, info in instrument_types().items())
        == EXPECTED_TYPES
    )


def test_type_ids_match_the_instrument_kind_vocabulary() -> None:
    assert set(instrument_types()) == {kind.value for kind in InstrumentKind}


def test_public_mapping_cannot_be_mutated() -> None:
    types = instrument_types()
    with pytest.raises(TypeError):
        types["american_option"] = types["european_option"]  # type: ignore[index]
    with pytest.raises(TypeError):
        del types["european_option"]  # type: ignore[attr-defined]
    with pytest.raises(AttributeError):
        types.clear()  # type: ignore[attr-defined]


def test_mutating_a_copy_does_not_touch_the_registry() -> None:
    copy = dict(instrument_types())
    copy.pop("european_option")
    assert "european_option" in instrument_types()


def test_type_info_is_frozen() -> None:
    info = instrument_type("european_option")
    with pytest.raises(dataclasses.FrozenInstanceError):
        info.type_id = "other"  # type: ignore[misc]


def test_repeated_calls_return_the_same_view() -> None:
    assert instrument_types() is instrument_types()


@pytest.mark.parametrize("type_id", sorted(EXPECTED_TYPES))
def test_lookup_by_type_id(type_id: str) -> None:
    info = instrument_type(type_id)
    assert isinstance(info, InstrumentTypeInfo)
    assert info.type_id == type_id
    assert info.schema_version == 1
    assert isinstance(info.capabilities, CapabilitySet)
    assert info.description


def test_unknown_type_id_names_the_known_ones() -> None:
    with pytest.raises(UnsupportedInstrumentError) as excinfo:
        instrument_type("american_option")
    message = str(excinfo.value)
    assert "american_option" in message
    assert "european_option" in message


@pytest.mark.parametrize(("type_id", "cls"), sorted(EXPECTED_TYPES.items()))
def test_type_id_for_round_trips(type_id: str, cls: type) -> None:
    assert type_id_for(cls) == type_id


def test_type_id_for_rejects_a_foreign_class() -> None:
    with pytest.raises(UnsupportedInstrumentError, match="not an instrument type"):
        type_id_for(dict)  # type: ignore[arg-type]


# --- payoff requirements -----------------------------------------------------


def test_payoff_requirements_are_declared_per_type() -> None:
    requirements = {tid: info.payoff_requirement for tid, info in instrument_types().items()}
    assert requirements == {
        "asset": None,
        "forward": PayoffRequirement.TERMINAL,
        "future": PayoffRequirement.TERMINAL,
        "european_option": PayoffRequirement.TERMINAL,
        "binary_option": PayoffRequirement.TERMINAL,
        "asian_option": PayoffRequirement.PATH,
        "barrier_option": PayoffRequirement.PATH,
        "lookback_option": PayoffRequirement.PATH,
        "variance_swap": PayoffRequirement.PATH,
        # A bond's payments are dated rather than terminal, so it is not
        # evaluated through the payoff dispatcher at all.
        "zero_coupon_bond": None,
        "fixed_rate_bond": None,
    }


def test_registry_capabilities_agree_with_the_capabilities_function() -> None:
    for info in instrument_types().values():
        assert info.capabilities == capabilities(info.python_type)


# --- capability content ------------------------------------------------------


def test_european_option_capabilities() -> None:
    caps = capabilities(EuropeanOption)
    assert caps.payoff is True
    assert caps.price == frozenset(PricingModel)
    assert caps.greeks == frozenset(PricingModel)
    assert set(caps.implied_volatility) == set(PricingModel)
    for model in PricingModel:
        assert caps.solvers_for(model) == frozenset(IVSolver)
    assert caps.simulate is True


def test_implied_volatility_capability_keeps_the_solver_dimension() -> None:
    """A boolean would erase the distinction a caller has to dispatch on."""
    caps = capabilities(EuropeanOption)
    assert not isinstance(caps.implied_volatility, bool)
    with pytest.raises(TypeError):
        caps.implied_volatility[PricingModel.BLACK] = frozenset()  # type: ignore[index]


def test_native_autodiff_is_only_ever_the_jackel_solver() -> None:
    for _model, solver, _backend in capabilities(EuropeanOption).native_autodiff:
        assert solver is IVSolver.JACKEL


def test_native_autodiff_tracks_installed_backends() -> None:
    import importlib.util

    caps = capabilities(EuropeanOption)
    backends = {backend for _m, _s, backend in caps.native_autodiff}
    for name in ("torch", "jax"):
        installed = importlib.util.find_spec(name) is not None
        assert (name in backends) is installed


@pytest.mark.parametrize("cls", [Forward, Future], ids=["forward", "future"])
def test_linear_contracts_are_recognized_but_have_no_analytic_kernel(cls: type) -> None:
    caps = capabilities(cls)
    assert caps.payoff is True
    assert caps.price == frozenset()
    assert caps.greeks == frozenset()
    assert caps.implied_volatility == {}
    assert caps.native_autodiff == frozenset()


def test_a_forward_can_be_simulated_but_a_future_cannot() -> None:
    """Their terminal formulas coincide; their economics do not."""
    assert capabilities(Forward).simulate is True
    assert capabilities(Future).simulate is False
    assert capabilities(Future).simulation_autodiff == frozenset()


def test_simulation_autodiff_tracks_installed_backends() -> None:
    import importlib.util

    caps = capabilities(EuropeanOption)
    for name in ("torch", "jax"):
        installed = importlib.util.find_spec(name) is not None
        assert caps.supports_simulation_autodiff(name) is installed
    assert caps.supports_simulation_autodiff("numpy") is False


def test_asset_has_no_operations() -> None:
    caps = capabilities(Asset)
    assert caps.payoff is False
    assert caps.price == frozenset()
    assert caps.simulate is False


def test_capabilities_accepts_an_instance() -> None:
    option = EuropeanOption(underlier="ACME", option_type="call", strike=1.0, maturity=1.0)
    assert capabilities(option) == capabilities(EuropeanOption)


def test_capabilities_rejects_a_foreign_type() -> None:
    with pytest.raises(UnsupportedInstrumentError, match="not an instrument type"):
        capabilities(dict)  # type: ignore[arg-type]
