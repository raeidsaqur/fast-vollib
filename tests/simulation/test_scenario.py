"""Scenario construction, ownership, access, and contract compatibility."""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import pytest

from fast_vollib.instruments import (
    Asset,
    AssetClass,
    EuropeanOption,
    Forward,
    InstrumentRef,
    InstrumentValidationError,
    SerializationError,
    instrument_to_dict,
    payoff,
)
from fast_vollib.simulation import Scenario, ScenarioMismatchError, SimulationValidationError

GRID = [0.0, 0.5, 1.0]
STATES = np.array(
    [
        [[100.0], [110.0], [121.0]],
        [[100.0], [90.0], [81.0]],
    ]
)


def scenario(**overrides: Any) -> Scenario:
    call: dict[str, Any] = {"underlier": "ACME", "time_grid": GRID, "states": STATES}
    call.update(overrides)
    underlier = call.pop("underlier")
    return Scenario.from_states(underlier, **call)


# --- shape and value invariants -----------------------------------------------


def test_construction_records_grid_states_and_names() -> None:
    built = scenario()
    assert built.underlier == InstrumentRef(identifier="ACME")
    assert built.state_names == ("spot",)
    np.testing.assert_array_equal(built.time_grid, np.asarray(GRID))
    np.testing.assert_array_equal(built.states, STATES)


def test_path_and_step_counts() -> None:
    built = scenario()
    assert built.n_paths == 2
    assert built.n_steps == 2


def test_state_spot_and_terminal_accessors() -> None:
    built = scenario()
    np.testing.assert_array_equal(built.spot, STATES[:, :, 0])
    np.testing.assert_array_equal(built.state("spot"), STATES[:, :, 0])
    np.testing.assert_array_equal(built.terminal(), np.array([121.0, 81.0]))
    np.testing.assert_array_equal(built.terminal("spot"), np.array([121.0, 81.0]))


def test_a_missing_state_name_says_what_is_there() -> None:
    built = scenario()
    with pytest.raises(SimulationValidationError) as excinfo:
        built.state("variance")
    assert "'spot'" in str(excinfo.value)


def test_spot_raises_clearly_when_there_is_no_spot() -> None:
    built = scenario(state_names=("variance",))
    with pytest.raises(SimulationValidationError, match="no 'spot' state"):
        _ = built.spot


def test_multiple_states_are_addressed_by_name() -> None:
    states = np.stack([STATES[:, :, 0], STATES[:, :, 0] * 0.01], axis=-1)
    built = scenario(states=states, state_names=("spot", "variance"))
    np.testing.assert_allclose(built.state("variance"), STATES[:, :, 0] * 0.01)


def test_the_repr_does_not_dump_the_buffer() -> None:
    text = repr(scenario())
    assert "ACME" in text and "n_paths=2" in text
    assert "121.0" not in text


# --- grid validation ----------------------------------------------------------


@pytest.mark.parametrize(
    ("grid", "match"),
    [
        ([[0.0, 1.0]], "one-dimensional"),
        ([0.0], "at least two points"),
        ([0.0, float("inf")], "finite"),
        ([0.1, 1.0], "must start at exactly 0.0"),
        ([0.0, 1.0, 0.5], "strictly increasing"),
        ([0.0, 0.5, 0.5], "strictly increasing"),
    ],
    ids=["2d", "single", "infinite", "late-start", "unordered", "repeated"],
)
def test_invalid_grids_are_refused(grid: Any, match: str) -> None:
    """The grid is checked before the states, so the message names the grid."""
    with pytest.raises(SimulationValidationError, match=match):
        Scenario.from_states("ACME", time_grid=grid, states=np.ones((2, 3, 1)))


def test_a_grid_starting_at_a_tiny_positive_time_is_refused() -> None:
    with pytest.raises(SimulationValidationError, match="must start at exactly 0.0"):
        Scenario.from_states("ACME", time_grid=[1e-18, 1.0], states=np.ones((1, 2, 1)))


# --- state validation ---------------------------------------------------------


def test_states_must_be_three_dimensional() -> None:
    with pytest.raises(SimulationValidationError, match="n_paths, n_times, n_state"):
        Scenario.from_states("ACME", time_grid=GRID, states=STATES[:, :, 0])


def test_states_must_match_the_grid_length() -> None:
    with pytest.raises(SimulationValidationError, match="but the grid has"):
        Scenario.from_states("ACME", time_grid=[0.0, 1.0], states=STATES)


def test_states_must_match_the_state_name_count() -> None:
    with pytest.raises(SimulationValidationError, match="state variables but 2 were named"):
        Scenario.from_states("ACME", time_grid=GRID, states=STATES, state_names=("spot", "v"))


@pytest.mark.parametrize(
    "states",
    [
        np.ones((2, 3, 1), dtype=bool),
        np.ones((2, 3, 1), dtype=np.int64),
        np.ones((2, 3, 1), dtype=np.complex128),
    ],
    ids=["bool", "int", "complex"],
)
def test_non_floating_states_are_refused(states: np.ndarray) -> None:
    with pytest.raises(SimulationValidationError, match="real and floating-point"):
        Scenario.from_states("ACME", time_grid=GRID, states=states)


def test_python_sequences_normalize_to_a_floating_dtype() -> None:
    built = Scenario.from_states("ACME", time_grid=(0.0, 1.0), states=[[[100], [110]]])
    assert built.states.dtype == np.float64
    assert built.time_grid.dtype == np.float64


def test_a_native_integer_grid_is_safely_cast_like_an_integer_sequence() -> None:
    built = Scenario.from_states(
        "ACME", time_grid=np.array([0, 1], dtype=np.int64), states=[[[100.0], [110.0]]]
    )
    assert built.time_grid.dtype == np.float64
    np.testing.assert_array_equal(built.time_grid, np.array([0.0, 1.0]))


@pytest.mark.parametrize(
    ("names", "match"),
    [
        ((), "at least one state"),
        (("",), "non-empty string"),
        (("spot", "spot"), "unique"),
        ("spot", "tuple of names"),
    ],
    ids=["empty", "blank", "duplicate", "bare-string"],
)
def test_invalid_state_names_are_refused(names: Any, match: str) -> None:
    with pytest.raises(SimulationValidationError, match=match):
        Scenario.from_states("ACME", time_grid=GRID, states=STATES, state_names=names)


def test_one_path_is_enough() -> None:
    assert Scenario.from_states("ACME", time_grid=GRID, states=STATES[:1]).n_paths == 1


# --- ownership ----------------------------------------------------------------


def test_from_states_copies_caller_owned_numpy_buffers() -> None:
    """A later edit to the caller's array must not change what a scenario means."""
    grid = np.array(GRID)
    states = STATES.copy()
    built = Scenario.from_states("ACME", time_grid=grid, states=states)
    states[0, 0, 0] = -1.0
    grid[1] = 99.0
    assert built.states[0, 0, 0] == 100.0
    assert built.time_grid[1] == 0.5


def test_stored_numpy_buffers_are_read_only() -> None:
    built = scenario()
    assert not built.states.flags.writeable
    assert not built.time_grid.flags.writeable
    with pytest.raises(ValueError):
        built.states[0, 0, 0] = 1.0


def test_freezing_does_not_reach_back_into_the_callers_array() -> None:
    grid = np.array(GRID)
    states = STATES.copy()
    Scenario.from_states("ACME", time_grid=grid, states=states)
    assert grid.flags.writeable
    assert states.flags.writeable


def test_the_scenario_itself_is_frozen_and_slotted() -> None:
    built = scenario()
    with pytest.raises(dataclasses.FrozenInstanceError):
        built.states = STATES  # type: ignore[misc]
    with pytest.raises((AttributeError, TypeError)):
        built.cached = 1  # type: ignore[attr-defined]


def test_equality_and_hashing_are_by_identity() -> None:
    first, second = scenario(), scenario()
    np.testing.assert_array_equal(first.states, second.states)
    assert first != second
    assert first == first
    assert hash(first) != hash(second)
    assert {first, second} == {first, second}


def test_a_scenario_is_not_serializable() -> None:
    with pytest.raises(SerializationError, match="not a serializable instrument type"):
        instrument_to_dict(scenario())  # type: ignore[arg-type]


# --- native namespaces --------------------------------------------------------


def test_torch_states_are_stored_undetached() -> None:
    torch = pytest.importorskip("torch", reason="torch not installed")
    states = torch.tensor(STATES, dtype=torch.float64, requires_grad=True)
    built = Scenario.from_states(
        "ACME", time_grid=torch.tensor(GRID, dtype=torch.float64), states=states
    )
    assert built.states is states
    assert built.states.requires_grad
    built.terminal().sum().backward()
    assert states.grad is not None


def test_a_python_grid_normalizes_into_the_state_namespace() -> None:
    torch = pytest.importorskip("torch", reason="torch not installed")
    built = Scenario.from_states(
        "ACME",
        time_grid=GRID,
        states=torch.tensor(STATES, dtype=torch.float32),
    )
    assert torch.is_tensor(built.time_grid)
    assert built.time_grid.dtype == torch.float32


def test_mixed_namespaces_are_refused() -> None:
    torch = pytest.importorskip("torch", reason="torch not installed")
    pytest.importorskip("jax", reason="jax not installed")
    import jax.numpy as jnp

    with pytest.raises(SimulationValidationError, match="more than one array namespace"):
        Scenario.from_states("ACME", time_grid=jnp.asarray(GRID), states=torch.tensor(STATES))


def test_jax_states_round_trip() -> None:
    jax = pytest.importorskip("jax", reason="jax not installed")
    jax.config.update("jax_enable_x64", True)
    jnp = jax.numpy
    built = Scenario.from_states(
        "ACME",
        time_grid=jnp.asarray(GRID, dtype=jnp.float64),
        states=jnp.asarray(STATES, dtype=jnp.float64),
    )
    assert isinstance(built.states, jax.Array)
    np.testing.assert_array_equal(np.asarray(built.terminal()), np.array([121.0, 81.0]))


# --- compatibility with a contract --------------------------------------------


def option(**overrides: Any) -> EuropeanOption:
    call: dict[str, Any] = {
        "underlier": "ACME",
        "option_type": "call",
        "strike": 100.0,
        "maturity": 1.0,
    }
    call.update(overrides)
    return EuropeanOption(**call)


def test_a_matching_contract_is_evaluated_at_the_horizon() -> None:
    np.testing.assert_array_equal(scenario().payoff(option()), np.array([21.0, 0.0]))


def test_a_forward_is_evaluated_at_the_horizon_too() -> None:
    forward = Forward(underlier="ACME", delivery_price=100.0, maturity=1.0)
    np.testing.assert_array_equal(scenario().payoff(forward), np.array([21.0, -19.0]))


def test_a_different_underlier_is_refused() -> None:
    with pytest.raises(ScenarioMismatchError, match="written on 'OTHER'"):
        scenario().payoff(option(underlier="OTHER"))


def test_asset_class_and_currency_are_compared_only_when_both_specify() -> None:
    equity = Asset(identifier="ACME", asset_class=AssetClass.EQUITY, currency="USD")
    detailed = Scenario.from_states(equity, time_grid=GRID, states=STATES)
    # The contract's reference says nothing about class or currency: no conflict.
    assert detailed.payoff(option()) is not None
    with pytest.raises(ScenarioMismatchError, match="asset_class"):
        detailed.payoff(
            option(underlier=InstrumentRef(identifier="ACME", asset_class=AssetClass.FX))
        )
    with pytest.raises(ScenarioMismatchError, match="currency"):
        detailed.payoff(option(underlier=InstrumentRef(identifier="ACME", currency="EUR")))


def test_a_bare_scenario_underlier_does_not_constrain_a_detailed_contract() -> None:
    detailed = InstrumentRef(identifier="ACME", asset_class=AssetClass.EQUITY, currency="USD")
    assert scenario().payoff(option(underlier=detailed)) is not None


@pytest.mark.parametrize("maturity", [0.5, 1.5, 1.0 + 1e-6], ids=["short", "long", "near"])
def test_a_horizon_that_is_not_the_maturity_is_refused(maturity: float) -> None:
    with pytest.raises(ScenarioMismatchError, match="ends at t="):
        scenario().payoff(option(maturity=maturity))


def test_a_horizon_within_tolerance_is_accepted() -> None:
    assert scenario().payoff(option(maturity=1.0 + 5e-13)) is not None


def test_a_zero_maturity_contract_cannot_be_evaluated_on_a_scenario() -> None:
    with pytest.raises(ScenarioMismatchError, match="strictly positive"):
        scenario().payoff(option(maturity=0.0))


def test_an_asset_has_no_payoff_on_a_scenario() -> None:
    from fast_vollib.instruments import UnsupportedInstrumentError

    with pytest.raises(UnsupportedInstrumentError):
        scenario().payoff(Asset(identifier="ACME", asset_class="equity"))


def test_a_terminal_payoff_refuses_a_scenario_and_names_the_alternative() -> None:
    """Nothing is inferred: the routing choice is explicit, not guessed."""
    with pytest.raises(InstrumentValidationError) as excinfo:
        payoff(option(), scenario())
    assert "scenario.payoff(instrument)" in str(excinfo.value)


# --- dtype discipline ----------------------------------------------------------


@pytest.mark.parametrize(
    "states",
    [[[[True], [False]]], [[[1j], [2j]]], [[["a"], ["b"]]]],
    ids=["bool", "complex", "string"],
)
def test_a_non_real_sequence_is_refused_rather_than_cast(states: Any) -> None:
    """A mask read as a trajectory of ones and zeros is a plausible wrong answer.

    The check has to run on what the caller passed: converting first would turn
    booleans into 1.0 and 0.0 and then find a floating dtype and accept them.
    """
    with pytest.raises(SimulationValidationError, match="real and floating-point|numeric"):
        Scenario.from_states("ACME", time_grid=[0.0, 1.0], states=states)


def test_a_sequence_wrapping_a_boolean_array_is_refused_too() -> None:
    mask = np.ones((2, 1), dtype=bool)
    with pytest.raises(SimulationValidationError, match="real and floating-point"):
        Scenario.from_states("ACME", time_grid=[0.0, 1.0], states=[mask])


def test_the_grid_takes_the_dtype_of_the_states_it_indexes() -> None:
    """One scenario carries one precision, not two."""
    single = np.asarray(STATES, dtype=np.float32)
    built = Scenario.from_states("ACME", time_grid=np.array(GRID, dtype=np.float64), states=single)
    assert built.states.dtype == np.float32
    assert built.time_grid.dtype == np.float32


def test_the_states_buffer_is_never_cast_to_match_the_grid() -> None:
    """Casting the large buffer to match the small one is the worse mistake."""
    torch = pytest.importorskip("torch", reason="torch not installed")
    states = torch.tensor(STATES, dtype=torch.float64)
    built = Scenario.from_states(
        "ACME", time_grid=torch.tensor(GRID, dtype=torch.float32), states=states
    )
    assert built.states is states
    assert built.time_grid.dtype == torch.float64


def test_a_zero_path_state_buffer_is_refused() -> None:
    with pytest.raises(SimulationValidationError, match="at least one path"):
        Scenario.from_states("ACME", time_grid=GRID, states=np.zeros((0, 3, 1)))


# --- horizon tolerance across precisions ---------------------------------------


def test_the_horizon_tolerance_is_unchanged_in_double_precision() -> None:
    """Four float64 ulps are below the floor, so the rule reduces to the spec's."""
    from fast_vollib.simulation.scenario import horizon_tolerance

    for maturity in (1e-6, 0.1, 1.0, 30.0):
        assert horizon_tolerance(maturity) == horizon_tolerance(
            maturity, epsilon=float(np.finfo(np.float64).eps)
        )
        assert horizon_tolerance(maturity) == max(1e-12, 1e-12 * abs(maturity))


def test_a_single_precision_grid_is_held_to_a_tolerance_it_can_deliver() -> None:
    """0.1 is not representable in binary32; the contract is not thereby wrong."""
    maturity = 0.1
    grid = np.linspace(0.0, maturity, 9, dtype=np.float32)
    representation_error = abs(float(grid[-1]) - maturity)
    assert representation_error > 1e-12, "the float64 tolerance would refuse this"

    built = Scenario.from_states(
        "ACME", time_grid=grid, states=np.full((2, 9, 1), 100.0, dtype=np.float32)
    )
    contract = EuropeanOption(underlier="ACME", option_type="call", strike=100.0, maturity=maturity)
    assert built.payoff(contract) is not None


def test_a_single_precision_grid_still_refuses_a_genuinely_wrong_horizon() -> None:
    """Widening for representation error must not widen into a real mismatch."""
    built = Scenario.from_states(
        "ACME",
        time_grid=np.array([0.0, 0.05, 0.11], dtype=np.float32),
        states=np.full((2, 3, 1), 100.0, dtype=np.float32),
    )
    contract = EuropeanOption(underlier="ACME", option_type="call", strike=100.0, maturity=0.1)
    with pytest.raises(ScenarioMismatchError, match="ends at t="):
        built.payoff(contract)


def test_the_epsilon_comes_from_the_grid_dtype_in_every_namespace() -> None:
    from fast_vollib.simulation.scenario import dtype_epsilon

    assert dtype_epsilon(np.zeros(2, dtype=np.float32)) == pytest.approx(
        float(np.finfo(np.float32).eps)
    )
    assert dtype_epsilon(np.zeros(2)) == pytest.approx(float(np.finfo(np.float64).eps))
    assert dtype_epsilon([0.0, 1.0]) == 0.0
    torch = pytest.importorskip("torch", reason="torch not installed")
    assert dtype_epsilon(torch.zeros(2, dtype=torch.float32)) == pytest.approx(
        float(torch.finfo(torch.float32).eps)
    )
