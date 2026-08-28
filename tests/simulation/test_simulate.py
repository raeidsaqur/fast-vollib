"""``simulate``: validation, purity, backends, and what it deliberately does not know."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from fast_vollib.instruments import Asset, AssetClass, EuropeanOption, InstrumentRef
from fast_vollib.processes import GBM
from fast_vollib.simulation import (
    Scenario,
    SimulationValidationError,
    UnsupportedProcessError,
    simulate,
)

GRID = np.linspace(0.0, 1.0, 5)
PROCESS = GBM.risk_neutral(rate=0.03, volatility=0.2)


def run(**overrides: Any) -> Scenario:
    call: dict[str, Any] = {
        "underlier": "ACME",
        "process": PROCESS,
        "initial_state": 100.0,
        "time_grid": GRID,
        "n_paths": 16,
        "rng": 7,
    }
    call.update(overrides)
    underlier = call.pop("underlier")
    process = call.pop("process")
    return simulate(underlier, process, **call)


# --- what comes back ----------------------------------------------------------


def test_a_scenario_is_returned_with_the_requested_shape() -> None:
    scenario = run(n_paths=32)
    assert isinstance(scenario, Scenario)
    assert scenario.states.shape == (32, len(GRID), 1)
    assert scenario.state_names == ("spot",)


def test_the_underlier_is_recorded_as_a_reference() -> None:
    equity = Asset(identifier="ACME", asset_class=AssetClass.EQUITY, currency="USD")
    assert run(underlier=equity).underlier == equity.ref()
    assert run(underlier="ACME").underlier == InstrumentRef(identifier="ACME")


def test_the_grid_is_stored_read_only_and_matches_the_input() -> None:
    scenario = run()
    np.testing.assert_array_equal(scenario.time_grid, GRID)
    assert not scenario.time_grid.flags.writeable
    assert not scenario.states.flags.writeable


def test_normalizing_the_grid_does_not_freeze_the_callers_array() -> None:
    grid = np.linspace(0.0, 1.0, 5)
    run(time_grid=grid)
    assert grid.flags.writeable


def test_a_python_list_grid_is_accepted() -> None:
    scenario = run(time_grid=[0.0, 0.25, 1.0])
    np.testing.assert_array_equal(scenario.time_grid, np.array([0.0, 0.25, 1.0]))


def test_an_irregular_grid_is_accepted() -> None:
    grid = [0.0, 0.001, 0.5, 0.99, 1.0]
    assert run(time_grid=grid).n_steps == 4


# --- the initial state --------------------------------------------------------


def test_a_bare_number_is_shorthand_for_a_one_state_process() -> None:
    assert np.all(run(initial_state=123.0).spot[:, 0] == 123.0)


def test_a_mapping_is_accepted() -> None:
    assert np.all(run(initial_state={"spot": 123.0}).spot[:, 0] == 123.0)


def test_the_initial_state_is_mandatory() -> None:
    with pytest.raises(SimulationValidationError, match="initial_state is required"):
        run(initial_state=None)


def test_a_missing_state_is_named() -> None:
    with pytest.raises(SimulationValidationError, match="missing 'spot'"):
        run(initial_state={"forward": 100.0})


def test_an_unexpected_state_is_refused_rather_than_ignored() -> None:
    with pytest.raises(SimulationValidationError, match="does not evolve"):
        run(initial_state={"spot": 100.0, "variance": 0.04})


# --- path counts --------------------------------------------------------------


def test_one_path_is_permitted() -> None:
    assert run(n_paths=1).n_paths == 1


@pytest.mark.parametrize("bad", [0, -4], ids=["zero", "negative"])
def test_a_non_positive_path_count_is_refused(bad: int) -> None:
    with pytest.raises(SimulationValidationError, match="at least 1"):
        run(n_paths=bad)


@pytest.mark.parametrize("bad", [True, 4.0, "8"], ids=["bool", "float", "string"])
def test_a_non_integer_path_count_is_refused(bad: Any) -> None:
    with pytest.raises(SimulationValidationError, match="must be an integer"):
        run(n_paths=bad)


@pytest.mark.parametrize("bad", [1, 3, 0], ids=["one", "odd", "zero"])
def test_antithetic_requires_an_even_path_count_of_at_least_two(bad: int) -> None:
    with pytest.raises(SimulationValidationError, match="even and at least 2"):
        run(n_paths=bad, antithetic=True)


@pytest.mark.parametrize("bad", [1, "false", None], ids=["integer", "string", "none"])
def test_antithetic_must_be_a_boolean(bad: Any) -> None:
    with pytest.raises(SimulationValidationError, match="antithetic must be a bool"):
        run(antithetic=bad)


def test_antithetic_pairs_the_halves() -> None:
    scenario = run(n_paths=10, antithetic=True)
    log_paths = np.log(scenario.spot)
    steps = np.diff(GRID)
    deterministic = (PROCESS.drift - 0.5 * 0.2**2) * steps
    left = np.diff(log_paths[:5], axis=1) - deterministic
    right = np.diff(log_paths[5:], axis=1) - deterministic
    np.testing.assert_allclose(left, -right, rtol=1e-12, atol=1e-12)


# --- the grid contract --------------------------------------------------------


@pytest.mark.parametrize(
    ("grid", "match"),
    [
        ([0.0], "at least two points"),
        ([0.5, 1.0], "start at exactly 0.0"),
        ([0.0, 1.0, 0.5], "strictly increasing"),
        ([[0.0, 1.0]], "one-dimensional"),
        ([0.0, np.nan], "finite"),
    ],
    ids=["single", "late-start", "unordered", "2d", "nan"],
)
def test_invalid_grids_are_refused_before_sampling(grid: Any, match: str) -> None:
    with pytest.raises(SimulationValidationError, match=match):
        run(time_grid=grid)


def test_no_maturity_is_consulted() -> None:
    """simulate knows no contract: a horizon is whatever the caller asked for."""
    scenario = run(time_grid=[0.0, 3.0])
    assert float(scenario.time_grid[-1]) == 3.0


# --- the process contract -----------------------------------------------------


class _NoStateNames:
    def sample(self, **_kwargs: Any) -> Any:  # pragma: no cover - never reached
        raise AssertionError("must not be sampled")


class _BlankStateName:
    state_names = (" ",)

    def sample(self, **_kwargs: Any) -> Any:  # pragma: no cover - never reached
        raise AssertionError("must not be sampled")


class _WrongShape:
    state_names = ("spot",)

    def params(self) -> dict[str, Any]:
        return {}

    def sample(self, **_kwargs: Any) -> Any:
        return np.zeros((2, 2))


class _WrongDtype:
    state_names = ("spot",)

    def params(self) -> dict[str, Any]:
        return {}

    def sample(self, *, n_paths: int, time_grid: Any, **_kwargs: Any) -> Any:
        return np.ones((n_paths, len(time_grid), 1), dtype=np.int64)


class _WrongPrecision:
    state_names = ("spot",)

    def params(self) -> dict[str, Any]:
        return {}

    def sample(self, *, n_paths: int, time_grid: Any, **_kwargs: Any) -> Any:
        return np.ones((n_paths, len(time_grid), 1), dtype=np.float32)


def test_a_process_without_state_names_is_refused() -> None:
    with pytest.raises(UnsupportedProcessError, match="state_names"):
        run(process=_NoStateNames())


def test_a_process_with_a_blank_state_name_is_refused() -> None:
    with pytest.raises(UnsupportedProcessError, match="non-empty strings"):
        run(process=_BlankStateName())


def test_a_process_returning_the_wrong_shape_is_refused() -> None:
    with pytest.raises(UnsupportedProcessError, match="returned shape"):
        run(process=_WrongShape())


def test_a_process_returning_non_floating_states_is_refused() -> None:
    with pytest.raises(UnsupportedProcessError, match="real floating-point states"):
        run(process=_WrongDtype())


def test_a_process_returning_the_wrong_precision_is_refused() -> None:
    with pytest.raises(UnsupportedProcessError, match="one precision"):
        run(process=_WrongPrecision())


def test_a_process_returning_the_wrong_namespace_is_refused() -> None:
    torch = pytest.importorskip("torch", reason="torch not installed")
    with pytest.raises(UnsupportedProcessError, match="inputs selected torch"):
        run(
            process=_WrongPrecision(),
            initial_state=torch.tensor(100.0),
            time_grid=torch.tensor(GRID),
            rng=torch.Generator().manual_seed(1),
        )


def test_the_process_is_sampled_exactly_once(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}
    original = GBM.sample

    def counting(self: GBM, **kwargs: Any) -> Any:
        calls["n"] += 1
        return original(self, **kwargs)

    monkeypatch.setattr(GBM, "sample", counting)
    run()
    assert calls["n"] == 1


def test_validation_failures_never_reach_the_sampler(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("sampling must not start before validation passes")

    monkeypatch.setattr(GBM, "sample", forbidden)
    for call in (
        {"time_grid": [1.0, 2.0]},
        {"n_paths": 0},
        {"n_paths": 3, "antithetic": True},
        {"initial_state": None},
    ):
        with pytest.raises(SimulationValidationError):
            run(**call)


# --- purity -------------------------------------------------------------------


def test_simulate_attaches_nothing_to_its_inputs() -> None:
    equity = Asset(identifier="ACME", asset_class=AssetClass.EQUITY)
    before_asset = set(dir(equity))
    before_process = dict(PROCESS.params())
    simulate(equity, PROCESS, initial_state=100.0, time_grid=GRID, n_paths=4, rng=1)
    assert set(dir(equity)) == before_asset
    assert dict(PROCESS.params()) == before_process


def test_each_call_returns_a_new_scenario() -> None:
    first, second = run(rng=5), run(rng=5)
    assert first is not second
    np.testing.assert_array_equal(first.states, second.states)


def test_the_same_seed_reproduces_the_same_paths() -> None:
    np.testing.assert_array_equal(run(rng=42).states, run(rng=42).states)


def test_a_supplied_generator_advances_between_calls() -> None:
    generator = np.random.default_rng(3)
    first = run(rng=generator)
    second = run(rng=generator)
    assert not np.array_equal(first.states, second.states)


def test_output_has_no_formatting_option() -> None:
    import inspect

    assert "return_as" not in inspect.signature(simulate).parameters
    assert "return_native" not in inspect.signature(simulate).parameters


# --- backends -----------------------------------------------------------------


def test_all_python_input_defaults_to_numpy() -> None:
    assert isinstance(run(time_grid=[0.0, 1.0], initial_state=100.0).states, np.ndarray)


def test_torch_input_keeps_dtype_device_and_tape() -> None:
    torch = pytest.importorskip("torch", reason="torch not installed")
    spot = torch.tensor(100.0, dtype=torch.float64, requires_grad=True)
    scenario = run(
        initial_state=spot,
        time_grid=torch.tensor(GRID, dtype=torch.float64),
        rng=torch.Generator().manual_seed(2),
    )
    assert torch.is_tensor(scenario.states)
    assert scenario.states.dtype == torch.float64
    assert scenario.states.requires_grad
    scenario.terminal().sum().backward()
    assert spot.grad is not None and torch.isfinite(spot.grad).all()


def test_jax_input_stays_in_jax() -> None:
    jax = pytest.importorskip("jax", reason="jax not installed")
    jax.config.update("jax_enable_x64", True)
    jnp = jax.numpy
    scenario = run(
        initial_state=jnp.float64(100.0),
        time_grid=jnp.asarray(GRID, dtype=jnp.float64),
        rng=jax.random.key(0),
    )
    assert isinstance(scenario.states, jax.Array)
    assert scenario.states.dtype == jnp.float64


def test_a_mixed_namespace_is_refused_before_sampling() -> None:
    torch = pytest.importorskip("torch", reason="torch not installed")
    pytest.importorskip("jax", reason="jax not installed")
    import jax.numpy as jnp

    with pytest.raises(SimulationValidationError, match="more than one array namespace"):
        run(initial_state=torch.tensor(100.0), time_grid=jnp.asarray(GRID))


def test_a_jax_key_is_required_for_jax_input() -> None:
    jax = pytest.importorskip("jax", reason="jax not installed")
    jnp = jax.numpy
    with pytest.raises(SimulationValidationError, match="PRNG key"):
        run(time_grid=jnp.asarray(GRID), rng=0)


# --- the scenario a simulation produces evaluates a contract -------------------


def test_a_simulated_scenario_prices_a_matching_contract() -> None:
    scenario = run(n_paths=4096, rng=2026)
    option = EuropeanOption(underlier="ACME", option_type="call", strike=100.0, maturity=1.0)
    cashflow = scenario.payoff(option)
    assert cashflow.shape == (4096,)
    assert np.all(cashflow >= 0.0)


# --- precision ----------------------------------------------------------------


def test_a_single_precision_grid_and_a_double_spot_resolve_to_one_precision() -> None:
    """A NumPy scalar is neutral for backend *selection* and not for promotion.

    Under NEP 50 ``float32_array * np.float64(x)`` is float64, so treating the
    scalar as if it were not there would draw single-precision normals and then
    multiply them into a double-precision path: a buffer twice the size whose
    extra mantissa carries nothing.
    """
    scenario = run(
        initial_state=np.float64(100.0), time_grid=np.linspace(0.0, 1.0, 5, dtype=np.float32)
    )
    assert scenario.states.dtype == np.float64
    assert scenario.time_grid.dtype == np.float64


def test_a_wholly_single_precision_request_stays_single_precision() -> None:
    scenario = run(
        initial_state=np.float32(100.0),
        time_grid=np.linspace(0.0, 1.0, 5, dtype=np.float32),
        process=GBM(np.float32(0.05), np.float32(0.2)),
    )
    assert scenario.states.dtype == np.float32
    assert scenario.time_grid.dtype == np.float32


def test_a_numpy_scalar_does_not_promote_a_torch_simulation() -> None:
    """torch treats a NumPy scalar as a weak number, and so does this."""
    torch = pytest.importorskip("torch", reason="torch not installed")
    scenario = run(
        initial_state=torch.tensor(100.0, dtype=torch.float32),
        time_grid=torch.linspace(0.0, 1.0, 5, dtype=torch.float32),
        process=GBM(np.float64(0.05), 0.2),
        rng=torch.Generator().manual_seed(1),
    )
    assert scenario.states.dtype == torch.float32
    assert scenario.time_grid.dtype == torch.float32


def test_a_jax_simulation_promotes_the_way_jax_does() -> None:
    jax = pytest.importorskip("jax", reason="jax not installed")
    jax.config.update("jax_enable_x64", True)
    jnp = jax.numpy
    scenario = run(
        initial_state=np.float64(100.0),
        time_grid=jnp.asarray(np.linspace(0.0, 1.0, 5), dtype=jnp.float32),
        rng=jax.random.key(0),
    )
    assert scenario.states.dtype == jnp.float64
    assert scenario.time_grid.dtype == jnp.float64


@pytest.mark.parametrize("scalar", [np.float32(0.2), np.float64(0.2)], ids=["f32", "f64"])
def test_a_numpy_scalar_is_a_valid_process_parameter(scalar: Any) -> None:
    """Accepting np.float64 and refusing np.float32 would be an accident of
    inheritance -- the first subclasses ``float`` and the second does not."""
    assert run(process=GBM(0.05, scalar)).n_paths == 16


def test_a_numpy_bool_is_still_not_a_process_parameter() -> None:
    with pytest.raises(SimulationValidationError, match="not a bool"):
        GBM(np.bool_(True), 0.2)


# --- the rest of the process contract -----------------------------------------


class _DuplicateStates:
    state_names = ("spot", "spot")

    def params(self) -> dict[str, Any]:
        return {}

    def sample(self, **_kwargs: Any) -> Any:  # pragma: no cover - never reached
        raise AssertionError("must not be sampled")


class _NoSampleMethod:
    state_names = ("spot",)

    def params(self) -> dict[str, Any]:
        return {}


class _NoParams:
    state_names = ("spot",)

    def sample(self, *, time_grid: Any, n_paths: int, **_kwargs: Any) -> Any:
        return np.full((n_paths, len(time_grid), 1), 100.0)


def test_duplicate_state_names_are_refused() -> None:
    with pytest.raises(UnsupportedProcessError, match="duplicate state names"):
        run(process=_DuplicateStates())


def test_a_process_without_a_sample_method_is_refused() -> None:
    with pytest.raises(UnsupportedProcessError, match="has no sample"):
        run(process=_NoSampleMethod())


def test_a_process_without_params_can_still_be_simulated() -> None:
    """params() informs backend inference; a process that has none is not broken."""
    assert run(process=_NoParams()).n_paths == 16
