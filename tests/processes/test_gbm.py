"""Exact GBM: parameters, the transition, backends, and transform safety."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from fast_vollib.processes import GBM, StochasticProcess
from fast_vollib.simulation.errors import SimulationValidationError

GRID = np.array([0.0, 0.25, 0.75, 1.0])
IRREGULAR = np.array([0.0, 0.01, 0.3, 0.31, 1.0])


def paths(process: GBM = GBM(0.05, 0.2), **overrides: Any) -> np.ndarray:
    call = {
        "initial_state": {"spot": 100.0},
        "time_grid": GRID,
        "n_paths": 8,
        "rng": 12345,
    }
    call.update(overrides)
    return process.sample(**call)  # type: ignore[arg-type]


# --- the protocol -------------------------------------------------------------


def test_gbm_satisfies_the_process_protocol() -> None:
    assert isinstance(GBM(0.05, 0.2), StochasticProcess)


def test_state_names_is_a_single_spot() -> None:
    assert GBM(0.05, 0.2).state_names == ("spot",)


def test_state_names_is_not_a_constructor_argument() -> None:
    import dataclasses

    assert {f.name for f in dataclasses.fields(GBM)} == {"drift", "volatility"}


def test_params_returns_the_original_objects_read_only() -> None:
    torch = pytest.importorskip("torch", reason="torch not installed")
    volatility = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    process = GBM(0.05, volatility)
    assert process.params()["volatility"] is volatility
    with pytest.raises(TypeError):
        process.params()["drift"] = 1.0  # type: ignore[index]


def test_a_process_is_frozen_and_slotted() -> None:
    import dataclasses

    process = GBM(0.05, 0.2)
    with pytest.raises(dataclasses.FrozenInstanceError):
        process.drift = 0.1  # type: ignore[misc]
    with pytest.raises((AttributeError, TypeError)):
        process.paths = []  # type: ignore[attr-defined]


# --- parameter validation -----------------------------------------------------


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_drift_must_be_finite(bad: float) -> None:
    with pytest.raises(SimulationValidationError, match="drift must be finite"):
        GBM(bad, 0.2)


def test_volatility_must_be_finite_and_non_negative() -> None:
    with pytest.raises(SimulationValidationError, match="volatility must be finite"):
        GBM(0.05, float("nan"))
    with pytest.raises(SimulationValidationError, match="volatility must be non-negative"):
        GBM(0.05, -1e-12)


def test_zero_volatility_is_admissible() -> None:
    assert GBM(0.05, 0.0).volatility == 0.0


@pytest.mark.parametrize("bad", [True, 1 + 2j], ids=["bool", "complex"])
def test_bool_and_complex_parameters_are_refused(bad: Any) -> None:
    with pytest.raises(SimulationValidationError, match="real number"):
        GBM(bad, 0.2)


def test_a_non_scalar_parameter_is_refused() -> None:
    with pytest.raises(SimulationValidationError, match="must be a scalar"):
        GBM(np.array([0.05, 0.06]), 0.2)


def test_an_integer_tensor_parameter_is_refused() -> None:
    torch = pytest.importorskip("torch", reason="torch not installed")
    with pytest.raises(SimulationValidationError, match="floating-point"):
        GBM(torch.tensor(1, dtype=torch.int64), 0.2)


def test_a_scalar_float_array_parameter_is_accepted_and_stored_unchanged() -> None:
    drift = np.array(0.05)
    assert isinstance(drift, np.ndarray) and drift.ndim == 0
    process = GBM(drift, 0.2)
    assert process.drift is drift


def test_a_string_parameter_is_refused() -> None:
    with pytest.raises(SimulationValidationError, match="real number or a scalar array"):
        GBM("0.05", 0.2)


def test_risk_neutral_drift_is_rate_minus_dividend_yield() -> None:
    assert GBM.risk_neutral(0.04, 0.2).drift == 0.04
    assert GBM.risk_neutral(0.04, 0.2, dividend_yield=0.015).drift == pytest.approx(0.025)


def test_risk_neutral_validates_its_own_arguments() -> None:
    with pytest.raises(SimulationValidationError, match="rate must be finite"):
        GBM.risk_neutral(float("inf"), 0.2)
    with pytest.raises(SimulationValidationError, match="dividend_yield"):
        GBM.risk_neutral(0.04, 0.2, dividend_yield=float("nan"))


# --- the transition -----------------------------------------------------------


def test_output_shape_is_paths_times_state() -> None:
    assert paths(n_paths=7).shape == (7, len(GRID), 1)


def test_the_first_state_is_the_initial_state_exactly() -> None:
    sampled = paths(initial_state={"spot": 123.456})
    assert np.all(sampled[:, 0, 0] == 123.456)


def test_zero_volatility_gives_the_deterministic_path() -> None:
    sampled = paths(GBM(0.07, 0.0), time_grid=IRREGULAR, n_paths=3)
    expected = 100.0 * np.exp(0.07 * IRREGULAR)
    np.testing.assert_allclose(sampled[:, :, 0], np.broadcast_to(expected, (3, len(IRREGULAR))))


def test_paths_are_strictly_positive() -> None:
    assert np.all(paths(n_paths=64) > 0.0)


def test_a_spot_must_be_supplied_and_positive() -> None:
    with pytest.raises(SimulationValidationError, match="must supply"):
        paths(initial_state={"forward": 100.0})
    with pytest.raises(SimulationValidationError, match="strictly positive"):
        paths(initial_state={"spot": 0.0})


def test_sampling_is_reproducible_from_a_seed() -> None:
    np.testing.assert_array_equal(paths(rng=99), paths(rng=99))


def test_different_seeds_give_different_paths() -> None:
    assert not np.array_equal(paths(rng=1), paths(rng=2))


def test_sampling_adds_no_state_to_the_process() -> None:
    """A process is stateless across calls: sampling twice cannot change it."""
    process = GBM(0.05, 0.2)
    before = dict(process.params())
    paths(process, rng=3)
    assert dict(process.params()) == before
    np.testing.assert_array_equal(paths(process, rng=3), paths(process, rng=3))


# --- Tier 2: the exact log-return moments -------------------------------------


@pytest.mark.parametrize("grid", [GRID, IRREGULAR], ids=["regular", "irregular"])
def test_log_return_mean_and_variance_match_the_exact_transition(grid: np.ndarray) -> None:
    """Each step's log return is N((mu - sigma^2/2) dt, sigma^2 dt), exactly.

    Checked as a statistical band: five standard errors of the sample mean,
    which a correct sampler clears with overwhelming probability and a biased
    Euler step of the same grid would not.
    """
    drift, volatility, n_paths = 0.07, 0.25, 40_000
    sampled = paths(GBM(drift, volatility), time_grid=grid, n_paths=n_paths, rng=20260825)[:, :, 0]
    log_returns = np.log(sampled[:, 1:] / sampled[:, :-1])
    steps = np.diff(grid)

    expected_mean = (drift - 0.5 * volatility**2) * steps
    expected_variance = volatility**2 * steps
    mean_stderr = np.sqrt(expected_variance / n_paths)
    np.testing.assert_array_less(
        np.abs(log_returns.mean(axis=0) - expected_mean), 5.0 * mean_stderr
    )

    # var(s^2) = 2 sigma^4 dt^2 / (n - 1) for a Gaussian sample.
    variance_stderr = expected_variance * np.sqrt(2.0 / (n_paths - 1))
    np.testing.assert_array_less(
        np.abs(log_returns.var(axis=0, ddof=1) - expected_variance), 5.0 * variance_stderr
    )


def test_increments_are_independent_across_steps() -> None:
    """Correlation between consecutive log returns is zero to sampling error."""
    n_paths = 40_000
    sampled = paths(GBM(0.0, 0.3), time_grid=GRID, n_paths=n_paths, rng=7)[:, :, 0]
    log_returns = np.log(sampled[:, 1:] / sampled[:, :-1])
    normalized = (log_returns - log_returns.mean(axis=0)) / log_returns.std(axis=0, ddof=1)
    correlation = float(np.mean(normalized[:, 0] * normalized[:, 1]))
    assert abs(correlation) < 5.0 / np.sqrt(n_paths)


def test_the_terminal_expectation_matches_the_lognormal_mean() -> None:
    drift, n_paths = 0.05, 100_000
    sampled = paths(GBM(drift, 0.2), n_paths=n_paths, rng=4242)[:, -1, 0]
    expected = 100.0 * np.exp(drift * GRID[-1])
    stderr = sampled.std(ddof=1) / np.sqrt(n_paths)
    assert abs(sampled.mean() - expected) <= 5.0 * stderr


# --- antithetic sampling ------------------------------------------------------


def test_antithetic_paths_are_exactly_mirrored_in_log_space() -> None:
    sampled = paths(n_paths=10, antithetic=True)[:, :, 0]
    first, second = sampled[:5], sampled[5:]
    drift, volatility = 0.05, 0.2
    steps = np.diff(GRID)
    deterministic = (drift - 0.5 * volatility**2) * steps
    left = np.log(first[:, 1:] / first[:, :-1]) - deterministic
    right = np.log(second[:, 1:] / second[:, :-1]) - deterministic
    np.testing.assert_allclose(left, -right, rtol=1e-12, atol=1e-12)


def test_antithetic_requires_an_even_path_count() -> None:
    with pytest.raises(SimulationValidationError, match="even number of paths"):
        paths(n_paths=5, antithetic=True)


# --- backends -----------------------------------------------------------------


def test_torch_paths_match_numpy_shapes_dtypes_and_devices() -> None:
    torch = pytest.importorskip("torch", reason="torch not installed")
    generator = torch.Generator()
    generator.manual_seed(5)
    sampled = GBM(torch.tensor(0.05, dtype=torch.float64), 0.2).sample(
        initial_state={"spot": torch.tensor(100.0, dtype=torch.float64)},
        time_grid=torch.tensor(GRID, dtype=torch.float64),
        n_paths=6,
        rng=generator,
    )
    assert torch.is_tensor(sampled)
    assert sampled.shape == (6, len(GRID), 1)
    assert sampled.dtype == torch.float64
    torch.testing.assert_close(
        sampled[:, 0, 0], torch.full((6,), 100.0, dtype=torch.float64), rtol=0, atol=0
    )


def test_torch_float32_stays_single_precision() -> None:
    torch = pytest.importorskip("torch", reason="torch not installed")
    sampled = GBM(0.05, 0.2).sample(
        initial_state={"spot": torch.tensor(100.0, dtype=torch.float32)},
        time_grid=torch.tensor(GRID, dtype=torch.float32),
        n_paths=4,
        rng=1,
    )
    assert sampled.dtype == torch.float32


def test_torch_gradients_reach_spot_drift_and_volatility() -> None:
    torch = pytest.importorskip("torch", reason="torch not installed")
    spot = torch.tensor(100.0, dtype=torch.float64, requires_grad=True)
    drift = torch.tensor(0.05, dtype=torch.float64, requires_grad=True)
    volatility = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    sampled = GBM(drift, volatility).sample(
        initial_state={"spot": spot},
        time_grid=torch.tensor(GRID, dtype=torch.float64),
        n_paths=8,
        rng=3,
    )
    assert sampled.requires_grad
    sampled.sum().backward()
    for name, value in (("spot", spot), ("drift", drift), ("volatility", volatility)):
        assert value.grad is not None, name
        assert torch.isfinite(value.grad).all(), name


def test_torch_cuda_paths_stay_on_the_device() -> None:
    torch = pytest.importorskip("torch", reason="torch not installed")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    generator = torch.Generator(device="cuda")
    generator.manual_seed(11)
    sampled = GBM(0.05, 0.2).sample(
        initial_state={"spot": torch.tensor(100.0, dtype=torch.float64, device="cuda")},
        time_grid=torch.tensor(GRID, dtype=torch.float64, device="cuda"),
        n_paths=4,
        rng=generator,
    )
    assert sampled.device.type == "cuda"


def test_jax_paths_and_gradients() -> None:
    jax = pytest.importorskip("jax", reason="jax not installed")
    jax.config.update("jax_enable_x64", True)
    jnp = jax.numpy

    def terminal_mean(spot: Any, drift: Any, volatility: Any) -> Any:
        sampled = GBM(drift, volatility).sample(
            initial_state={"spot": spot},
            time_grid=jnp.asarray(GRID, dtype=jnp.float64),
            n_paths=8,
            rng=jax.random.key(0),
        )
        return sampled[:, -1, 0].mean()

    arguments = (
        jnp.float64(100.0),
        jnp.float64(0.05),
        jnp.float64(0.2),
    )
    value = terminal_mean(*arguments)
    assert jnp.isfinite(value)
    for argnum in range(3):
        gradient = jax.grad(terminal_mean, argnums=argnum)(*arguments)
        assert jnp.isfinite(gradient), argnum


def test_jax_jit_traces_sampling_without_truth_testing_a_tracer() -> None:
    """Validation must not concretize a traced parameter."""
    jax = pytest.importorskip("jax", reason="jax not installed")
    jax.config.update("jax_enable_x64", True)
    jnp = jax.numpy

    @jax.jit
    def run(spot: Any, drift: Any, volatility: Any) -> Any:
        return GBM(drift, volatility).sample(
            initial_state={"spot": spot},
            time_grid=jnp.asarray(GRID, dtype=jnp.float64),
            n_paths=4,
            rng=jax.random.key(1),
        )

    sampled = run(jnp.float64(100.0), jnp.float64(0.05), jnp.float64(0.2))
    assert sampled.shape == (4, len(GRID), 1)
    np.testing.assert_allclose(np.asarray(sampled[:, 0, 0]), 100.0)


def test_mixed_namespaces_are_refused_before_sampling() -> None:
    torch = pytest.importorskip("torch", reason="torch not installed")
    jax = pytest.importorskip("jax", reason="jax not installed")
    with pytest.raises(SimulationValidationError, match="more than one array namespace"):
        GBM(0.05, 0.2).sample(
            initial_state={"spot": torch.tensor(100.0, dtype=torch.float64)},
            time_grid=jax.numpy.asarray(GRID),
            n_paths=4,
            rng=0,
        )
