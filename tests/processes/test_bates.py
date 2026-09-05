"""``Bates``: the lattice, the draw-order contract, and the reductions.

The reductions are the point of this file.  A configurable model is only worth
having if switching a feature off gives back *exactly* the model without it,
and "exactly" here means bitwise wherever the arithmetic allows it -- because a
reduction that agreed only statistically could not distinguish a correct
sampler from one that had reordered its draws.

Three of the four corners are checked against the library's own existing
implementation of the same model rather than against a second copy written
here, which is what makes the check independent.
"""

from __future__ import annotations

import numpy as np
import pytest

from fast_vollib._random_api import random_stream, split, standard_normal
from fast_vollib._simulation_errors import SimulationValidationError
from fast_vollib.processes import (
    SCHEMES,
    Bates,
    ConstantVariance,
    Heston,
    HestonVariance,
    LognormalJumps,
    NoJumps,
    StochasticProcess,
)

HESTON_PARAMS = {"kappa": 2.0, "theta": 0.04, "vol_of_vol": 0.3, "rho": -0.7}
JUMP_PARAMS = {"jump_intensity": 1.5, "mean_log_jump": -0.05, "jump_volatility": 0.2}
DRIFT = 0.03
INITIAL = {"spot": 100.0, "variance": 0.04}
GRID = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
SEED = 20260904


def variance():
    return HestonVariance(**HESTON_PARAMS)


def jumps(**overrides):
    return LognormalJumps(**{**JUMP_PARAMS, **overrides})


def sample(process, *, n_paths=8, **kwargs):
    return process.sample(
        initial_state=INITIAL, time_grid=GRID, n_paths=n_paths, rng=SEED, **kwargs
    )


# --- the contract --------------------------------------------------------------


def test_it_satisfies_the_structural_protocol() -> None:
    assert isinstance(Bates(variance=variance(), jumps=NoJumps()), StochasticProcess)


@pytest.mark.parametrize("component", [variance(), ConstantVariance()], ids=["heston", "constant"])
def test_the_state_names_do_not_depend_on_the_configuration(component) -> None:
    """So a scenario's shape is the same across every reduction."""
    process = Bates(variance=component, jumps=NoJumps())
    assert process.state_names == ("spot", "variance")
    assert sample(process).shape == (8, len(GRID), 2)


def test_a_constant_variance_column_holds_its_initial_value_at_every_time() -> None:
    paths = sample(Bates(variance=ConstantVariance(), jumps=NoJumps(), drift=DRIFT))
    assert np.all(paths[:, :, 1] == INITIAL["variance"])


def test_column_zero_is_the_initial_spot_exactly() -> None:
    for component in (variance(), ConstantVariance()):
        paths = sample(Bates(variance=component, jumps=jumps(), drift=DRIFT))
        assert np.all(paths[:, 0, 0] == INITIAL["spot"])


def test_params_is_flat_dotted_and_holds_the_original_objects() -> None:
    """The engine validates each value as a scalar; a component object there is
    a hard error, and a bare ``kappa`` would not say which component it came
    from once two components have one."""
    kappa = np.float64(2.0)
    process = Bates(
        variance=HestonVariance(**{**HESTON_PARAMS, "kappa": kappa}),
        jumps=jumps(),
        drift=DRIFT,
    )
    flat = process.params()
    assert set(flat) == {
        "variance.kappa",
        "variance.theta",
        "variance.vol_of_vol",
        "variance.rho",
        "jumps.jump_intensity",
        "jumps.mean_log_jump",
        "jumps.jump_volatility",
        "drift",
    }
    assert flat["variance.kappa"] is kappa
    assert all(np.isscalar(v) or np.ndim(v) == 0 for v in flat.values()), flat
    with pytest.raises(TypeError):
        flat["drift"] = 1.0  # type: ignore[index]


def test_a_reduced_configuration_reports_only_the_parameters_it_has() -> None:
    """A constant marker stores no level, so there is nothing to report."""
    assert set(Bates(variance=ConstantVariance(), jumps=NoJumps()).params()) == {"drift"}


def test_risk_neutral_records_the_drift_without_compensating_twice() -> None:
    """The sampler subtracts the compensator; doing it here as well is the
    mistake the arrangement exists to prevent."""
    process = Bates.risk_neutral(rate=0.05, dividend_yield=0.02, variance=variance(), jumps=jumps())
    assert process.drift == pytest.approx(0.03)


def test_a_process_is_a_value() -> None:
    one = Bates(variance=variance(), jumps=jumps(), drift=DRIFT)
    assert one == Bates(variance=variance(), jumps=jumps(), drift=DRIFT)
    with pytest.raises(Exception):
        one.drift = 0.1  # type: ignore[misc]


# --- the closed unions ---------------------------------------------------------


def test_a_component_outside_its_union_is_refused_at_construction() -> None:
    """Before arithmetic and before randomness, naming what is admissible."""

    class LooksLikeVariance:
        kappa = theta = vol_of_vol = rho = 0.1

        def params(self):
            return {}

    with pytest.raises(SimulationValidationError, match="HestonVariance, ConstantVariance"):
        Bates(variance=LooksLikeVariance(), jumps=NoJumps())  # type: ignore[arg-type]


def test_a_heston_process_is_not_accepted_as_a_variance_component() -> None:
    """It carries a drift, and a facade that ignored it would silently price a
    different model from the one the caller described."""
    with pytest.raises(SimulationValidationError, match="Heston"):
        Bates(variance=Heston(**HESTON_PARAMS, drift=0.9), jumps=NoJumps())  # type: ignore[arg-type]


def test_a_jump_component_outside_its_union_is_refused() -> None:
    with pytest.raises(SimulationValidationError, match="LognormalJumps, NoJumps"):
        Bates(variance=variance(), jumps=object())  # type: ignore[arg-type]


def test_an_unknown_scheme_is_refused_even_for_a_constant_variance() -> None:
    """Which has no scheme to choose, so a typo would otherwise pass silently."""
    with pytest.raises(SimulationValidationError, match="scheme must be one of"):
        sample(Bates(variance=ConstantVariance(), jumps=NoJumps()), scheme="milstein")


@pytest.mark.parametrize("missing", ["spot", "variance"])
def test_a_missing_initial_state_is_refused_naming_it(missing) -> None:
    """``variance`` is required even for ``ConstantVariance``: that is where its
    level comes from, because the marker deliberately does not hold one."""
    state = {k: v for k, v in INITIAL.items() if k != missing}
    with pytest.raises(SimulationValidationError, match=missing):
        Bates(variance=ConstantVariance(), jumps=NoJumps()).sample(
            initial_state=state, time_grid=GRID, n_paths=4, rng=SEED
        )


# --- reduction: Heston ---------------------------------------------------------


@pytest.mark.parametrize("scheme", SCHEMES)
@pytest.mark.parametrize("antithetic", [False, True])
def test_no_jumps_reduces_to_heston_bitwise(scheme, antithetic) -> None:
    """Against the library's own ``Heston``, not a copy written here.

    Bitwise, which is the strongest available statement and the one that
    distinguishes a shared transition from a re-derived one.
    """
    reduced = sample(
        Bates(variance=variance(), jumps=NoJumps(), drift=DRIFT),
        scheme=scheme,
        antithetic=antithetic,
    )
    reference = sample(Heston(**HESTON_PARAMS, drift=DRIFT), scheme=scheme, antithetic=antithetic)
    np.testing.assert_array_equal(reduced, reference)


def test_the_heston_reduction_holds_at_a_feller_violating_parameter_set() -> None:
    """Where the two schemes diverge, so the shared step is exercised there too."""
    rough = {"kappa": 0.5, "theta": 0.06, "vol_of_vol": 0.9, "rho": -0.9}
    np.testing.assert_array_equal(
        sample(Bates(variance=HestonVariance(**rough), jumps=NoJumps(), drift=DRIFT)),
        sample(Heston(**rough, drift=DRIFT)),
    )


# --- reduction: zero intensity -------------------------------------------------


@pytest.mark.parametrize("antithetic", [False, True])
def test_zero_jump_intensity_reduces_to_no_jumps_bitwise(antithetic) -> None:
    """``K`` is exactly zero, and ``x + 0.0`` is ``x`` bit for bit.

    Not reasoned about -- measured. The aggregate log jump at ``K = 0`` is
    ``0 * m + delta * sqrt(0) * Z``, which is a signed zero, and adding it to a
    log return has to leave every bit alone for this to pass.
    """
    zero = sample(
        Bates(variance=variance(), jumps=jumps(jump_intensity=0.0), drift=DRIFT),
        antithetic=antithetic,
    )
    none = sample(Bates(variance=variance(), jumps=NoJumps(), drift=DRIFT), antithetic=antithetic)
    np.testing.assert_array_equal(zero, none)


def test_zero_intensity_reduces_for_a_constant_variance_too() -> None:
    zero = sample(Bates(variance=ConstantVariance(), jumps=jumps(jump_intensity=0.0), drift=DRIFT))
    none = sample(Bates(variance=ConstantVariance(), jumps=NoJumps(), drift=DRIFT))
    np.testing.assert_array_equal(zero, none)


def test_the_aggregate_log_jump_is_exactly_zero_at_zero_counts() -> None:
    """The arithmetic the reduction above rests on, checked on its own."""
    process = Bates(variance=variance(), jumps=jumps(), drift=DRIFT)
    xp_zero = np.zeros(4)
    from fast_vollib._array_api import get_namespace

    contribution = process._jump(get_namespace(xp_zero), xp_zero, np.array([1.0, -2.0, 3.0, 0.5]))
    assert np.all(contribution == 0.0)
    assert np.all(
        np.asarray([1.5, -0.2, 0.0, 7.0]) + contribution == np.asarray([1.5, -0.2, 0.0, 7.0])
    )


# --- reduction: log-normal spot ------------------------------------------------


def test_a_constant_variance_spot_is_the_exact_log_normal_transition() -> None:
    """Not an Euler approximation: with ``v`` fixed the transition is closed form.

    Reconstructed from column 1 of the diffusion block, which is simultaneously
    the check that a constant-variance configuration uses that column and
    leaves column 0 alone.
    """
    paths = sample(Bates(variance=ConstantVariance(), jumps=NoJumps(), drift=DRIFT), n_paths=512)
    stream = split(random_stream(SEED, namespace="numpy", dtype=np.float64), 2)[0]
    normals = standard_normal(stream, (512, len(GRID) - 1, 2))[:, :, 1]

    variance_level = INITIAL["variance"]
    steps = np.diff(GRID)
    increments = (DRIFT - 0.5 * variance_level) * steps + np.sqrt(variance_level * steps) * normals
    expected = INITIAL["spot"] * np.exp(
        np.concatenate([np.zeros((512, 1)), np.cumsum(increments, axis=1)], axis=1)
    )
    # One ulp: the sampler accumulates step by step and this sums at the end.
    np.testing.assert_allclose(paths[:, :, 0], expected, rtol=4e-16)


def test_the_constant_variance_spot_is_lognormal_with_the_right_moments() -> None:
    """``E[S_T] = S_0 e^{drift T}`` exactly under the risk-neutral drift."""
    paths = sample(
        Bates(variance=ConstantVariance(), jumps=NoJumps(), drift=DRIFT), n_paths=200_000
    )
    terminal = paths[:, -1, 0]
    expected = INITIAL["spot"] * np.exp(DRIFT * GRID[-1])
    stderr = terminal.std(ddof=1) / np.sqrt(len(terminal))
    assert abs(terminal.mean() - expected) < 3.0 * stderr


# --- the draw-order contract ---------------------------------------------------


def test_the_diffusion_block_comes_from_the_first_split_slot() -> None:
    """Block 1 is the identical call ``Heston`` makes, from slot 0.

    Verified through the sampler rather than asserted: the constant-variance
    configuration's spot is reconstructible from slot 0's column 1, and the
    Heston configuration's paths equal ``Heston``'s, which draws the same block.
    """
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    key = jax.random.key(SEED)
    produced = Bates(variance=variance(), jumps=NoJumps(), drift=DRIFT).sample(
        initial_state=INITIAL, time_grid=jnp.asarray(GRID), n_paths=8, rng=key
    )
    reference = Heston(**HESTON_PARAMS, drift=DRIFT).sample(
        initial_state=INITIAL,
        time_grid=jnp.asarray(GRID),
        n_paths=8,
        rng=split(random_stream(key, namespace="jax", dtype=produced.dtype), 3)[0].handle,
    )
    np.testing.assert_array_equal(np.asarray(produced), np.asarray(reference))


@pytest.mark.parametrize("partitionable", [False, True])
def test_bates_reserves_the_same_slots_as_bcc97(partitionable) -> None:
    """Reductions hold for both legacy and partitionable Threefry layouts."""
    jax = pytest.importorskip("jax")
    from fast_vollib.processes import BCC97, ConstantShortRate

    previous = jax.config.jax_threefry_partitionable
    try:
        jax.config.update("jax_threefry_partitionable", partitionable)
        kwargs = dict(time_grid=jax.numpy.asarray(GRID), n_paths=8)
        bates = Bates(variance=variance(), jumps=NoJumps(), drift=DRIFT).sample(
            initial_state=INITIAL,
            rng=jax.random.key(SEED),
            **kwargs,
        )
        bcc = BCC97(
            variance=variance(),
            jumps=NoJumps(),
            rates=ConstantShortRate(),
        ).sample(
            initial_state={**INITIAL, "short_rate": DRIFT},
            rng=jax.random.key(SEED),
            **kwargs,
        )
        np.testing.assert_array_equal(np.asarray(bates), np.asarray(bcc)[:, :, :2])
    finally:
        jax.config.update("jax_threefry_partitionable", previous)


def test_the_jump_block_uses_one_intensity_per_step_of_the_grid() -> None:
    """A non-uniform grid is handled by construction, not by assuming ``dt``."""
    uneven = np.array([0.0, 0.01, 1.0])
    paths = Bates(
        variance=ConstantVariance(), jumps=jumps(jump_intensity=50.0), drift=DRIFT
    ).sample(initial_state=INITIAL, time_grid=uneven, n_paths=20_000, rng=SEED)
    # The long step should carry far more jump activity than the short one.
    first_move = np.log(paths[:, 1, 0] / paths[:, 0, 0])
    second_move = np.log(paths[:, 2, 0] / paths[:, 1, 0])
    assert second_move.var() > 20.0 * first_move.var()


def test_the_compound_poisson_admits_more_than_one_jump_per_step() -> None:
    """Not an at-most-one Bernoulli approximation, which would misprice a short
    maturity at a high intensity."""
    process = Bates(variance=ConstantVariance(), jumps=jumps(jump_intensity=40.0), drift=0.0)
    single_step = process.sample(
        initial_state=INITIAL, time_grid=np.array([0.0, 1.0]), n_paths=50_000, rng=SEED
    )
    moves = np.log(single_step[:, 1, 0] / INITIAL["spot"])
    # With lambda = 40 over a year, a Bernoulli approximation caps the jump
    # contribution near one jump; the compound Poisson reaches many.
    jump_only = moves - (0.0 - 0.5 * INITIAL["variance"]) * 1.0
    assert jump_only.min() < 10.0 * JUMP_PARAMS["mean_log_jump"], jump_only.min()


# --- jump statistics -----------------------------------------------------------


def test_the_compensated_drift_makes_the_discounted_spot_a_martingale() -> None:
    """The reason the compensator is subtracted, stated as the property it buys.

    ``E[S_T] = S_0 e^{drift T}`` must hold *with* jumps as well as without,
    which is exactly what ``drift - lambda * mean_relative_jump`` arranges.
    """
    process = Bates(variance=variance(), jumps=jumps(), drift=DRIFT)
    paths = process.sample(
        initial_state=INITIAL, time_grid=np.array([0.0, 1.0]), n_paths=400_000, rng=SEED
    )
    terminal = paths[:, -1, 0]
    expected = INITIAL["spot"] * np.exp(DRIFT * 1.0)
    stderr = terminal.std(ddof=1) / np.sqrt(len(terminal))
    assert abs(terminal.mean() - expected) < 4.0 * stderr, (terminal.mean(), expected, stderr)


def test_jumps_add_variance_and_a_heavier_left_tail() -> None:
    """A property no reduction test would catch: the jumps have to actually fire.

    Variance and the *left tail*, deliberately, rather than standardized
    skewness -- which moves the wrong way here and is worth knowing why.
    Heston at ``rho = -0.7`` is already skewed to about -0.8; adding jumps at
    ``m = -0.05`` with ``delta = 0.2`` more than doubles the variance, and most
    of that addition is symmetric, so the *standardized* third moment rises to
    about -0.47 while the distribution has unambiguously gained downside. The
    quantile says what happened; the skewness would have said the opposite.
    """
    common = {
        "initial_state": INITIAL,
        "time_grid": np.array([0.0, 1.0]),
        "n_paths": 100_000,
        "rng": SEED,
    }
    with_jumps = Bates(variance=variance(), jumps=jumps(), drift=DRIFT).sample(**common)
    without = Bates(variance=variance(), jumps=NoJumps(), drift=DRIFT).sample(**common)
    returns_with = np.log(with_jumps[:, -1, 0] / INITIAL["spot"])
    returns_without = np.log(without[:, -1, 0] / INITIAL["spot"])

    assert returns_with.var() > 1.5 * returns_without.var()
    assert np.percentile(returns_with, 1) < np.percentile(returns_without, 1)


def test_a_crash_jump_deepens_the_left_tail_without_moving_the_mean_spot() -> None:
    """The compensator absorbs the drift effect; the shape change is what is left."""
    crash = LognormalJumps(jump_intensity=0.5, mean_log_jump=-0.30, jump_volatility=0.05)
    common = {
        "initial_state": INITIAL,
        "time_grid": np.array([0.0, 1.0]),
        "n_paths": 200_000,
        "rng": SEED,
    }
    with_crash = Bates(variance=variance(), jumps=crash, drift=DRIFT).sample(**common)
    without = Bates(variance=variance(), jumps=NoJumps(), drift=DRIFT).sample(**common)

    assert np.percentile(with_crash[:, -1, 0], 1) < np.percentile(without[:, -1, 0], 1)
    expected = INITIAL["spot"] * np.exp(DRIFT)
    terminal = with_crash[:, -1, 0]
    stderr = terminal.std(ddof=1) / np.sqrt(len(terminal))
    assert abs(terminal.mean() - expected) < 4.0 * stderr


# --- backends ------------------------------------------------------------------


@pytest.mark.parametrize("scheme", SCHEMES)
def test_it_samples_on_torch(scheme) -> None:
    torch = pytest.importorskip("torch")
    paths = Bates(variance=variance(), jumps=jumps(), drift=DRIFT).sample(
        initial_state={
            "spot": torch.tensor(100.0, dtype=torch.float64),
            "variance": torch.tensor(0.04, dtype=torch.float64),
        },
        time_grid=torch.as_tensor(GRID),
        n_paths=64,
        rng=SEED,
        scheme=scheme,
    )
    assert paths.shape == (64, len(GRID), 2)
    assert paths.dtype == torch.float64


def test_it_samples_on_jax_and_the_reduction_still_holds() -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    key = jax.random.key(SEED)
    common = {"initial_state": INITIAL, "time_grid": jnp.asarray(GRID), "n_paths": 32, "rng": key}
    zero = Bates(variance=variance(), jumps=jumps(jump_intensity=0.0), drift=DRIFT).sample(**common)
    none = Bates(variance=variance(), jumps=NoJumps(), drift=DRIFT).sample(**common)
    np.testing.assert_array_equal(np.asarray(zero), np.asarray(none))


# --- through simulate() --------------------------------------------------------
#
# Every test above calls ``Bates.sample`` directly.  ``simulate()`` is the layer
# the engine goes through, and it is the layer that has never been handed a
# process with more than one state: it validates the initial-state mapping
# against ``state_names`` and hands the result to ``Scenario``, which labels the
# trailing axis.  Checking it here separates a multi-state ``Scenario`` problem
# from an engine problem later.


def test_simulate_labels_both_states_and_agrees_with_sample() -> None:
    from fast_vollib.simulation import Scenario, simulate

    process = Bates(variance=variance(), jumps=jumps(), drift=DRIFT)
    scenario = simulate(
        "ACME",
        process,
        initial_state=INITIAL,
        time_grid=GRID,
        n_paths=64,
        rng=SEED,
    )
    assert isinstance(scenario, Scenario)
    assert scenario.state_names == ("spot", "variance")

    direct = process.sample(initial_state=INITIAL, time_grid=GRID, n_paths=64, rng=SEED)
    np.testing.assert_array_equal(np.asarray(scenario.states), direct)
    np.testing.assert_array_equal(np.asarray(scenario.terminal("spot")), direct[:, -1, 0])
    np.testing.assert_array_equal(np.asarray(scenario.terminal("variance")), direct[:, -1, 1])


def test_simulate_requires_both_states_by_name() -> None:
    from fast_vollib.simulation import simulate

    process = Bates(variance=variance(), jumps=jumps(), drift=DRIFT)
    with pytest.raises(SimulationValidationError):
        simulate(
            "ACME",
            process,
            initial_state={"spot": 100.0},
            time_grid=GRID,
            n_paths=8,
            rng=SEED,
        )
