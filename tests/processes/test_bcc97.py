r"""``BCC97``: the third factor, the third slot, and the reductions it must keep.

Everything :class:`Bates` promised has to survive the short rate becoming a
state, and one new thing has to be true: the rate the spot's drift uses must be
*the same number* the discount factor integrates.  In continuous time that is
automatic; in discrete time it is a choice, and getting it wrong produces a
model that is arbitrage-free on paper and not in the simulation.

``test_the_rate_cancels_pathwise_between_drift_and_discounting`` is the sharp
form of that check and the most important test in this file.  It is not a Monte
Carlo comparison inside an error budget: with the same seed, a CIR-rate run and
a zero-rate run share their spot, variance and jump draws exactly, so
``discount_factor * S_T`` from the first must equal ``S_T`` from the second
*path by path*, to floating-point.  A drift that integrated the rate any other
way than the discounting does would fail it by orders of magnitude rather than
by a standard error.
"""

from __future__ import annotations

import numpy as np
import pytest

from fast_vollib._random_api import random_stream, split
from fast_vollib._simulation_errors import SimulationValidationError, UnsupportedProcessError
from fast_vollib.processes import (
    BCC97,
    CIR_SCHEMES,
    SCHEMES,
    Bates,
    CIRShortRate,
    ConstantShortRate,
    ConstantVariance,
    Heston,
    HestonVariance,
    LognormalJumps,
    NoJumps,
    StochasticProcess,
)
from fast_vollib.simulation.discounting import PathwiseShortRateDiscounting

HESTON_PARAMS = {"kappa": 2.0, "theta": 0.04, "vol_of_vol": 0.3, "rho": -0.7}
JUMP_PARAMS = {"jump_intensity": 1.5, "mean_log_jump": -0.05, "jump_volatility": 0.2}
RATE_PARAMS = {"kappa": 0.5, "theta": 0.05, "volatility": 0.15}
RATE = 0.03
DIVIDEND_YIELD = 0.01
INITIAL = {"spot": 100.0, "variance": 0.04, "short_rate": RATE}
GRID = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
SEED = 20260904


def variance():
    return HestonVariance(**HESTON_PARAMS)


def jumps(**overrides):
    return LognormalJumps(**{**JUMP_PARAMS, **overrides})


def rates():
    return CIRShortRate(**RATE_PARAMS)


def process(**overrides):
    fields = {
        "variance": variance(),
        "jumps": jumps(),
        "rates": rates(),
        "dividend_yield": DIVIDEND_YIELD,
    }
    fields.update(overrides)
    return BCC97(**fields)


#: Every corner of the lattice that has a constant rate, with the ``Bates``
#: configuration it must reduce to.
CONSTANT_RATE_CORNERS = [
    (HestonVariance(**HESTON_PARAMS), LognormalJumps(**JUMP_PARAMS)),
    (HestonVariance(**HESTON_PARAMS), NoJumps()),
    (ConstantVariance(), LognormalJumps(**JUMP_PARAMS)),
    (ConstantVariance(), NoJumps()),
]
CORNER_IDS = ["heston+jumps", "heston", "constant+jumps", "constant"]


# --- the shape of the thing -----------------------------------------------------


def test_it_satisfies_the_process_protocol() -> None:
    assert isinstance(process(), StochasticProcess)


def test_the_state_order_is_fixed_for_every_configuration() -> None:
    """A reduction must not change a scenario's shape or its column meanings."""
    for var, jm in CONSTANT_RATE_CORNERS:
        for rate_model in (rates(), ConstantShortRate()):
            assert BCC97(variance=var, jumps=jm, rates=rate_model).state_names == (
                "spot",
                "variance",
                "short_rate",
            )


def test_params_are_flat_scalars_under_dotted_keys() -> None:
    flat = process().params()
    assert sorted(flat) == [
        "dividend_yield",
        "jumps.jump_intensity",
        "jumps.jump_volatility",
        "jumps.mean_log_jump",
        "rates.kappa",
        "rates.theta",
        "rates.volatility",
        "variance.kappa",
        "variance.rho",
        "variance.theta",
        "variance.vol_of_vol",
    ]
    assert flat["rates.kappa"] == RATE_PARAMS["kappa"]
    assert flat["variance.kappa"] == HESTON_PARAMS["kappa"]
    with pytest.raises(TypeError):
        flat["variance.kappa"] = 1.0  # type: ignore[index]


def test_a_constant_configuration_still_carries_its_states() -> None:
    """Nothing disappears from ``params`` or the columns when a model reduces."""
    reduced = BCC97(variance=ConstantVariance(), jumps=NoJumps(), rates=ConstantShortRate())
    assert sorted(reduced.params()) == ["dividend_yield"]
    paths = reduced.sample(initial_state=INITIAL, time_grid=GRID, n_paths=8, rng=SEED)
    assert paths.shape == (8, len(GRID), 3)
    assert np.all(paths[:, :, 1] == INITIAL["variance"])
    assert np.all(paths[:, :, 2] == RATE)


def test_column_zero_is_the_initial_state_exactly() -> None:
    paths = process().sample(initial_state=INITIAL, time_grid=GRID, n_paths=16, rng=SEED)
    assert np.all(paths[:, 0, 0] == INITIAL["spot"])
    assert np.all(paths[:, 0, 1] == INITIAL["variance"])
    assert np.all(paths[:, 0, 2] == INITIAL["short_rate"])


def test_a_one_point_grid_is_the_initial_state_and_nothing_else() -> None:
    paths = process().sample(initial_state=INITIAL, time_grid=np.array([0.0]), n_paths=8, rng=SEED)
    assert paths.shape == (8, 1, 3)


def test_the_rate_stays_non_negative() -> None:
    """A property of the square-root model, not of the scheme."""
    paths = process().sample(initial_state=INITIAL, time_grid=GRID, n_paths=4096, rng=SEED)
    assert np.all(paths[:, :, 2] >= 0.0)


# --- validation -----------------------------------------------------------------


def test_a_foreign_component_is_refused_by_name() -> None:
    with pytest.raises(SimulationValidationError, match="rates"):
        BCC97(
            variance=variance(),
            jumps=jumps(),
            rates=Heston(kappa=1.0, theta=0.04, vol_of_vol=0.3, rho=-0.5),
        )  # type: ignore[arg-type]
    with pytest.raises(SimulationValidationError, match="variance"):
        BCC97(variance=rates(), jumps=jumps(), rates=rates())  # type: ignore[arg-type]


@pytest.mark.parametrize("missing", ["spot", "variance", "short_rate"])
def test_every_state_must_be_supplied(missing) -> None:
    """Including ``short_rate`` for a constant rate: that is where its level is."""
    partial = {key: value for key, value in INITIAL.items() if key != missing}
    with pytest.raises(SimulationValidationError, match=missing):
        process().sample(initial_state=partial, time_grid=GRID, n_paths=8, rng=SEED)


def test_an_unknown_scheme_names_both_vocabularies() -> None:
    """The two state variables do not offer the same schemes, so they are two
    arguments; a typo in either is refused rather than silently ignored."""
    with pytest.raises(SimulationValidationError, match=str(SCHEMES[0])):
        process().sample(initial_state=INITIAL, time_grid=GRID, n_paths=8, rng=SEED, scheme="euler")
    with pytest.raises(SimulationValidationError, match="rate_scheme"):
        process().sample(
            initial_state=INITIAL, time_grid=GRID, n_paths=8, rng=SEED, rate_scheme="euler"
        )


def test_the_variance_scheme_is_not_a_rate_scheme() -> None:
    """``exact_transition`` exists for the square-root rate and not for the
    variance, which is the whole reason the two are separate arguments."""
    assert "exact_transition" in CIR_SCHEMES
    assert "exact_transition" not in SCHEMES
    with pytest.raises(SimulationValidationError):
        process().sample(
            initial_state=INITIAL,
            time_grid=GRID,
            n_paths=8,
            rng=SEED,
            scheme="exact_transition",
        )


def test_a_scheme_is_validated_even_when_the_component_is_constant() -> None:
    reduced = BCC97(variance=ConstantVariance(), jumps=NoJumps(), rates=ConstantShortRate())
    with pytest.raises(SimulationValidationError):
        reduced.sample(
            initial_state=INITIAL, time_grid=GRID, n_paths=8, rng=SEED, rate_scheme="nonsense"
        )


def test_a_negative_initial_rate_is_refused() -> None:
    with pytest.raises(SimulationValidationError, match="short_rate"):
        process().sample(
            initial_state={**INITIAL, "short_rate": -0.01},
            time_grid=GRID,
            n_paths=8,
            rng=SEED,
        )


# --- the reduction to Bates, bitwise on every backend ---------------------------


@pytest.mark.parametrize(("var", "jm"), CONSTANT_RATE_CORNERS, ids=CORNER_IDS)
@pytest.mark.parametrize("scheme", SCHEMES)
def test_a_constant_rate_is_bitwise_bates(var, jm, scheme) -> None:
    r"""``ConstantShortRate`` must give back ``Bates(drift=rate - dividend_yield)``.

    Bitwise on the spot and variance columns, which requires two things at once:
    nothing may be drawn from the rate slot, so the diffusion and jump blocks are
    the same numbers in the same order; and the drift must be *associated* the
    same way -- ``(r - q) - compensator``, not ``r - (q + compensator)`` -- so
    the arithmetic is the same operations on the same values.
    """
    common = {"time_grid": GRID, "n_paths": 32, "rng": SEED, "scheme": scheme}
    reference = Bates(variance=var, jumps=jm, drift=RATE - DIVIDEND_YIELD).sample(
        initial_state={"spot": INITIAL["spot"], "variance": INITIAL["variance"]}, **common
    )
    got = BCC97(
        variance=var, jumps=jm, rates=ConstantShortRate(), dividend_yield=DIVIDEND_YIELD
    ).sample(initial_state=INITIAL, **common)
    np.testing.assert_array_equal(got[:, :, :2], reference)
    assert np.all(got[:, :, 2] == RATE)


@pytest.mark.parametrize(("var", "jm"), CONSTANT_RATE_CORNERS, ids=CORNER_IDS)
def test_the_bates_reduction_holds_on_torch(var, jm) -> None:
    torch = pytest.importorskip("torch")
    grid = torch.as_tensor(GRID)
    spot = torch.tensor(INITIAL["spot"], dtype=torch.float64)
    v0 = torch.tensor(INITIAL["variance"], dtype=torch.float64)
    r0 = torch.tensor(RATE, dtype=torch.float64)
    common = {"time_grid": grid, "n_paths": 16, "rng": SEED}
    reference = Bates(variance=var, jumps=jm, drift=RATE - DIVIDEND_YIELD).sample(
        initial_state={"spot": spot, "variance": v0}, **common
    )
    got = BCC97(
        variance=var, jumps=jm, rates=ConstantShortRate(), dividend_yield=DIVIDEND_YIELD
    ).sample(initial_state={"spot": spot, "variance": v0, "short_rate": r0}, **common)
    assert torch.equal(got[:, :, :2], reference)


@pytest.mark.parametrize(("var", "jm"), CONSTANT_RATE_CORNERS, ids=CORNER_IDS)
def test_the_bates_reduction_holds_on_jax(var, jm) -> None:
    """Both facades reserve three slots, so their first two keys agree."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    grid = jnp.asarray(GRID)
    key = jax.random.key(SEED)
    common = {"time_grid": grid, "n_paths": 16, "rng": key}
    reference = Bates(variance=var, jumps=jm, drift=RATE - DIVIDEND_YIELD).sample(
        initial_state={"spot": INITIAL["spot"], "variance": INITIAL["variance"]}, **common
    )
    got = BCC97(
        variance=var, jumps=jm, rates=ConstantShortRate(), dividend_yield=DIVIDEND_YIELD
    ).sample(initial_state=INITIAL, **common)
    np.testing.assert_array_equal(np.asarray(got)[:, :, :2], np.asarray(reference))


# --- the rate block -------------------------------------------------------------


def test_the_rate_column_is_the_short_rate_process_itself_on_jax() -> None:
    """Not "distributed like": the same code, handed the slot's own key.

    Checkable exactly on JAX, where slot 2 is a derived key a test can rebuild.
    On NumPy and torch ``split`` hands back the same advancing generator, so the
    slot has no name of its own and this identity has nothing to compare
    against -- the law is still the same because the call is the same call.
    """
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    key = jax.random.key(SEED)
    grid = jnp.asarray(GRID)
    paths = process().sample(initial_state=INITIAL, time_grid=grid, n_paths=32, rng=key)

    stream = random_stream(key, namespace="jax", device=None, dtype=None)
    rate_slot = split(stream, 3)[2]
    alone = rates().sample(
        initial_state={"short_rate": RATE}, time_grid=grid, n_paths=32, rng=rate_slot.handle
    )
    np.testing.assert_array_equal(np.asarray(paths[:, :, 2]), np.asarray(alone[:, :, 0]))


def test_zero_intensity_jumps_reduce_to_no_jumps_on_jax() -> None:
    """Fixed slots, so switching a component off does not move the ones after it."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    key = jax.random.key(SEED)
    common = {"time_grid": jnp.asarray(GRID), "n_paths": 16, "rng": key}
    zero = BCC97(variance=variance(), jumps=jumps(jump_intensity=0.0), rates=rates()).sample(
        initial_state=INITIAL, **common
    )
    none = BCC97(variance=variance(), jumps=NoJumps(), rates=rates()).sample(
        initial_state=INITIAL, **common
    )
    np.testing.assert_array_equal(np.asarray(zero), np.asarray(none))


def test_on_numpy_a_skipped_block_moves_the_ones_after_it() -> None:
    """The documented limit of the slot contract, asserted rather than implied.

    A NumPy or torch generator is a position in one stream, so a configuration
    that draws *fewer* blocks leaves every later block reading different
    numbers. The variance is driven by the diffusion block alone and is
    therefore untouched; the rate block is not, and neither is the spot that
    reads it. Nothing here is wrong -- both runs are correct samples -- and
    stating it is what stops a reader from expecting the JAX guarantee
    everywhere.
    """
    common = {"time_grid": GRID, "n_paths": 16, "rng": SEED}
    zero = BCC97(variance=variance(), jumps=jumps(jump_intensity=0.0), rates=rates()).sample(
        initial_state=INITIAL, **common
    )
    none = BCC97(variance=variance(), jumps=NoJumps(), rates=rates()).sample(
        initial_state=INITIAL, **common
    )
    np.testing.assert_array_equal(zero[:, :, 1], none[:, :, 1])
    assert not np.array_equal(zero[:, :, 2], none[:, :, 2])


@pytest.mark.parametrize("jm", [LognormalJumps(**JUMP_PARAMS), NoJumps()], ids=["jumps", "none"])
def test_switching_the_rate_component_leaves_the_diffusion_block_alone(jm) -> None:
    """The draw-order contract's normative consequence for the third slot.

    Switching ``CIRShortRate`` for ``ConstantShortRate`` must leave block 1
    bitwise identical, so the diffusion innovations of a reduced model are the
    innovations of the full one. The variance column is block 1's witness under
    :class:`HestonVariance`: it is driven by column 0 of the diffusion normals
    and by nothing else, so it moves if and only if that block moves.

    The :class:`ConstantVariance` corner has no such witness -- its variance is
    held fixed and its spot reads the rate -- and is pinned instead by
    ``test_a_constant_rate_is_bitwise_bates``, which compares the whole spot
    column against a run that has no rate slot at all.
    """
    common = {"initial_state": INITIAL, "time_grid": GRID, "n_paths": 32, "rng": SEED}
    with_cir = BCC97(variance=variance(), jumps=jm, rates=rates()).sample(**common)
    with_constant = BCC97(variance=variance(), jumps=jm, rates=ConstantShortRate()).sample(**common)
    np.testing.assert_array_equal(with_cir[:, :, 1], with_constant[:, :, 1])
    # And the rate slot really was used, so the equality above is not vacuous.
    assert not np.array_equal(with_cir[:, :, 2], with_constant[:, :, 2])


# --- the rate quadrature is the discounting quadrature --------------------------


@pytest.mark.parametrize(("var", "jm"), CONSTANT_RATE_CORNERS, ids=CORNER_IDS)
@pytest.mark.parametrize("scheme", SCHEMES)
def test_the_rate_cancels_pathwise_between_drift_and_discounting(var, jm, scheme) -> None:
    r"""``exp(-int_0^T r) * S_T`` at rate ``r`` equals ``S_T`` at rate zero, per path.

    The rate enters the log-spot additively and the rate draws come from their
    own slot, so a CIR-rate run and a zero-rate run at the same seed share every
    spot, variance and jump number.  Their log-spots therefore differ by exactly
    the drift's accumulated rate -- and if that is the same quantity
    :class:`PathwiseShortRateDiscounting` accumulates, the discount factor
    cancels it path by path.

    This is what "the same rate path in drift and discounting" means, and it is
    a floating-point identity rather than a statistical one: a mismatched
    quadrature would show up here at 1e-2, not at 1e-15.
    """
    horizon = np.linspace(0.0, 2.0, 49)
    volatile = CIRShortRate(kappa=0.5, theta=0.05, volatility=0.25)
    common = {"time_grid": horizon, "n_paths": 2000, "rng": 99, "scheme": scheme}

    with_rate = BCC97(variance=var, jumps=jm, rates=volatile, dividend_yield=DIVIDEND_YIELD).sample(
        initial_state=INITIAL, **common
    )
    without = BCC97(
        variance=var, jumps=jm, rates=ConstantShortRate(), dividend_yield=DIVIDEND_YIELD
    ).sample(initial_state={**INITIAL, "short_rate": 0.0}, **common)

    factors = PathwiseShortRateDiscounting().discount_factors(
        states=with_rate, time_grid=horizon, state_names=BCC97.state_names
    )
    np.testing.assert_allclose(
        factors * with_rate[:, -1, 0], without[:, -1, 0], rtol=1e-13, atol=0.0
    )
    # The variance is driven by the diffusion block alone, so it is identical.
    np.testing.assert_array_equal(with_rate[:, :, 1], without[:, :, 1])


def test_left_riemann_discounting_is_a_different_quantity() -> None:
    """Not a worse approximation of the same number: a different number.

    The drift integrates the trapezoid, so discounting the same paths with the
    left rule leaves an uncancelled residual. It shrinks with the grid -- about
    linearly, as a first-order rule against a second-order one -- and shrinking
    is exactly why it must not be silently substituted.
    """
    volatile = CIRShortRate(kappa=0.5, theta=0.05, volatility=0.25)
    model = BCC97(variance=ConstantVariance(), jumps=NoJumps(), rates=volatile, dividend_yield=0.0)
    residuals = []
    for n_steps in (8, 16, 32):
        horizon = np.linspace(0.0, 2.0, n_steps + 1)
        paths = model.sample(initial_state=INITIAL, time_grid=horizon, n_paths=2000, rng=3)
        common = {
            "states": paths,
            "time_grid": horizon,
            "state_names": BCC97.state_names,
        }
        trapezoid = PathwiseShortRateDiscounting(rule="trapezoid").discount_factors(**common)
        left = PathwiseShortRateDiscounting(rule="left_riemann").discount_factors(**common)
        residuals.append(float(np.abs(left / trapezoid - 1.0).mean()))

    assert residuals[0] > 1e-4
    for coarse, fine in zip(residuals, residuals[1:]):
        assert fine < 0.7 * coarse


def test_the_discounted_spot_is_a_martingale() -> None:
    r"""``E[e^{-int r} S_T] = S_0 e^{-qT}``, at the configuration where it is exact.

    With a constant variance the log-spot step is closed-form, the jump
    compensation is exact per step, and the trapezoid rate cancels: the identity
    then holds in the *discretized* model, not merely in its limit, so the only
    error left is Monte Carlo.

    The Heston corner carries the quadratic-exponential scheme's own martingale
    bias on top of that and is checked separately, with a budget.
    """
    horizon = np.linspace(0.0, 1.0, 33)
    model = BCC97(
        variance=ConstantVariance(),
        jumps=jumps(),
        rates=rates(),
        dividend_yield=DIVIDEND_YIELD,
    )
    paths = model.sample(initial_state=INITIAL, time_grid=horizon, n_paths=200_000, rng=11)
    factors = PathwiseShortRateDiscounting().discount_factors(
        states=paths, time_grid=horizon, state_names=BCC97.state_names
    )
    discounted = factors * paths[:, -1, 0]
    expected = INITIAL["spot"] * np.exp(-DIVIDEND_YIELD * horizon[-1])
    stderr = discounted.std(ddof=1) / np.sqrt(len(discounted))
    assert abs(float(discounted.mean()) - expected) < 4.0 * stderr


def test_the_heston_corner_is_a_martingale_within_the_scheme_s_own_bias() -> None:
    """Averaged over seeds, because one heavy-tailed sample is not a measurement.

    The estimator has jumps and stochastic volatility in it, so its fourth
    moment is large and a single run's standard error understates the spread.
    Five independent runs give a mean bias and a spread of that mean, which is
    what the bound is stated against.
    """
    horizon = np.linspace(0.0, 1.0, 65)
    model = process()
    expected = INITIAL["spot"] * np.exp(-DIVIDEND_YIELD * horizon[-1])
    biases = []
    for seed in (1, 2, 3, 4, 5):
        paths = model.sample(initial_state=INITIAL, time_grid=horizon, n_paths=100_000, rng=seed)
        factors = PathwiseShortRateDiscounting().discount_factors(
            states=paths, time_grid=horizon, state_names=BCC97.state_names
        )
        biases.append(float((factors * paths[:, -1, 0]).mean()) - expected)
    mean = float(np.mean(biases))
    spread = float(np.std(biases, ddof=1) / np.sqrt(len(biases)))
    assert abs(mean) < 4.0 * spread + 0.02


# --- the rate component's refusals travel with it --------------------------------


def test_the_exact_transition_is_refused_on_torch_before_any_draw() -> None:
    """The component's own refusal, raised through the configuration.

    Before the stream exists, so a caller's generator is not advanced by a call
    that failed -- which is what makes a refused run costless and repeatable.
    """
    torch = pytest.importorskip("torch")
    generator = torch.Generator().manual_seed(SEED)
    state = generator.get_state()
    model = process(rates=CIRShortRate(kappa=1.0, theta=0.04, volatility=0.05))
    with pytest.raises(UnsupportedProcessError, match="generator"):
        model.sample(
            initial_state={
                "spot": torch.tensor(100.0, dtype=torch.float64),
                "variance": torch.tensor(0.04, dtype=torch.float64),
                "short_rate": torch.tensor(RATE, dtype=torch.float64),
            },
            time_grid=torch.as_tensor(GRID),
            n_paths=8,
            rng=generator,
            rate_scheme="exact_transition",
        )
    assert torch.equal(generator.get_state(), state)


def test_antithetic_is_refused_below_one_degree_of_freedom() -> None:
    """Feller ratio at or under one leaves no normal to mirror; see ``CIRShortRate``."""
    # 4*kappa*theta/sigma**2 = 4*0.5*0.02/0.5**2 = 0.16
    thin = CIRShortRate(kappa=0.5, theta=0.02, volatility=0.5)
    with pytest.raises(UnsupportedProcessError, match="degrees of freedom"):
        process(rates=thin).sample(
            initial_state=INITIAL,
            time_grid=GRID,
            n_paths=8,
            rng=SEED,
            antithetic=True,
            rate_scheme="exact_transition",
        )


@pytest.mark.parametrize("rate_scheme", CIR_SCHEMES)
def test_every_rate_scheme_runs(rate_scheme) -> None:
    paths = process().sample(
        initial_state=INITIAL, time_grid=GRID, n_paths=64, rng=SEED, rate_scheme=rate_scheme
    )
    assert paths.shape == (64, len(GRID), 3)
    assert np.all(np.isfinite(paths))


def test_the_rate_scheme_changes_the_rate_and_not_the_diffusion_block_on_jax() -> None:
    """Slot 0 is drawn before slot 2 and from its own key, so it cannot move."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    key = jax.random.key(SEED)
    common = {"time_grid": jnp.asarray(GRID), "n_paths": 32, "rng": key}
    quadratic = process().sample(
        initial_state=INITIAL, rate_scheme="quadratic_exponential", **common
    )
    euler = process().sample(initial_state=INITIAL, rate_scheme="full_truncation_euler", **common)
    np.testing.assert_array_equal(np.asarray(quadratic[:, :, 1]), np.asarray(euler[:, :, 1]))
    assert not np.array_equal(np.asarray(quadratic[:, :, 2]), np.asarray(euler[:, :, 2]))


# --- backends and grids ----------------------------------------------------------


def test_a_non_uniform_grid_uses_each_step_s_own_length() -> None:
    """A jump intensity is per year, so a longer step must carry more jumps."""
    uneven = np.array([0.0, 0.05, 2.0])
    heavy = BCC97(
        variance=ConstantVariance(),
        jumps=jumps(jump_intensity=20.0),
        rates=ConstantShortRate(),
    )
    paths = heavy.sample(initial_state=INITIAL, time_grid=uneven, n_paths=20_000, rng=SEED)
    first = np.log(paths[:, 1, 0] / paths[:, 0, 0])
    second = np.log(paths[:, 2, 0] / paths[:, 1, 0])
    assert second.var() > 10.0 * first.var()


def test_it_samples_on_torch() -> None:
    torch = pytest.importorskip("torch")
    paths = process().sample(
        initial_state={
            "spot": torch.tensor(100.0, dtype=torch.float64),
            "variance": torch.tensor(0.04, dtype=torch.float64),
            "short_rate": torch.tensor(RATE, dtype=torch.float64),
        },
        time_grid=torch.as_tensor(GRID),
        n_paths=64,
        rng=SEED,
    )
    assert paths.shape == (64, len(GRID), 3)
    assert paths.dtype == torch.float64


def test_it_samples_on_jax() -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    paths = process().sample(
        initial_state=INITIAL,
        time_grid=jnp.asarray(GRID),
        n_paths=64,
        rng=jax.random.key(SEED),
    )
    assert paths.shape == (64, len(GRID), 3)
