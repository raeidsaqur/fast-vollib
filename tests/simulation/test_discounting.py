"""Discounting rules: the arithmetic, the refusals, and the bit-level agreement.

The most important test in this file is not about a rule at all.  It is
``test_the_constant_rule_agrees_with_the_engine_to_the_bit``: the engine
already discounts, and when the engine later learns to take a rule as an
argument, a caller who names the behaviour they were already getting must get
the same number back -- not a number that rounds the same way.  Anything less
turns "this argument defaults to the old behaviour" into a claim nobody can
check.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from fast_vollib.simulation import (
    RULES,
    ConstantRateDiscounting,
    DiscountingRule,
    PathwiseShortRateDiscounting,
    SimulationValidationError,
)

GRID = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
NAMES = ("short_rate",)


def rates(values) -> np.ndarray:
    """Paths shaped ``(n_paths, n_times, 1)`` from a list of per-path rate rows."""
    return np.asarray(values, dtype=np.float64)[:, :, None]


# --- the protocol --------------------------------------------------------------


@pytest.mark.parametrize(
    "rule", [ConstantRateDiscounting(rate=0.03), PathwiseShortRateDiscounting()]
)
def test_both_shipped_rules_satisfy_the_protocol(rule) -> None:
    assert isinstance(rule, DiscountingRule)


def test_the_protocol_is_structural_and_needs_no_registration() -> None:
    class OurOwnRule:
        def discount_factors(self, *, states, time_grid, state_names):
            return np.ones(states.shape[0])

    assert isinstance(OurOwnRule(), DiscountingRule)


def test_an_object_without_the_method_is_not_a_rule() -> None:
    assert not isinstance(object(), DiscountingRule)


@pytest.mark.parametrize(
    "rule", [ConstantRateDiscounting(rate=0.03), PathwiseShortRateDiscounting()]
)
def test_a_rule_is_a_frozen_value_holding_no_simulation_state(rule) -> None:
    with pytest.raises(Exception):
        rule.rule = "x"  # type: ignore[misc]
    forbidden = {"states", "paths", "rng", "process", "engine", "instrument"}
    assert forbidden.isdisjoint(set(rule.__dataclass_fields__))


# --- the constant rule ---------------------------------------------------------


@pytest.mark.parametrize("rate", [-0.01, 0.0, 0.03, 0.12])
def test_the_constant_rule_is_the_exponential_of_the_grid_horizon(rate) -> None:
    states = np.zeros((4, len(GRID), 1))
    factors = ConstantRateDiscounting(rate=rate).discount_factors(
        states=states, time_grid=GRID, state_names=NAMES
    )
    assert factors.shape == (4,)
    assert np.all(factors == math.exp(-rate * 1.0))


def test_the_constant_rule_ignores_the_states_entirely() -> None:
    """It is a *constant* rate; a state that changed it would be a different rule."""
    one = ConstantRateDiscounting(rate=0.05).discount_factors(
        states=np.zeros((3, 2, 1)), time_grid=np.array([0.0, 1.0]), state_names=NAMES
    )
    two = ConstantRateDiscounting(rate=0.05).discount_factors(
        states=np.full((3, 2, 1), 9.9), time_grid=np.array([0.0, 1.0]), state_names=NAMES
    )
    np.testing.assert_array_equal(one, two)


@pytest.mark.parametrize("rate", [-0.01, 0.0, 0.005, 0.03, 0.12])
@pytest.mark.parametrize("maturity", [0.25, 1.0, 5.0, 30.0])
def test_the_constant_rule_agrees_with_the_engine_to_the_bit(rate, maturity) -> None:
    """The claim an engine extension will rest on, checked before it is made.

    ``MonteCarloEngine._discount`` and this rule compute the same factor, and
    "the same" has to mean the same float64. ``math.exp`` and ``numpy.exp``
    disagree in the last bit at some arguments -- measured, at two of the
    forty-two pairs swept in ``tests/rates/test_curves.py`` -- so a rule that
    reached for the obvious ``xp.exp`` would move a caller's price in the last
    digits the moment they named the behaviour they already had.

    The engine multiplies a cashflow by the factor and this rule returns the
    factor, so the comparison is done on a unit cashflow.
    """
    from fast_vollib.simulation.monte_carlo import _discount

    grid = np.array([0.0, 0.5 * maturity, maturity])
    factors = ConstantRateDiscounting(rate=rate).discount_factors(
        states=np.zeros((2, 3, 1)), time_grid=grid, state_names=NAMES
    )
    engine = _discount(1.0, rate=rate, maturity=maturity)
    assert float(factors[0]) == engine
    assert float(factors[1]) == engine


def test_the_horizon_the_constant_rule_uses_is_the_grid_and_it_says_so() -> None:
    """A grid ending short of maturity discounts over the grid, not the contract.

    The engine accepts a caller-supplied grid whose last time sits within a
    tolerance of maturity, so the two horizons can differ. The rule cannot see
    the contract, and rather than pretend otherwise this pins which one it
    uses.
    """
    grid = np.array([0.0, 0.5, 0.999])
    factors = ConstantRateDiscounting(rate=0.04).discount_factors(
        states=np.zeros((1, 3, 1)), time_grid=grid, state_names=NAMES
    )
    assert float(factors[0]) == math.exp(-0.04 * 0.999)
    assert float(factors[0]) != math.exp(-0.04 * 1.0)


# --- the pathwise rule ---------------------------------------------------------


def test_the_trapezoid_rule_is_the_trapezoid_rule() -> None:
    """Computed by hand on two paths, which is the point of a small fixture."""
    paths = rates([[0.02, 0.04, 0.06], [0.05, 0.05, 0.01]])
    grid = np.array([0.0, 1.0, 3.0])
    got = PathwiseShortRateDiscounting().discount_factors(
        states=paths, time_grid=grid, state_names=NAMES
    )
    first = 0.5 * (0.02 + 0.04) * 1.0 + 0.5 * (0.04 + 0.06) * 2.0
    second = 0.5 * (0.05 + 0.05) * 1.0 + 0.5 * (0.05 + 0.01) * 2.0
    np.testing.assert_allclose(got, np.exp([-first, -second]), rtol=1e-15)


def test_the_left_riemann_rule_is_the_left_riemann_rule() -> None:
    paths = rates([[0.02, 0.04, 0.06], [0.05, 0.05, 0.01]])
    grid = np.array([0.0, 1.0, 3.0])
    got = PathwiseShortRateDiscounting(rule="left_riemann").discount_factors(
        states=paths, time_grid=grid, state_names=NAMES
    )
    first = 0.02 * 1.0 + 0.04 * 2.0
    second = 0.05 * 1.0 + 0.05 * 2.0
    np.testing.assert_allclose(got, np.exp([-first, -second]), rtol=1e-15)


@pytest.mark.parametrize("rule", RULES)
@pytest.mark.parametrize("rate", [0.0, 0.03, -0.01])
def test_both_rules_are_exact_for_a_rate_that_does_not_move(rule, rate) -> None:
    """The one case where the quadrature error is zero, so it pins the setup."""
    paths = np.full((3, len(GRID), 1), rate)
    got = PathwiseShortRateDiscounting(rule=rule).discount_factors(
        states=paths, time_grid=GRID, state_names=NAMES
    )
    np.testing.assert_allclose(got, np.exp(-rate * 1.0), rtol=1e-15)


def test_the_two_rules_disagree_on_a_rate_that_does_move() -> None:
    """Otherwise ``rule`` would be a setting with no effect."""
    paths = rates([[0.01, 0.03, 0.09]])
    grid = np.array([0.0, 0.5, 1.0])
    kwargs = {"states": paths, "time_grid": grid, "state_names": NAMES}
    trapezoid = PathwiseShortRateDiscounting(rule="trapezoid").discount_factors(**kwargs)
    riemann = PathwiseShortRateDiscounting(rule="left_riemann").discount_factors(**kwargs)
    assert float(trapezoid[0]) != float(riemann[0])
    # A rising rate is under-integrated from the left, so the factor is larger.
    assert float(riemann[0]) > float(trapezoid[0])


def test_an_unevenly_spaced_grid_is_integrated_over_its_own_steps() -> None:
    """The rule reads the times it was given rather than assuming a spacing."""
    paths = rates([[0.02, 0.02, 0.02]])
    grid = np.array([0.0, 0.1, 5.0])
    got = PathwiseShortRateDiscounting().discount_factors(
        states=paths, time_grid=grid, state_names=NAMES
    )
    np.testing.assert_allclose(got, np.exp(-0.02 * 5.0), rtol=1e-15)


def test_the_named_state_is_found_by_name_and_not_by_position() -> None:
    """A multi-state process puts the rate wherever its ``state_names`` say."""
    states = np.stack(
        [np.full((2, 3), 100.0), np.full((2, 3), 0.05), np.full((2, 3), 0.04)], axis=2
    )
    got = PathwiseShortRateDiscounting().discount_factors(
        states=states,
        time_grid=np.array([0.0, 0.5, 1.0]),
        state_names=("spot", "variance", "short_rate"),
    )
    np.testing.assert_allclose(got, np.exp(-0.04), rtol=1e-15)


def test_a_custom_state_name_is_honoured() -> None:
    states = np.full((2, 3, 1), 0.02)
    got = PathwiseShortRateDiscounting(state_name="r").discount_factors(
        states=states, time_grid=np.array([0.0, 0.5, 1.0]), state_names=("r",)
    )
    np.testing.assert_allclose(got, np.exp(-0.02), rtol=1e-15)


# --- refusals ------------------------------------------------------------------


def test_an_unknown_rule_is_refused_at_construction() -> None:
    with pytest.raises(SimulationValidationError, match="rule must be one of"):
        PathwiseShortRateDiscounting(rule="simpson")


@pytest.mark.parametrize("name", ["", None, 3])
def test_an_unusable_state_name_is_refused_at_construction(name) -> None:
    with pytest.raises(SimulationValidationError, match="state_name"):
        PathwiseShortRateDiscounting(state_name=name)


def test_a_state_the_process_does_not_have_is_refused_rather_than_substituted() -> None:
    """Discounting a payoff by a spot returns a number that is not a price."""
    with pytest.raises(SimulationValidationError, match="not among the simulated states"):
        PathwiseShortRateDiscounting().discount_factors(
            states=np.zeros((2, 3, 1)),
            time_grid=np.array([0.0, 0.5, 1.0]),
            state_names=("spot",),
        )


def test_the_state_name_is_checked_before_any_arithmetic() -> None:
    """So a mismatched process fails on the name rather than on a shape later."""
    with pytest.raises(SimulationValidationError, match="not among the simulated states"):
        PathwiseShortRateDiscounting().discount_factors(
            states=np.zeros((2, 99, 1)),  # a grid mismatch as well
            time_grid=np.array([0.0, 0.5, 1.0]),
            state_names=("spot",),
        )


@pytest.mark.parametrize(
    "rule", [ConstantRateDiscounting(rate=0.03), PathwiseShortRateDiscounting()]
)
def test_paths_and_times_from_different_runs_are_refused(rule) -> None:
    with pytest.raises(SimulationValidationError, match="describe different runs"):
        rule.discount_factors(
            states=np.zeros((2, 4, 1)), time_grid=np.array([0.0, 0.5, 1.0]), state_names=NAMES
        )


@pytest.mark.parametrize(
    "rule", [ConstantRateDiscounting(rate=0.03), PathwiseShortRateDiscounting()]
)
def test_a_grid_that_does_not_start_at_the_valuation_date_is_refused(rule) -> None:
    """An integral from a later start discounts over the wrong interval."""
    with pytest.raises(SimulationValidationError, match="must start at the valuation date"):
        rule.discount_factors(
            states=np.zeros((2, 3, 1)), time_grid=np.array([0.5, 0.75, 1.0]), state_names=NAMES
        )


@pytest.mark.parametrize(
    "rule", [ConstantRateDiscounting(rate=0.03), PathwiseShortRateDiscounting()]
)
def test_a_single_time_is_refused_because_a_horizon_is_an_interval(rule) -> None:
    with pytest.raises(SimulationValidationError, match="at least two times"):
        rule.discount_factors(
            states=np.zeros((2, 1, 1)), time_grid=np.array([0.0]), state_names=NAMES
        )


@pytest.mark.parametrize(
    "rule", [ConstantRateDiscounting(rate=0.03), PathwiseShortRateDiscounting()]
)
def test_states_of_the_wrong_rank_are_refused(rule) -> None:
    with pytest.raises(SimulationValidationError, match="n_paths, n_times, n_state"):
        rule.discount_factors(
            states=np.zeros((2, 3)), time_grid=np.array([0.0, 0.5, 1.0]), state_names=NAMES
        )


@pytest.mark.parametrize(
    "rule", [ConstantRateDiscounting(rate=0.03), PathwiseShortRateDiscounting()]
)
def test_a_two_dimensional_time_grid_is_refused(rule) -> None:
    with pytest.raises(SimulationValidationError, match="one-dimensional"):
        rule.discount_factors(
            states=np.zeros((2, 3, 1)),
            time_grid=np.zeros((3, 1)),
            state_names=NAMES,
        )


# --- backends ------------------------------------------------------------------


def test_the_pathwise_rule_runs_in_torch_and_keeps_a_gradient() -> None:
    """The factor is a smooth function of the sampled rates, so it differentiates."""
    torch = pytest.importorskip("torch")
    states = torch.full((4, 3, 1), 0.03, dtype=torch.float64, requires_grad=True)
    grid = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
    factors = PathwiseShortRateDiscounting().discount_factors(
        states=states, time_grid=grid, state_names=NAMES
    )
    assert factors.shape == (4,)
    factors.sum().backward()
    assert states.grad is not None and torch.all(torch.isfinite(states.grad))


def test_the_constant_rule_keeps_a_torch_rate_native_and_differentiable() -> None:
    torch = pytest.importorskip("torch")
    rate = torch.tensor(0.03, dtype=torch.float64, requires_grad=True)
    factors = ConstantRateDiscounting(rate=rate).discount_factors(
        states=torch.zeros((3, 2, 1), dtype=torch.float64),
        time_grid=torch.tensor([0.0, 4.0], dtype=torch.float64),
        state_names=NAMES,
    )
    factors.sum().backward()
    # d/dr sum_i exp(-4r) = -4 * 3 * exp(-4r), the three paths sharing one rate.
    assert float(rate.grad) == pytest.approx(-3.0 * 4.0 * math.exp(-0.12), rel=1e-12)


def test_the_pathwise_rule_traces_and_differentiates_under_jax() -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    grid = jnp.asarray([0.0, 0.5, 1.0])

    def total(level):
        states = jnp.full((4, 3, 1), level)
        return (
            PathwiseShortRateDiscounting()
            .discount_factors(states=states, time_grid=grid, state_names=NAMES)
            .sum()
        )

    assert float(jax.jit(total)(jnp.asarray(0.03))) == pytest.approx(
        4.0 * math.exp(-0.03), rel=1e-6
    )
    assert float(jax.grad(total)(jnp.asarray(0.03))) == pytest.approx(
        -4.0 * math.exp(-0.03), rel=1e-5
    )
