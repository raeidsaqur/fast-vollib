"""Path-dependent payoffs: dispatch, conventions, identities, and domains."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from fast_vollib.instruments import (
    AsianOption,
    EuropeanOption,
    InstrumentValidationError,
    PayoffRequirement,
    payoff,
    payoff_requirement,
)
from fast_vollib.simulation import Scenario, ScenarioMismatchError

GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
#: Two hand-built paths with easy averages: fixings 90, 100, 110, 120 (mean 105)
#: and 120, 110, 100, 90 (mean 105), so the arithmetic average is the same and
#: the terminal value is not.
PATHS = np.array(
    [
        [[100.0], [90.0], [100.0], [110.0], [120.0]],
        [[100.0], [120.0], [110.0], [100.0], [90.0]],
    ]
)
FIXINGS = PATHS[:, 1:, 0]


def scenario(states: np.ndarray = PATHS, grid: Any = GRID) -> Scenario:
    return Scenario.from_states("ACME", time_grid=grid, states=states)


def asian(**overrides: Any) -> AsianOption:
    call: dict[str, Any] = {
        "underlier": "ACME",
        "option_type": "call",
        "strike": 100.0,
        "averaging_method": "arithmetic",
        "strike_convention": "fixed",
        "maturity": 1.0,
    }
    call.update(overrides)
    return AsianOption(**call)


def geometric_mean(values: np.ndarray) -> np.ndarray:
    return np.exp(np.mean(np.log(values), axis=1))


# --- dispatch ------------------------------------------------------------------


def test_a_path_contract_declares_a_path_requirement() -> None:
    assert payoff_requirement(AsianOption) is PayoffRequirement.PATH
    assert payoff_requirement(asian()) is PayoffRequirement.PATH


def test_a_bare_array_is_refused_before_any_arithmetic() -> None:
    with pytest.raises(InstrumentValidationError) as excinfo:
        payoff(asian(), FIXINGS)
    message = str(excinfo.value)
    assert "needs a simulated scenario" in message
    assert "no underlier" in message


@pytest.mark.parametrize(
    "given", [100.0, [100.0, 110.0], np.array([100.0])], ids=["float", "list", "array"]
)
def test_no_shape_of_bare_state_becomes_a_scenario(given: Any) -> None:
    with pytest.raises(InstrumentValidationError, match="needs a simulated scenario"):
        payoff(asian(), given)


def test_a_scenario_of_the_wrong_underlier_is_refused() -> None:
    with pytest.raises(ScenarioMismatchError, match="written on 'OTHER'"):
        payoff(asian(underlier="OTHER"), scenario())


def test_a_scenario_whose_horizon_is_not_the_maturity_is_refused() -> None:
    with pytest.raises(ScenarioMismatchError, match="ends at t="):
        payoff(asian(maturity=2.0), scenario())


def test_the_scenario_method_and_the_function_agree() -> None:
    contract = asian()
    np.testing.assert_array_equal(scenario().payoff(contract), payoff(contract, scenario()))


def test_a_terminal_contract_is_still_refused_a_scenario() -> None:
    european = EuropeanOption(underlier="ACME", option_type="call", strike=100.0, maturity=1.0)
    with pytest.raises(InstrumentValidationError, match="scenario.payoff"):
        payoff(european, scenario())


# --- averaging conventions -----------------------------------------------------


def test_fixings_exclude_the_valuation_date() -> None:
    """S_0 is known when the contract is written and is not a fixing.

    Built so that including it would change the answer: the average of all five
    points differs from the average of the last four.
    """
    result = payoff(asian(strike=0.001), scenario())
    np.testing.assert_allclose(result, FIXINGS.mean(axis=1) - 0.001)
    including_start = PATHS[:, :, 0].mean(axis=1)
    assert not np.allclose(FIXINGS.mean(axis=1), including_start)


def test_fixed_strike_arithmetic_call_and_put() -> None:
    average = FIXINGS.mean(axis=1)
    np.testing.assert_allclose(
        payoff(asian(strike=100.0), scenario()), np.maximum(average - 100.0, 0.0)
    )
    np.testing.assert_allclose(
        payoff(asian(strike=110.0, option_type="put"), scenario()),
        np.maximum(110.0 - average, 0.0),
    )


def test_fixed_strike_geometric_call_uses_the_geometric_mean() -> None:
    average = geometric_mean(FIXINGS)
    np.testing.assert_allclose(
        payoff(asian(averaging_method="geometric", strike=100.0), scenario()),
        np.maximum(average - 100.0, 0.0),
    )


def test_floating_strike_call_and_put_compare_against_the_average() -> None:
    average = FIXINGS.mean(axis=1)
    terminal = PATHS[:, -1, 0]
    floating = asian(strike=None, strike_convention="floating")
    np.testing.assert_allclose(payoff(floating, scenario()), np.maximum(terminal - average, 0.0))
    floating_put = asian(strike=None, strike_convention="floating", option_type="put")
    np.testing.assert_allclose(
        payoff(floating_put, scenario()), np.maximum(average - terminal, 0.0)
    )


def test_the_two_paths_separate_fixed_from_floating() -> None:
    """Same arithmetic average, different terminal values, so the two differ."""
    average = FIXINGS.mean(axis=1)
    assert average[0] == pytest.approx(average[1])
    fixed = payoff(asian(strike=100.0), scenario())
    floating = payoff(asian(strike=None, strike_convention="floating"), scenario())
    assert fixed[0] == pytest.approx(fixed[1])
    assert floating[0] != pytest.approx(floating[1])


def test_payoffs_are_undiscounted() -> None:
    """A longer-dated contract on the same path pays the same at maturity."""
    long_grid = [0.0, 2.5, 5.0, 7.5, 10.0]
    same = payoff(asian(maturity=10.0), scenario(grid=long_grid))
    np.testing.assert_allclose(same, payoff(asian(), scenario()))


# --- identities ----------------------------------------------------------------


@pytest.mark.parametrize("notional", [1.0, 3.5, -2.0], ids=["unit", "long", "short"])
@pytest.mark.parametrize("method", ["arithmetic", "geometric"], ids=["arith", "geom"])
def test_fixed_call_minus_put_is_the_notional_times_average_minus_strike(
    notional: float, method: str
) -> None:
    common = {"averaging_method": method, "strike": 105.0, "notional": notional}
    call = payoff(asian(option_type="call", **common), scenario())
    put = payoff(asian(option_type="put", **common), scenario())
    average = geometric_mean(FIXINGS) if method == "geometric" else FIXINGS.mean(axis=1)
    np.testing.assert_allclose(call - put, notional * (average - 105.0))


@pytest.mark.parametrize("notional", [1.0, 3.5, -2.0], ids=["unit", "long", "short"])
def test_floating_call_minus_put_is_the_notional_times_terminal_minus_average(
    notional: float,
) -> None:
    common = {"strike": None, "strike_convention": "floating", "notional": notional}
    call = payoff(asian(option_type="call", **common), scenario())
    put = payoff(asian(option_type="put", **common), scenario())
    np.testing.assert_allclose(call - put, notional * (PATHS[:, -1, 0] - FIXINGS.mean(axis=1)))


def test_geometric_never_exceeds_arithmetic_for_a_long_call() -> None:
    """AM-GM, stated for a positive notional; a short position reverses it."""
    common = {"strike": 100.0, "notional": 2.0}
    arithmetic = payoff(asian(averaging_method="arithmetic", **common), scenario())
    geometric = payoff(asian(averaging_method="geometric", **common), scenario())
    assert np.all(geometric <= arithmetic)


def test_geometric_is_never_below_arithmetic_for_a_long_put() -> None:
    common = {"strike": 110.0, "option_type": "put", "notional": 2.0}
    arithmetic = payoff(asian(averaging_method="arithmetic", **common), scenario())
    geometric = payoff(asian(averaging_method="geometric", **common), scenario())
    assert np.all(geometric >= arithmetic)


def test_a_short_position_reverses_the_ordering() -> None:
    """The economics, spelled out rather than left as a trap in the previous test."""
    common = {"strike": 100.0, "notional": -2.0}
    arithmetic = payoff(asian(averaging_method="arithmetic", **common), scenario())
    geometric = payoff(asian(averaging_method="geometric", **common), scenario())
    assert np.all(geometric >= arithmetic)


@pytest.mark.parametrize("strikes", [(90.0, 100.0), (100.0, 130.0)], ids=["low", "high"])
def test_a_long_call_is_non_increasing_in_the_strike(strikes: tuple[float, float]) -> None:
    lower, higher = strikes
    assert np.all(
        payoff(asian(strike=higher, notional=2.0), scenario())
        <= payoff(asian(strike=lower, notional=2.0), scenario())
    )


def test_payoffs_are_non_negative_before_a_negative_notional() -> None:
    assert np.all(payoff(asian(strike=100.0), scenario()) >= 0.0)
    assert np.all(payoff(asian(strike=100.0, option_type="put"), scenario()) >= 0.0)


# --- domains -------------------------------------------------------------------


def test_geometric_averaging_refuses_a_non_positive_path() -> None:
    dipping = PATHS.copy()
    dipping[0, 2, 0] = 0.0
    with pytest.raises(InstrumentValidationError, match="strictly positive"):
        payoff(asian(averaging_method="geometric"), scenario(states=dipping))


def test_arithmetic_averaging_has_no_positivity_requirement() -> None:
    """Nothing here takes a logarithm, so nothing needs to be refused."""
    dipping = PATHS.copy()
    dipping[0, 2, 0] = 1e-8
    assert np.all(np.isfinite(payoff(asian(), scenario(states=dipping))))


# --- native namespaces ---------------------------------------------------------


@pytest.mark.parametrize("method", ["arithmetic", "geometric"], ids=["arith", "geom"])
@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
def test_numpy_dtype_is_preserved(method: str, dtype_name: str) -> None:
    states = PATHS.astype(dtype_name)
    result = payoff(asian(averaging_method=method), scenario(states=states))
    assert result.dtype == states.dtype


@pytest.mark.parametrize("method", ["arithmetic", "geometric"], ids=["arith", "geom"])
def test_torch_dtype_device_and_tape_are_preserved(method: str) -> None:
    torch = pytest.importorskip("torch", reason="torch not installed")
    states = torch.tensor(PATHS, dtype=torch.float64, requires_grad=True)
    built = Scenario.from_states(
        "ACME", time_grid=torch.tensor(GRID, dtype=torch.float64), states=states
    )
    result = payoff(asian(averaging_method=method, strike=50.0), built)
    assert torch.is_tensor(result) and result.requires_grad
    result.sum().backward()
    assert states.grad is not None and torch.isfinite(states.grad).all()
    np.testing.assert_allclose(
        result.detach().numpy(),
        payoff(asian(averaging_method=method, strike=50.0), scenario()),
    )


def test_torch_cuda_paths_stay_on_the_device() -> None:
    torch = pytest.importorskip("torch", reason="torch not installed")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    built = Scenario.from_states(
        "ACME",
        time_grid=torch.tensor(GRID, dtype=torch.float64, device="cuda"),
        states=torch.tensor(PATHS, dtype=torch.float64, device="cuda"),
    )
    assert payoff(asian(), built).device.type == "cuda"


def test_no_host_staging_in_the_path_payoff(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = pytest.importorskip("torch", reason="torch not installed")
    states = torch.tensor(PATHS, dtype=torch.float64, requires_grad=True)
    built = Scenario.from_states(
        "ACME", time_grid=torch.tensor(GRID, dtype=torch.float64), states=states
    )
    for name in ("numpy", "detach", "cpu", "item", "tolist"):
        monkeypatch.setattr(torch.Tensor, name, forbidden_host_call, raising=True)
    assert payoff(asian(strike=50.0), built).requires_grad


def forbidden_host_call(*_args: object, **_kwargs: object) -> object:
    raise AssertionError("payoff evaluation must not stage through host memory")


@pytest.mark.parametrize("method", ["arithmetic", "geometric"], ids=["arith", "geom"])
def test_jax_values_and_gradients(method: str) -> None:
    jax = pytest.importorskip("jax", reason="jax not installed")
    jax.config.update("jax_enable_x64", True)
    jnp = jax.numpy
    contract = asian(averaging_method=method, strike=50.0)

    def evaluate(states: Any) -> Any:
        built = Scenario.from_states(
            "ACME", time_grid=jnp.asarray(GRID, dtype=jnp.float64), states=states
        )
        return payoff(contract, built).sum()

    states = jnp.asarray(PATHS, dtype=jnp.float64)
    np.testing.assert_allclose(float(evaluate(states)), float(payoff(contract, scenario()).sum()))
    gradient = jax.grad(evaluate)(states)
    assert np.all(np.isfinite(np.asarray(gradient)))


def test_jax_jit_traces_a_path_payoff() -> None:
    """Tracer-safe: the positivity check must not concretize a traced path."""
    jax = pytest.importorskip("jax", reason="jax not installed")
    jax.config.update("jax_enable_x64", True)
    jnp = jax.numpy
    contract = asian(averaging_method="geometric", strike=50.0)

    @jax.jit
    def evaluate(states: Any) -> Any:
        built = Scenario.from_states(
            "ACME", time_grid=jnp.asarray(GRID, dtype=jnp.float64), states=states
        )
        return payoff(contract, built)

    result = evaluate(jnp.asarray(PATHS, dtype=jnp.float64))
    np.testing.assert_allclose(np.asarray(result), payoff(contract, scenario()))


# --- barrier monitoring --------------------------------------------------------

from fast_vollib.instruments import BarrierOption, BarrierType, LookbackOption  # noqa: E402

#: Path 0 rises to 135 then settles at 120; path 1 peaks at 105 and ends at 90.
#: With a barrier at 130 only the first is knocked; with one at 90 only the
#: second is, and only at its last observation -- which the inclusive rule
#: counts and an exclusive one would miss.
BARRIER_PATHS = np.array(
    [
        [[100.0], [120.0], [135.0], [128.0], [120.0]],
        [[100.0], [105.0], [98.0], [95.0], [90.0]],
    ]
)


def barrier_scenario() -> Scenario:
    return Scenario.from_states("ACME", time_grid=GRID, states=BARRIER_PATHS)


def barrier(**overrides: Any) -> BarrierOption:
    call: dict[str, Any] = {
        "underlier": "ACME",
        "option_type": "call",
        "strike": 100.0,
        "barrier": 130.0,
        "barrier_type": "up_and_out",
        "maturity": 1.0,
    }
    call.update(overrides)
    return BarrierOption(**call)


def vanilla_intrinsic(option_type: str = "call", strike: float = 100.0) -> np.ndarray:
    terminal = BARRIER_PATHS[:, -1, 0]
    if option_type == "call":
        return np.maximum(terminal - strike, 0.0)
    return np.maximum(strike - terminal, 0.0)


def test_an_up_barrier_knocks_on_the_path_that_reaches_it() -> None:
    knock_in = payoff(barrier(barrier_type="up_and_in"), barrier_scenario())
    knock_out = payoff(barrier(barrier_type="up_and_out"), barrier_scenario())
    np.testing.assert_allclose(knock_in, [vanilla_intrinsic()[0], 0.0])
    np.testing.assert_allclose(knock_out, [0.0, vanilla_intrinsic()[1]])


def test_a_down_barrier_knocks_on_the_path_that_reaches_it() -> None:
    knock_in = payoff(barrier(barrier=90.0, barrier_type="down_and_in"), barrier_scenario())
    np.testing.assert_allclose(knock_in, vanilla_intrinsic() * np.array([0.0, 1.0]))


@pytest.mark.parametrize(
    ("barrier_type", "level"),
    [("up_and_in", 130.0), ("down_and_in", 90.0)],
    ids=["up", "down"],
)
@pytest.mark.parametrize("option_type", ["call", "put"], ids=["call", "put"])
def test_in_plus_out_reconstructs_the_vanilla_payoff(
    barrier_type: str, level: float, option_type: str
) -> None:
    """The defining identity of a barrier pair, on every path.

    Knock-in and knock-out are complementary by construction, so their sum must
    be the unconditional payoff for any monitoring rule -- which makes this a
    check on the indicator rather than on the rule.
    """
    out_type = barrier_type.replace("_in", "_out")
    common = {"barrier": level, "option_type": option_type, "notional": 3.0}
    knock_in = payoff(barrier(barrier_type=barrier_type, **common), barrier_scenario())
    knock_out = payoff(barrier(barrier_type=out_type, **common), barrier_scenario())
    np.testing.assert_allclose(knock_in + knock_out, 3.0 * vanilla_intrinsic(option_type))


def test_a_touch_exactly_at_the_barrier_counts() -> None:
    """Inclusive on both sides: reaching the level is touching it."""
    exact = np.array([[[100.0], [110.0], [120.0], [110.0], [105.0]]])
    built = Scenario.from_states("ACME", time_grid=GRID, states=exact)
    knocked = payoff(barrier(barrier=120.0, barrier_type="up_and_out"), built)
    np.testing.assert_allclose(knocked, [0.0])
    just_above = payoff(barrier(barrier=120.0 + 1e-9, barrier_type="up_and_out"), built)
    np.testing.assert_allclose(just_above, [5.0])


def test_a_touch_exactly_at_a_down_barrier_counts() -> None:
    """The same inclusive rule downwards, which the up case cannot establish.

    ``min(S) <= barrier`` and ``min(S) < barrier`` differ on exactly one kind of
    path: one that reaches the level and does not pass it.
    """
    exact = np.array([[[100.0], [95.0], [90.0], [100.0], [115.0]]])
    built = Scenario.from_states("ACME", time_grid=GRID, states=exact)
    knocked_out = payoff(barrier(barrier=90.0, barrier_type="down_and_out"), built)
    np.testing.assert_allclose(knocked_out, [0.0])
    knocked_in = payoff(barrier(barrier=90.0, barrier_type="down_and_in"), built)
    np.testing.assert_allclose(knocked_in, [15.0])
    # A hair below the level is not reached, so the contract survives.
    just_below = payoff(barrier(barrier=90.0 - 1e-9, barrier_type="down_and_out"), built)
    np.testing.assert_allclose(just_below, [15.0])


def test_the_first_and_last_observations_are_monitored() -> None:
    """Both endpoints count; excluding either would silently drop a touch."""
    starts_at_barrier = np.array([[[100.0], [95.0], [96.0], [97.0], [98.0]]])
    ends_at_barrier = np.array([[[95.0], [96.0], [97.0], [98.0], [100.0]]])
    for states in (starts_at_barrier, ends_at_barrier):
        built = Scenario.from_states("ACME", time_grid=GRID, states=states)
        contract = barrier(barrier=100.0, barrier_type="up_and_out", strike=50.0)
        np.testing.assert_allclose(payoff(contract, built), [0.0])


def test_monitoring_is_discrete_so_a_coarser_grid_can_miss_a_touch() -> None:
    """Stated as behaviour, not apologized for: the schedule is the contract.

    A grid that does not observe the excursion prices a differently monitored
    instrument. Nothing applies a continuity correction to pretend otherwise.
    """
    contract = barrier(barrier_type="up_and_out")
    observed = payoff(contract, barrier_scenario())
    coarse = Scenario.from_states("ACME", time_grid=[0.0, 1.0], states=BARRIER_PATHS[:, [0, -1], :])
    assert observed[0] == 0.0
    assert payoff(contract, coarse)[0] > 0.0


def test_a_knocked_out_barrier_pays_nothing_rather_than_a_rebate() -> None:
    np.testing.assert_array_equal(
        payoff(barrier(barrier_type="up_and_out"), barrier_scenario())[0], 0.0
    )


@pytest.mark.parametrize("barrier_type", list(BarrierType), ids=lambda b: b.value)
def test_barrier_payoffs_never_exceed_the_vanilla(barrier_type: BarrierType) -> None:
    result = payoff(barrier(barrier_type=barrier_type), barrier_scenario())
    assert np.all(result <= vanilla_intrinsic() + 1e-12)
    assert np.all(result >= 0.0)


# --- lookback extremes ---------------------------------------------------------


def lookback(**overrides: Any) -> LookbackOption:
    call: dict[str, Any] = {
        "underlier": "ACME",
        "option_type": "call",
        "strike": 100.0,
        "strike_convention": "fixed",
        "maturity": 1.0,
    }
    call.update(overrides)
    return LookbackOption(**call)


def test_fixed_lookback_uses_the_running_extreme() -> None:
    highest = BARRIER_PATHS[:, :, 0].max(axis=1)
    lowest = BARRIER_PATHS[:, :, 0].min(axis=1)
    np.testing.assert_allclose(
        payoff(lookback(strike=110.0), barrier_scenario()), np.maximum(highest - 110.0, 0.0)
    )
    np.testing.assert_allclose(
        payoff(lookback(strike=99.0, option_type="put"), barrier_scenario()),
        np.maximum(99.0 - lowest, 0.0),
    )


def test_floating_lookback_settles_against_the_extreme() -> None:
    spot = BARRIER_PATHS[:, :, 0]
    terminal = spot[:, -1]
    floating = lookback(strike=None, strike_convention="floating")
    np.testing.assert_allclose(payoff(floating, barrier_scenario()), terminal - spot.min(axis=1))
    floating_put = lookback(strike=None, strike_convention="floating", option_type="put")
    np.testing.assert_allclose(
        payoff(floating_put, barrier_scenario()), spot.max(axis=1) - terminal
    )


def test_a_floating_lookback_is_never_negative_by_construction() -> None:
    """The terminal value is one of the observations the extreme was taken over."""
    for option_type in ("call", "put"):
        result = payoff(
            lookback(strike=None, strike_convention="floating", option_type=option_type),
            barrier_scenario(),
        )
        assert np.all(result >= 0.0)


def test_the_extremes_include_both_endpoints() -> None:
    """Each extreme is pinned at *each* end, which needs both directions.

    A rising path puts its maximum at the last observation and its minimum at
    the first, so on its own it can only show that the maximum sees the end and
    the minimum sees the start. A falling path establishes the other two;
    without it a running maximum that quietly skipped the valuation date would
    go unnoticed.
    """
    rising = np.array([[[80.0], [85.0], [90.0], [95.0], [100.0]]])
    built = Scenario.from_states("ACME", time_grid=GRID, states=rising)
    np.testing.assert_allclose(payoff(lookback(strike=99.0), built), [1.0])
    np.testing.assert_allclose(payoff(lookback(strike=81.0, option_type="put"), built), [1.0])

    falling = np.array([[[100.0], [95.0], [90.0], [85.0], [80.0]]])
    built = Scenario.from_states("ACME", time_grid=GRID, states=falling)
    # The maximum is now the *first* observation: dropping it would give 95.
    np.testing.assert_allclose(payoff(lookback(strike=99.0), built), [1.0])
    # The minimum is now the *last*: dropping it would give 85.
    np.testing.assert_allclose(payoff(lookback(strike=81.0, option_type="put"), built), [1.0])


def test_each_lookback_extreme_reads_the_whole_path() -> None:
    """A direct comparison, so no one-sided truncation can hide.

    The maximum sits at index 0 and the minimum at index -1, which makes every
    dropped endpoint change the answer.
    """
    path = np.array([[[130.0], [120.0], [125.0], [110.0], [70.0]]])
    built = Scenario.from_states("ACME", time_grid=GRID, states=path)
    spot = path[0, :, 0]
    assert spot.argmax() == 0 and spot.argmin() == len(spot) - 1
    np.testing.assert_allclose(payoff(lookback(strike=100.0), built), [spot.max() - 100.0])
    np.testing.assert_allclose(
        payoff(lookback(strike=100.0, option_type="put"), built), [100.0 - spot.min()]
    )
    np.testing.assert_allclose(
        payoff(lookback(strike=None, strike_convention="floating", option_type="put"), built),
        [spot.max() - spot[-1]],
    )


def test_a_fixed_lookback_call_dominates_the_vanilla() -> None:
    """The best level reached is at least the level at maturity."""
    fixed = payoff(lookback(strike=100.0), barrier_scenario())
    assert np.all(fixed >= vanilla_intrinsic() - 1e-12)


def test_a_flat_path_makes_every_lookback_variant_worthless() -> None:
    flat = np.full((1, 5, 1), 100.0)
    built = Scenario.from_states("ACME", time_grid=GRID, states=flat)
    for kwargs in (
        {"strike": 100.0, "strike_convention": "fixed"},
        {"strike": 100.0, "strike_convention": "fixed", "option_type": "put"},
        {"strike": None, "strike_convention": "floating"},
        {"strike": None, "strike_convention": "floating", "option_type": "put"},
    ):
        np.testing.assert_allclose(payoff(lookback(**kwargs), built), [0.0])


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
def test_barrier_and_lookback_preserve_the_numpy_dtype(dtype_name: str) -> None:
    states = BARRIER_PATHS.astype(dtype_name)
    built = Scenario.from_states("ACME", time_grid=GRID, states=states)
    assert payoff(barrier(), built).dtype == states.dtype
    assert payoff(lookback(), built).dtype == states.dtype


def test_barrier_and_lookback_keep_the_torch_graph() -> None:
    torch = pytest.importorskip("torch", reason="torch not installed")
    states = torch.tensor(BARRIER_PATHS, dtype=torch.float64, requires_grad=True)
    built = Scenario.from_states(
        "ACME", time_grid=torch.tensor(GRID, dtype=torch.float64), states=states
    )
    for contract in (barrier(strike=50.0, barrier_type="up_and_in"), lookback(strike=50.0)):
        result = payoff(contract, built)
        assert result.requires_grad
        result.sum().backward(retain_graph=True)
    assert states.grad is not None and torch.isfinite(states.grad).all()


def test_barrier_and_lookback_trace_under_jax_jit() -> None:
    jax = pytest.importorskip("jax", reason="jax not installed")
    jax.config.update("jax_enable_x64", True)
    jnp = jax.numpy

    for contract in (barrier(), lookback()):

        @jax.jit
        def evaluate(states: Any, contract: Any = contract) -> Any:
            built = Scenario.from_states(
                "ACME", time_grid=jnp.asarray(GRID, dtype=jnp.float64), states=states
            )
            return payoff(contract, built)

        result = evaluate(jnp.asarray(BARRIER_PATHS, dtype=jnp.float64))
        np.testing.assert_allclose(np.asarray(result), payoff(contract, barrier_scenario()))


# --- realized variance ---------------------------------------------------------

from fast_vollib.instruments import VarianceSwap  # noqa: E402


def variance_swap(**overrides: Any) -> VarianceSwap:
    call: dict[str, Any] = {
        "underlier": "ACME",
        "strike_variance": 0.04,
        "maturity": 1.0,
    }
    call.update(overrides)
    return VarianceSwap(**call)


def realized_variance(states: np.ndarray, horizon: float) -> np.ndarray:
    spot = states[:, :, 0]
    return (np.log(spot[:, 1:] / spot[:, :-1]) ** 2).sum(axis=1) / horizon


def test_realized_variance_is_the_sum_of_squared_log_returns_over_the_horizon() -> None:
    expected = realized_variance(PATHS, 1.0)
    np.testing.assert_allclose(payoff(variance_swap(strike_variance=0.0), scenario()), expected)


def test_the_payoff_is_realized_minus_strike_scaled_by_notional() -> None:
    expected = realized_variance(PATHS, 1.0)
    np.testing.assert_allclose(
        payoff(variance_swap(strike_variance=0.05, notional=1_000_000.0), scenario()),
        1_000_000.0 * (expected - 0.05),
    )


def test_the_payoff_is_zero_when_realized_equals_the_strike() -> None:
    realized = float(realized_variance(PATHS[:1], 1.0)[0])
    single = Scenario.from_states("ACME", time_grid=GRID, states=PATHS[:1])
    np.testing.assert_allclose(
        payoff(variance_swap(strike_variance=realized), single), [0.0], atol=1e-15
    )


def test_the_payoff_can_be_negative_and_is_not_clipped() -> None:
    """A swap is two-sided: the short leg has to be able to receive."""
    result = payoff(variance_swap(strike_variance=100.0), scenario())
    assert np.all(result < 0.0)


def test_the_convention_has_no_sample_mean_subtraction() -> None:
    """The contract pays on the sum of squares, not on a variance estimate.

    A trending path makes the two differ: subtracting the sample mean would
    remove the drift's contribution, which the traded convention keeps.
    """
    trending = np.array([[[100.0], [110.0], [121.0], [133.1], [146.41]]])
    built = Scenario.from_states("ACME", time_grid=GRID, states=trending)
    returns = np.log(trending[0, 1:, 0] / trending[0, :-1, 0])
    plain_sum = float((returns**2).sum())
    mean_adjusted = float(((returns - returns.mean()) ** 2).sum())
    assert mean_adjusted < plain_sum
    np.testing.assert_allclose(payoff(variance_swap(strike_variance=0.0), built), [plain_sum / 1.0])


def test_dividing_by_the_year_fraction_already_annualizes() -> None:
    """No 252 anywhere: applying one as well would annualize twice.

    The same daily returns over a shorter year fraction are worth more per
    year, by exactly the ratio of the horizons.
    """
    states = PATHS[:1]
    short = Scenario.from_states("ACME", time_grid=[0.0, 0.25, 0.5, 0.75, 1.0], states=states)
    quarter = Scenario.from_states(
        "ACME", time_grid=[0.0, 0.0625, 0.125, 0.1875, 0.25], states=states
    )
    annual = payoff(variance_swap(strike_variance=0.0), short)
    quarterly = payoff(variance_swap(strike_variance=0.0, maturity=0.25), quarter)
    np.testing.assert_allclose(quarterly, 4.0 * annual)


def test_realized_variance_refuses_a_non_positive_path() -> None:
    dipping = PATHS.copy()
    dipping[0, 3, 0] = -1.0
    with pytest.raises(InstrumentValidationError, match="strictly positive"):
        payoff(variance_swap(), scenario(states=dipping))


def test_the_first_state_is_part_of_the_first_return() -> None:
    """S_0 is not a fixing, but it is the denominator of the first log return."""
    moved_start = PATHS[:1].copy()
    moved_start[0, 0, 0] = 50.0
    built = Scenario.from_states("ACME", time_grid=GRID, states=moved_start)
    assert not np.allclose(
        payoff(variance_swap(), built), payoff(variance_swap(), scenario(states=PATHS[:1]))
    )


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
def test_variance_preserves_the_numpy_dtype(dtype_name: str) -> None:
    states = PATHS.astype(dtype_name)
    built = Scenario.from_states("ACME", time_grid=np.array(GRID, dtype=dtype_name), states=states)
    assert payoff(variance_swap(), built).dtype == states.dtype


def test_variance_keeps_the_torch_graph_and_is_smooth() -> None:
    """The only payoff here with a useful gradient everywhere on a positive path."""
    torch = pytest.importorskip("torch", reason="torch not installed")
    states = torch.tensor(PATHS, dtype=torch.float64, requires_grad=True)
    built = Scenario.from_states(
        "ACME", time_grid=torch.tensor(GRID, dtype=torch.float64), states=states
    )
    result = payoff(variance_swap(), built)
    assert result.requires_grad
    result.sum().backward()
    assert states.grad is not None
    assert torch.isfinite(states.grad).all()
    assert torch.any(states.grad != 0.0)


def test_variance_traces_under_jax_jit() -> None:
    jax = pytest.importorskip("jax", reason="jax not installed")
    jax.config.update("jax_enable_x64", True)
    jnp = jax.numpy

    @jax.jit
    def evaluate(states: Any) -> Any:
        built = Scenario.from_states(
            "ACME", time_grid=jnp.asarray(GRID, dtype=jnp.float64), states=states
        )
        return payoff(variance_swap(), built)

    result = evaluate(jnp.asarray(PATHS, dtype=jnp.float64))
    np.testing.assert_allclose(np.asarray(result), payoff(variance_swap(), scenario()))


# --- what the module docstring claims about host memory ------------------------


def test_the_arithmetic_routes_touch_the_host_not_at_all(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No computed value stages through host memory, on any path payoff."""
    torch = pytest.importorskip("torch", reason="torch not installed")
    states = torch.tensor(BARRIER_PATHS, dtype=torch.float64, requires_grad=True)
    built = Scenario.from_states(
        "ACME", time_grid=torch.tensor(GRID, dtype=torch.float64), states=states
    )
    contracts = [
        asian(strike=50.0),
        barrier(strike=50.0, barrier_type="up_and_in"),
        lookback(strike=50.0),
        lookback(strike=None, strike_convention="floating"),
    ]
    for name in ("numpy", "detach", "cpu", "item", "tolist"):
        monkeypatch.setattr(torch.Tensor, name, forbidden_host_call, raising=True)
    for contract in contracts:
        assert payoff(contract, built).requires_grad, type(contract).__name__


def test_the_positivity_check_reads_one_scalar_and_nothing_more(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Geometric averaging and realized variance do read a flag back; only that.

    The module docstring says so rather than claiming they touch nothing. What
    must stay true is that the read is a single zero-dimensional reduction — a
    decision about whether to raise — and never the path itself.
    """
    torch = pytest.importorskip("torch", reason="torch not installed")
    states = torch.tensor(PATHS, dtype=torch.float64, requires_grad=True)
    built = Scenario.from_states(
        "ACME", time_grid=torch.tensor(GRID, dtype=torch.float64), states=states
    )
    shapes: list[tuple[int, ...]] = []
    original = torch.Tensor.detach

    def recording(self: Any) -> Any:
        shapes.append(tuple(self.shape))
        return original(self)

    monkeypatch.setattr(torch.Tensor, "detach", recording, raising=True)
    for contract in (asian(averaging_method="geometric", strike=50.0), variance_swap()):
        shapes.clear()
        result = payoff(contract, built)
        assert result.requires_grad
        assert shapes == [()], (type(contract).__name__, shapes)
