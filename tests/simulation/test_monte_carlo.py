"""The Monte Carlo engine: support, validation, routing, estimator, and oracles."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from fast_vollib.instruments import (
    AsianOption,
    Asset,
    BarrierOption,
    BinaryOption,
    EuropeanOption,
    Forward,
    Future,
    LookbackOption,
    MissingMarketInputError,
    UnsupportedInstrumentError,
    VanillaMarketInputs,
    VarianceSwap,
    capabilities,
    price_instrument,
)
from fast_vollib.processes import GBM
from fast_vollib.simulation import (
    MCResult,
    MonteCarloEngine,
    SimulationValidationError,
    UnsupportedProcessError,
)

# Chosen so that the log-drift ``a = r - sigma^2 / 2`` is not zero. With
# ``r = 0.02`` and ``sigma = 0.2`` it is zero to machine precision, which
# silently removes the drift term from the geometric-Asian and variance-swap
# closed forms and makes an at-the-money digital call and put worth exactly the
# same -- so those oracles could not see a sign error in the drift correction or
# a call/put swap. ``a = 0.02 - 0.03125 = -0.01125`` here.
SPOT, STRIKE, RATE, VOLATILITY, MATURITY = 100.0, 100.0, 0.02, 0.25, 1.0

#: The log-drift the oracles below depend on, asserted rather than assumed.
LOG_DRIFT = RATE - 0.5 * VOLATILITY**2

CALL = EuropeanOption(underlier="ACME", option_type="call", strike=STRIKE, maturity=MATURITY)
PUT = EuropeanOption(underlier="ACME", option_type="put", strike=STRIKE, maturity=MATURITY)
FORWARD = Forward(underlier="ACME", delivery_price=STRIKE, maturity=MATURITY)


def market(**overrides: Any) -> VanillaMarketInputs:
    call: dict[str, Any] = {"underlying": SPOT, "rate": RATE, "volatility": VOLATILITY}
    call.update(overrides)
    return VanillaMarketInputs(**call)


def risk_neutral(volatility: float = VOLATILITY) -> GBM:
    return GBM.risk_neutral(rate=RATE, volatility=volatility)


def run(
    instrument: Any = CALL, engine: MonteCarloEngine | None = None, **overrides: Any
) -> MCResult:
    call: dict[str, Any] = {
        "process": risk_neutral(),
        "n_paths": 40_000,
        "n_steps": 1,
        "rng": 20260825,
    }
    call.update(overrides)
    inputs = call.pop("market", market())
    return (engine or MonteCarloEngine()).price(instrument, inputs, **call)


def analytic(instrument: EuropeanOption) -> float:
    return float(price_instrument(instrument, market(), model="black_scholes")[0])


# --- the support table --------------------------------------------------------


@pytest.mark.parametrize(
    "cls",
    [
        EuropeanOption,
        Forward,
        BinaryOption,
        AsianOption,
        BarrierOption,
        LookbackOption,
        VarianceSwap,
    ],
    ids=["european", "forward", "binary", "asian", "barrier", "lookback", "variance"],
)
def test_supported_types(cls: type) -> None:
    assert MonteCarloEngine().supports(cls) is True


@pytest.mark.parametrize("cls", [Future, Asset, dict], ids=["future", "asset", "foreign"])
def test_unsupported_types(cls: type) -> None:
    assert MonteCarloEngine().supports(cls) is False


def test_an_instance_also_needs_a_positive_maturity() -> None:
    """The type-level and instance-level questions have different answers."""
    engine = MonteCarloEngine()
    expiring = EuropeanOption(underlier="ACME", option_type="call", strike=STRIKE, maturity=0.0)
    assert engine.supports(EuropeanOption) is True
    assert engine.supports(expiring) is False
    assert engine.supports(CALL) is True


def test_a_subclass_is_not_supported_by_inheritance() -> None:
    class Exotic(EuropeanOption):
        pass

    assert MonteCarloEngine().supports(Exotic) is False


def test_capability_simulate_agrees_with_the_engine() -> None:
    """The type-level summary and the engine's own answer must not disagree."""
    from fast_vollib.instruments import instrument_types

    engine = MonteCarloEngine()
    for info in instrument_types().values():
        assert info.capabilities.simulate == engine.supports(info.python_type), info.type_id


def test_a_future_is_refused_and_the_message_says_why() -> None:
    future = Future(underlier="ACME", contract_price=STRIKE, maturity=MATURITY)
    with pytest.raises(UnsupportedInstrumentError) as excinfo:
        run(future)
    message = str(excinfo.value)
    assert "variation margin" in message
    assert "Forward" in message


def test_a_futures_terminal_payoff_still_works() -> None:
    """Refusing to price it must not break the payoff it does have."""
    from fast_vollib.instruments import payoff

    future = Future(underlier="ACME", contract_price=STRIKE, maturity=MATURITY)
    np.testing.assert_array_equal(payoff(future, np.array([120.0])), np.array([20.0]))


def test_a_zero_maturity_contract_is_refused_and_points_at_the_payoff() -> None:
    expiring = EuropeanOption(underlier="ACME", option_type="call", strike=STRIKE, maturity=0.0)
    with pytest.raises(UnsupportedInstrumentError) as excinfo:
        run(expiring)
    assert "payoff(instrument, terminal_state)" in str(excinfo.value)


def test_a_zero_maturity_contract_is_still_a_valid_contract() -> None:
    from fast_vollib.instruments import payoff

    expiring = EuropeanOption(underlier="ACME", option_type="call", strike=STRIKE, maturity=0.0)
    np.testing.assert_array_equal(payoff(expiring, np.array([120.0])), np.array([20.0]))


def test_an_unsupported_type_lists_the_supported_ones() -> None:
    with pytest.raises(
        UnsupportedInstrumentError,
        match="AsianOption, BarrierOption, BinaryOption, EuropeanOption, Forward",
    ):
        run(Asset(identifier="ACME", asset_class="equity"))


# --- input validation ---------------------------------------------------------


def test_exactly_one_grid_source_is_required() -> None:
    with pytest.raises(SimulationValidationError, match="exactly one"):
        run(n_steps=None)
    with pytest.raises(SimulationValidationError, match="exactly one"):
        run(n_steps=4, time_grid=[0.0, MATURITY])


@pytest.mark.parametrize("bad", [0, -1, 2.5, True], ids=["zero", "negative", "float", "bool"])
def test_n_steps_must_be_a_positive_integer(bad: Any) -> None:
    with pytest.raises(SimulationValidationError, match="n_steps must be a positive integer"):
        run(n_steps=bad)


def test_an_explicit_grid_must_end_at_maturity() -> None:
    with pytest.raises(SimulationValidationError, match="ends at t="):
        run(n_steps=None, time_grid=[0.0, 0.5])
    with pytest.raises(SimulationValidationError, match="ends at t="):
        run(n_steps=None, time_grid=[0.0, 0.5, 2.0])


def test_an_explicit_grid_at_maturity_is_accepted() -> None:
    assert run(n_steps=None, time_grid=[0.0, 0.25, 0.75, MATURITY]).n_paths == 40_000


def test_a_built_grid_lands_exactly_on_maturity(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}
    from fast_vollib.simulation import monte_carlo

    original = monte_carlo.simulate

    def capturing(*args: Any, **kwargs: Any) -> Any:
        seen["grid"] = kwargs["time_grid"]
        return original(*args, **kwargs)

    monkeypatch.setattr(monte_carlo, "simulate", capturing)
    run(n_steps=8)
    grid = seen["grid"]
    assert len(grid) == 9
    assert float(grid[0]) == 0.0
    assert float(grid[-1]) == MATURITY


@pytest.mark.parametrize(
    "bad", [1, 0, -2, 2.0, True], ids=["one", "zero", "negative", "float", "bool"]
)
def test_ordinary_sampling_needs_at_least_two_paths(bad: Any) -> None:
    with pytest.raises(SimulationValidationError):
        run(n_paths=bad)


@pytest.mark.parametrize("bad", [2, 3, 5], ids=["one-pair", "odd", "odd-larger"])
def test_antithetic_sampling_needs_two_full_pairs(bad: int) -> None:
    with pytest.raises(SimulationValidationError, match="at least 4"):
        run(engine=MonteCarloEngine(antithetic=True), n_paths=bad)


@pytest.mark.parametrize("bad", [1, "false", None], ids=["integer", "string", "none"])
def test_engine_antithetic_must_be_a_boolean(bad: Any) -> None:
    with pytest.raises(SimulationValidationError, match="antithetic must be a bool"):
        MonteCarloEngine(antithetic=bad)


def test_a_missing_market_field_is_named() -> None:
    with pytest.raises(MissingMarketInputError, match="underlying"):
        run(market=VanillaMarketInputs(underlying=None, rate=RATE))  # type: ignore[arg-type]
    with pytest.raises(MissingMarketInputError, match="rate"):
        run(market=VanillaMarketInputs(underlying=SPOT, rate=None))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "value", [np.array([SPOT, SPOT]), [SPOT], (SPOT,)], ids=["array", "list", "tuple"]
)
def test_a_non_scalar_market_is_refused(value: Any) -> None:
    with pytest.raises(SimulationValidationError, match="must be a scalar"):
        run(market=market(underlying=value))


def test_a_zero_dimensional_array_counts_as_a_scalar() -> None:
    assert math.isfinite(run(market=market(underlying=np.array(SPOT))).price)


def test_a_non_positive_spot_is_refused() -> None:
    with pytest.raises(SimulationValidationError, match="strictly positive spot"):
        run(market=market(underlying=-1.0))


def test_a_non_finite_rate_is_refused() -> None:
    with pytest.raises(SimulationValidationError, match="market.rate must be finite"):
        run(market=market(rate=float("nan")))


class _TwoFactor:
    state_names = ("spot", "variance")

    def params(self) -> dict[str, Any]:
        return {}

    def sample(self, **_kwargs: Any) -> Any:  # pragma: no cover - never reached
        raise AssertionError("must not be sampled")


def test_a_multi_state_process_is_refused_with_a_reason() -> None:
    with pytest.raises(UnsupportedProcessError) as excinfo:
        run(process=_TwoFactor())
    message = str(excinfo.value)
    assert "('spot', 'variance')" in message
    assert "simulate() itself is not restricted" in message


def test_an_unusable_rng_is_refused_before_sampling(monkeypatch: pytest.MonkeyPatch) -> None:
    from fast_vollib.simulation import monte_carlo

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("no paths may be drawn for an invalid request")

    monkeypatch.setattr(monte_carlo, "simulate", forbidden)
    with pytest.raises(SimulationValidationError, match="numpy.random.Generator"):
        run(rng="seed")


def test_no_request_that_fails_validation_reaches_the_sampler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fast_vollib.simulation import monte_carlo

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("no paths may be drawn for an invalid request")

    monkeypatch.setattr(monte_carlo, "simulate", forbidden)
    invalid: list[dict[str, Any]] = [
        {"n_paths": 1},
        {"n_steps": 0},
        {"n_steps": None},
        {"n_steps": None, "time_grid": [0.0, 0.5]},
        {"market": market(underlying=np.zeros(3))},
        {"market": market(rate=float("inf"))},
        {"process": _TwoFactor()},
    ]
    for call in invalid:
        with pytest.raises((SimulationValidationError, UnsupportedProcessError)):
            run(**call)
    for instrument in (
        Future(underlier="ACME", contract_price=STRIKE, maturity=MATURITY),
        EuropeanOption(underlier="ACME", option_type="call", strike=STRIKE, maturity=0.0),
    ):
        with pytest.raises(UnsupportedInstrumentError):
            run(instrument)


# --- routing discipline -------------------------------------------------------


def test_the_underlier_is_simulated_once_and_the_payoff_dispatched_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fast_vollib.simulation import monte_carlo

    counts = {"simulate": 0, "payoff": 0}
    original_simulate, original_payoff = monte_carlo.simulate, monte_carlo.payoff

    def counting_simulate(*args: Any, **kwargs: Any) -> Any:
        counts["simulate"] += 1
        return original_simulate(*args, **kwargs)

    def counting_payoff(*args: Any, **kwargs: Any) -> Any:
        counts["payoff"] += 1
        return original_payoff(*args, **kwargs)

    monkeypatch.setattr(monte_carlo, "simulate", counting_simulate)
    monkeypatch.setattr(monte_carlo, "payoff", counting_payoff)
    run(n_paths=256)
    assert counts == {"simulate": 1, "payoff": 1}


def test_a_terminal_contract_is_evaluated_at_the_horizon(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Terminal routing passes an array of terminal states, never the scenario."""
    from fast_vollib.simulation import monte_carlo

    seen: dict[str, Any] = {}
    original = monte_carlo.payoff

    def capturing(instrument: Any, state: Any) -> Any:
        seen["state"] = state
        return original(instrument, state)

    monkeypatch.setattr(monte_carlo, "payoff", capturing)
    run(n_paths=64)
    assert np.shape(seen["state"]) == (64,)


def test_the_simulated_underlier_is_the_contracts_own() -> None:
    other = EuropeanOption(underlier="OTHER", option_type="call", strike=STRIKE, maturity=MATURITY)
    assert run(other).price >= 0.0


# --- what the engine reads, and what it refuses to read -----------------------


def test_market_volatility_is_ignored_because_the_process_owns_it() -> None:
    """Two disagreeing volatilities must not silently pick one."""
    quiet = run(market=market(volatility=0.01), rng=11)
    loud = run(market=market(volatility=5.0), rng=11)
    assert quiet.price == loud.price


def test_market_volatility_is_not_even_required() -> None:
    assert math.isfinite(run(market=VanillaMarketInputs(underlying=SPOT, rate=RATE)).price)


def test_the_rate_discounts_and_does_not_become_a_drift() -> None:
    """A physical-measure process discounted at r is not a risk-neutral price."""
    physical = run(process=GBM(0.25, VOLATILITY), n_paths=100_000, rng=5)
    neutral = run(process=risk_neutral(), n_paths=100_000, rng=5)
    assert physical.price - neutral.price > 5.0 * (physical.stderr + neutral.stderr)


def test_the_dividend_yield_on_the_market_is_not_consumed() -> None:
    with_yield = run(market=market(dividend_yield=0.05), rng=13)
    without = run(market=market(dividend_yield=None), rng=13)
    assert with_yield.price == without.price


def test_an_analytic_request_for_a_forward_still_raises() -> None:
    """The engine's existence must not turn into a fallback for the adapters."""
    with pytest.raises(UnsupportedInstrumentError, match="no analytic pricing kernel"):
        price_instrument(FORWARD, market(), model="black_scholes")


# --- the estimator ------------------------------------------------------------


def test_ordinary_estimator_matches_the_textbook_formula() -> None:
    samples = np.array([1.0, 2.0, 3.0, 4.0, 6.0])
    result = MonteCarloEngine()._estimate(samples, n_paths=5, return_native=False)
    assert result.price == pytest.approx(3.2)
    assert result.stderr == pytest.approx(np.std(samples, ddof=1) / math.sqrt(5))
    assert result.effective_samples == 5
    assert result.n_paths == 5


def test_antithetic_estimator_averages_the_pairs_first() -> None:
    # The pairs must not come out equal: [1,2,3,9,8,7] averages to (5,5,5), and
    # a zero spread makes the standard-error half of this test vacuous.
    samples = np.array([1.0, 2.0, 3.0, 11.0, 6.0, 4.0])
    result = MonteCarloEngine(antithetic=True)._estimate(samples, n_paths=6, return_native=False)
    pairs = np.array([(1.0 + 11.0) / 2, (2.0 + 6.0) / 2, (3.0 + 4.0) / 2])
    assert np.std(pairs, ddof=1) > 0.0, "a degenerate fixture cannot check a standard error"
    assert result.price == pytest.approx(pairs.mean())
    assert result.stderr == pytest.approx(np.std(pairs, ddof=1) / math.sqrt(3))
    assert result.effective_samples == 3
    assert result.n_paths == 6


def test_antithetic_effective_samples_is_half_the_paths() -> None:
    result = run(engine=MonteCarloEngine(antithetic=True), n_paths=4_000)
    assert result.n_paths == 4_000
    assert result.effective_samples == 2_000


def test_a_constant_payoff_has_zero_standard_error() -> None:
    result = MonteCarloEngine()._estimate(np.full(8, 2.5), n_paths=8, return_native=False)
    assert result.price == pytest.approx(2.5)
    assert result.stderr == pytest.approx(0.0)


# --- Tier 2 oracles -----------------------------------------------------------


@pytest.mark.parametrize("option", [CALL, PUT], ids=["call", "put"])
def test_european_monte_carlo_matches_the_analytic_kernel(option: EuropeanOption) -> None:
    """Under an explicitly risk-neutral GBM, and only then."""
    result = run(option, n_paths=100_000, rng=31337)
    exact = analytic(option)
    assert abs(result.price - exact) <= 5.0 * result.stderr + 1e-12


def test_a_multi_step_grid_does_not_change_the_european_answer() -> None:
    """Exact sampling has no discretization bias, so more steps is only noise."""
    exact = analytic(CALL)
    for steps in (1, 4, 64):
        result = run(n_paths=50_000, n_steps=steps, rng=99)
        assert abs(result.price - exact) <= 5.0 * result.stderr + 1e-12, steps


def test_forward_monte_carlo_matches_its_discounted_expectation() -> None:
    result = run(FORWARD, n_paths=100_000, rng=808)
    exact = math.exp(-RATE * MATURITY) * (SPOT * math.exp(RATE * MATURITY) - STRIKE)
    assert abs(result.price - exact) <= 5.0 * result.stderr + 1e-12


def test_a_deterministic_process_prices_exactly() -> None:
    """Zero volatility removes the noise, so the answer is the closed form."""
    result = run(FORWARD, process=GBM.risk_neutral(rate=RATE, volatility=0.0), n_paths=4, n_steps=3)
    exact = math.exp(-RATE * MATURITY) * (SPOT * math.exp(RATE * MATURITY) - STRIKE)
    assert result.price == pytest.approx(exact, rel=1e-12)
    assert result.stderr == pytest.approx(0.0, abs=1e-12)


def test_put_call_parity_holds_pathwise_across_one_simulation_seed() -> None:
    """C - P is the forward on the same paths, exactly, not just in the limit.

    Comparing against the closed-form parity value would only hold to sampling
    error, because it assumes the sample mean of the terminal spot equals its
    expectation. The pathwise identity holds sample by sample, so this is an
    exact check on the routing and the discounting.
    """
    call = run(CALL, n_paths=50_000, rng=4242)
    put = run(PUT, n_paths=50_000, rng=4242)
    forward = run(FORWARD, n_paths=50_000, rng=4242)
    assert call.price - put.price == pytest.approx(forward.price, rel=1e-12)


# --- output form --------------------------------------------------------------


def test_the_default_result_is_python_floats() -> None:
    result = run(n_paths=1_000)
    assert isinstance(result.price, float)
    assert isinstance(result.stderr, float)


def test_the_result_is_frozen() -> None:
    import dataclasses

    result = run(n_paths=1_000)
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.price = 0.0  # type: ignore[misc]


def test_capabilities_report_simulation_autodiff_only_for_installed_backends() -> None:
    import importlib.util

    caps = capabilities(EuropeanOption)
    for name in ("torch", "jax"):
        installed = importlib.util.find_spec(name) is not None
        assert caps.supports_simulation_autodiff(name) is installed
    assert caps.supports_simulation_autodiff("numpy") is False


# --- native routes and gradients ----------------------------------------------


def torch_inputs(spot: float = SPOT) -> Any:
    torch = pytest.importorskip("torch", reason="torch not installed")
    return torch, {
        "spot": torch.tensor(spot, dtype=torch.float64, requires_grad=True),
        "rate": torch.tensor(RATE, dtype=torch.float64, requires_grad=True),
        "drift": torch.tensor(RATE, dtype=torch.float64, requires_grad=True),
        "volatility": torch.tensor(VOLATILITY, dtype=torch.float64, requires_grad=True),
    }


def test_torch_returns_a_native_result_that_keeps_its_graph() -> None:
    torch, values = torch_inputs()
    result = run(
        market=VanillaMarketInputs(underlying=values["spot"], rate=values["rate"]),
        process=GBM(values["drift"], values["volatility"]),
        n_paths=512,
        n_steps=4,
        rng=torch.Generator().manual_seed(3),
        return_native=True,
    )
    assert torch.is_tensor(result.price) and torch.is_tensor(result.stderr)
    assert result.price.dtype == torch.float64
    assert result.price.requires_grad


def test_torch_gradients_are_finite_for_all_four_advertised_inputs() -> None:
    torch, values = torch_inputs()
    result = run(
        market=VanillaMarketInputs(underlying=values["spot"], rate=values["rate"]),
        process=GBM(values["drift"], values["volatility"]),
        n_paths=2_048,
        n_steps=4,
        rng=11,
        return_native=True,
    )
    result.price.backward()
    for name, tensor in values.items():
        assert tensor.grad is not None, name
        assert torch.isfinite(tensor.grad).all(), name


def test_torch_gradcheck_on_a_smooth_route() -> None:
    """A forward has no kink anywhere, so finite differences are meaningful.

    The seed is an integer, so every function evaluation rebuilds the same
    local generator and the perturbed runs share their random numbers. Without
    that, gradcheck would be differentiating sampling noise.
    """
    torch = pytest.importorskip("torch", reason="torch not installed")

    def priced(spot: Any, rate: Any, drift: Any, volatility: Any) -> Any:
        return (
            MonteCarloEngine()
            .price(
                FORWARD,
                VanillaMarketInputs(underlying=spot, rate=rate),
                process=GBM(drift, volatility),
                n_paths=64,
                n_steps=4,
                rng=2024,
                return_native=True,
            )
            .price
        )

    arguments = tuple(
        torch.tensor(value, dtype=torch.float64, requires_grad=True)
        for value in (SPOT, RATE, RATE, VOLATILITY)
    )
    assert torch.autograd.gradcheck(priced, arguments, eps=1e-6, atol=1e-6, rtol=1e-4)


def test_torch_gradcheck_on_a_deep_in_the_money_option() -> None:
    """Every path finishes in the money, so the payoff kink is never touched."""
    torch = pytest.importorskip("torch", reason="torch not installed")
    deep = EuropeanOption(underlier="ACME", option_type="call", strike=1.0, maturity=MATURITY)

    def priced(spot: Any, rate: Any, drift: Any, volatility: Any) -> Any:
        return (
            MonteCarloEngine()
            .price(
                deep,
                VanillaMarketInputs(underlying=spot, rate=rate),
                process=GBM(drift, volatility),
                n_paths=64,
                n_steps=2,
                rng=77,
                return_native=True,
            )
            .price
        )

    arguments = tuple(
        torch.tensor(value, dtype=torch.float64, requires_grad=True)
        for value in (SPOT, RATE, RATE, 0.15)
    )
    assert torch.autograd.gradcheck(priced, arguments, eps=1e-6, atol=1e-6, rtol=1e-4)


def test_torch_spot_gradient_matches_the_pathwise_formula() -> None:
    """The forward's pathwise delta is exp(-rT) * mean(S_T) / S_0, exactly.

    Not the closed-form ``exp((mu - r) T)``: that is what the estimator
    converges to, and comparing against it would test the sample size rather
    than the derivative. Re-simulating with the same seed gives the very paths
    the gradient was taken through, so the identity is exact.
    """
    torch, values = torch_inputs()
    from fast_vollib.simulation import simulate

    result = run(
        FORWARD,
        market=VanillaMarketInputs(underlying=values["spot"], rate=values["rate"]),
        process=GBM(values["drift"], values["volatility"]),
        n_paths=1_024,
        n_steps=4,
        rng=6,
        return_native=True,
    )
    result.price.backward()

    # Re-simulated in the *same* backend: a seed produces a torch stream here
    # and a NumPy one there, and the library never claims the two agree.
    same_paths = simulate(
        "ACME",
        GBM(values["drift"].detach(), values["volatility"].detach()),
        initial_state=values["spot"].detach(),
        time_grid=torch.linspace(0.0, MATURITY, 5, dtype=torch.float64),
        n_paths=1_024,
        rng=6,
    )
    expected = math.exp(-RATE * MATURITY) * float(same_paths.terminal().mean()) / SPOT
    assert float(values["spot"].grad) == pytest.approx(expected, rel=1e-9)


def test_host_output_ends_the_graph() -> None:
    torch, values = torch_inputs()
    result = run(
        market=VanillaMarketInputs(underlying=values["spot"], rate=values["rate"]),
        process=GBM(values["drift"], values["volatility"]),
        n_paths=256,
        n_steps=2,
        rng=8,
        return_native=False,
    )
    assert isinstance(result.price, float)
    assert values["spot"].grad is None


def test_no_host_staging_before_the_final_extraction(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = pytest.importorskip("torch", reason="torch not installed")

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("the numerical path must stay on the device")

    values = {
        "spot": torch.tensor(SPOT, dtype=torch.float64, requires_grad=True),
        "rate": torch.tensor(RATE, dtype=torch.float64),
    }
    for name in ("numpy", "cpu", "tolist"):
        monkeypatch.setattr(torch.Tensor, name, forbidden, raising=True)
    result = run(
        market=VanillaMarketInputs(underlying=values["spot"], rate=values["rate"]),
        process=risk_neutral(),
        n_paths=256,
        n_steps=2,
        rng=torch.Generator().manual_seed(4),
        return_native=True,
    )
    assert result.price.requires_grad


def test_jax_gradients_use_common_random_numbers() -> None:
    jax = pytest.importorskip("jax", reason="jax not installed")
    jax.config.update("jax_enable_x64", True)
    jnp = jax.numpy
    key = jax.random.key(19)

    def priced(spot: Any, rate: Any, drift: Any, volatility: Any) -> Any:
        return (
            MonteCarloEngine()
            .price(
                FORWARD,
                VanillaMarketInputs(underlying=spot, rate=rate),
                process=GBM(drift, volatility),
                n_paths=256,
                n_steps=4,
                rng=key,
                return_native=True,
            )
            .price
        )

    arguments = tuple(jnp.float64(value) for value in (SPOT, RATE, RATE, VOLATILITY))
    for argnum in range(4):
        gradient = jax.grad(priced, argnums=argnum)(*arguments)
        assert jnp.isfinite(gradient), argnum

    # The immutable key means the same draws on both sides of a difference, so
    # a finite difference measures the derivative rather than sampling noise.
    step = 1e-6
    bumped = (arguments[0] + step,) + arguments[1:]
    lowered = (arguments[0] - step,) + arguments[1:]
    numeric = (priced(*bumped) - priced(*lowered)) / (2 * step)
    assert float(jax.grad(priced, argnums=0)(*arguments)) == pytest.approx(float(numeric), rel=1e-6)


def test_jax_result_is_native_and_traceable() -> None:
    jax = pytest.importorskip("jax", reason="jax not installed")
    jax.config.update("jax_enable_x64", True)
    jnp = jax.numpy
    result = run(
        market=VanillaMarketInputs(underlying=jnp.float64(SPOT), rate=jnp.float64(RATE)),
        process=GBM(jnp.float64(RATE), jnp.float64(VOLATILITY)),
        n_paths=512,
        n_steps=2,
        rng=jax.random.key(1),
        return_native=True,
    )
    assert isinstance(result.price, jax.Array)
    assert result.price.dtype == jnp.float64


def test_a_mixed_namespace_request_is_refused() -> None:
    torch = pytest.importorskip("torch", reason="torch not installed")
    pytest.importorskip("jax", reason="jax not installed")
    import jax.numpy as jnp

    with pytest.raises(SimulationValidationError, match="more than one array namespace"):
        run(
            market=VanillaMarketInputs(
                underlying=torch.tensor(SPOT, dtype=torch.float64), rate=jnp.float64(RATE)
            ),
            process=risk_neutral(),
            n_paths=64,
            n_steps=2,
            rng=0,
        )


def test_the_rate_participates_in_backend_inference() -> None:
    """A native rate alone is enough to select a backend."""
    torch = pytest.importorskip("torch", reason="torch not installed")
    result = run(
        market=VanillaMarketInputs(underlying=SPOT, rate=torch.tensor(RATE, dtype=torch.float64)),
        process=risk_neutral(),
        n_paths=256,
        n_steps=2,
        rng=torch.Generator().manual_seed(1),
        return_native=True,
    )
    assert torch.is_tensor(result.price)


# --- the digital slice --------------------------------------------------------

DIGITAL_CALL = BinaryOption(
    underlier="ACME", option_type="call", strike=STRIKE, maturity=MATURITY, cash_amount=1.0
)
DIGITAL_PUT = BinaryOption(
    underlier="ACME", option_type="put", strike=STRIKE, maturity=MATURITY, cash_amount=1.0
)


def cash_or_nothing(
    option: BinaryOption,
    *,
    spot: float = SPOT,
    rate: float = RATE,
    volatility: float = VOLATILITY,
) -> float:
    """Independent closed form for a cash-or-nothing digital under Black-Scholes.

    ``C = X e^{-rT} N(d2)`` and ``P = X e^{-rT} N(-d2)``, with the usual
    ``d2``. Written out here rather than taken from the library's kernels: an
    oracle that shares an implementation with the thing it checks is not an
    oracle.
    """
    from scipy.special import ndtr

    d2 = (math.log(SPOT / option.strike) + (RATE - 0.5 * VOLATILITY**2) * MATURITY) / (
        VOLATILITY * math.sqrt(MATURITY)
    )
    sign = 1.0 if option.option_type.value == "call" else -1.0
    return option.cash_amount * math.exp(-RATE * MATURITY) * float(ndtr(sign * d2))


@pytest.mark.parametrize("digital", [DIGITAL_CALL, DIGITAL_PUT], ids=["call", "put"])
def test_digital_monte_carlo_matches_the_closed_form(digital: BinaryOption) -> None:
    result = run(digital, n_paths=100_000, rng=515)
    exact = cash_or_nothing(digital)
    assert abs(result.price - exact) <= 5.0 * result.stderr + 1e-12


def test_a_digital_call_and_put_sum_to_the_discount_factor() -> None:
    """On one set of paths, exactly one of the pair pays unless S_T is the strike."""
    call = run(DIGITAL_CALL, n_paths=20_000, rng=616)
    put = run(DIGITAL_PUT, n_paths=20_000, rng=616)
    assert call.price + put.price == pytest.approx(math.exp(-RATE * MATURITY), rel=1e-12)


def test_a_digital_with_a_cash_amount_scales_linearly() -> None:
    unit = run(DIGITAL_CALL, n_paths=20_000, rng=717)
    ten = run(
        BinaryOption(
            underlier="ACME",
            option_type="call",
            strike=STRIKE,
            maturity=MATURITY,
            cash_amount=10.0,
        ),
        n_paths=20_000,
        rng=717,
    )
    assert ten.price == pytest.approx(10.0 * unit.price, rel=1e-12)


def test_a_zero_maturity_digital_is_refused_but_keeps_its_payoff() -> None:
    from fast_vollib.instruments import payoff

    expiring = BinaryOption(
        underlier="ACME", option_type="call", strike=STRIKE, maturity=0.0, cash_amount=5.0
    )
    with pytest.raises(UnsupportedInstrumentError, match="payoff\\(instrument, terminal_state\\)"):
        run(expiring)
    np.testing.assert_array_equal(payoff(expiring, np.array([120.0])), np.array([5.0]))


def test_a_digital_keeps_a_graph_whose_spot_gradient_is_zero() -> None:
    """Tape retention is not the same claim as a useful Greek."""
    torch, values = torch_inputs()
    result = run(
        DIGITAL_CALL,
        market=VanillaMarketInputs(underlying=values["spot"], rate=values["rate"]),
        process=GBM(values["drift"], values["volatility"]),
        n_paths=512,
        n_steps=2,
        rng=9,
        return_native=True,
    )
    assert result.price.requires_grad
    result.price.backward()
    assert values["spot"].grad is not None
    assert float(values["spot"].grad) == 0.0
    # The discount rate still enters smoothly, so that gradient is not zero.
    assert values["rate"].grad is not None
    assert float(values["rate"].grad) != 0.0


# --- the average-rate slice ---------------------------------------------------


def asian(**overrides: Any) -> AsianOption:
    call: dict[str, Any] = {
        "underlier": "ACME",
        "option_type": "call",
        "strike": STRIKE,
        "averaging_method": "geometric",
        "strike_convention": "fixed",
        "maturity": MATURITY,
    }
    call.update(overrides)
    return AsianOption(**call)


def discrete_geometric_asian(
    option: AsianOption,
    n_steps: int,
    *,
    spot: float = SPOT,
    rate: float = RATE,
    volatility: float = VOLATILITY,
) -> float:
    """Exact value of a discretely monitored geometric Asian under GBM.

    The geometric average of lognormal fixings is itself lognormal, so the
    contract has a closed form for the *discrete* schedule it is actually
    monitored on -- which is the only comparison that means anything here. A
    continuous-averaging formula would be a different contract, and matching
    against one would test the number of steps rather than the payoff.

    With fixings at ``t_1 .. t_n``, ``ln G`` is normal with mean
    ``ln S0 + (mu - sigma^2/2) * mean(t_i)`` and variance
    ``sigma^2 * sum_ij min(t_i, t_j) / n^2``.
    """
    from scipy.special import ndtr

    times = np.linspace(0.0, option.maturity, n_steps + 1)[1:]
    mean_log = math.log(spot) + (rate - 0.5 * volatility**2) * float(times.mean())
    variance = volatility**2 * float(np.minimum.outer(times, times).sum()) / n_steps**2
    spread = math.sqrt(variance)
    d2 = (mean_log - math.log(option.strike)) / spread
    d1 = d2 + spread
    forward = math.exp(mean_log + 0.5 * variance)
    discount = math.exp(-rate * option.maturity)
    if option.option_type.value == "call":
        return discount * (forward * float(ndtr(d1)) - option.strike * float(ndtr(d2)))
    return discount * (option.strike * float(ndtr(-d2)) - forward * float(ndtr(-d1)))


@pytest.mark.parametrize("option_type", ["call", "put"], ids=["call", "put"])
@pytest.mark.parametrize("n_steps", [4, 8], ids=["four", "eight"])
def test_geometric_asian_matches_its_discrete_closed_form(option_type: str, n_steps: int) -> None:
    contract = asian(option_type=option_type)
    result = run(contract, n_paths=100_000, n_steps=n_steps, rng=5150)
    exact = discrete_geometric_asian(contract, n_steps)
    assert abs(result.price - exact) <= 5.0 * result.stderr + 1e-12


def test_a_path_contract_is_routed_the_whole_scenario(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fast_vollib.simulation import monte_carlo
    from fast_vollib.simulation.scenario import Scenario

    seen: dict[str, Any] = {}
    original = monte_carlo.payoff

    def capturing(instrument: Any, state: Any) -> Any:
        seen["state"] = state
        return original(instrument, state)

    monkeypatch.setattr(monte_carlo, "payoff", capturing)
    run(asian(), n_paths=64, n_steps=4)
    assert isinstance(seen["state"], Scenario)
    assert seen["state"].n_steps == 4


def test_an_asian_price_depends_on_the_monitoring_schedule() -> None:
    """The grid is part of the contract's meaning, not a convergence knob."""
    coarse = run(asian(), n_paths=60_000, n_steps=2, rng=2718)
    fine = run(asian(), n_paths=60_000, n_steps=32, rng=2718)
    assert abs(coarse.price - fine.price) > 3.0 * (coarse.stderr + fine.stderr)


def test_the_fixed_asian_call_minus_put_identity_holds_on_one_seed() -> None:
    call = run(asian(option_type="call"), n_paths=20_000, n_steps=8, rng=161)
    put = run(asian(option_type="put"), n_paths=20_000, n_steps=8, rng=161)
    exact = discrete_geometric_asian(asian(option_type="call"), 8) - discrete_geometric_asian(
        asian(option_type="put"), 8
    )
    combined = math.sqrt(call.stderr**2 + put.stderr**2)
    assert abs((call.price - put.price) - exact) <= 5.0 * combined + 1e-12


def test_a_floating_strike_asian_prices_without_a_strike() -> None:
    floating = asian(strike=None, strike_convention="floating", averaging_method="arithmetic")
    result = run(floating, n_paths=20_000, n_steps=8, rng=1123)
    assert result.price > 0.0
    assert math.isfinite(result.stderr)


def test_an_arithmetic_asian_is_worth_at_least_its_geometric_twin() -> None:
    """AM-GM path by path, so the ordering survives the average on one seed."""
    arithmetic = run(asian(averaging_method="arithmetic"), n_paths=40_000, n_steps=8, rng=3141)
    geometric = run(asian(averaging_method="geometric"), n_paths=40_000, n_steps=8, rng=3141)
    assert arithmetic.price >= geometric.price


def test_an_asian_keeps_its_torch_graph_through_all_four_inputs() -> None:
    torch, values = torch_inputs()
    result = run(
        asian(averaging_method="arithmetic"),
        market=VanillaMarketInputs(underlying=values["spot"], rate=values["rate"]),
        process=GBM(values["drift"], values["volatility"]),
        n_paths=1_024,
        n_steps=8,
        rng=17,
        return_native=True,
    )
    assert result.price.requires_grad
    result.price.backward()
    for name, tensor in values.items():
        assert tensor.grad is not None, name
        assert torch.isfinite(tensor.grad).all(), name
        assert float(tensor.grad) != 0.0, name


# --- the barrier and lookback slices ------------------------------------------


def barrier(**overrides: Any) -> BarrierOption:
    call: dict[str, Any] = {
        "underlier": "ACME",
        "option_type": "call",
        "strike": STRIKE,
        "barrier": 130.0,
        "barrier_type": "up_and_out",
        "maturity": MATURITY,
    }
    call.update(overrides)
    return BarrierOption(**call)


def lookback(**overrides: Any) -> LookbackOption:
    call: dict[str, Any] = {
        "underlier": "ACME",
        "option_type": "call",
        "strike": STRIKE,
        "strike_convention": "fixed",
        "maturity": MATURITY,
    }
    call.update(overrides)
    return LookbackOption(**call)


@pytest.mark.parametrize("option_type", ["call", "put"], ids=["call", "put"])
@pytest.mark.parametrize(
    ("in_type", "out_type", "level"),
    [("up_and_in", "up_and_out", 130.0), ("down_and_in", "down_and_out", 80.0)],
    ids=["up", "down"],
)
def test_in_plus_out_equals_the_european_price_on_one_seed(
    option_type: str, in_type: str, out_type: str, level: float
) -> None:
    """An exact identity, not a statistical one: the same paths price all three.

    Any monitoring rule satisfies it, so a failure means the indicator is
    wrong rather than that the sample was unlucky.
    """
    common = {"option_type": option_type, "barrier": level}
    knock_in = run(barrier(barrier_type=in_type, **common), n_paths=20_000, n_steps=16, rng=90210)
    knock_out = run(barrier(barrier_type=out_type, **common), n_paths=20_000, n_steps=16, rng=90210)
    european = run(
        EuropeanOption(underlier="ACME", option_type=option_type, strike=STRIKE, maturity=MATURITY),
        n_paths=20_000,
        n_steps=16,
        rng=90210,
    )
    assert knock_in.price + knock_out.price == pytest.approx(european.price, rel=1e-12)


def test_a_barrier_is_never_worth_more_than_the_vanilla() -> None:
    for barrier_type in ("up_and_in", "up_and_out"):
        knocked = run(barrier(barrier_type=barrier_type), n_paths=20_000, n_steps=16, rng=112)
        european = run(CALL, n_paths=20_000, n_steps=16, rng=112)
        assert knocked.price <= european.price + 1e-12


def test_a_barrier_price_depends_on_the_monitoring_frequency() -> None:
    """Discrete monitoring is the contract, so the schedule moves the price."""
    coarse = run(barrier(), n_paths=60_000, n_steps=2, rng=1414)
    fine = run(barrier(), n_paths=60_000, n_steps=64, rng=1414)
    assert coarse.price > fine.price + 3.0 * (coarse.stderr + fine.stderr)


def test_an_unreachable_barrier_leaves_the_vanilla_price_intact() -> None:
    """A knock-out far above every path is economically the European option."""
    far = run(barrier(barrier=1e6, barrier_type="up_and_out"), n_paths=20_000, n_steps=8, rng=1618)
    european = run(CALL, n_paths=20_000, n_steps=8, rng=1618)
    assert far.price == pytest.approx(european.price, rel=1e-12)


def test_a_fixed_lookback_call_dominates_the_european_on_one_seed() -> None:
    look = run(lookback(), n_paths=20_000, n_steps=16, rng=2718)
    european = run(CALL, n_paths=20_000, n_steps=16, rng=2718)
    assert look.price >= european.price - 1e-12


def test_a_floating_lookback_is_worth_more_the_more_often_it_is_observed() -> None:
    """More observations can only widen the running extreme."""
    floating = lookback(strike=None, strike_convention="floating")
    coarse = run(floating, n_paths=40_000, n_steps=4, rng=3141)
    fine = run(floating, n_paths=40_000, n_steps=64, rng=3141)
    assert fine.price > coarse.price


def test_a_floating_lookback_price_is_non_negative() -> None:
    for option_type in ("call", "put"):
        result = run(
            lookback(strike=None, strike_convention="floating", option_type=option_type),
            n_paths=20_000,
            n_steps=8,
            rng=161,
        )
        assert result.price >= 0.0


def test_a_deterministic_process_prices_a_barrier_exactly() -> None:
    """Zero volatility gives one known path, so the answer is arithmetic.

    The forward path rises monotonically from 100 to 100 * exp(rT), which stays
    below the barrier, so an up-and-out call is simply the European intrinsic.
    """
    result = run(
        barrier(barrier=1e6),
        process=GBM.risk_neutral(rate=RATE, volatility=0.0),
        n_paths=4,
        n_steps=8,
    )
    terminal = SPOT * math.exp(RATE * MATURITY)
    exact = math.exp(-RATE * MATURITY) * max(terminal - STRIKE, 0.0)
    assert result.price == pytest.approx(exact, rel=1e-12)
    assert result.stderr == pytest.approx(0.0, abs=1e-12)


def test_a_knocked_out_deterministic_path_prices_to_zero() -> None:
    result = run(
        barrier(barrier=SPOT, barrier_type="up_and_out"),
        process=GBM.risk_neutral(rate=RATE, volatility=0.0),
        n_paths=4,
        n_steps=8,
    )
    assert result.price == 0.0


def test_barrier_and_lookback_keep_a_native_graph() -> None:
    torch, values = torch_inputs()
    for contract in (barrier(barrier=1e6), lookback(strike=50.0)):
        fresh = {
            name: tensor.detach().clone().requires_grad_(True) for name, tensor in values.items()
        }
        result = run(
            contract,
            market=VanillaMarketInputs(underlying=fresh["spot"], rate=fresh["rate"]),
            process=GBM(fresh["drift"], fresh["volatility"]),
            n_paths=512,
            n_steps=8,
            rng=21,
            return_native=True,
        )
        assert result.price.requires_grad
        result.price.backward()
        assert fresh["spot"].grad is not None
        assert torch.isfinite(fresh["spot"].grad).all()


# --- the variance slice -------------------------------------------------------


def variance_swap(**overrides: Any) -> VarianceSwap:
    call: dict[str, Any] = {
        "underlier": "ACME",
        "strike_variance": VOLATILITY**2,
        "maturity": MATURITY,
    }
    call.update(overrides)
    return VarianceSwap(**call)


def expected_realized_variance(process: GBM, n_steps: int, *, maturity: float = MATURITY) -> float:
    """``E[RV] = sigma^2 + a^2 * sum(dt_i^2) / T`` with ``a = mu - sigma^2 / 2``.

    Each step's log return is ``a dt + sigma sqrt(dt) Z``, so its second moment
    is ``a^2 dt^2 + sigma^2 dt``. Summing and dividing by ``T`` leaves the
    volatility squared plus a drift term that vanishes as the grid refines --
    the exact expectation for the schedule actually monitored, not its limit.
    """
    steps = np.diff(np.linspace(0.0, maturity, n_steps + 1))
    adjusted = float(process.drift) - 0.5 * float(process.volatility) ** 2
    return float(process.volatility) ** 2 + adjusted**2 * float((steps**2).sum()) / maturity


@pytest.mark.parametrize("n_steps", [4, 32], ids=["coarse", "fine"])
def test_a_variance_swap_matches_its_analytic_expectation(n_steps: int) -> None:
    process = risk_neutral()
    contract = variance_swap(strike_variance=0.0)
    result = run(contract, process=process, n_paths=100_000, n_steps=n_steps, rng=2024)
    exact = math.exp(-RATE * MATURITY) * expected_realized_variance(process, n_steps)
    assert abs(result.price - exact) <= 5.0 * result.stderr + 1e-12


def test_a_fair_strike_prices_to_zero_within_noise() -> None:
    process = risk_neutral()
    fair = expected_realized_variance(process, 16)
    result = run(
        variance_swap(strike_variance=fair),
        process=process,
        n_paths=100_000,
        n_steps=16,
        rng=1234,
    )
    assert abs(result.price) <= 5.0 * result.stderr + 1e-12


def test_the_drift_term_shrinks_as_the_grid_refines() -> None:
    """``sum(dt^2)/T`` is ``T/n`` on an even grid, so the correction is O(1/n)."""
    process = GBM(0.5, VOLATILITY)
    coarse = expected_realized_variance(process, 2)
    fine = expected_realized_variance(process, 256)
    assert coarse > fine > VOLATILITY**2


def test_a_variance_swap_is_linear_in_its_notional() -> None:
    unit = run(variance_swap(), n_paths=20_000, n_steps=16, rng=77)
    short = run(variance_swap(notional=-2.5), n_paths=20_000, n_steps=16, rng=77)
    assert short.price == pytest.approx(-2.5 * unit.price, rel=1e-12)
    assert short.stderr == pytest.approx(2.5 * unit.stderr, rel=1e-12)


def test_a_deterministic_process_realizes_only_the_drift_term() -> None:
    """With zero volatility every path is the same, so the answer is exact."""
    n_steps = 8
    process = GBM.risk_neutral(rate=RATE, volatility=0.0)
    result = run(
        variance_swap(strike_variance=0.0),
        process=process,
        n_paths=4,
        n_steps=n_steps,
    )
    exact = math.exp(-RATE * MATURITY) * expected_realized_variance(process, n_steps)
    assert result.price == pytest.approx(exact, rel=1e-10)
    assert result.stderr == pytest.approx(0.0, abs=1e-14)


def test_a_variance_swap_carries_a_useful_native_gradient() -> None:
    """Smooth on a positive path: every advertised input moves the price."""
    torch, values = torch_inputs()
    result = run(
        variance_swap(),
        market=VanillaMarketInputs(underlying=values["spot"], rate=values["rate"]),
        process=GBM(values["drift"], values["volatility"]),
        n_paths=1_024,
        n_steps=16,
        rng=31,
        return_native=True,
    )
    assert result.price.requires_grad
    result.price.backward()
    for name, tensor in values.items():
        assert tensor.grad is not None, name
        assert torch.isfinite(tensor.grad).all(), name
    # Realized variance is scale-invariant, so the spot only enters through
    # nothing at all: the gradient is exactly zero rather than merely small.
    assert float(values["spot"].grad) == pytest.approx(0.0, abs=1e-12)
    assert float(values["volatility"].grad) > 0.0


def test_a_single_precision_path_contract_can_be_priced() -> None:
    """A maturity not representable in binary32 is not a contract error.

    The horizon check is held to the precision the grid is actually stored in,
    so single precision remains usable end to end rather than failing on every
    maturity that is not a dyadic rational.
    """
    contract = asian(averaging_method="arithmetic", maturity=0.1, strike=100.0)
    result = MonteCarloEngine().price(
        contract,
        VanillaMarketInputs(underlying=np.float32(SPOT), rate=np.float32(RATE)),
        process=GBM(np.float32(RATE), np.float32(VOLATILITY)),
        n_paths=4_096,
        n_steps=8,
        rng=0,
        return_native=True,
    )
    assert result.price.dtype == np.float32
    assert float(result.price) > 0.0


def test_an_explicit_single_precision_grid_is_accepted_at_its_own_precision() -> None:
    contract = asian(averaging_method="arithmetic", maturity=0.1, strike=100.0)
    grid = np.linspace(0.0, 0.1, 9, dtype=np.float32)
    result = MonteCarloEngine().price(
        contract,
        VanillaMarketInputs(underlying=np.float32(SPOT), rate=np.float32(RATE)),
        process=GBM(np.float32(RATE), np.float32(VOLATILITY)),
        n_paths=1_024,
        n_steps=None,
        time_grid=grid,
        rng=1,
    )
    assert math.isfinite(result.price)


def test_a_grid_that_really_ends_elsewhere_is_still_refused_in_single_precision() -> None:
    contract = asian(averaging_method="arithmetic", maturity=0.1, strike=100.0)
    with pytest.raises(SimulationValidationError, match="ends at t="):
        MonteCarloEngine().price(
            contract,
            VanillaMarketInputs(underlying=np.float32(SPOT), rate=np.float32(RATE)),
            process=GBM(np.float32(RATE), np.float32(VOLATILITY)),
            n_paths=1_024,
            n_steps=None,
            time_grid=np.linspace(0.0, 0.11, 9, dtype=np.float32),
            rng=1,
        )


# --- do the oracles have the power they claim? --------------------------------
#
# An oracle that agrees with the code for the wrong reason is worse than none.
# These pin the conditions each Tier-2 comparison depends on, so a later edit to
# the shared constants cannot quietly disarm them again.


def test_the_shared_market_keeps_the_log_drift_away_from_zero() -> None:
    """``r - sigma^2 / 2`` at ``r=0.02, sigma=0.2`` is zero to machine precision.

    That is not a rounding curiosity: it deletes the drift term from the
    geometric-Asian and variance-swap closed forms, and makes an at-the-money
    digital call and put worth exactly the same. Two oracles then agree with a
    sign-flipped drift correction and with a call/put swap.
    """
    assert LOG_DRIFT == pytest.approx(RATE - 0.5 * VOLATILITY**2)
    assert abs(LOG_DRIFT) > 1e-3, LOG_DRIFT


def test_the_binary_oracle_can_tell_a_call_from_a_put() -> None:
    call = cash_or_nothing(DIGITAL_CALL)
    put = cash_or_nothing(DIGITAL_PUT)
    assert call != pytest.approx(put, abs=1e-6)
    assert call + put == pytest.approx(math.exp(-RATE * MATURITY), rel=1e-12)


def test_the_binary_oracle_is_sensitive_to_the_sign_of_the_drift_correction() -> None:
    """The exact mutation the degenerate market hid: ``-sigma^2/2`` flipped."""
    from scipy.special import ndtr

    def flipped(option: BinaryOption) -> float:
        d2 = (math.log(SPOT / option.strike) + (RATE + 0.5 * VOLATILITY**2) * MATURITY) / (
            VOLATILITY * math.sqrt(MATURITY)
        )
        sign = 1.0 if option.option_type.value == "call" else -1.0
        return option.cash_amount * math.exp(-RATE * MATURITY) * float(ndtr(sign * d2))

    result = run(DIGITAL_CALL, n_paths=100_000, rng=515)
    correct = cash_or_nothing(DIGITAL_CALL)
    mutated = flipped(DIGITAL_CALL)
    assert abs(result.price - correct) <= 5.0 * result.stderr
    assert abs(result.price - mutated) > 5.0 * result.stderr, (
        "the oracle cannot distinguish the sign of the drift correction"
    )


def test_the_geometric_asian_oracle_depends_on_the_drift_term() -> None:
    contract = asian(option_type="call")
    with_drift = discrete_geometric_asian(contract, 8)
    without_drift = discrete_geometric_asian(
        contract, 8, rate=0.5 * VOLATILITY**2, volatility=VOLATILITY
    )
    assert abs(with_drift - without_drift) > 1e-3


def test_the_variance_oracle_exercises_its_drift_term() -> None:
    """``a^2 * sum(dt^2) / T`` is a real contribution, not a rounding artefact.

    Under a risk-neutral GBM at a coarse grid it is small but measurable; under
    a strongly drifting process it dominates. Both are checked, because the
    formula is wrong in different ways at each end.
    """
    coarse = expected_realized_variance(risk_neutral(), 2)
    assert coarse > VOLATILITY**2
    assert coarse - VOLATILITY**2 == pytest.approx(LOG_DRIFT**2 * 0.5, rel=1e-12)

    drifting = GBM(1.0, VOLATILITY)
    contribution = expected_realized_variance(drifting, 4) - VOLATILITY**2
    assert contribution > 0.05 * VOLATILITY**2


def test_a_strongly_drifting_variance_swap_matches_its_expectation() -> None:
    """The drift term is checked where it is large enough to see."""
    process = GBM(1.0, VOLATILITY)
    n_steps = 4
    result = run(
        variance_swap(strike_variance=0.0),
        process=process,
        n_paths=100_000,
        n_steps=n_steps,
        rng=606,
    )
    exact = math.exp(-RATE * MATURITY) * expected_realized_variance(process, n_steps)
    without_drift = math.exp(-RATE * MATURITY) * VOLATILITY**2
    assert abs(result.price - exact) <= 5.0 * result.stderr + 1e-12
    assert abs(result.price - without_drift) > 5.0 * result.stderr, (
        "at this drift the term is large enough that dropping it must be visible"
    )


def test_the_variance_convention_is_sensitive_to_the_horizon_divisor() -> None:
    """Dividing by ``n_steps`` instead of ``T`` would pass at T=1 and only there."""
    process = GBM.risk_neutral(rate=RATE, volatility=VOLATILITY)
    short = VarianceSwap(underlier="ACME", strike_variance=0.0, maturity=0.25)
    n_steps = 8
    result = MonteCarloEngine().price(
        short,
        market(),
        process=process,
        n_paths=100_000,
        n_steps=n_steps,
        rng=707,
    )
    exact = math.exp(-RATE * 0.25) * expected_realized_variance(process, n_steps, maturity=0.25)
    divided_by_steps = exact * 0.25 / n_steps
    assert abs(result.price - exact) <= 5.0 * result.stderr + 1e-12
    assert abs(result.price - divided_by_steps) > 5.0 * result.stderr


def test_the_forward_oracle_is_sensitive_to_the_discount_factor() -> None:
    """With r*T small the discount is a subtle effect; make it unmissable."""
    long_dated = Forward(underlier="ACME", delivery_price=STRIKE, maturity=5.0)
    process = GBM.risk_neutral(rate=0.15, volatility=VOLATILITY)
    result = MonteCarloEngine().price(
        long_dated,
        VanillaMarketInputs(underlying=SPOT, rate=0.15),
        process=process,
        n_paths=100_000,
        n_steps=4,
        rng=808,
    )
    exact = math.exp(-0.15 * 5.0) * (SPOT * math.exp(0.15 * 5.0) - STRIKE)
    undiscounted = SPOT * math.exp(0.15 * 5.0) - STRIKE
    assert abs(result.price - exact) <= 5.0 * result.stderr + 1e-12
    assert abs(result.price - undiscounted) > 5.0 * result.stderr, (
        "dropping the discount factor must be visible to this oracle"
    )


def test_the_european_oracle_is_sensitive_to_the_drift_correction() -> None:
    """A sign-flipped ``-sigma^2/2`` must move the price out of the band."""
    result = run(CALL, n_paths=100_000, rng=909)
    exact = analytic(CALL)
    mutated = float(
        price_instrument(
            CALL,
            VanillaMarketInputs(
                underlying=SPOT * math.exp(VOLATILITY**2 * MATURITY),
                rate=RATE,
                volatility=VOLATILITY,
            ),
            model="black_scholes",
        )[0]
    )
    assert abs(result.price - exact) <= 5.0 * result.stderr + 1e-12
    assert abs(result.price - mutated) > 5.0 * result.stderr


def test_a_discretely_monitored_barrier_is_checked_against_a_direct_computation() -> None:
    """The in/out identity holds for *any* monitoring rule, so it is not an oracle.

    This is: simulate once, then evaluate the barrier indicator by hand from the
    same scenario, and require the engine's discounted mean to match exactly.
    A mutation to the comparison direction or its strictness moves this number.
    """
    from fast_vollib.simulation import simulate

    contract = barrier(barrier_type="up_and_out", barrier=115.0)
    grid = np.linspace(0.0, MATURITY, 17)
    scenario = simulate(
        "ACME",
        risk_neutral(),
        initial_state=SPOT,
        time_grid=grid,
        n_paths=20_000,
        rng=1001,
    )
    spot = scenario.spot
    alive = spot.max(axis=1) < 115.0
    intrinsic = np.maximum(spot[:, -1] - STRIKE, 0.0)
    expected = math.exp(-RATE * MATURITY) * float((intrinsic * alive).mean())

    result = MonteCarloEngine().price(
        contract, market(), process=risk_neutral(), n_paths=20_000, n_steps=16, rng=1001
    )
    assert result.price == pytest.approx(expected, rel=1e-12)
    # An exclusive rule, or the wrong direction, gives a different number.
    exclusive = math.exp(-RATE * MATURITY) * float((intrinsic * (spot.max(axis=1) <= 115.0)).mean())
    down = math.exp(-RATE * MATURITY) * float((intrinsic * (spot.min(axis=1) > 115.0)).mean())
    assert expected != pytest.approx(down, abs=1e-9)
    assert exclusive >= expected
