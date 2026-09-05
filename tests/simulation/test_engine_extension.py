r"""``MonteCarloEngine.price``'s two new keywords, and what they may not change.

The engine was written for one state and one discount factor.  ``initial_state``
and ``discounting`` are what let it drive a three-factor model, and the whole
risk of adding them is that they change something for a caller who does not use
them.  So the first section here is about *absence*: a call that omits both must
produce the same number it always did, and the freeze in
``test_engine_reference_fixtures.py`` says the same thing over a wider grid.

The rest is about the two rules the keywords encode.

**No silent selection.**  ``initial_state`` may not carry ``spot`` -- the engine
takes that from ``market.underlying``, and two values for one quantity would
pick one.  It may not carry a name the process does not evolve.  It must carry
every name the process *does* evolve, because an engine that defaulted a missing
variance would be choosing a model.  And when a ``discounting`` rule is supplied,
``market.rate`` is not read at all, for the reason ``market.volatility`` is never
read.

**The capstone.**  BCC97 priced through the engine, against ``bcc97_price``.
This connects the three-factor model in one call: three states from ``initial_state``, a rate
path used in the drift and in the discount factor through
``PathwiseShortRateDiscounting``, and a closed form to check it against.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from fast_vollib.instruments import (
    EuropeanOption,
    MissingMarketInputError,
    VanillaMarketInputs,
)
from fast_vollib.pricing import bcc97_price, heston_price
from fast_vollib.processes import (
    BCC97,
    GBM,
    CIRShortRate,
    ConstantShortRate,
    Heston,
    HestonVariance,
    LognormalJumps,
)
from fast_vollib.simulation import (
    ConstantRateDiscounting,
    MonteCarloEngine,
    PathwiseShortRateDiscounting,
    SimulationValidationError,
    UnsupportedProcessError,
)

SPOT, STRIKE, RATE, VOLATILITY, MATURITY = 100.0, 100.0, 0.02, 0.25, 1.0
SEED = 20260904

CALL = EuropeanOption(underlier="ACME", option_type="call", strike=STRIKE, maturity=MATURITY)


def market(**overrides: Any) -> VanillaMarketInputs:
    fields: dict[str, Any] = {"underlying": SPOT, "rate": RATE, "volatility": VOLATILITY}
    fields.update(overrides)
    return VanillaMarketInputs(**fields)


def heston_process(drift: float = RATE) -> Heston:
    return Heston(kappa=2.0, theta=0.04, vol_of_vol=0.3, rho=-0.7, drift=drift)


# --- absence changes nothing ------------------------------------------------------


@pytest.mark.parametrize(
    "factors", [np.ones((8, 1)), np.ones(1), np.ones(8) * np.nan, np.ones(8) * np.inf, -np.ones(8)]
)
def test_invalid_discount_outputs_are_refused(factors):
    class Rule:
        def discount_factors(self, **kwargs):
            return factors

    with pytest.raises(SimulationValidationError, match="discount_factors"):
        MonteCarloEngine().price(
            CALL,
            market(),
            process=GBM(volatility=0.2, drift=RATE),
            n_paths=8,
            n_steps=1,
            rng=SEED,
            discounting=Rule(),
        )


@pytest.mark.parametrize(
    "rule", [object(), PathwiseShortRateDiscounting(), ConstantRateDiscounting(np.nan)]
)
def test_invalid_discount_request_does_not_advance_rng(rule):
    rng = np.random.default_rng(SEED)
    before = repr(rng.bit_generator.state)
    with pytest.raises(SimulationValidationError):
        MonteCarloEngine().price(
            CALL,
            market(),
            process=GBM(volatility=0.2, drift=RATE),
            n_paths=8,
            n_steps=1,
            rng=rng,
            discounting=rule,
        )
    assert repr(rng.bit_generator.state) == before


def test_omitting_both_keywords_is_the_call_that_was_always_made() -> None:
    """Bitwise, against the same call written without them.

    The default value of a keyword is not evidence that the default *path* is
    unchanged: what matters is that no extra input entered the namespace
    resolution and no extra operation entered the arithmetic.
    """
    engine = MonteCarloEngine()
    common = dict(
        process=GBM.risk_neutral(rate=RATE, volatility=VOLATILITY),
        n_paths=8192,
        n_steps=4,
        rng=SEED,
    )
    plain = engine.price(CALL, market(), **common)
    explicit = engine.price(CALL, market(), initial_state=None, discounting=None, **common)
    assert float(plain.price).hex() == float(explicit.price).hex()
    assert float(plain.stderr).hex() == float(explicit.stderr).hex()


def test_an_empty_initial_state_is_accepted_for_a_one_state_process() -> None:
    """``{}`` says "nothing further", which is true, and must not be an error."""
    engine = MonteCarloEngine()
    common = dict(
        process=GBM.risk_neutral(rate=RATE, volatility=VOLATILITY),
        n_paths=4096,
        n_steps=2,
        rng=SEED,
    )
    plain = engine.price(CALL, market(), **common)
    empty = engine.price(CALL, market(), initial_state={}, **common)
    assert float(plain.price).hex() == float(empty.price).hex()


def test_the_default_discounting_is_the_constant_rate_rule() -> None:
    """Stated as a bitwise identity rather than as a comment.

    The two routes reach the factor through the same
    ``constant_rate_factor``, and the horizons agree because ``n_steps`` builds a
    grid ending exactly at the contract's maturity. On a grid that ended merely
    *within tolerance* they would differ in the last bits, which
    :class:`ConstantRateDiscounting` documents.
    """
    engine = MonteCarloEngine()
    common = dict(
        process=GBM.risk_neutral(rate=RATE, volatility=VOLATILITY),
        n_paths=4096,
        n_steps=4,
        rng=SEED,
    )
    implicit = engine.price(CALL, market(), **common)
    explicit = engine.price(CALL, market(), discounting=ConstantRateDiscounting(RATE), **common)
    assert float(implicit.price).hex() == float(explicit.price).hex()


# --- no silent selection -----------------------------------------------------------


def test_a_supplied_rule_means_market_rate_is_not_read() -> None:
    """Not "read and ignored": not read. A market with no rate at all prices."""
    engine = MonteCarloEngine()
    common = dict(
        process=GBM.risk_neutral(rate=RATE, volatility=VOLATILITY),
        n_paths=4096,
        n_steps=2,
        rng=SEED,
    )
    priced = engine.price(
        CALL, market(rate=None), discounting=ConstantRateDiscounting(RATE), **common
    )
    assert math.isfinite(priced.price)
    with pytest.raises(MissingMarketInputError, match="rate"):
        engine.price(CALL, market(rate=None), **common)


def test_a_rule_beside_a_market_rate_uses_the_rule() -> None:
    """The two disagree on purpose, and the answer must come from the argument."""
    engine = MonteCarloEngine()
    common = dict(
        process=GBM.risk_neutral(rate=RATE, volatility=VOLATILITY),
        n_paths=4096,
        n_steps=2,
        rng=SEED,
    )
    priced = engine.price(
        CALL, market(rate=0.02), discounting=ConstantRateDiscounting(0.10), **common
    )
    reference = engine.price(CALL, market(rate=0.10), **common)
    assert float(priced.price).hex() == float(reference.price).hex()


def test_a_non_finite_market_rate_is_not_even_looked_at_under_a_rule() -> None:
    """The validation that would have refused it is part of reading it."""
    engine = MonteCarloEngine()
    priced = engine.price(
        CALL,
        market(rate=float("nan")),
        process=GBM.risk_neutral(rate=RATE, volatility=VOLATILITY),
        discounting=ConstantRateDiscounting(RATE),
        n_paths=4096,
        n_steps=2,
        rng=SEED,
    )
    assert math.isfinite(priced.price)


def test_initial_state_may_not_carry_the_spot() -> None:
    with pytest.raises(SimulationValidationError, match="market.underlying"):
        MonteCarloEngine().price(
            CALL,
            market(),
            process=heston_process(),
            initial_state={"spot": 90.0, "variance": 0.04},
            n_paths=1024,
            n_steps=2,
            rng=SEED,
        )


def test_initial_state_may_not_carry_a_state_the_process_does_not_evolve() -> None:
    with pytest.raises(SimulationValidationError, match="short_rate"):
        MonteCarloEngine().price(
            CALL,
            market(),
            process=heston_process(),
            initial_state={"variance": 0.04, "short_rate": 0.03},
            n_paths=1024,
            n_steps=2,
            rng=SEED,
        )


def test_a_missing_state_is_refused_and_names_what_the_process_evolves() -> None:
    with pytest.raises(UnsupportedProcessError) as excinfo:
        MonteCarloEngine().price(
            CALL, market(), process=heston_process(), n_paths=1024, n_steps=2, rng=SEED
        )
    message = str(excinfo.value)
    assert "('spot', 'variance')" in message
    assert "variance" in message
    assert "simulate() itself is not restricted" in message


def test_a_non_scalar_state_is_refused() -> None:
    with pytest.raises(SimulationValidationError, match="initial_state"):
        MonteCarloEngine().price(
            CALL,
            market(),
            process=heston_process(),
            initial_state={"variance": np.zeros(3)},
            n_paths=1024,
            n_steps=2,
            rng=SEED,
        )


def test_a_non_mapping_initial_state_is_refused() -> None:
    with pytest.raises(SimulationValidationError, match="mapping"):
        MonteCarloEngine().price(
            CALL,
            market(),
            process=heston_process(),
            initial_state=[0.04],
            n_paths=1024,
            n_steps=2,
            rng=SEED,
        )


def test_a_process_whose_first_state_is_not_spot_is_still_refused() -> None:
    """The relaxation is "first state is spot", not "any states at all"."""
    with pytest.raises(UnsupportedProcessError, match="first state must be 'spot'"):
        MonteCarloEngine().price(
            CALL,
            market(),
            process=CIRShortRate(kappa=0.5, theta=0.04, volatility=0.1),
            initial_state={"short_rate": 0.03},
            n_paths=1024,
            n_steps=2,
            rng=SEED,
        )


def test_no_invalid_request_reaches_the_sampler(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every new refusal costs nothing, like every old one."""
    from fast_vollib.simulation import monte_carlo

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("no paths may be drawn for an invalid request")

    monkeypatch.setattr(monte_carlo, "simulate", forbidden)
    invalid: list[dict[str, Any]] = [
        {"initial_state": {"spot": 90.0, "variance": 0.04}},
        {"initial_state": {"variance": 0.04, "elephant": 1.0}},
        {"initial_state": None},
        {"initial_state": {"variance": np.zeros(3)}},
        {"initial_state": "0.04"},
    ]
    for call in invalid:
        with pytest.raises((SimulationValidationError, UnsupportedProcessError)):
            MonteCarloEngine().price(
                CALL, market(), process=heston_process(), n_paths=1024, n_steps=2, rng=SEED, **call
            )


# --- a second state, end to end ------------------------------------------------------


def test_heston_prices_through_the_engine_against_its_closed_form() -> None:
    """The first multi-state valuation the engine has ever done.

    Checked against ``heston_price`` rather than against another simulation, so
    the agreement says the engine assembled the right initial state rather than
    that two runs of the same code agree.
    """
    variance = 0.04
    process = heston_process(drift=RATE)
    result = MonteCarloEngine().price(
        CALL,
        market(),
        process=process,
        initial_state={"variance": variance},
        n_paths=200_000,
        n_steps=64,
        rng=7,
    )
    exact = float(
        heston_price(
            forward=SPOT * math.exp(RATE * MATURITY),
            strike=STRIKE,
            maturity=MATURITY,
            discount=math.exp(-RATE * MATURITY),
            v0=variance,
            kappa=2.0,
            theta=0.04,
            vol_of_vol=0.3,
            rho=-0.7,
        )
    )
    assert abs(result.price - exact) < 4.0 * result.stderr + 0.01


def test_the_initial_state_joins_the_namespace_resolution() -> None:
    """A torch variance makes the whole valuation torch, as a torch spot would.

    ``initial_state`` is an *input*, so it has to be resolved with the others;
    an implementation that passed it through untouched would leave a NumPy
    scalar in a torch simulation and be refused one layer down.
    """
    torch = pytest.importorskip("torch")
    result = MonteCarloEngine().price(
        CALL,
        market(underlying=torch.tensor(SPOT, dtype=torch.float64), rate=RATE),
        process=heston_process(),
        initial_state={"variance": torch.tensor(0.04, dtype=torch.float64)},
        n_paths=4096,
        n_steps=4,
        rng=SEED,
        return_native=True,
    )
    assert isinstance(result.price, torch.Tensor)
    assert result.price.dtype == torch.float64


def test_a_mixed_namespace_in_the_initial_state_is_refused() -> None:
    """A NumPy spot and a torch variance is two backends, and it is named as one.

    The spot is a *zero-dimensional array* rather than a Python float on
    purpose: a Python float belongs to no namespace and is therefore no
    conflict, which is why the test above -- a float spot with a torch variance
    -- runs on torch instead of failing.
    """
    torch = pytest.importorskip("torch")
    with pytest.raises(SimulationValidationError, match="more than one array namespace"):
        MonteCarloEngine().price(
            CALL,
            market(underlying=np.array(SPOT)),
            process=heston_process(),
            initial_state={"variance": torch.tensor(0.04, dtype=torch.float64)},
            n_paths=1024,
            n_steps=2,
            rng=SEED,
        )


# --- three states, a rate path, and a closed form ---------------------------------------


def test_the_two_discounting_routes_agree_on_a_constant_rate() -> None:
    """A constant rate reached two ways: as ``market.rate``, and as a path.

    With ``ConstantShortRate`` the simulated rate column holds ``market.rate`` at
    every time, so integrating it with the trapezoid gives the same factor the
    default route applies as a closed form. The paths are identical -- nothing
    comes from the rate slot either way -- so the only thing under test is that
    the two discounting routes agree, which they must.

    Not asserted bitwise: one route forms ``exp(-r*T)`` once and the other
    accumulates ``sum_k 0.5*(r + r)*dt`` and exponentiates, which is a different
    sequence of roundings for the same number.
    """
    engine = MonteCarloEngine()
    common = dict(n_paths=8192, n_steps=8, rng=SEED)
    lattice = engine.price(
        CALL,
        market(),
        process=BCC97(
            variance=HestonVariance(kappa=2.0, theta=0.04, vol_of_vol=0.3, rho=-0.7),
            jumps=LognormalJumps(jump_intensity=0.0, mean_log_jump=0.0, jump_volatility=0.0),
            rates=ConstantShortRate(),
        ),
        initial_state={"variance": 0.04, "short_rate": RATE},
        **common,
    )
    # The same paths, discounted the same way, reached through the rule instead.
    with_rule = engine.price(
        CALL,
        market(rate=None),
        process=BCC97(
            variance=HestonVariance(kappa=2.0, theta=0.04, vol_of_vol=0.3, rho=-0.7),
            jumps=LognormalJumps(jump_intensity=0.0, mean_log_jump=0.0, jump_volatility=0.0),
            rates=ConstantShortRate(),
        ),
        initial_state={"variance": 0.04, "short_rate": RATE},
        discounting=PathwiseShortRateDiscounting(),
        **common,
    )
    assert abs(lattice.price - with_rule.price) < 1e-12 * abs(lattice.price)


def test_bcc97_prices_through_the_engine_against_the_fourier_price() -> None:
    r"""The capstone: three states, a stochastic rate, and a closed form.

    The rate path drives the drift *and* the discount factor, and it is the same
    path in both roles -- the engine hands ``PathwiseShortRateDiscounting`` the
    very states it simulated, and that rule's trapezoid is the quadrature
    ``BCC97``'s drift uses. Anything else would leave an uncancelled rate and
    show up here as a bias far outside the budget.

    The budget is four standard errors plus a small allowance for the scheme's
    discretization at 64 steps, sized from the direct measurement in
    ``tests/test_pricing/test_bcc97.py``.
    """
    heston = {"v0": 0.04, "kappa": 2.0, "theta": 0.04, "vol_of_vol": 0.3, "rho": -0.7}
    jumps = {"jump_intensity": 0.5, "mean_log_jump": -0.05, "jump_volatility": 0.2}
    rates = {"rate_kappa": 0.5, "rate_theta": 0.05, "rate_volatility": 0.2, "initial_rate": 0.03}
    dividend_yield = 0.01

    exact = float(
        bcc97_price(
            spot=SPOT,
            strike=STRIKE,
            maturity=MATURITY,
            dividend_yield=dividend_yield,
            **heston,
            **jumps,
            **rates,
        )
    )
    process = BCC97(
        variance=HestonVariance(
            kappa=heston["kappa"],
            theta=heston["theta"],
            vol_of_vol=heston["vol_of_vol"],
            rho=heston["rho"],
        ),
        jumps=LognormalJumps(**jumps),
        rates=CIRShortRate(
            kappa=rates["rate_kappa"],
            theta=rates["rate_theta"],
            volatility=rates["rate_volatility"],
        ),
        dividend_yield=dividend_yield,
    )
    result = MonteCarloEngine().price(
        CALL,
        # No market rate at all: under a stochastic-rate model the discount
        # factor is an output, and there is no constant for the market to carry.
        market(rate=None),
        process=process,
        initial_state={"variance": heston["v0"], "short_rate": rates["initial_rate"]},
        discounting=PathwiseShortRateDiscounting(),
        n_paths=200_000,
        n_steps=64,
        rng=2026,
    )
    assert abs(result.price - exact) < 4.0 * result.stderr + 0.02


def test_the_left_riemann_rule_is_measurably_a_different_number() -> None:
    """Same paths, same contract, a different quadrature -- and it shows.

    Not an assertion that one is right: it is an assertion that the engine
    applies the rule it was given rather than one it picked, which a silent
    default would make untestable.
    """
    engine = MonteCarloEngine()
    process = BCC97(
        variance=HestonVariance(kappa=2.0, theta=0.04, vol_of_vol=0.3, rho=-0.7),
        jumps=LognormalJumps(jump_intensity=0.0, mean_log_jump=0.0, jump_volatility=0.0),
        rates=CIRShortRate(kappa=0.5, theta=0.06, volatility=0.4),
    )
    common = dict(
        process=process,
        initial_state={"variance": 0.04, "short_rate": 0.01},
        n_paths=40_000,
        n_steps=8,
        rng=11,
    )
    trapezoid = engine.price(
        CALL, market(rate=None), discounting=PathwiseShortRateDiscounting(), **common
    )
    left = engine.price(
        CALL,
        market(rate=None),
        discounting=PathwiseShortRateDiscounting(rule="left_riemann"),
        **common,
    )
    assert trapezoid.price != left.price
    # Same paths, so the difference is the quadrature and nothing else.
    assert abs(trapezoid.price - left.price) < 0.05 * abs(trapezoid.price)
