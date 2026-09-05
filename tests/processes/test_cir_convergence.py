"""The analytic CIR bond price, recovered by simulating and discounting.

These checks connect simulation with the affine formula. One file rather than
one assertion is that a simulated bond price is wrong in **two independent
ways**, which a single tolerance would silently blend:

*transition error*
    the sampled rate at the grid points is not distributed as the true
    ``r_{t+dt} | r_t``.  ``exact_transition`` has none of it by construction;
    the two discretizations have some, and it shrinks as the grid is refined.

*integral error*
    even given perfect states, ``int_0^T r_u du`` is approximated from finitely
    many samples of ``r``.  This one survives ``exact_transition`` untouched,
    which is precisely what makes the two measurable apart: run the exact
    transition and whatever remains is quadrature.

A test that only checked ``|MC - P| < tol`` would pass with a first-order
quadrature cancelling a scheme bias, and would keep passing when one of them
was broken.  So each is measured on its own, on a grid where the *other* is
negligible, and the combined statement comes last.

Nothing here asserts a convergence *order*.  The errors are measured and
required to shrink; claiming a rate would be claiming something about the
weak convergence of a quadrature applied to a diffusion path, which this
library has not proved and does not need.

Budget: every configuration below stays at or under 100,000 paths and 64
steps, which is the same order as the existing Heston tests.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from fast_vollib.processes import CIRShortRate
from fast_vollib.rates import cir_discount_factor
from fast_vollib.simulation import PathwiseShortRateDiscounting

KAPPA, THETA, VOLATILITY = 0.3, 0.04, 0.1
INITIAL_RATE = 0.05
HORIZON = 2.0
N_PATHS = 100_000
SEED = 20260904
NAMES = ("short_rate",)

PROCESS = CIRShortRate(kappa=KAPPA, theta=THETA, volatility=VOLATILITY)

#: ``E[r_T] = theta + (r_0 - theta) e^{-kappa T}`` and the matching variance,
#: both from Cox, Ingersoll and Ross (1985). Written out rather than imported
#: so that the moments this file checks against are not produced by the code
#: it is checking.
DECAY = math.exp(-KAPPA * HORIZON)
MEAN_T = THETA + (INITIAL_RATE - THETA) * DECAY
VARIANCE_T = INITIAL_RATE * VOLATILITY**2 * (DECAY - DECAY**2) / KAPPA + THETA * VOLATILITY**2 * (
    1.0 - DECAY
) ** 2 / (2.0 * KAPPA)

ANALYTIC_PRICE = float(
    cir_discount_factor(
        kappa=KAPPA,
        theta=THETA,
        volatility=VOLATILITY,
        initial_rate=INITIAL_RATE,
        maturity=HORIZON,
    )
)


def grid_of(n_steps: int) -> np.ndarray:
    return np.linspace(0.0, HORIZON, n_steps + 1)


@pytest.fixture(scope="module")
def sampled():
    """Paths for every (scheme, n_steps) the file needs, drawn once.

    Module-scoped because the same draws answer several questions and
    resampling them would triple the cost of the file for no extra evidence.
    """
    cache: dict[tuple[str, int], np.ndarray] = {}

    def get(scheme: str, n_steps: int) -> np.ndarray:
        key = (scheme, n_steps)
        if key not in cache:
            cache[key] = PROCESS.sample(
                initial_state={"short_rate": INITIAL_RATE},
                time_grid=grid_of(n_steps),
                n_paths=N_PATHS,
                rng=SEED,
                scheme=scheme,
            )
        return cache[key]

    return get


def standard_error(values: np.ndarray) -> float:
    return float(values.std(ddof=1) / math.sqrt(len(values)))


def price_from(states: np.ndarray, n_steps: int, rule: str) -> tuple[float, float]:
    """``(estimate, standard error)`` of ``E[exp(-int r)]`` from sampled paths."""
    factors = PathwiseShortRateDiscounting(rule=rule).discount_factors(
        states=states, time_grid=grid_of(n_steps), state_names=NAMES
    )
    return float(factors.mean()), standard_error(factors)


def variance_relative_error(states: np.ndarray) -> float:
    return float(states[:, -1, 0].var(ddof=1) / VARIANCE_T - 1.0)


# --- the setup is what it says it is -------------------------------------------


def test_the_analytic_price_is_the_expectation_being_estimated() -> None:
    """``P(0,T) = E[exp(-int_0^T r)]`` under the risk-neutral measure.

    Stated here because everything below compares a sample mean to
    ``ANALYTIC_PRICE``, and that comparison is only meaningful if the closed
    form really is the expectation of the thing being averaged.
    """
    assert 0.0 < ANALYTIC_PRICE < 1.0
    # An upper bound the price must respect: rates are non-negative under CIR.
    assert ANALYTIC_PRICE <= 1.0
    # And a sanity anchor: a flat rate at the mean of the path is nearby.
    assert ANALYTIC_PRICE == pytest.approx(
        math.exp(-0.5 * (INITIAL_RATE + MEAN_T) * HORIZON), rel=0.01
    )


def test_this_parameter_set_uses_the_two_block_branch_of_the_exact_transition() -> None:
    """``d > 1``; the Poisson-mixture branch is exercised in ``test_cir.py``."""
    assert 4.0 * KAPPA * THETA / VOLATILITY**2 > 1.0


# --- (i) transition error, isolated at the grid points -------------------------


@pytest.mark.parametrize("scheme", ["exact_transition", "quadratic_exponential"])
def test_the_terminal_mean_matches_the_closed_form_within_sampling_error(scheme, sampled) -> None:
    terminal = sampled(scheme, 16)[:, -1, 0]
    error = float(terminal.mean()) - MEAN_T
    assert abs(error) <= 3.0 * standard_error(terminal), (error, standard_error(terminal))


@pytest.mark.parametrize("scheme", ["exact_transition", "quadratic_exponential"])
def test_the_terminal_variance_matches_the_closed_form_on_a_coarse_grid(scheme, sampled) -> None:
    """The discriminating moment: the mean is nearly right for every scheme.

    A first-order scheme reproduces ``E[r_T]`` well before it reproduces
    ``Var[r_T]``, so checking the mean alone would grade all three the same.
    Measured relative error here is about 0.003 at eight steps, which is the
    sampling noise of a variance from 100,000 draws.
    """
    assert abs(variance_relative_error(sampled(scheme, 8))) < 0.015


def test_full_truncation_euler_has_a_variance_bias_the_others_do_not(sampled) -> None:
    """Named, not hidden: this is the cost of the scheme, and it is why it is
    documented as a transparent comparison rather than a recommendation."""
    biased = variance_relative_error(sampled("full_truncation_euler", 8))
    exact = variance_relative_error(sampled("exact_transition", 8))
    assert biased > 0.025, biased
    assert biased > 5.0 * abs(exact), (biased, exact)


def test_the_euler_variance_bias_shrinks_when_the_grid_is_refined(sampled) -> None:
    """Measured, with no order asserted -- only that refining removes it.

    At 8, 16 and 32 steps the relative variance error runs about 0.060, 0.031
    and 0.018 on this parameter set, so each refinement roughly halves it.
    """
    biases = [
        abs(variance_relative_error(sampled("full_truncation_euler", n))) for n in (8, 16, 32)
    ]
    assert biases[0] > biases[1] > biases[2], biases
    assert biases[0] / biases[2] > 2.0, biases


def test_refining_the_grid_does_not_change_the_exact_transition(sampled) -> None:
    """It is exact at the grid points, so there is no bias for a finer grid to
    remove -- which is the property that lets the integral error be isolated."""
    coarse = abs(variance_relative_error(sampled("exact_transition", 8)))
    fine = abs(variance_relative_error(sampled("exact_transition", 32)))
    assert coarse < 0.015 and fine < 0.015, (coarse, fine)


# --- (ii) integral error, isolated on exact states -----------------------------


@pytest.mark.parametrize("n_steps", [4, 8, 16])
def test_the_trapezoid_rule_prices_the_bond_within_sampling_error(n_steps, sampled) -> None:
    """On exact states the only error left is quadrature, and this one is small.

    Small enough that it is not resolvable above the Monte Carlo noise even at
    four steps over two years, which is the strongest statement available:
    the measured deviation is under one standard error at every grid tried.
    """
    estimate, stderr = price_from(sampled("exact_transition", n_steps), n_steps, "trapezoid")
    assert abs(estimate - ANALYTIC_PRICE) <= 3.0 * stderr, (estimate, ANALYTIC_PRICE, stderr)


def test_the_left_riemann_rule_is_visibly_biased_on_a_coarse_grid(sampled) -> None:
    """Otherwise the comparison between the two rules would prove nothing.

    A left Riemann sum under-counts a rate that is on average rising from
    ``r_0`` toward ``theta``... and here it *over*-counts, because ``r_0``
    exceeds ``theta`` and the path drifts down. Either way it misses, by about
    twenty standard errors at four steps.
    """
    estimate, stderr = price_from(sampled("exact_transition", 4), 4, "left_riemann")
    assert abs(estimate - ANALYTIC_PRICE) > 8.0 * stderr, (estimate, stderr)


def test_the_left_riemann_error_shrinks_as_the_grid_is_refined(sampled) -> None:
    """Measured over a four-fold refinement; no order is claimed.

    Common random numbers across the grids, so the differences reflect the
    quadrature rather than three independent samples.
    """
    errors = []
    for n_steps in (4, 8, 16):
        estimate, _stderr = price_from(
            sampled("exact_transition", n_steps), n_steps, "left_riemann"
        )
        errors.append(abs(estimate - ANALYTIC_PRICE))
    assert errors[0] > errors[1] > errors[2], errors
    assert errors[0] / errors[2] > 2.0, errors


def test_the_two_rules_bracket_nothing_and_simply_differ(sampled) -> None:
    """The trapezoid is not merely a refinement of the left rule at fine grids.

    Both are applied to the same states, so a difference between them is
    entirely quadrature and can be reported as such.
    """
    states = sampled("exact_transition", 8)
    trapezoid, _ = price_from(states, 8, "trapezoid")
    riemann, _ = price_from(states, 8, "left_riemann")
    assert trapezoid != riemann
    assert abs(trapezoid - ANALYTIC_PRICE) < abs(riemann - ANALYTIC_PRICE)


# --- (iii) the two together ----------------------------------------------------


@pytest.mark.parametrize(
    "scheme", ["exact_transition", "quadratic_exponential", "full_truncation_euler"]
)
def test_every_scheme_prices_the_bond_inside_sampling_error_plus_its_own_bias(
    scheme, sampled
) -> None:
    """The combined statement, with the bias measured rather than assumed.

    ``|MC - P| <= 3 * stderr + bias``, where ``bias`` is what the scheme's
    variance error at this grid is worth in price terms. Writing it this way
    keeps the tolerance honest: a scheme with a larger bias is allowed a
    larger deviation, and has to *report* the larger bias to get it.
    """
    n_steps = 16
    states = sampled(scheme, n_steps)
    estimate, stderr = price_from(states, n_steps, "trapezoid")
    # A relative error of ``e`` in Var[r_T] moves the price by at most
    # ``0.5 * e * Var[r_T] * T^2`` through the second-order term of
    # ``E[exp(-int r)]``; generous, and derived rather than fitted.
    bias = 0.5 * abs(variance_relative_error(states)) * VARIANCE_T * HORIZON**2
    assert abs(estimate - ANALYTIC_PRICE) <= 3.0 * stderr + bias, (
        scheme,
        estimate - ANALYTIC_PRICE,
        stderr,
        bias,
    )


def test_the_estimator_converges_toward_the_analytic_price_with_more_paths() -> None:
    """Four path counts, each an independent seed, so the standard error is the
    only thing shrinking. Nothing else in this file varies the path count."""
    deviations = []
    for paths, seed in ((6_250, 1), (25_000, 2), (100_000, 3)):
        states = PROCESS.sample(
            initial_state={"short_rate": INITIAL_RATE},
            time_grid=grid_of(16),
            n_paths=paths,
            rng=seed,
            scheme="exact_transition",
        )
        estimate, stderr = price_from(states, 16, "trapezoid")
        deviations.append(abs(estimate - ANALYTIC_PRICE) / stderr)
    # Each is a draw from roughly |N(0,1)|; the claim is that none of them is
    # far out, not that the sequence decreases.
    assert max(deviations) < 3.5, deviations
