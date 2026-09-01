"""Heston dynamics: the protocol, the parameters, and a sampler that states its bias."""

from __future__ import annotations

import numpy as np
import pytest

from fast_vollib.processes import SCHEMES, Heston, StochasticProcess
from fast_vollib.simulation import SimulationValidationError

GRID = np.linspace(0.0, 1.0, 33)
SPOT = 100.0
INITIAL = {"spot": SPOT, "variance": 0.04}

#: A Feller-satisfying set and one that violates it, because the schemes differ
#: most where the variance actually reaches zero.
CALM = {"kappa": 2.0, "theta": 0.04, "vol_of_vol": 0.3, "rho": -0.7}
ROUGH = {"kappa": 0.5, "theta": 0.06, "vol_of_vol": 0.9, "rho": -0.9}


def _process(parameters: dict, drift: float = 0.0) -> Heston:
    return Heston(drift=drift, **parameters)


# --- the protocol --------------------------------------------------------------


def test_heston_satisfies_the_process_protocol() -> None:
    assert isinstance(_process(CALM), StochasticProcess)


def test_state_names_is_not_a_constructor_argument() -> None:
    assert Heston.state_names == ("spot", "variance")
    with pytest.raises(TypeError):
        Heston(kappa=1.0, theta=0.04, vol_of_vol=0.3, rho=-0.5, state_names=("x",))


def test_params_returns_the_original_objects_read_only() -> None:
    process = _process(CALM)
    params = process.params()
    assert dict(params) == {**CALM, "drift": 0.0}
    with pytest.raises(TypeError):
        params["kappa"] = 3.0


def test_the_risk_neutral_constructor_records_a_decision_rather_than_making_one() -> None:
    process = Heston.risk_neutral(rate=0.04, dividend_yield=0.01, **CALM)
    assert process.drift == pytest.approx(0.03, rel=1e-12)


# --- parameter validation ------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"kappa": 0.0}, "kappa must be strictly positive"),
        ({"theta": -0.1}, "theta must be strictly positive"),
        ({"vol_of_vol": 0.0}, "vol_of_vol must be strictly positive"),
        ({"rho": 1.0}, "rho must lie strictly inside"),
        ({"rho": -1.5}, "rho must lie strictly inside"),
        ({"kappa": float("nan")}, "kappa must be finite"),
    ],
)
def test_invalid_parameters_are_refused(kwargs, message) -> None:
    with pytest.raises(SimulationValidationError, match=message):
        Heston(**{**CALM, **kwargs})


@pytest.mark.parametrize("bad", [True, 1 + 2j], ids=["bool", "complex"])
def test_a_parameter_must_be_a_real_number(bad) -> None:
    with pytest.raises(SimulationValidationError):
        Heston(**{**CALM, "kappa": bad})


def test_the_feller_condition_is_reported_and_not_enforced() -> None:
    # 2 kappa theta / xi^2: 2*0.5*0.06/0.81 = 0.0740...
    rough = _process(ROUGH)
    assert rough.feller_ratio == pytest.approx(2 * 0.5 * 0.06 / 0.81, rel=1e-12)
    assert rough.satisfies_feller is False
    assert _process(CALM).satisfies_feller is True


def test_the_integrated_variance_mean_is_the_closed_form() -> None:
    # E[int_0^T v ds] = theta T + (v0 - theta)(1 - e^{-kappa T})/kappa, and it
    # collapses to theta T exactly when v0 == theta.
    process = _process(CALM)
    assert process.integrated_variance_mean(2.0, 0.04) == pytest.approx(0.08, rel=1e-12)
    expected = 0.04 * 2.0 + (0.09 - 0.04) * (1 - np.exp(-2.0 * 2.0)) / 2.0
    assert process.integrated_variance_mean(2.0, 0.09) == pytest.approx(expected, rel=1e-12)


# --- the sample ----------------------------------------------------------------


@pytest.mark.parametrize("scheme", SCHEMES)
def test_the_sample_has_the_documented_shape(scheme) -> None:
    paths = _process(CALM).sample(
        initial_state=INITIAL, time_grid=GRID, n_paths=16, rng=7, scheme=scheme
    )
    assert paths.shape == (16, GRID.size, 2)


@pytest.mark.parametrize("scheme", SCHEMES)
def test_the_first_state_is_the_initial_state_exactly(scheme) -> None:
    paths = _process(CALM).sample(
        initial_state=INITIAL, time_grid=GRID, n_paths=16, rng=7, scheme=scheme
    )
    np.testing.assert_array_equal(paths[:, 0, 0], np.full(16, SPOT))
    np.testing.assert_array_equal(paths[:, 0, 1], np.full(16, 0.04))


@pytest.mark.parametrize("scheme", SCHEMES)
def test_sampling_is_reproducible_from_a_seed_and_adds_no_state(scheme) -> None:
    process = _process(CALM)

    def draw():
        return process.sample(
            initial_state=INITIAL, time_grid=GRID, n_paths=64, rng=4242, scheme=scheme
        )

    np.testing.assert_array_equal(draw(), draw())
    assert dict(process.params()) == {**CALM, "drift": 0.0}


@pytest.mark.parametrize("parameters", [CALM, ROUGH], ids=["calm", "rough"])
def test_the_quadratic_exponential_scheme_never_produces_a_negative_variance(
    parameters,
) -> None:
    """The defining property of the scheme, and the reason it is the default.

    Full-truncation Euler is allowed a negative variance and truncates it in the
    coefficients; QE samples from a distribution supported on the non-negative
    half-line, so it never produces one at all.  The contrast is asserted in both
    directions below.
    """
    paths = _process(parameters).sample(
        initial_state={"spot": SPOT, "variance": 0.04},
        time_grid=np.linspace(0.0, 2.0, 17),
        n_paths=20_000,
        rng=20260831,
        scheme="quadratic_exponential",
    )
    assert float(np.min(paths[:, :, 1])) >= 0.0


def test_full_truncation_euler_does_produce_negative_variances() -> None:
    # Not a defect: it is what "full truncation" means, and it is why the
    # module docstring declines to call either scheme exact.
    paths = _process(ROUGH).sample(
        initial_state={"spot": SPOT, "variance": 0.04},
        time_grid=np.linspace(0.0, 2.0, 9),
        n_paths=20_000,
        rng=20260831,
        scheme="full_truncation_euler",
    )
    assert float(np.min(paths[:, :, 1])) < 0.0


@pytest.mark.parametrize("scheme", SCHEMES)
def test_the_forward_is_a_martingale_within_the_sampling_error(scheme) -> None:
    """``E[S_T] = S_0`` at zero drift, checked as five standard errors.

    A statistical claim stated as a band on the sample mean, computed from the
    sample itself, so raising the path count tightens it automatically.
    """
    n_paths = 200_000
    paths = _process(CALM).sample(
        initial_state=INITIAL,
        time_grid=np.linspace(0.0, 1.0, 33),
        n_paths=n_paths,
        rng=20260831,
        scheme=scheme,
    )
    terminal = paths[:, -1, 0]
    stderr = terminal.std(ddof=1) / np.sqrt(n_paths)
    assert abs(terminal.mean() - SPOT) < 5.0 * stderr


@pytest.mark.parametrize("scheme", SCHEMES)
def test_the_mean_integrated_variance_matches_its_closed_form(scheme) -> None:
    """The sampler's bias has an exact target to be measured against.

    ``v0`` is deliberately different from ``theta`` here: with the two equal the
    expected integrated variance is ``theta T`` whatever ``kappa`` is, and the
    test would pass for a sampler that ignored the mean reversion entirely.
    """
    n_paths = 100_000
    grid = np.linspace(0.0, 2.0, 65)
    process = _process(CALM)
    paths = process.sample(
        initial_state={"spot": SPOT, "variance": 0.10},
        time_grid=grid,
        n_paths=n_paths,
        rng=20260831,
        scheme=scheme,
    )
    integrated = np.trapezoid(paths[:, :, 1], grid, axis=1)
    stderr = integrated.std(ddof=1) / np.sqrt(n_paths)
    expected = process.integrated_variance_mean(2.0, 0.10)
    assert abs(integrated.mean() - expected) < 5.0 * stderr + 1e-3


def test_refining_the_grid_reduces_the_euler_scheme_bias() -> None:
    """Neither scheme is exact, and the Euler bias is what shrinks with the step.

    Measured on the mean terminal *variance*, whose exact value is
    ``theta + (v0 - theta) e^{-kappa T}`` -- the variance drift is linear, so its
    mean has a closed form under the true dynamics whatever the vol-of-vol is.
    The terminal spot is not usable for this: at these parameters its sample
    standard error is a hundred times the bias, so a test written on it would be
    reading noise.

    The claim is stated in standard errors of the same sample, so it is a
    statistical statement rather than a fitted tolerance: a coarse grid is biased
    by more than ten standard errors and a fine one by less than three.
    """
    n_paths = 400_000
    exact = ROUGH["theta"] + (0.04 - ROUGH["theta"]) * np.exp(-ROUGH["kappa"] * 1.0)
    measured = {}
    for n_steps in (4, 128):
        paths = _process(ROUGH).sample(
            initial_state=INITIAL,
            time_grid=np.linspace(0.0, 1.0, n_steps + 1),
            n_paths=n_paths,
            rng=11,
            scheme="full_truncation_euler",
        )
        terminal = paths[:, -1, 1]
        stderr = terminal.std(ddof=1) / np.sqrt(n_paths)
        measured[n_steps] = (abs(float(terminal.mean()) - exact), float(stderr))
    coarse_bias, coarse_stderr = measured[4]
    fine_bias, fine_stderr = measured[128]
    assert coarse_bias > 10.0 * coarse_stderr, measured
    assert fine_bias < 3.0 * fine_stderr, measured
    assert fine_bias < coarse_bias / 5.0, measured


def test_the_quadratic_exponential_scheme_is_unbiased_in_the_mean_variance() -> None:
    """QE matches the exact transition's first two moments, so this holds at any step.

    The contrast with the Euler scheme above is the whole reason QE is the
    default: at four steps a year it is already within sampling noise of the
    closed-form mean, where the Euler scheme is thirty standard errors away.
    """
    n_paths = 400_000
    exact = ROUGH["theta"] + (0.04 - ROUGH["theta"]) * np.exp(-ROUGH["kappa"] * 1.0)
    for n_steps in (2, 8, 64):
        paths = _process(ROUGH).sample(
            initial_state=INITIAL,
            time_grid=np.linspace(0.0, 1.0, n_steps + 1),
            n_paths=n_paths,
            rng=11,
            scheme="quadratic_exponential",
        )
        terminal = paths[:, -1, 1]
        stderr = terminal.std(ddof=1) / np.sqrt(n_paths)
        assert abs(float(terminal.mean()) - exact) < 3.0 * stderr, n_steps


def test_antithetic_sampling_mirrors_both_drivers() -> None:
    # Half the normals are drawn and negated, which for a correlated model has to
    # flip the variance innovation as well as the spot's, or the pair is not an
    # antithetic pair at all.
    paths = _process(CALM).sample(
        initial_state=INITIAL,
        time_grid=np.linspace(0.0, 1.0, 5),
        n_paths=1000,
        rng=3,
        antithetic=True,
        scheme="full_truncation_euler",
    )
    first, second = paths[:500], paths[500:]
    # Step one of the log-spot is (drift - v0/2)dt +/- sqrt(v0 dt) z; the two
    # halves straddle that deterministic part symmetrically.
    centre = np.log(SPOT) + (0.0 - 0.5 * 0.04) * 0.25
    left = np.log(first[:, 1, 0]) - centre
    right = np.log(second[:, 1, 0]) - centre
    np.testing.assert_allclose(left, -right, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize(
    "state, message",
    [
        ({"variance": 0.04}, "'spot' is missing"),
        ({"spot": 100.0}, "'variance' is missing"),
        ({"spot": -1.0, "variance": 0.04}, "strictly positive"),
        ({"spot": 100.0, "variance": -0.01}, "non-negative"),
    ],
)
def test_an_invalid_initial_state_is_refused(state, message) -> None:
    with pytest.raises(SimulationValidationError, match=message):
        _process(CALM).sample(initial_state=state, time_grid=GRID, n_paths=4, rng=1)


def test_an_unknown_scheme_is_refused() -> None:
    with pytest.raises(SimulationValidationError, match="scheme must be one of"):
        _process(CALM).sample(
            initial_state=INITIAL, time_grid=GRID, n_paths=4, rng=1, scheme="milstein"
        )


def test_a_drift_moves_the_terminal_mean_by_the_stated_factor() -> None:
    n_paths = 200_000
    drift = 0.05
    paths = _process(CALM, drift=drift).sample(
        initial_state=INITIAL,
        time_grid=np.linspace(0.0, 1.0, 33),
        n_paths=n_paths,
        rng=20260831,
    )
    terminal = paths[:, -1, 0]
    stderr = terminal.std(ddof=1) / np.sqrt(n_paths)
    assert abs(terminal.mean() - SPOT * np.exp(drift)) < 5.0 * stderr
