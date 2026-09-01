"""Generative evaluation: every draw is checked, and the mean is not the answer."""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import numpy as np
import pytest

from fast_vollib.surface import (
    ForecastHorizon,
    SurfaceGridSpec,
    SurfaceObservations,
    SurfacePoints,
    SurfaceSamples,
    SurfaceValidationError,
    VerificationLevel,
)
from fast_vollib.surface.fitting import FlatIVSurface, SVICalibrator, SVIParameters
from fast_vollib.surface.generative import (
    CONDITIONS,
    SCHEMA_VERSION,
    GaussianFieldSurfaceDistribution,
    GaussianFieldSurfaceGenerator,
    evaluate_samples,
    generative_json_schema,
    wilson_interval,
)
from fast_vollib.surface.protocols import GenerativeSurfaceModel, SurfaceDistribution

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = ROOT / "docs" / "schemas" / "fast-vollib-generative-arbitrage-v1.schema.json"

GRID = SurfaceGridSpec(k=np.linspace(-0.3, 0.3, 11), T=[0.25, 0.5, 1.0])
SEED = 20260831

#: Draw count for the probability claims below.  At 200 trials a Wilson interval
#: on an observed zero reaches to 0.019, which is what "no violations" is worth
#: here and is stated in the assertions rather than rounded to "none".
N_SAMPLES = 200


def _smooth() -> GaussianFieldSurfaceDistribution:
    """A field whose correlation length dwarfs the grid: draws stay convex."""
    return GaussianFieldSurfaceDistribution(
        base=FlatIVSurface(level=0.2), volatility=0.02, length_scale_k=1.0, length_scale_T=2.0
    )


def _rough() -> GaussianFieldSurfaceDistribution:
    """A field at the grid's own spacing: draws are jagged and break convexity."""
    return GaussianFieldSurfaceDistribution(
        base=FlatIVSurface(level=0.2), volatility=0.35, length_scale_k=0.06, length_scale_T=2.0
    )


# --- the distribution ------------------------------------------------------------


def test_the_distribution_and_generator_satisfy_their_protocols() -> None:
    assert isinstance(_smooth(), SurfaceDistribution)
    assert isinstance(GaussianFieldSurfaceGenerator(), GenerativeSurfaceModel)


def test_a_draw_is_reproducible_from_its_seed() -> None:
    points = GRID.to_points()
    first = _smooth().sample(points, n_samples=8, rng=SEED)
    second = _smooth().sample(points, n_samples=8, rng=SEED)
    np.testing.assert_array_equal(first.iv, second.iv)


def test_a_different_seed_gives_a_different_draw() -> None:
    points = GRID.to_points()
    assert not np.array_equal(
        _smooth().sample(points, n_samples=8, rng=SEED).iv,
        _smooth().sample(points, n_samples=8, rng=SEED + 1).iv,
    )


def test_the_field_is_multiplicatively_unbiased_in_total_variance() -> None:
    """``E[exp(sigma Z - sigma^2/2)] = 1`` exactly, so the mean draw is the base.

    Checked as four standard errors of the sample mean, computed from the sample
    itself.  It matters because the whole argument for per-draw evaluation is
    that a well-centred mean says nothing about the draws -- so the mean had
    better actually be centred.
    """
    n_samples = 20_000
    points = SurfacePoints(k=np.array([-0.1, 0.0, 0.1]), T=np.full(3, 1.0))
    distribution = GaussianFieldSurfaceDistribution(
        base=FlatIVSurface(level=0.2), volatility=0.3, length_scale_k=0.5
    )
    samples = distribution.sample(points, n_samples=n_samples, rng=SEED)
    total_variance = samples.iv**2 * points.T[None, :]
    mean = total_variance.mean(axis=0)
    stderr = total_variance.std(axis=0, ddof=1) / np.sqrt(n_samples)
    np.testing.assert_array_less(np.abs(mean - 0.04), 4.0 * stderr)


def test_a_draw_is_a_surface_rather_than_a_point_cloud() -> None:
    # Adjacent nodes of one draw are far more alike than the same node across
    # draws, which is what a correlated field means and what makes a per-draw
    # convexity check meaningful.
    points = GRID.to_points()
    samples = _smooth().sample(points, n_samples=64, rng=SEED)
    along_k = np.std(np.diff(samples.iv[:, :11], axis=1))
    across_draws = np.std(samples.iv[:, 0])
    assert along_k < 0.25 * across_draws


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"volatility": 0.0}, "volatility must be finite and strictly positive"),
        ({"length_scale_k": -1.0}, "length_scale_k must be finite"),
        ({"base": "not-a-surface"}, "base must be a definite surface"),
    ],
)
def test_invalid_distribution_configuration_is_refused(kwargs, message) -> None:
    with pytest.raises(SurfaceValidationError, match=message):
        GaussianFieldSurfaceDistribution(**{"base": FlatIVSurface(level=0.2), **kwargs})


def test_an_oversized_joint_draw_is_refused_rather_than_attempted() -> None:
    from fast_vollib.surface.generative import MAX_JOINT_POINTS

    n = MAX_JOINT_POINTS + 1
    points = SurfacePoints(k=np.linspace(-1.0, 1.0, n), T=np.full(n, 1.0))
    with pytest.raises(SurfaceValidationError, match="dense Cholesky"):
        _smooth().sample(points, n_samples=1, rng=SEED)


def test_the_generator_conditions_on_the_most_recent_context() -> None:
    generator = GaussianFieldSurfaceGenerator(calibrator=SVICalibrator())
    reference = SVIParameters(a=0.04, b=0.4, rho=-0.4, m=0.0, sigma=0.1)
    k = np.linspace(-0.3, 0.3, 15)
    history = [
        SurfaceObservations(
            k=k, T=np.full(15, 1.0), iv=np.sqrt(reference.total_variance(k) * scale)
        )
        for scale in (1.5, 1.0)
    ]
    distribution = generator.distribution(history, horizon=ForecastHorizon(steps=5))
    fitted = distribution.base.evaluate(SurfacePoints(k=[0.0], T=[1.0])).iv[0]
    expected = float(np.sqrt(reference.total_variance(0.0)))
    assert abs(fitted - expected) < 1e-6


def test_the_generator_declines_an_empty_context() -> None:
    with pytest.raises(SurfaceValidationError, match="nothing to condition on"):
        GaussianFieldSurfaceGenerator().distribution([])


# --- the aggregation ------------------------------------------------------------


def test_a_smooth_field_almost_never_violates_and_the_interval_says_so() -> None:
    report = evaluate_samples(_smooth(), GRID, n_samples=N_SAMPLES, rng=SEED)
    assert report.any_violation_probability == 0.0
    lower, upper = report.any_violation_interval
    assert lower == 0.0
    # Zero out of two hundred is not "impossible"; it is "below about 2%".
    assert 0.01 < upper < 0.03


def test_a_rough_field_almost_always_violates() -> None:
    report = evaluate_samples(_rough(), GRID, n_samples=N_SAMPLES, rng=SEED)
    assert report.any_violation_probability > 0.9
    assert dict(report.condition_probability)["butterfly"] > 0.9
    assert report.worst_severity > report.expected_severity > 0.0


def test_the_mean_surface_can_pass_while_almost_every_draw_fails() -> None:
    """The failure this whole module exists to prevent, demonstrated.

    A rough field breaks convexity on essentially every draw, and its pointwise
    mean is smooth enough to pass every hard check.  A report that scored only
    the mean would call this model arbitrage-free.
    """
    report = evaluate_samples(_rough(), GRID, n_samples=N_SAMPLES, rng=SEED)
    assert report.any_violation_probability > 0.9
    assert report.mean_surface_metrics is not None
    assert report.mean_surface_metrics["passed"] == 1.0
    assert report.median_surface_metrics["passed"] == 1.0


def test_the_violation_rate_rises_monotonically_as_the_field_roughens() -> None:
    probabilities = []
    for length_scale in (1.0, 0.3, 0.12, 0.05):
        distribution = GaussianFieldSurfaceDistribution(
            base=FlatIVSurface(level=0.2),
            volatility=0.25,
            length_scale_k=length_scale,
            length_scale_T=2.0,
        )
        report = evaluate_samples(distribution, GRID, n_samples=100, rng=SEED)
        probabilities.append(report.any_violation_probability)
    assert probabilities == sorted(probabilities)
    assert probabilities[0] < 0.1 < 0.9 < probabilities[-1]


def test_every_condition_family_is_reported_in_a_fixed_order() -> None:
    report = evaluate_samples(_rough(), GRID, n_samples=32, rng=SEED)
    assert tuple(name for name, _ in report.condition_probability) == CONDITIONS
    assert tuple(name for name, _ in report.condition_expected_fraction) == CONDITIONS


def test_the_severity_quantiles_are_ordered_and_bounded_by_the_worst() -> None:
    report = evaluate_samples(
        _rough(), GRID, n_samples=N_SAMPLES, rng=SEED, severity_quantiles=(0.1, 0.5, 0.9)
    )
    levels = [level for level, _ in report.severity_quantiles]
    values = [value for _, value in report.severity_quantiles]
    assert levels == [0.1, 0.5, 0.9]
    assert values == sorted(values)
    assert values[-1] <= report.worst_severity


def test_the_monte_carlo_error_shrinks_with_the_square_root_of_the_count() -> None:
    # A property of the estimator, asserted as a ratio so it cannot be met by a
    # coincidence of scale.
    distribution = GaussianFieldSurfaceDistribution(
        base=FlatIVSurface(level=0.2), volatility=0.25, length_scale_k=0.12, length_scale_T=2.0
    )
    small = evaluate_samples(distribution, GRID, n_samples=100, rng=SEED)
    large = evaluate_samples(distribution, GRID, n_samples=400, rng=SEED)
    assert 1.5 < small.any_violation_stderr / large.any_violation_stderr < 2.5


def test_already_drawn_samples_can_be_evaluated_directly() -> None:
    samples = _rough().sample(GRID.to_points(), n_samples=64, rng=SEED)
    report = evaluate_samples(samples, GRID)
    assert report.n_samples == 64
    assert report.rng_policy is not None and "PCG64" in report.rng_policy


def test_the_reported_verification_level_is_the_one_that_was_measured() -> None:
    report = evaluate_samples(_smooth(), GRID, n_samples=16, rng=SEED)
    assert report.verification is VerificationLevel.EMPIRICAL_FINITE_GRID


def test_a_stronger_verification_level_must_be_asserted_deliberately() -> None:
    report = evaluate_samples(
        _smooth(),
        GRID,
        n_samples=16,
        rng=SEED,
        verification=VerificationLevel.TRAINING_PENALTY,
    )
    assert report.verification is VerificationLevel.TRAINING_PENALTY


def test_declined_points_reduce_the_reported_coverage() -> None:
    points = GRID.to_points()
    iv = np.full((8, points.n), 0.2)
    iv[:, :5] = np.nan
    samples = SurfaceSamples(points=points, iv=iv)
    report = evaluate_samples(samples, GRID)
    assert report.valid_sample_fraction == 0.0
    assert report.point_coverage == pytest.approx(1.0 - 5 / points.n, rel=1e-12)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"n_samples": None, "rng": SEED}, "needs an explicit n_samples"),
        ({"n_samples": 4, "rng": None}, "cannot be replayed is not evidence"),
        ({"n_samples": 4, "rng": SEED, "severity_quantiles": (0.0,)}, "strictly inside"),
    ],
)
def test_invalid_aggregation_requests_are_refused(kwargs, message) -> None:
    with pytest.raises(SurfaceValidationError, match=message):
        evaluate_samples(_smooth(), GRID, **kwargs)


def test_a_sample_count_alongside_ready_samples_is_refused() -> None:
    samples = _smooth().sample(GRID.to_points(), n_samples=8, rng=SEED)
    with pytest.raises(SurfaceValidationError, match="n_samples was given alongside"):
        evaluate_samples(samples, GRID, n_samples=99)


def test_samples_from_another_grid_are_refused() -> None:
    other = SurfaceGridSpec(k=np.linspace(-0.5, 0.5, 7), T=[1.0])
    samples = _smooth().sample(other.to_points(), n_samples=4, rng=SEED)
    with pytest.raises(SurfaceValidationError, match="other than this grid's nodes"):
        evaluate_samples(samples, GRID)


# --- the interval ---------------------------------------------------------------


@pytest.mark.parametrize("successes, trials", [(0, 10), (0, 200), (5, 10), (10, 10), (1, 3)])
def test_the_wilson_interval_stays_inside_the_unit_interval(successes, trials) -> None:
    lower, upper = wilson_interval(successes, trials)
    assert 0.0 <= lower <= successes / trials <= upper <= 1.0


def test_the_wilson_interval_is_not_degenerate_at_the_endpoints() -> None:
    # The failure the textbook normal interval has and this one does not.
    assert wilson_interval(0, 50)[1] > 0.0
    assert wilson_interval(50, 50)[0] < 1.0


@pytest.mark.parametrize(
    "successes, trials, message",
    [(0, 0, "trials must be at least 1"), (5, 3, r"successes must lie in \[0, 3\]")],
)
def test_invalid_interval_inputs_are_refused(successes, trials, message) -> None:
    with pytest.raises(SurfaceValidationError, match=message):
        wilson_interval(successes, trials)


# --- the wire contract ------------------------------------------------------------


def test_the_report_validates_against_its_committed_schema() -> None:
    report = evaluate_samples(_rough(), GRID, n_samples=32, rng=SEED)
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.Draft202012Validator(schema).validate(report.to_dict())
    assert report.to_dict()["schema"] == SCHEMA_VERSION


def test_the_generative_schema_and_the_module_agree() -> None:
    assert json.loads(SCHEMA_PATH.read_text(encoding="utf-8")) == generative_json_schema()


def test_the_report_renders_deterministically_with_no_non_standard_floats() -> None:
    report = evaluate_samples(_rough(), GRID, n_samples=16, rng=SEED)
    text = report.to_json(indent=2)
    assert text.endswith("\n")
    assert text == report.to_json(indent=2)
    json.dumps(report.to_dict(), allow_nan=False)
