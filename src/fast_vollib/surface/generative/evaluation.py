"""Scoring a distribution over surfaces by scoring every surface it draws.

*A mean surface is not an evaluation of a generative model.*  Average a hundred
arbitrageable surfaces and the butterfly violations, which point in different
directions on different draws, can cancel: the mean comes back convex and every
draw the model would actually produce is inadmissible.  Averaging can also work
the other way, manufacturing a violation the draws do not have.  Either way the
number describes the average, and nobody trades the average.

So every draw is materialized as its own definite surface on the declared grid,
checked by the same hard conditions a deterministic model is checked by, and only
then aggregated.  What comes out is a distribution of outcomes -- the probability
that a draw has any violation, the expected fraction and severity, the tail --
rather than one number about a surface that was never sampled.

The mean and median surface metrics *are* reported, because they are useful, and
they are labelled as summaries of the cloud rather than as members of it.

*The probability is estimated, so it carries its own error.*  A hundred draws
that all pass do not establish that violations are impossible; they establish
that the rate is below roughly three percent.  Every probability here comes with
a Monte Carlo standard error and a Wilson interval, which stays sensible at zero
and one where the textbook normal interval does not.

Examples
--------
>>> import numpy as np
>>> from fast_vollib.surface import SurfaceGridSpec
>>> from fast_vollib.surface.fitting import FlatIVSurface
>>> from fast_vollib.surface.generative import (
...     GaussianFieldSurfaceDistribution, evaluate_samples,
... )
>>> grid = SurfaceGridSpec(k=np.linspace(-0.3, 0.3, 9), T=[0.5, 1.0])
>>> distribution = GaussianFieldSurfaceDistribution(
...     base=FlatIVSurface(level=0.2), volatility=0.02, length_scale_k=1.0
... )
>>> report = evaluate_samples(distribution, grid, n_samples=16, rng=20260831)
>>> report.n_samples, report.verification.value
(16, 'empirical_finite_grid')
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import TYPE_CHECKING, Any

import numpy as np

from ..errors import SurfaceValidationError
from ..evaluation import VerificationLevel
from ..gridspec import SurfaceGridSpec
from ..materialize import materialize_samples
from ..prediction import SurfaceSamples

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..market import SurfaceMarket
    from ..protocols import RNGInput, SurfaceDistribution

__all__ = [
    "CONDITIONS",
    "DEFAULT_SEVERITY_QUANTILES",
    "SCHEMA_VERSION",
    "GenerativeArbitrageReport",
    "evaluate_samples",
    "generative_json_schema",
    "render_generative_json_schema",
    "wilson_interval",
]

#: The wire identifier of the serialized generative report.
SCHEMA_VERSION = "fast-vollib-generative-arbitrage-v1"

_SCHEMA_ID = (
    "https://raeidsaqur.github.io/fast-vollib/schemas/"
    "fast-vollib-generative-arbitrage-v1.schema.json"
)

#: The four no-arbitrage condition families, in the order they are reported.
CONDITIONS = ("butterfly", "calendar", "vertical", "bound")

#: Severity quantiles reported by default.  The median says what a typical draw
#: looks like and the upper two say what the bad ones look like, which is the
#: question a risk reader is actually asking.
DEFAULT_SEVERITY_QUANTILES = (0.5, 0.9, 0.99)

_FRACTION_KEYS = {
    "butterfly": "bfly_frac",
    "calendar": "cal_frac",
    "vertical": "vert_frac",
    "bound": "bound_frac",
}


def wilson_interval(
    successes: int, trials: int, *, z: float = 1.959963984540054
) -> tuple[float, float]:
    """A Wilson score interval for a binomial proportion.

    Parameters
    ----------
    successes, trials:
        Counts, with ``0 <= successes <= trials``.
    z:
        Normal quantile; the default is the two-sided 95% value.

    Returns
    -------
    ``(lower, upper)``, both inside ``[0, 1]``.

    Notes
    -----
    Used instead of the textbook ``p +/- z sqrt(p(1-p)/n)`` because that interval
    collapses to a point at ``p = 0`` and at ``p = 1``, which is exactly where a
    generative arbitrage report lands most often and exactly where the answer
    matters: "none of two hundred draws violated" is not "the probability is
    zero", and the Wilson upper bound says what it is instead.

    Examples
    --------
    >>> from fast_vollib.surface.generative import wilson_interval
    >>> lower, upper = wilson_interval(0, 200)
    >>> lower == 0.0, round(upper, 4)
    (True, 0.0188)
    """
    if trials <= 0:
        raise SurfaceValidationError(f"trials must be at least 1; got {trials}.")
    if not (0 <= successes <= trials):
        raise SurfaceValidationError(f"successes must lie in [0, {trials}]; got {successes}.")
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    centre = (proportion + z * z / (2 * trials)) / denominator
    spread = (
        z
        / denominator
        * math.sqrt(proportion * (1.0 - proportion) / trials + z * z / (4 * trials * trials))
    )
    # In exact arithmetic the interval always contains the point estimate. In
    # float64 at p = 1 the centre-plus-spread lands one ulp below it, and an
    # interval that excludes its own estimate is a defect however small: a caller
    # checking whether a rate is inside the reported band would be told no.
    lower = min(max(0.0, centre - spread), proportion)
    upper = max(min(1.0, centre + spread), proportion)
    return (lower, upper)


@dataclass(frozen=True, slots=True)
class GenerativeArbitrageReport:
    """Aggregate no-arbitrage statistics over the draws of a distribution.

    Attributes
    ----------
    n_samples : int
        Draws evaluated.  Every probability below is an estimate from this many
        Bernoulli trials, and its interval says how much that is worth.
    n_nodes : int
        Mesh nodes each draw was checked on.
    grid_shape : tuple[int, int]
    rng_policy : str | None
        What the producer declared about its randomness.  Recorded, not verified.
    valid_sample_fraction : float
        Fraction of draws in which every node was answered.  A draw with a hole
        is still checked, on the nodes it has.
    point_coverage : float
        Mean fraction of nodes answered across draws.
    any_violation_probability : float
        Fraction of draws with at least one violation of any condition.
    any_violation_interval : tuple[float, float]
        Wilson 95% interval for that probability.
    any_violation_stderr : float
        Monte Carlo standard error of the same estimate.
    expected_condition_violation_fraction : float
        Mean over draws of the mean over the four condition families of each
        family's violating-stencil fraction.  Stated that precisely because the
        four families are checked on differently shaped stencils and have no
        common denominator; the per-family means are reported separately below.
    expected_severity : float
        Mean composite severity across draws.
    severity_quantiles : tuple[tuple[float, float], ...]
        ``(level, value)`` pairs of the severity distribution.
    worst_severity : float
    condition_probability : tuple[tuple[str, float], ...]
        Per-family probability that a draw violates that family at least once.
    condition_expected_fraction : tuple[tuple[str, float], ...]
        Per-family mean violating fraction.
    mean_surface_metrics, median_surface_metrics : dict | None
        Hard-check metrics of the pointwise mean and median surface, labelled as
        summaries of the cloud rather than as draws from it.
    verification : VerificationLevel
        What kind of statement this is.  Always the empirical finite-grid level
        unless a caller overrides it, because that is what was measured.
    """

    n_samples: int
    n_nodes: int
    grid_shape: tuple[int, int]
    rng_policy: str | None
    valid_sample_fraction: float
    point_coverage: float
    any_violation_probability: float
    any_violation_interval: tuple[float, float]
    any_violation_stderr: float
    expected_condition_violation_fraction: float
    expected_severity: float
    severity_quantiles: tuple[tuple[float, float], ...]
    worst_severity: float
    condition_probability: tuple[tuple[str, float], ...]
    condition_expected_fraction: tuple[tuple[str, float], ...]
    mean_surface_metrics: dict[str, float] | None = None
    median_surface_metrics: dict[str, float] | None = None
    verification: VerificationLevel = VerificationLevel.EMPIRICAL_FINITE_GRID

    def to_dict(self) -> dict[str, Any]:
        """The report as a canonical, JSON-safe mapping."""
        return {
            "schema": SCHEMA_VERSION,
            "samples": {
                "n_samples": int(self.n_samples),
                "n_nodes": int(self.n_nodes),
                "n_moneyness": int(self.grid_shape[0]),
                "n_maturities": int(self.grid_shape[1]),
                "rng_policy": self.rng_policy,
                "valid_sample_fraction": float(self.valid_sample_fraction),
                "point_coverage": float(self.point_coverage),
            },
            "any_violation": {
                "probability": float(self.any_violation_probability),
                "stderr": float(self.any_violation_stderr),
                "interval_lower": float(self.any_violation_interval[0]),
                "interval_upper": float(self.any_violation_interval[1]),
            },
            "severity": {
                "expected": float(self.expected_severity),
                "worst": float(self.worst_severity),
                "expected_condition_violation_fraction": float(
                    self.expected_condition_violation_fraction
                ),
                "quantiles": [
                    {"level": float(level), "value": float(value)}
                    for level, value in self.severity_quantiles
                ],
            },
            "conditions": [
                {
                    "condition": name,
                    "probability": float(probability),
                    "expected_fraction": float(fraction),
                }
                for (name, probability), (_, fraction) in zip(
                    self.condition_probability, self.condition_expected_fraction
                )
            ],
            "summary_surfaces": {
                "mean": self.mean_surface_metrics,
                "median": self.median_surface_metrics,
            },
            "verification": self.verification.value,
        }

    def to_json(self, *, indent: int | None = None) -> str:
        """Canonical, newline-terminated JSON."""
        separators = (",", ":") if indent is None else (",", ": ")
        return (
            json.dumps(
                self.to_dict(),
                ensure_ascii=False,
                allow_nan=False,
                indent=indent,
                separators=separators,
                sort_keys=False,
            )
            + "\n"
        )


def evaluate_samples(
    source: "SurfaceSamples | SurfaceDistribution",
    grid: SurfaceGridSpec,
    *,
    n_samples: int | None = None,
    rng: "RNGInput" = None,
    market: "SurfaceMarket | None" = None,
    severity_quantiles: tuple[float, ...] = DEFAULT_SEVERITY_QUANTILES,
    summary_surfaces: bool = True,
    verification: VerificationLevel | None = None,
    tolerance: float | None = None,
) -> GenerativeArbitrageReport:
    """Check every draw on ``grid`` and aggregate.

    Parameters
    ----------
    source:
        Either already-drawn :class:`~fast_vollib.surface.prediction.SurfaceSamples`
        on the grid's own nodes, or a
        :class:`~fast_vollib.surface.protocols.SurfaceDistribution` to draw from,
        in which case ``n_samples`` and ``rng`` are required.
    grid:
        The declared mesh.  Everything reported is a statement about these nodes.
    n_samples, rng:
        Draw count and randomness, when ``source`` is a distribution.
    market:
        Market state passed to the distribution and used when materializing.
    severity_quantiles:
        Levels of the severity distribution to report, each strictly inside
        ``(0, 1)``.
    summary_surfaces:
        Whether to also check the pointwise mean and median surfaces.
    verification:
        Override the reported verification level.  Defaults to
        :attr:`~fast_vollib.surface.evaluation.VerificationLevel.EMPIRICAL_FINITE_GRID`,
        which is what a finite-grid check establishes.
    tolerance:
        Passed to :func:`~fast_vollib.surface.metrics.validate_surface`.

    Raises
    ------
    SurfaceValidationError
        If a distribution is given without a sample count or randomness, if the
        samples were not drawn on the grid's nodes, or if a quantile level lies
        outside ``(0, 1)``.
    """
    if not isinstance(grid, SurfaceGridSpec):
        raise SurfaceValidationError(f"grid must be a SurfaceGridSpec; got {type(grid).__name__}.")
    for level in severity_quantiles:
        if not (0.0 < float(level) < 1.0):
            raise SurfaceValidationError(
                f"Every severity quantile level must lie strictly inside (0, 1); got {level!r}."
            )

    samples = _draw(source, grid, n_samples=n_samples, rng=rng, market=market)
    if samples.n_samples == 0:
        raise SurfaceValidationError("There are no draws to evaluate.")

    validate_kwargs: dict[str, Any] = {"compute_trust": False}
    if tolerance is not None:
        validate_kwargs["tolerance"] = tolerance

    any_violation = 0
    condition_hits = dict.fromkeys(CONDITIONS, 0)
    condition_fractions: dict[str, list[float]] = {name: [] for name in CONDITIONS}
    severities: list[float] = []
    complete_samples = 0
    for index, surface in enumerate(materialize_samples(samples, grid, market=market)):
        report = surface.validate(**validate_kwargs)
        severities.append(float(report.sas))
        violated = False
        for name in CONDITIONS:
            fraction = float(report.metrics[_FRACTION_KEYS[name]])
            condition_fractions[name].append(fraction)
            hit = bool(report.by_condition.get(name, {}).get("count", 0))
            if hit:
                condition_hits[name] += 1
                violated = True
        if violated or not report.passed:
            any_violation += 1
        if bool(np.all(samples.valid[index])):
            complete_samples += 1

    n = samples.n_samples
    probability = any_violation / n
    quantile_values = tuple(
        (float(level), float(np.quantile(severities, level))) for level in severity_quantiles
    )
    per_condition_mean = {
        name: float(np.mean(values)) for name, values in condition_fractions.items()
    }

    mean_metrics = median_metrics = None
    if summary_surfaces:
        mean_metrics = _summary_metrics(samples.mean_prediction(), grid, market, validate_kwargs)
        median_metrics = _summary_metrics(
            samples.median_prediction(), grid, market, validate_kwargs
        )

    return GenerativeArbitrageReport(
        n_samples=n,
        n_nodes=samples.n_points,
        grid_shape=grid.shape,
        rng_policy=samples.rng_policy,
        valid_sample_fraction=complete_samples / n,
        point_coverage=float(np.mean(samples.valid)),
        any_violation_probability=probability,
        any_violation_interval=wilson_interval(any_violation, n),
        any_violation_stderr=math.sqrt(max(probability * (1.0 - probability), 0.0) / n),
        expected_condition_violation_fraction=float(np.mean(list(per_condition_mean.values()))),
        expected_severity=float(np.mean(severities)),
        severity_quantiles=quantile_values,
        worst_severity=float(np.max(severities)),
        condition_probability=tuple((name, condition_hits[name] / n) for name in CONDITIONS),
        condition_expected_fraction=tuple((name, per_condition_mean[name]) for name in CONDITIONS),
        mean_surface_metrics=mean_metrics,
        median_surface_metrics=median_metrics,
        verification=(
            VerificationLevel.EMPIRICAL_FINITE_GRID if verification is None else verification
        ),
    )


def _draw(
    source: Any,
    grid: SurfaceGridSpec,
    *,
    n_samples: int | None,
    rng: Any,
    market: Any,
) -> SurfaceSamples:
    """Either the samples given, or a fresh draw from the distribution given."""
    if isinstance(source, SurfaceSamples):
        if n_samples is not None:
            raise SurfaceValidationError(
                "n_samples was given alongside already-drawn samples. Drop it, or pass "
                "the distribution instead -- silently ignoring it would let a caller "
                "believe a different number of draws was evaluated."
            )
        return source
    if not hasattr(source, "sample"):
        raise SurfaceValidationError(
            f"source must be SurfaceSamples or a SurfaceDistribution; got {type(source).__name__}."
        )
    if n_samples is None:
        raise SurfaceValidationError(
            "Sampling a distribution needs an explicit n_samples: the sample count is "
            "what the reported Monte Carlo error is a function of."
        )
    if rng is None:
        raise SurfaceValidationError(
            "Sampling a distribution needs explicit randomness. An evaluation that "
            "cannot be replayed is not evidence."
        )
    return source.sample(grid.to_points(), n_samples=n_samples, rng=rng, market=market)


def _summary_metrics(
    prediction: Any, grid: SurfaceGridSpec, market: Any, validate_kwargs: dict[str, Any]
) -> dict[str, float]:
    """Hard-check metrics of a pointwise summary surface.

    Labelled as a summary everywhere it appears.  The mean of a set of
    arbitrageable surfaces need not be arbitrageable and the reverse is equally
    possible, so this number answers a different question from the aggregate
    probabilities and is never reported instead of them.
    """
    from ..grid import IVSurface

    state = market if market is not None else grid.market
    forward: Any = 1.0
    rate: Any = 0.0
    if state is not None:
        forward = np.asarray(state.forward_at(grid.T), dtype=np.float64)
        rate = np.asarray(state.rate_at(grid.T), dtype=np.float64)
    iv = np.where(prediction.valid, prediction.iv, np.nan).reshape(grid.shape)
    surface = IVSurface(
        k=np.array(grid.k, copy=True),
        T=np.array(grid.T, copy=True),
        iv=iv,
        forward=forward,
        r=rate,
        native_mask=grid.native_mask,
    )
    report = surface.validate(**validate_kwargs)
    return {"passed": float(report.passed), "sas": float(report.sas), **report.metrics}


# --- the wire schema ----------------------------------------------------------


def _closed(properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": sorted(properties),
        "properties": properties,
    }


def generative_json_schema() -> dict[str, Any]:
    """The closed Draft 2020-12 schema for ``fast-vollib-generative-arbitrage-v1``."""
    probability = {"type": "number", "minimum": 0.0, "maximum": 1.0}
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": _SCHEMA_ID,
        "title": "fast-vollib generative-arbitrage-v1",
        **_closed(
            {
                "schema": {"const": SCHEMA_VERSION},
                "samples": _closed(
                    {
                        "n_samples": {"type": "integer", "minimum": 1},
                        "n_nodes": {"type": "integer", "minimum": 1},
                        "n_moneyness": {"type": "integer", "minimum": 1},
                        "n_maturities": {"type": "integer", "minimum": 1},
                        "rng_policy": {"type": ["string", "null"]},
                        "valid_sample_fraction": probability,
                        "point_coverage": probability,
                    }
                ),
                "any_violation": _closed(
                    {
                        "probability": probability,
                        "stderr": {"type": "number", "minimum": 0.0},
                        "interval_lower": probability,
                        "interval_upper": probability,
                    }
                ),
                "severity": _closed(
                    {
                        "expected": {"type": "number", "minimum": 0.0},
                        "worst": {"type": "number", "minimum": 0.0},
                        "expected_condition_violation_fraction": probability,
                        "quantiles": {
                            "type": "array",
                            "items": _closed(
                                {
                                    "level": {
                                        "type": "number",
                                        "exclusiveMinimum": 0.0,
                                        "exclusiveMaximum": 1.0,
                                    },
                                    "value": {"type": "number", "minimum": 0.0},
                                }
                            ),
                        },
                    }
                ),
                "conditions": {
                    "type": "array",
                    "items": _closed(
                        {
                            "condition": {"type": "string", "enum": list(CONDITIONS)},
                            "probability": probability,
                            "expected_fraction": probability,
                        }
                    ),
                },
                "summary_surfaces": _closed(
                    {
                        "mean": {"type": ["object", "null"]},
                        "median": {"type": ["object", "null"]},
                    }
                ),
                "verification": {
                    "type": "string",
                    "enum": [level.value for level in VerificationLevel],
                },
            }
        ),
    }


def render_generative_json_schema() -> str:
    """The schema as the exact text checked in at ``docs/schemas``."""
    return (
        json.dumps(generative_json_schema(), indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
