"""A distribution over surfaces that is simple enough to check the evaluation with.

Production generative surface models -- HyperIV, conditional neural processes,
neural operators, diffusion -- are later milestones.  What is needed *now* is a
distribution over definite surfaces that is real rather than a stub: something
whose draws are genuine surfaces, whose randomness is explicit and replayable,
and which produces arbitrage violations often enough that an evaluation designed
to find them can be shown to find them.

:class:`GaussianFieldSurfaceDistribution` is that model.  It takes a fitted
definite surface and multiplies its total variance by a log-normal random field:

.. math::

    w_s(k, T) = w_0(k, T) \\exp\\!\\left(\\sigma Z_s(k, T) - \\tfrac{1}{2}\\sigma^2\\right),

where :math:`Z` is a zero-mean, unit-variance Gaussian field with a
squared-exponential covariance in :math:`(k, T)`.  The correction term makes the
field multiplicatively unbiased, so the *mean* total variance is the base
surface's -- which matters, because the whole point of the generative evaluation
is that a well-centred mean says nothing about whether the draws are admissible.

*The draws are surfaces, not point clouds.*  The field is drawn jointly across
the queried points from one Cholesky factor, so a sample is a smooth surface and
its butterfly and calendar conditions mean something.  Drawing each point
independently would produce noise, and noise violates every convexity condition
trivially, which would make the evaluation measure the sampler rather than the
model.

*Short length scales break arbitrage, long ones do not.*  That is the knob a test
uses to demonstrate that the aggregation reports what it claims: at a length
scale comparable to the grid spacing nearly every draw has a butterfly
violation; at a long one nearly none does.

Examples
--------
>>> import numpy as np
>>> from fast_vollib.surface import SurfacePoints
>>> from fast_vollib.surface.fitting import FlatIVSurface
>>> from fast_vollib.surface.generative import GaussianFieldSurfaceDistribution
>>> distribution = GaussianFieldSurfaceDistribution(
...     base=FlatIVSurface(level=0.2), volatility=0.05
... )
>>> points = SurfacePoints(k=np.linspace(-0.2, 0.2, 5), T=np.full(5, 1.0))
>>> samples = distribution.sample(points, n_samples=4, rng=20260831)
>>> samples.n_samples, samples.n_points
(4, 5)
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
import math
from typing import TYPE_CHECKING, cast

import numpy as np

from ..errors import SurfaceValidationError
from ..points import SurfacePoints
from ..prediction import SurfaceSamples

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..market import SurfaceMarket
    from ..observations import SurfaceObservations
    from ..protocols import DefiniteIVSurface, ForecastHorizon, RNGInput, SurfaceCalibrator

__all__ = [
    "MAX_JOINT_POINTS",
    "GaussianFieldSurfaceDistribution",
    "GaussianFieldSurfaceGenerator",
]

#: Largest number of points the joint draw will factorize.  The covariance is
#: dense and the Cholesky is cubic, so a caller asking for a hundred thousand
#: points is asking for an hour; refusing is more useful than delivering it.
MAX_JOINT_POINTS = 4000

#: Relative jitter added to the covariance diagonal before factorization.  A
#: squared-exponential kernel on nearly coincident points is numerically
#: singular, and the alternative to a stated jitter is a Cholesky that fails on
#: data the caller cannot see anything wrong with.
_JITTER = 1e-10


@dataclass(frozen=True, slots=True)
class GaussianFieldSurfaceDistribution:
    """A log-normal random field multiplying a base surface's total variance.

    Parameters
    ----------
    base:
        The definite surface the field perturbs.  Its own validity carries
        through: a point the base declines is declined in every draw.
    volatility:
        Standard deviation of the log shock, strictly positive.  0.05 moves total
        variance by about five percent, which is a plausible day-to-day move and
        rarely breaks convexity; 0.5 breaks it often.
    length_scale_k, length_scale_T:
        Correlation lengths of the field in log-moneyness and in maturity, both
        strictly positive.  Short relative to the grid spacing gives rough draws
        that violate butterfly conditions; long gives smooth draws that mostly do
        not.
    market:
        Optional market state passed through to the base surface.

    Notes
    -----
    Satisfies :class:`~fast_vollib.surface.protocols.SurfaceDistribution`.
    ``rng`` is a required keyword on :meth:`sample`, not an optional one: a
    distribution that drew from a hidden stream could not be replayed, and an
    evaluation that cannot be replayed is not evidence.
    """

    base: "DefiniteIVSurface"
    volatility: float = 0.05
    length_scale_k: float = 0.25
    length_scale_T: float = 1.0

    def __post_init__(self) -> None:
        if not hasattr(self.base, "evaluate"):
            raise SurfaceValidationError(
                f"base must be a definite surface with an evaluate method; got "
                f"{type(self.base).__name__}."
            )
        for name in ("volatility", "length_scale_k", "length_scale_T"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise SurfaceValidationError(
                    f"{name} must be finite and strictly positive; got {getattr(self, name)!r}."
                )
            object.__setattr__(self, name, value)

    def sample(
        self,
        points: SurfacePoints,
        *,
        n_samples: int,
        rng: "RNGInput",
        market: "SurfaceMarket | None" = None,
    ) -> SurfaceSamples:
        """Draw ``n_samples`` surfaces jointly across ``points``.

        Raises
        ------
        SurfaceValidationError
            On a non-positive sample count, or more than
            :data:`MAX_JOINT_POINTS` points.
        """
        if not isinstance(points, SurfacePoints):
            raise SurfaceValidationError(
                f"points must be a SurfacePoints; got {type(points).__name__}."
            )
        if isinstance(n_samples, bool) or not isinstance(n_samples, int):
            raise SurfaceValidationError(
                f"n_samples must be an integer; got {type(n_samples).__name__}."
            )
        if n_samples < 1:
            raise SurfaceValidationError(f"n_samples must be at least 1; got {n_samples}.")
        if points.n > MAX_JOINT_POINTS:
            raise SurfaceValidationError(
                f"A joint draw over {points.n} points needs a dense Cholesky of that "
                f"size; the limit is {MAX_JOINT_POINTS}. Sample on a coarser grid, or "
                f"draw the surfaces one mesh at a time."
            )
        generator = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
        prediction = self.base.evaluate(points, market=market)
        base_variance = np.where(prediction.valid, prediction.iv**2 * points.T, np.nan)

        factor = self._cholesky(points)
        normals = generator.standard_normal((n_samples, points.n))
        field = normals @ factor.T
        shocked = base_variance[None, :] * np.exp(
            self.volatility * field - 0.5 * self.volatility * self.volatility
        )
        with np.errstate(invalid="ignore"):
            iv = np.sqrt(shocked / points.T[None, :])
        return SurfaceSamples(
            points=points,
            iv=iv,
            valid=np.isfinite(iv) & (iv > 0.0),
            rng_policy=(
                f"numpy.random.Generator(PCG64) seeded {rng!r}; "
                f"joint squared-exponential field, volatility={self.volatility}, "
                f"length_scale_k={self.length_scale_k}, length_scale_T={self.length_scale_T}"
            ),
        )

    def _cholesky(self, points: SurfacePoints) -> np.ndarray:
        """Lower Cholesky factor of the squared-exponential covariance."""
        dk = (points.k[:, None] - points.k[None, :]) / self.length_scale_k
        dt = (points.T[:, None] - points.T[None, :]) / self.length_scale_T
        covariance = np.exp(-0.5 * (dk * dk + dt * dt))
        covariance[np.diag_indices_from(covariance)] += _JITTER
        return np.linalg.cholesky(covariance)


@dataclass(frozen=True, slots=True)
class GaussianFieldSurfaceGenerator:
    """Conditions on quotes by fitting, then perturbs the fit.

    Parameters
    ----------
    calibrator:
        Turns the context into a definite surface.  Defaults to
        :class:`~fast_vollib.surface.fitting.FlatVolatilityCalibrator`.
    volatility, length_scale_k, length_scale_T:
        Passed to the distribution.

    Notes
    -----
    Satisfies :class:`~fast_vollib.surface.protocols.GenerativeSurfaceModel`.
    Conditioning and sampling are separate calls, so the context is encoded once
    and the sample count is an argument to *measurement* rather than to
    inference.

    The horizon is accepted and ignored, and the docstring says so rather than
    the code implying otherwise: the field has no dynamics, so a one-day-ahead
    distribution and a one-year-ahead distribution are the same distribution.
    Anything else would be a model this class does not contain.
    """

    calibrator: "SurfaceCalibrator" = field(default=None)  # type: ignore[assignment]
    volatility: float = 0.05
    length_scale_k: float = 0.25
    length_scale_T: float = 1.0

    def __post_init__(self) -> None:
        if self.calibrator is None:
            from ..fitting.prior import FlatVolatilityCalibrator

            object.__setattr__(self, "calibrator", FlatVolatilityCalibrator())
        if not hasattr(self.calibrator, "fit"):
            raise SurfaceValidationError(
                f"calibrator must have a fit method; got {type(self.calibrator).__name__}."
            )

    def distribution(
        self,
        context: "SurfaceObservations | Sequence[SurfaceObservations]",
        *,
        horizon: "ForecastHorizon | None" = None,
    ) -> GaussianFieldSurfaceDistribution:
        """Fit the most recent context and wrap the fit in a field.

        ``horizon`` is accepted and ignored; see the class notes.
        """
        del horizon
        if isinstance(context, Sequence) and not hasattr(context, "iv"):
            if len(context) == 0:
                raise SurfaceValidationError(
                    "context is empty, so there is nothing to condition on."
                )
            latest: "SurfaceObservations" = context[-1]
        else:
            latest = cast("SurfaceObservations", context)
        return GaussianFieldSurfaceDistribution(
            base=self.calibrator.fit(latest),
            volatility=self.volatility,
            length_scale_k=self.length_scale_k,
            length_scale_T=self.length_scale_T,
        )
