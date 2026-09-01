"""What a model says: one prediction per point, or a set of draws per point.

:class:`SurfacePrediction` is the deterministic answer -- one implied
volatility per queried point, optionally with a pointwise standard deviation or
quantiles.  :class:`SurfaceSamples` is the stochastic answer -- a sample axis of
draws, each row of which is a whole surface on the queried points.

*A model is allowed to fail at a point, and the failure is data.*  Both
containers accept negative and non-finite implied volatilities, because that is
what an unconstrained network, a diverged optimizer, or an extrapolating spline
actually emits.  They are recorded and flagged invalid rather than dropped: an
evaluation that silently discarded them would report the error of the points a
model happened to get right and call it the model's error.

The distinction between the two containers is the distinction the evaluation
layer needs.  A deterministic model is scored against observations point by
point.  A generative model is scored by materializing *every draw* as its own
definite surface, checking each one, and aggregating -- because the mean of a
set of arbitrageable surfaces can itself be arbitrage-free, and reporting only
the mean would hide exactly the defect the check exists to find.

Examples
--------
>>> import numpy as np
>>> from fast_vollib.surface import SurfacePoints, SurfacePrediction
>>> points = SurfacePoints(k=[-0.1, 0.0, 0.1], T=[0.5, 0.5, 0.5])
>>> prediction = SurfacePrediction(points=points, iv=[0.21, 0.20, np.nan])
>>> prediction.valid.tolist()
[True, True, False]
>>> prediction.invalid_count
1
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ._validate import owned_bool_1d, owned_float_1d, owned_float_2d, read_only
from .errors import SurfaceValidationError
from .points import SurfacePoints

__all__ = ["SurfacePrediction", "SurfaceSamples"]


def _default_valid(iv: np.ndarray) -> np.ndarray:
    """The default validity rule: finite and strictly positive.

    A zero implied volatility is not a usable prediction -- it prices every
    option at intrinsic and inverts to nothing -- so it is invalid rather than
    an extreme but acceptable value.
    """
    with np.errstate(invalid="ignore"):
        return read_only(np.isfinite(iv) & (iv > 0.0))


@dataclass(frozen=True, slots=True)
class SurfacePrediction:
    """One predicted implied volatility per point, with optional uncertainty.

    Parameters
    ----------
    points:
        The coordinates the prediction answers.
    iv:
        Predicted implied volatility, shape ``(N,)``.  Non-finite and
        non-positive values are permitted and are marked invalid.
    sd:
        Optional pointwise standard deviation, shape ``(N,)``, non-negative and
        finite where the prediction is valid.
    quantiles:
        Optional predictive quantiles, shape ``(Q, N)``, non-decreasing down the
        level axis at every point.
    quantile_levels:
        The levels the rows of ``quantiles`` correspond to, shape ``(Q,)``,
        strictly increasing and strictly inside ``(0, 1)``.  Required whenever
        ``quantiles`` is given; a quantile without its level is a number nobody
        can interpret.
    valid:
        Optional explicit validity mask, shape ``(N,)``.  Defaults to
        ``isfinite(iv) & (iv > 0)``.  Supplying it lets a model declare a point
        unanswered for its own reason -- outside its fitted domain, below its
        liquidity floor -- rather than only by emitting a NaN.

    Notes
    -----
    Stored arrays are owned, read-only copies.  ``iv`` is *not* sanitized: the
    value a model produced is the value this container reports.

    Examples
    --------
    >>> import numpy as np
    >>> from fast_vollib.surface import SurfacePoints, SurfacePrediction
    >>> points = SurfacePoints(k=[0.0], T=[1.0])
    >>> SurfacePrediction(points=points, iv=[-0.3]).valid.tolist()
    [False]
    """

    points: SurfacePoints
    iv: Any
    sd: Any = None
    quantiles: Any = None
    quantile_levels: Any = None
    valid: Any = None

    def __post_init__(self) -> None:
        if not isinstance(self.points, SurfacePoints):
            raise SurfaceValidationError(
                f"points must be a SurfacePoints; got {type(self.points).__name__}."
            )
        n = self.points.n
        iv = owned_float_1d(self.iv, "iv", n)
        if iv.size != n:
            raise SurfaceValidationError(f"iv must have one value per point ({n}); got {iv.size}.")

        sd = None
        if self.sd is not None:
            sd = owned_float_1d(self.sd, "sd", n)
            if sd.size != n:
                raise SurfaceValidationError(
                    f"sd must have one value per point ({n}); got {sd.size}."
                )
            present = ~np.isnan(sd)
            if not bool(np.all(np.isfinite(sd[present]))):
                raise SurfaceValidationError(
                    "sd must be NaN (not reported) or finite; infinities are rejected."
                )
            if not bool(np.all(sd[present] >= 0.0)):
                raise SurfaceValidationError("sd must be non-negative.")

        quantiles = quantile_levels = None
        if (self.quantiles is None) != (self.quantile_levels is None):
            raise SurfaceValidationError(
                "quantiles and quantile_levels must be given together; a quantile "
                "without its level cannot be interpreted."
            )
        if self.quantiles is not None:
            quantile_levels = owned_float_1d(self.quantile_levels, "quantile_levels")
            q = quantile_levels.size
            quantiles = owned_float_2d(self.quantiles, "quantiles", (q, n))
            if not bool(np.all(np.isfinite(quantile_levels))):
                raise SurfaceValidationError("quantile_levels must be finite everywhere.")
            if not bool(np.all((quantile_levels > 0.0) & (quantile_levels < 1.0))):
                raise SurfaceValidationError(
                    f"quantile_levels must lie strictly inside (0, 1); "
                    f"got {quantile_levels.tolist()}."
                )
            if q > 1 and not bool(np.all(np.diff(quantile_levels) > 0.0)):
                raise SurfaceValidationError(
                    f"quantile_levels must be strictly increasing; got {quantile_levels.tolist()}."
                )
            finite_rows = np.isfinite(quantiles)
            monotone = np.diff(np.where(finite_rows, quantiles, -np.inf), axis=0) >= 0.0
            if q > 1 and not bool(np.all(monotone)):
                raise SurfaceValidationError(
                    "quantiles must be non-decreasing in the level axis at every point; "
                    "a crossing quantile pair describes no distribution."
                )

        valid = _default_valid(iv) if self.valid is None else owned_bool_1d(self.valid, "valid", n)

        object.__setattr__(self, "iv", iv)
        object.__setattr__(self, "sd", sd)
        object.__setattr__(self, "quantiles", quantiles)
        object.__setattr__(self, "quantile_levels", quantile_levels)
        object.__setattr__(self, "valid", valid)

    @property
    def n(self) -> int:
        """Number of predicted points."""
        return self.points.n

    def __len__(self) -> int:
        return self.n

    @property
    def valid_count(self) -> int:
        """How many points carry a usable implied volatility."""
        return int(np.count_nonzero(self.valid))

    @property
    def invalid_count(self) -> int:
        """How many points do not."""
        return self.n - self.valid_count

    @property
    def coverage(self) -> float | None:
        """Fraction of points answered, or ``None`` when there are no points."""
        return None if self.n == 0 else self.valid_count / self.n

    @property
    def has_uncertainty(self) -> bool:
        """Whether a standard deviation or a quantile block is carried."""
        return self.sd is not None or self.quantiles is not None

    def subset(self, index: Any) -> SurfacePrediction:
        """The predictions selected by a boolean mask or an index array."""
        index = np.asarray(index)
        return SurfacePrediction(
            points=self.points.subset(index),
            iv=self.iv[index],
            sd=None if self.sd is None else self.sd[index],
            quantiles=None if self.quantiles is None else self.quantiles[:, index],
            quantile_levels=self.quantile_levels,
            valid=self.valid[index],
        )


@dataclass(frozen=True, slots=True)
class SurfaceSamples:
    """Draws from a distribution over surfaces: ``(n_samples, n_points)``.

    Parameters
    ----------
    points:
        The coordinates every draw answers.  All draws share them, which is what
        makes row ``s`` a definite surface on the same domain as row ``s + 1``
        and lets the aggregate statistics be computed at all.
    iv:
        Sampled implied volatility, shape ``(S, N)``.
    valid:
        Optional validity mask, shape ``(S, N)``.  Defaults to
        ``isfinite(iv) & (iv > 0)`` per entry.
    rng_policy:
        Free text naming the generator and seed the draws came from, e.g.
        ``"numpy.random.default_rng(20260831)"``.  Recorded rather than
        enforced: this container cannot verify a claim about randomness, and
        says only what the producer declared.

    Notes
    -----
    A draw is *not* summarized here.  Aggregation across the sample axis --
    violation probabilities, expected severities, tail quantiles -- belongs to
    :mod:`fast_vollib.surface.generative`, which evaluates each draw as its own
    definite surface first.

    Examples
    --------
    >>> import numpy as np
    >>> from fast_vollib.surface import SurfacePoints, SurfaceSamples
    >>> points = SurfacePoints(k=[-0.1, 0.1], T=[1.0, 1.0])
    >>> samples = SurfaceSamples(points=points, iv=np.full((4, 2), 0.2))
    >>> samples.n_samples, samples.n_points
    (4, 2)
    """

    points: SurfacePoints
    iv: Any
    valid: Any = None
    rng_policy: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.points, SurfacePoints):
            raise SurfaceValidationError(
                f"points must be a SurfacePoints; got {type(self.points).__name__}."
            )
        n = self.points.n
        iv = np.array(self.iv, dtype=np.float64, copy=True)
        if iv.ndim != 2:
            raise SurfaceValidationError(
                f"iv must be two-dimensional (n_samples, n_points); got shape {iv.shape}."
            )
        if iv.shape[1] != n:
            raise SurfaceValidationError(
                f"iv must have one column per point ({n}); got {iv.shape[1]}."
            )
        iv = read_only(iv)

        if self.valid is None:
            with np.errstate(invalid="ignore"):
                valid = read_only(np.isfinite(iv) & (iv > 0.0))
        else:
            valid = np.asarray(self.valid)
            if valid.dtype != np.bool_:
                raise SurfaceValidationError(
                    f"valid must be a boolean array; got dtype {valid.dtype}."
                )
            if valid.shape != iv.shape:
                raise SurfaceValidationError(
                    f"valid must have shape {iv.shape}; got {valid.shape}."
                )
            valid = read_only(np.array(valid, dtype=bool, copy=True))

        if self.rng_policy is not None and not isinstance(self.rng_policy, str):
            raise SurfaceValidationError(
                f"rng_policy must be a string; got {type(self.rng_policy).__name__}."
            )

        object.__setattr__(self, "iv", iv)
        object.__setattr__(self, "valid", valid)

    @property
    def n_samples(self) -> int:
        """Number of draws."""
        return int(self.iv.shape[0])

    @property
    def n_points(self) -> int:
        """Number of points each draw answers."""
        return self.points.n

    def __len__(self) -> int:
        return self.n_samples

    def sample(self, index: int) -> SurfacePrediction:
        """Draw ``index`` as a definite prediction on the shared points."""
        return SurfacePrediction(points=self.points, iv=self.iv[index], valid=self.valid[index])

    def __iter__(self):
        for index in range(self.n_samples):
            yield self.sample(index)

    def mean_prediction(self) -> SurfacePrediction:
        """The pointwise mean over *valid* draws, with its standard deviation.

        Labelled as what it is: a summary of the sample cloud, not a member of
        it.  The mean of arbitrageable surfaces need not be arbitrageable and
        the reverse is equally possible, so this is reported alongside the
        per-draw aggregates and never instead of them.
        """
        contributions = np.where(self.valid, self.iv, 0.0)
        counts = np.count_nonzero(self.valid, axis=0)
        safe_counts = np.maximum(counts, 1)
        mean = contributions.sum(axis=0) / safe_counts
        deviations = np.where(self.valid, self.iv - mean, 0.0)
        variance = (deviations * deviations).sum(axis=0) / np.maximum(counts - 1, 1)
        return SurfacePrediction(
            points=self.points,
            iv=np.where(counts > 0, mean, np.nan),
            sd=np.where(counts > 1, np.sqrt(variance), np.nan),
        )

    def median_prediction(self) -> SurfacePrediction:
        """The pointwise median over *valid* draws."""
        counts = np.count_nonzero(self.valid, axis=0)
        median = np.full(self.n_points, np.nan)
        for column in range(self.n_points):
            if counts[column]:
                median[column] = float(np.median(self.iv[self.valid[:, column], column]))
        return SurfacePrediction(points=self.points, iv=median)
