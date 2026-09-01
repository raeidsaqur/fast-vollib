"""The contracts a surface model satisfies, and the one thing they all produce.

A model joins this ecosystem by producing a **definite surface** -- an
evaluable map from points to implied volatilities -- or a distribution over
definite surfaces.  Everything downstream (arbitrage checks, fit and price
metrics, differentiable penalties, the benchmark, the application) consumes
those two outputs and nothing model-specific.  That is the whole architectural
claim, and these protocols are where it is written down.

*One hierarchy would have been the wrong shape.*  Calibrating an SVI slice to
today's quotes, conditioning a trained amortized network on a context set,
forecasting tomorrow's surface from a history, and sampling from a generative
model are four different lifecycles.  Forcing them through a single
``fit(data) -> surface`` method makes three of the four lie: a trained network's
``fit`` would mean "train", a forecaster's would silently drop the horizon, and
a generative model would have to collapse a distribution to a point estimate
before anyone asked it to.  So there are four small structural protocols with
one shared output type, rather than one base class with four meanings.

*Training is not conditioning.*  A protocol here never means "learn parameters
from a corpus".  :class:`SurfaceCalibrator` fits *one* surface to *one* set of
observations; :class:`ConditionalSurfaceEstimator` applies an already-trained
model to a context set.  Training lifecycles differ too much between families
to have a useful common contract yet, and inventing one now would force every
future learned model through an interface written before any of them existed.
Dataset selection, schedules, and checkpoint policy live in the benchmark layer
that runs experiments, not in the library that defines the numbers.

These are :func:`~typing.runtime_checkable` Protocols, so ``isinstance`` checks
that the members exist -- not that they behave.  It is a convenience for error
messages, not a verification.

Examples
--------
>>> import numpy as np
>>> from fast_vollib.surface import DefiniteIVSurface, SurfacePoints, SurfacePrediction
>>> class FlatSurface:
...     def __init__(self, level): self.level = level
...     def evaluate(self, points, *, market=None):
...         return SurfacePrediction(points=points, iv=np.full(points.n, self.level))
>>> isinstance(FlatSurface(0.2), DefiniteIVSurface)
True
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, Union, runtime_checkable

import numpy as np

from .errors import SurfaceValidationError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .market import SurfaceMarket
    from .observations import SurfaceObservations
    from .points import SurfacePoints
    from .prediction import SurfacePrediction, SurfaceSamples

__all__ = [
    "ConditionalSurfaceEstimator",
    "DefiniteIVSurface",
    "ForecastHorizon",
    "GenerativeSurfaceModel",
    "RNGInput",
    "SurfaceCalibrator",
    "SurfaceDistribution",
    "SurfaceForecaster",
]

#: What a caller may pass as randomness.  An integer seed builds a local
#: generator; a generator is used as given and advances; ``None`` means the
#: callee is deterministic and will say so if it is not.  There is no hidden
#: module-global stream anywhere in this package.
RNGInput = Union[int, np.random.Generator, None]


@dataclass(frozen=True, slots=True)
class ForecastHorizon:
    """How far ahead a forecast reaches.

    Parameters
    ----------
    steps:
        Number of observation steps ahead, at least 1.  A forecaster's history
        is a sequence of surfaces; the horizon counts in the same units.
    step_years:
        Optional calendar spacing of one step in years.  Supplied when the
        forecaster needs it to age maturities -- a one-day-ahead forecast of a
        30-day option is a forecast of a 29-day option -- and omitted when the
        model works in step units alone.  It is never guessed: a model that
        needs it and is not given it raises.

    Examples
    --------
    >>> from fast_vollib.surface import ForecastHorizon
    >>> ForecastHorizon(steps=5, step_years=1 / 252).steps
    5
    """

    steps: int = 1
    step_years: float | None = None

    def __post_init__(self) -> None:
        if isinstance(self.steps, bool) or not isinstance(self.steps, int):
            raise SurfaceValidationError(
                f"steps must be an integer; got {type(self.steps).__name__}."
            )
        if self.steps < 1:
            raise SurfaceValidationError(f"steps must be at least 1; got {self.steps}.")
        if self.step_years is not None:
            value = float(self.step_years)
            if not (value == value) or value in (float("inf"), float("-inf")):
                raise SurfaceValidationError(f"step_years must be finite; got {value!r}.")
            if value <= 0.0:
                raise SurfaceValidationError(
                    f"step_years must be strictly positive; got {value!r}."
                )

    @property
    def years(self) -> float | None:
        """Total calendar reach in years, or ``None`` when the spacing is unknown."""
        return None if self.step_years is None else self.steps * float(self.step_years)

    def to_dict(self) -> dict[str, Any]:
        """The horizon as a JSON-safe mapping."""
        return {
            "steps": int(self.steps),
            "step_years": None if self.step_years is None else float(self.step_years),
        }


@runtime_checkable
class DefiniteIVSurface(Protocol):
    """An evaluable implied-volatility surface.

    The single output type of every deterministic path in this package.  It is
    a *map*, not a grid and not a training algorithm: given points, it returns
    one implied volatility per point.  A calibrated SVI slice, an SSVI surface,
    a penalized spline, a factor reconstruction, a Heston surface backed by
    characteristic-function pricing, a conditioned network, and one realized
    draw of a generative model are all definite surfaces, and the arbitrage
    harness cannot tell them apart -- which is the point.

    Notes
    -----
    A definite surface is free to decline a point.  Returning a
    :class:`~fast_vollib.surface.prediction.SurfacePrediction` whose ``valid``
    entry is ``False`` says "outside what I will speak for" and is preferable
    to extrapolating; raising
    :class:`~fast_vollib.surface.errors.SurfaceDomainError` is the stricter
    alternative for a surface with a hard domain.  What it must not do is
    return a plausible number it does not stand behind.

    An implementation that needs a market -- anything defined through prices --
    raises :class:`~fast_vollib.surface.errors.MissingMarketStateError` when
    ``market`` is ``None``, rather than assuming one.
    """

    def evaluate(
        self,
        points: "SurfacePoints",
        *,
        market: "SurfaceMarket | None" = None,
    ) -> "SurfacePrediction":
        """Implied volatility at ``points``, one value per row, in row order."""
        ...  # pragma: no cover - protocol declaration


@runtime_checkable
class SurfaceCalibrator(Protocol):
    """Fits one definite surface to one set of observations.

    The classical lifecycle: parameters are found for *these* quotes and mean
    nothing without them.  Calling it twice on two days gives two surfaces and
    changes nothing about the calibrator, which holds configuration only --
    never the last fit.  A stateful calibrator would make the second of two
    otherwise identical runs depend on the order they were run in, which is the
    reproducibility failure this signature is shaped to prevent.
    """

    def fit(
        self,
        observations: "SurfaceObservations",
        *,
        rng: RNGInput = None,
    ) -> DefiniteIVSurface:
        """Calibrate to ``observations`` and return the resulting surface."""
        ...  # pragma: no cover - protocol declaration


@runtime_checkable
class ConditionalSurfaceEstimator(Protocol):
    """Applies an already-trained model to a context set.

    Deliberately *not* spelled ``fit``.  An amortized model -- a conditional
    neural process, a hypernetwork, an operator -- has learned its parameters
    somewhere else; what happens here is inference on today's quotes, which is
    cheap, repeatable, and changes no weights.  Calling it ``fit`` would make
    "how long does fitting take" an unanswerable question across families.
    """

    def condition(
        self,
        context: "SurfaceObservations",
        *,
        rng: RNGInput = None,
    ) -> DefiniteIVSurface:
        """Condition on ``context`` and return the implied definite surface."""
        ...  # pragma: no cover - protocol declaration


@runtime_checkable
class SurfaceForecaster(Protocol):
    """Projects a history of surfaces forward by a stated horizon.

    Returns either a definite surface (a point forecast) or a
    :class:`SurfaceDistribution` (a predictive distribution).  Which one is a
    property of the model and is declared in its capability metadata, so a
    caller never has to inspect the return value to find out what kind of
    forecast it asked for.
    """

    def forecast(
        self,
        history: Sequence["SurfaceObservations"],
        horizon: ForecastHorizon,
        *,
        rng: RNGInput = None,
    ) -> "DefiniteIVSurface | SurfaceDistribution":
        """Forecast ``horizon`` ahead from ``history``, oldest observation first."""
        ...  # pragma: no cover - protocol declaration


@runtime_checkable
class SurfaceDistribution(Protocol):
    """A distribution over definite surfaces, sampled at chosen points.

    Every draw is a whole surface on the requested domain, not a cloud of
    independent pointwise draws -- that is what makes a per-draw arbitrage
    check meaningful.  ``rng`` is required rather than optional: a distribution
    that sampled from a hidden stream could not be replayed, and an evaluation
    that cannot be replayed is not evidence.
    """

    def sample(
        self,
        points: "SurfacePoints",
        *,
        n_samples: int,
        rng: RNGInput,
        market: "SurfaceMarket | None" = None,
    ) -> "SurfaceSamples":
        """Draw ``n_samples`` surfaces evaluated at ``points``."""
        ...  # pragma: no cover - protocol declaration


@runtime_checkable
class GenerativeSurfaceModel(Protocol):
    """Turns a context (and optionally a horizon) into a distribution over surfaces.

    Separating ``distribution`` from ``sample`` keeps the two costs apart:
    conditioning is done once, and drawing is done as many times as the
    evaluation's Monte Carlo error requires.  A model that fused them would
    re-encode the context on every draw and would make the sample count an
    argument to inference rather than to measurement.
    """

    def distribution(
        self,
        context: "SurfaceObservations | Sequence[SurfaceObservations]",
        *,
        horizon: ForecastHorizon | None = None,
    ) -> SurfaceDistribution:
        """The predictive distribution implied by ``context``."""
        ...  # pragma: no cover - protocol declaration
