"""The simplest surfaces that are still surfaces: a flat level, and a prior.

Every comparison needs a floor.  A model that cannot beat a single number
fitted to the same quotes has not demonstrated that its structure is doing
anything, and a benchmark without that floor reports differences between
elaborate models without establishing that elaboration was warranted.

:class:`FlatIVSurface` is one constant implied volatility, evaluable anywhere.
:class:`FlatVolatilityCalibrator` fits that constant to observations under a
declared objective -- averaging implied volatility and averaging variance are
different estimators and give different numbers, so the choice is a parameter
rather than a convention buried in the code.

*A flat surface is arbitrage-free, and that is not a virtue.*  Constant
volatility gives non-decreasing total variance and a positive Black density, so
it passes every hard check while fitting a real smile badly.  It is the control
that keeps an arbitrage report from being read as a quality score.

Examples
--------
>>> import numpy as np
>>> from fast_vollib.surface import SurfaceObservations, SurfacePoints
>>> from fast_vollib.surface.fitting import FlatVolatilityCalibrator
>>> observations = SurfaceObservations(
...     k=[-0.1, 0.0, 0.1], T=[1.0, 1.0, 1.0], iv=[0.22, 0.20, 0.24]
... )
>>> surface = FlatVolatilityCalibrator().fit(observations)
>>> float(round(surface.level, 6))
0.22
>>> float(round(surface.evaluate(SurfacePoints(k=[2.0], T=[5.0])).iv[0], 6))
0.22
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from ..errors import SurfaceCalibrationError, SurfaceValidationError
from ..points import SurfacePoints
from ..prediction import SurfacePrediction

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..market import SurfaceMarket
    from ..observations import SurfaceObservations
    from ..protocols import RNGInput

__all__ = ["OBJECTIVES", "FlatIVSurface", "FlatVolatilityCalibrator"]

#: How the single level is estimated.  ``'implied_volatility'`` averages
#: :math:`\\sigma`; ``'variance'`` averages :math:`\\sigma^2` and takes the root.
OBJECTIVES = ("implied_volatility", "variance")


@dataclass(frozen=True, slots=True)
class FlatIVSurface:
    """One implied volatility, everywhere.

    Parameters
    ----------
    level:
        The constant implied volatility, finite and strictly positive.

    Notes
    -----
    Satisfies :class:`~fast_vollib.surface.protocols.DefiniteIVSurface`.  It
    declines nothing: a constant is defined at every coordinate, so every point
    comes back valid.

    Examples
    --------
    >>> from fast_vollib.surface import SurfacePoints
    >>> from fast_vollib.surface.fitting import FlatIVSurface
    >>> FlatIVSurface(level=0.2).evaluate(SurfacePoints(k=[0.0], T=[1.0])).iv.tolist()
    [0.2]
    """

    level: float

    def __post_init__(self) -> None:
        value = float(self.level)
        if not (value == value) or value in (float("inf"), float("-inf")):
            raise SurfaceValidationError(f"level must be finite; got {self.level!r}.")
        if value <= 0.0:
            raise SurfaceValidationError(
                f"level must be strictly positive; got {value!r}. A zero implied "
                f"volatility prices every option at intrinsic and inverts to nothing."
            )
        object.__setattr__(self, "level", value)

    def evaluate(
        self,
        points: SurfacePoints,
        *,
        market: "SurfaceMarket | None" = None,
    ) -> SurfacePrediction:
        """The constant level at every point."""
        del market  # a constant needs no market state
        if not isinstance(points, SurfacePoints):
            raise SurfaceValidationError(
                f"points must be a SurfacePoints; got {type(points).__name__}."
            )
        return SurfacePrediction(points=points, iv=np.full(points.n, self.level))

    def parameters(self) -> dict[str, Any]:
        """The fitted parameters as a JSON-safe mapping."""
        return {"level": float(self.level)}


@dataclass(frozen=True, slots=True)
class FlatVolatilityCalibrator:
    """Fits one implied-volatility level to a set of observations.

    Parameters
    ----------
    objective:
        One of :data:`OBJECTIVES`.  ``'implied_volatility'`` (default) is the
        weighted mean of :math:`\\sigma`; ``'variance'`` is the root of the
        weighted mean of :math:`\\sigma^2`.  The two agree only when the smile
        is flat, so reporting which one produced a number is part of the number.
    use_weights:
        Whether to honour the observations' ``weight`` column.  A calibrator
        that ignored supplied weights without saying so would make a weighted
        experiment silently unweighted.

    Notes
    -----
    Satisfies :class:`~fast_vollib.surface.protocols.SurfaceCalibrator`.  Holds
    configuration only: two calls on two days share nothing, so the second
    cannot depend on the first.
    """

    objective: str = "implied_volatility"
    use_weights: bool = True

    def __post_init__(self) -> None:
        if self.objective not in OBJECTIVES:
            raise SurfaceValidationError(
                f"objective must be one of {OBJECTIVES}; got {self.objective!r}."
            )

    def fit(
        self,
        observations: "SurfaceObservations",
        *,
        rng: "RNGInput" = None,
    ) -> FlatIVSurface:
        """Calibrate the level to ``observations``.

        Raises
        ------
        SurfaceCalibrationError
            If no observation is usable -- every implied volatility missing, or
            every weight zero.  An empty fit has no level, and returning a
            plausible default would be inventing one.
        """
        del rng  # deterministic
        iv, weight = _usable(observations, self.use_weights)
        total = float(weight.sum())
        if total <= 0.0:
            raise SurfaceCalibrationError(
                "No usable observation: every implied volatility is missing or every "
                "weight is zero, so there is no level to fit."
            )
        if self.objective == "variance":
            level = float(np.sqrt(float((weight * iv * iv).sum()) / total))
        else:
            level = float((weight * iv).sum() / total)
        if not np.isfinite(level) or level <= 0.0:
            raise SurfaceCalibrationError(
                f"The fitted level is {level!r}, which is not a usable implied "
                f"volatility. Check the observations for zero or degenerate quotes."
            )
        return FlatIVSurface(level=level)


def _usable(observations: "SurfaceObservations", use_weights: bool) -> tuple[Any, Any]:
    """Observed implied volatilities and their weights, missing rows dropped."""
    observed = ~np.isnan(observations.iv)
    iv = observations.iv[observed]
    if use_weights and observations.weight is not None:
        weight = observations.weight[observed]
    else:
        weight = np.ones(iv.size, dtype=np.float64)
    return iv, weight
