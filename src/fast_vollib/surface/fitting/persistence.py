"""Tomorrow looks like today: the forecast every other forecast must beat.

Implied-volatility surfaces are highly persistent, so a random walk is a
genuinely strong forecast and a genuinely hard baseline.  A model that does not
beat it has not shown that its dynamics are worth their parameters, and a
benchmark that omits it can report a large improvement over a weak alternative
and mean nothing by it.

:class:`PersistenceForecaster` calibrates the most recent observation set and
returns that surface, unchanged, for any horizon.  The horizon is accepted and
deliberately ignored -- that *is* the model -- and the returned surface says so
rather than pretending the maturities were aged.

*The baseline is a composition, not a special case.*  It needs a calibrator to
turn the last scattered observations into something evaluable at arbitrary
points, and it takes any calibrator, so "persistence of an SVI fit" and
"persistence of a spline fit" are the same object with a different part.

Examples
--------
>>> from fast_vollib.surface import ForecastHorizon, SurfaceObservations, SurfacePoints
>>> from fast_vollib.surface.fitting import FlatVolatilityCalibrator, PersistenceForecaster
>>> history = [
...     SurfaceObservations(k=[0.0], T=[1.0], iv=[0.30]),
...     SurfaceObservations(k=[0.0], T=[1.0], iv=[0.20]),
... ]
>>> forecaster = PersistenceForecaster(calibrator=FlatVolatilityCalibrator())
>>> surface = forecaster.forecast(history, ForecastHorizon(steps=5))
>>> float(round(surface.evaluate(SurfacePoints(k=[0.1], T=[2.0])).iv[0], 6))
0.2
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..errors import SurfaceValidationError
from .prior import FlatVolatilityCalibrator

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..observations import SurfaceObservations
    from ..protocols import (
        DefiniteIVSurface,
        ForecastHorizon,
        RNGInput,
        SurfaceCalibrator,
    )

__all__ = ["PersistenceForecaster"]


@dataclass(frozen=True, slots=True)
class PersistenceForecaster:
    """Forecasts the last observed surface, for every horizon.

    Parameters
    ----------
    calibrator:
        How the most recent observations become an evaluable surface.  Defaults
        to :class:`~fast_vollib.surface.fitting.FlatVolatilityCalibrator`, which
        makes the baseline as weak as it can honestly be; pass a richer
        calibrator to ask the sharper question -- does this model's *dynamics*
        beat no dynamics at all, holding the representation fixed?

    Notes
    -----
    Satisfies :class:`~fast_vollib.surface.protocols.SurfaceForecaster` and
    returns a definite surface, never a distribution: a random walk with no
    innovation distribution attached has a point forecast and nothing else, and
    manufacturing an interval here would be inventing a model.
    """

    calibrator: "SurfaceCalibrator" = field(default_factory=FlatVolatilityCalibrator)

    def forecast(
        self,
        history: Sequence["SurfaceObservations"],
        horizon: "ForecastHorizon",
        *,
        rng: "RNGInput" = None,
    ) -> "DefiniteIVSurface":
        """Calibrate the last element of ``history`` and return it unchanged.

        Parameters
        ----------
        history:
            Observation sets, oldest first.  Only the last is used; the rest are
            required so the signature matches every other forecaster and a
            caller cannot accidentally pass a single surface where a history was
            expected.
        horizon:
            Accepted and ignored.  Persistence is horizon-invariant by
            construction.

        Raises
        ------
        SurfaceValidationError
            If ``history`` is empty.  There is nothing to persist.
        """
        del horizon, rng
        if isinstance(history, str) or not isinstance(history, Sequence):
            raise SurfaceValidationError(
                f"history must be a sequence of SurfaceObservations, oldest first; "
                f"got {type(history).__name__}."
            )
        if len(history) == 0:
            raise SurfaceValidationError(
                "history is empty, so there is no last surface to persist."
            )
        return self.calibrator.fit(history[-1])
