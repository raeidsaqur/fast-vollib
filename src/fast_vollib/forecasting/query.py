"""Physical-time queries with an explicit origin and actual-day convention."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
from typing import Any

from ._validate import instant, names
from .errors import ForecastQueryError

_DAYS = {"act_365": 365.0, "act_365_25": 365.25, "act_360": 360.0}


@dataclass(frozen=True, slots=True, kw_only=True)
class ForecastQuery:
    origin: datetime
    times: Any
    convention: str
    targets: tuple[str, ...]
    kind: str

    def __post_init__(self) -> None:
        origin = instant(self.origin)
        object.__setattr__(self, "origin", origin)
        if self.convention not in _DAYS:
            raise ForecastQueryError("explicit supported actual-day convention required")
        object.__setattr__(self, "targets", names(self.targets, "targets"))
        times = tuple(self.times)
        if not times:
            raise ForecastQueryError("query requires at least one time")
        if all(isinstance(t, datetime) for t in times):
            years = tuple(
                (instant(t) - origin).total_seconds() / (86400 * _DAYS[self.convention])
                for t in times
            )
        elif all(isinstance(t, (int, float)) and not isinstance(t, bool) for t in times):
            years = tuple(float(t) for t in times)
        else:
            raise ForecastQueryError("times must be all aware instants or all real elapsed years")
        if any(not math.isfinite(t) or t < 0 for t in years) or any(
            a >= b for a, b in zip(years, years[1:])
        ):
            raise ForecastQueryError("times must be finite, nonnegative and strictly increasing")
        if self.kind not in ("path", "marginal"):
            raise ForecastQueryError("kind must be path or marginal")
        if self.kind == "path" and (len(years) < 2 or years[0] != 0):
            raise ForecastQueryError("paths require at least two times starting at zero")
        object.__setattr__(self, "times", years)
