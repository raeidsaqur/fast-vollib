"""Structural forecast contracts; models and training frameworks stay external."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol, runtime_checkable

from .query import ForecastQuery
from .results import ForecastResult
from .targets import ForecastTarget


@runtime_checkable
class ForecastDrawSet(Protocol):
    sample_ids: Any
    weights: Any

    def query(self, query: ForecastQuery) -> ForecastResult: ...


@runtime_checkable
class ForecastDistribution(Protocol):
    targets: tuple[ForecastTarget, ...]
    support: str
    origin: datetime
    domain: Any
    configured_query: ForecastQuery | None

    def draw(self, *, n_samples: int, rng: Any) -> ForecastDrawSet: ...


class GenerativeForecastModel(Protocol):
    def distribution(
        self, context: Any, *, targets: tuple[ForecastTarget, ...], horizon: Any
    ) -> ForecastDistribution: ...
