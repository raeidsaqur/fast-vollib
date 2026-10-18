"""A single-date, fixed-point wrapper over one joint surface draw."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import Any, ClassVar
from uuid import uuid4

from .._validate import sample_count
from ..draws import FixedGridDrawSet
from ..errors import ForecastError
from ..query import ForecastQuery
from ..results import ForecastMeta, ForecastResult, SurfaceBlock, _sealed_points
from ..targets import ForecastTarget, Quantity


@dataclass(frozen=True, slots=True, kw_only=True)
class SurfaceDistributionAdapter:
    surface_distribution: Any
    points: Any
    date: datetime
    query: ForecastQuery
    target: ForecastTarget
    meta: ForecastMeta
    market: Any = None
    support: ClassVar[str] = "fixed_grid"

    def __post_init__(self) -> None:
        from .._validate import instant

        if (
            self.query.kind != "marginal"
            or len(self.query.times) != 1
            or self.query.targets != (self.target.target_id,)
        ):
            raise ForecastError("surface adapter requires one date and one target")
        if self.target.quantity != Quantity.IV_SURFACE or self.target.coordinates != ("k", "T"):
            raise ForecastError(
                "surface adapter requires an IV target with canonical (k, T) coordinates"
            )
        date = instant(self.date)
        dated = ForecastQuery(
            origin=self.query.origin,
            times=(date,),
            convention=self.query.convention,
            targets=self.query.targets,
            kind="marginal",
        )
        if dated.times != self.query.times or self.meta.conditioning_cutoff > self.origin:
            raise ForecastError("surface date or conditioning cutoff disagrees with query")
        if not callable(getattr(self.surface_distribution, "sample", None)):
            raise ForecastError("surface distribution must expose sample")
        if self.meta.weighting_scheme == "importance":
            raise ForecastError("surface sampler supplies no importance weights")
        object.__setattr__(self, "date", date)
        object.__setattr__(self, "points", _sealed_points(self.points))

    @property
    def targets(self) -> tuple[ForecastTarget, ...]:
        return (self.target,)

    @property
    def configured_query(self) -> ForecastQuery:
        return self.query

    @property
    def origin(self) -> datetime:
        return self.query.origin

    @property
    def domain(self) -> Any:
        return MappingProxyType({"time": self.query.times[0], "points": self.points})

    def draw(self, *, n_samples: int, rng: Any) -> FixedGridDrawSet:
        from dataclasses import replace

        from ...surface import SurfaceSamples

        n_samples = sample_count(n_samples)
        samples = self.surface_distribution.sample(
            self.points, n_samples=n_samples, rng=rng, market=self.market
        )
        if not isinstance(samples, SurfaceSamples) or samples.n_samples != n_samples:
            raise ForecastError("surface sampler returned wrong type or sample count")
        import numpy as np

        for name in ("k", "T", "surface_id", "point_id"):
            if not np.array_equal(getattr(samples.points, name), getattr(self.points, name)):
                raise ForecastError("surface sampler changed query point identity/order")
        draw_id = uuid4().hex
        block = SurfaceBlock(
            target_id=self.target.target_id,
            time=self.query.times[0],
            points=self.points,
            iv=samples.iv,
            valid=samples.valid,
        )
        meta = replace(self.meta, draw_set_id=draw_id, seed=rng if type(rng) is int else None)
        return FixedGridDrawSet(
            ForecastResult(
                query=self.query,
                targets=self.targets,
                blocks={block.target_id: block},
                sample_ids=[f"{draw_id}:{i}" for i in range(n_samples)],
                weights=None,
                meta=meta,
            )
        )
