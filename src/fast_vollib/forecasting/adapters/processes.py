"""Fixed-grid adapter delegating simulation to the existing process API."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import Any, ClassVar, Mapping
from uuid import uuid4

from ...observations.records import freeze
from .._validate import sample_count
from ..draws import FixedGridDrawSet
from ..errors import ForecastError
from ..query import ForecastQuery
from ..results import ForecastMeta, ForecastResult, TrajectoryBlock
from ..targets import ForecastTarget, Quantity


@dataclass(frozen=True, slots=True, kw_only=True)
class ProcessDistribution:
    process: Any
    initial_state: Any
    underlier: Any
    query: ForecastQuery
    measure: str
    backend_metadata: Mapping[str, Any]
    conditioning_cutoff: datetime
    observations_digest: str
    target: ForecastTarget | None = None
    antithetic: bool = False
    support: ClassVar[str] = "fixed_grid"

    def __post_init__(self) -> None:
        from ...instruments import InstrumentRef

        if self.query.kind != "path" or len(self.query.targets) != 1:
            raise ForecastError("process adapter requires one target and a complete path grid")
        underlier = (
            InstrumentRef(identifier=self.underlier)
            if isinstance(self.underlier, str)
            else self.underlier
        )
        if not isinstance(underlier, InstrumentRef):
            raise ForecastError("underlier must be an InstrumentRef or identifier")
        target = self.target or ForecastTarget(
            target_id=self.query.targets[0],
            entity=underlier,
            quantity=Quantity.SPOT,
            units=underlier.currency or "price_units",
            resolution="configured_grid",
            currency=underlier.currency,
        )
        if target.entity != underlier or self.query.targets != (target.target_id,):
            raise ForecastError("process underlier and target identity must agree")
        if target.quantity in (Quantity.RETURN, Quantity.IV_SURFACE, Quantity.DISCOUNT_CURVE):
            raise ForecastError(
                "process adapter emits state channels, not return/surface/curve projections"
            )
        if type(self.antithetic) is not bool:
            raise ForecastError("antithetic must be boolean")
        if target.quantity == Quantity.SPOT and "spot" not in getattr(
            self.process, "state_names", ()
        ):
            raise ForecastError("a spot target requires a process with a spot state")
        object.__setattr__(self, "underlier", underlier)
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "backend_metadata", freeze(dict(self.backend_metadata)))
        self._meta("validation", None)

    @property
    def targets(self) -> tuple[ForecastTarget, ...]:
        assert self.target is not None
        return (self.target,)

    @property
    def configured_query(self) -> ForecastQuery:
        return self.query

    @property
    def origin(self) -> datetime:
        return self.query.origin

    @property
    def domain(self) -> Any:
        return MappingProxyType({"times": self.query.times, "targets": self.query.targets})

    def _meta(self, draw_id: str, seed: int | None) -> ForecastMeta:
        meta = ForecastMeta(
            measure=self.measure,
            weighting_scheme="none" if self.measure == "stress_only" else "equal",
            sampling_dependence="dependent" if self.antithetic else "unknown",
            coupling="joint",
            initial_state="point",
            generator=type(self.process).__name__,
            draw_set_id=draw_id,
            conditioning_cutoff=self.conditioning_cutoff,
            observations_digest=self.observations_digest,
            seed=seed,
            backend_metadata=self.backend_metadata,
        )
        if meta.conditioning_cutoff > self.origin:
            raise ForecastError("conditioning cutoff exceeds origin")
        return meta

    def draw(self, *, n_samples: int, rng: Any) -> FixedGridDrawSet:
        from ...simulation import simulate

        n_samples = sample_count(n_samples)
        scenario = simulate(
            self.underlier,
            self.process,
            initial_state=self.initial_state,
            time_grid=list(self.query.times),
            n_paths=n_samples,
            rng=rng,
            antithetic=self.antithetic,
        )
        draw_id = uuid4().hex
        block = TrajectoryBlock(
            target_id=self.query.targets[0],
            times=self.query.times,
            values=scenario.states,
            channels=scenario.state_names,
        )
        result = ForecastResult(
            query=self.query,
            targets=self.targets,
            blocks={block.target_id: block},
            sample_ids=[f"{draw_id}:{i}" for i in range(n_samples)],
            weights=None,
            meta=self._meta(draw_id, rng if type(rng) is int else None),
        )
        return FixedGridDrawSet(result)
