"""Retained fixed-grid samples: query selection never calls a generator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from .errors import ForecastQueryError
from .query import ForecastQuery
from .results import Block, EventBlock, ForecastResult, MarginalBlock, TrajectoryBlock


@dataclass(frozen=True, slots=True)
class FixedGridDrawSet:
    """An owned result answering only subsets of its original grid and targets."""

    result: ForecastResult
    support: ClassVar[str] = "fixed_grid"

    def __post_init__(self) -> None:
        if not isinstance(self.result, ForecastResult):
            raise ForecastQueryError("retained draws require a validated ForecastResult")

    @property
    def sample_ids(self):
        return self.result.sample_ids

    @property
    def weights(self):
        return self.result.weights

    @property
    def configured_query(self) -> ForecastQuery:
        return self.result.query

    def query(self, query: ForecastQuery) -> ForecastResult:
        source = self.result.query
        if (
            not isinstance(query, ForecastQuery)
            or query.origin != source.origin
            or query.convention != source.convention
        ):
            raise ForecastQueryError("origin and day count cannot change on retained draws")
        if not set(query.targets).issubset(source.targets):
            raise ForecastQueryError("query requests an unretained target")
        if not set(query.times).issubset(source.times):
            raise ForecastQueryError("fixed_grid cannot refine, interpolate, extend or resample")
        if source.kind == "marginal" and query.kind != "marginal":
            raise ForecastQueryError("marginals cannot be promoted into paths")
        indices = [source.times.index(t) for t in query.times]
        if query.kind == "marginal" and len(indices) != 1:
            raise ForecastQueryError("single-date typed blocks require one marginal time per query")
        blocks: dict[str, Block] = {}
        for key in query.targets:
            block = self.result.blocks[key]
            if isinstance(block, TrajectoryBlock):
                if query.kind == "path":
                    blocks[key] = TrajectoryBlock(
                        target_id=key,
                        times=query.times,
                        values=block.values[:, indices, :],
                        channels=block.channels,
                        valid=block.valid[:, indices],
                    )
                else:
                    blocks[key] = MarginalBlock(
                        target_id=key,
                        time=query.times[0],
                        values=block.values[:, indices[0], :],
                        channels=block.channels,
                        valid=block.valid[:, indices[0]],
                    )
            elif isinstance(block, EventBlock):
                if query.kind != "path":
                    raise ForecastQueryError("event paths cannot be relabelled as marginal state")
                blocks[key] = EventBlock(
                    target_id=key,
                    events_by_sample=tuple(
                        tuple(e for e in row if e.time <= query.times[-1])
                        for row in block.events_by_sample
                    ),
                )
            else:
                blocks[key] = block
        targets = {t.target_id: t for t in self.result.targets}
        return ForecastResult(
            query=query,
            targets=tuple(targets[k] for k in query.targets),
            blocks=blocks,
            sample_ids=self.sample_ids,
            weights=self.weights,
            meta=self.result.meta,
            original_weight_mass=self.result.original_weight_mass,
        )
