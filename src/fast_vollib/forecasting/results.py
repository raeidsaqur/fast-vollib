"""Owned host blocks and generic scenario views; no inventory or valuation state."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from ..observations.records import freeze
from ._validate import instant, names, owned, text, values_and_valid
from .errors import ForecastError, ForecastProjectionError
from .query import ForecastQuery
from .targets import ForecastTarget, Quantity


@dataclass(frozen=True, slots=True, kw_only=True)
class ForecastMeta:
    measure: str | None
    weighting_scheme: str
    sampling_dependence: str
    coupling: str
    initial_state: str
    generator: str
    draw_set_id: str
    conditioning_cutoff: datetime
    observations_digest: str
    checkpoint: str | None = None
    seed: int | None = None
    backend_metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.measure not in (None, "physical", "risk_neutral", "stress_only"):
            raise ForecastError("unknown forecast measure")
        if self.weighting_scheme not in ("equal", "importance", "none"):
            raise ForecastError("unknown weighting scheme")
        if self.measure == "stress_only" and self.weighting_scheme != "none":
            raise ForecastError("stress draws have no probability weights")
        if self.measure in ("physical", "risk_neutral") and self.weighting_scheme == "none":
            raise ForecastError("probability laws require probability weighting")
        if self.sampling_dependence not in ("independent", "dependent", "unknown"):
            raise ForecastError("unknown sampling dependence")
        if self.initial_state not in ("point", "distributed"):
            raise ForecastError("initial_state must be point or distributed")
        if self.coupling not in ("joint", "independent_targets") and not self.coupling.startswith(
            "declared:"
        ):
            raise ForecastError("declare target coupling")
        for field in ("generator", "draw_set_id", "observations_digest"):
            text(getattr(self, field), field)
        if self.checkpoint is not None:
            text(self.checkpoint, "checkpoint")
        if self.seed is not None and (
            isinstance(self.seed, bool) or not isinstance(self.seed, int)
        ):
            raise ForecastError("seed must be an integer or missing")
        object.__setattr__(self, "conditioning_cutoff", instant(self.conditioning_cutoff))
        object.__setattr__(self, "backend_metadata", freeze(dict(self.backend_metadata or {})))


@dataclass(frozen=True, slots=True, kw_only=True)
class TrajectoryBlock:
    target_id: str
    times: Any
    values: Any
    channels: tuple[str, ...]
    valid: Any = None

    def __post_init__(self) -> None:
        text(self.target_id, "target_id")
        channels = names(self.channels, "channels")
        values, valid = values_and_valid(self.values, self.valid, 3)
        times = owned(self.times, dtype=float)
        if (
            times.ndim != 1
            or times.size < 2
            or times[0] != 0
            or not np.isfinite(times).all()
            or (np.diff(times) <= 0).any()
        ):
            raise ForecastError("trajectory times must increase strictly from zero")
        if values.shape[1:] != (times.size, len(channels)):
            raise ForecastError("trajectory axes disagree with times/channels")
        for name, value in (
            ("values", values),
            ("valid", valid),
            ("times", times),
            ("channels", channels),
        ):
            object.__setattr__(self, name, value)


@dataclass(frozen=True, slots=True, kw_only=True)
class MarginalBlock:
    target_id: str
    time: float
    values: Any
    channels: tuple[str, ...]
    valid: Any = None

    def __post_init__(self) -> None:
        text(self.target_id, "target_id")
        channels = names(self.channels, "channels")
        values, valid = values_and_valid(self.values, self.valid, 2)
        if not np.isfinite(self.time) or self.time < 0 or values.shape[1] != len(channels):
            raise ForecastError("invalid marginal time/channels")
        for name, value in (("values", values), ("valid", valid), ("channels", channels)):
            object.__setattr__(self, name, value)


def _sealed_points(points: Any) -> Any:
    from ..surface import CoordinateConvention, SurfacePoints

    if not isinstance(points, SurfacePoints):
        raise ForecastError("surface blocks require SurfacePoints")
    convention = points.convention.to_dict()
    if any(v is not None and not isinstance(v, str) for v in convention.values()):
        raise ForecastError("surface coordinate provenance must be immutable text")
    copy = SurfacePoints(
        k=points.k,
        T=points.T,
        surface_id=points.surface_id,
        point_id=points.point_id,
        convention=CoordinateConvention(**convention),
    )
    for name in ("k", "T", "surface_id", "point_id"):
        value = getattr(copy, name)
        if value is not None:
            object.__setattr__(copy, name, owned(value))
    return copy


@dataclass(frozen=True, slots=True, kw_only=True)
class SurfaceBlock:
    target_id: str
    time: float
    points: Any
    iv: Any
    valid: Any = None
    surfaces: tuple | None = None

    def __post_init__(self) -> None:
        text(self.target_id, "target_id")
        points = _sealed_points(self.points)
        iv = owned(self.iv, dtype=float)
        if (
            iv.ndim != 2
            or not iv.shape[0]
            or iv.shape[1] != points.n
            or not np.isfinite(self.time)
            or self.time < 0
        ):
            raise ForecastError("invalid surface sample/point/time axes")
        valid = owned(np.isfinite(iv) & (iv > 0) if self.valid is None else self.valid)
        if (
            valid.dtype != np.dtype(bool)
            or valid.shape != iv.shape
            or np.any(valid & (~np.isfinite(iv) | (iv <= 0)))
        ):
            raise ForecastError("invalid surface validity")
        if self.surfaces is not None:
            raise ForecastError(
                "retaining arbitrary evaluable surface objects has no ownership contract"
            )
        for name, value in (("points", points), ("iv", iv), ("valid", valid)):
            object.__setattr__(self, name, value)


@dataclass(frozen=True, slots=True, kw_only=True)
class CurveBlock:
    target_id: str
    time: float
    tenors: Any
    values: Any
    valid: Any = None

    def __post_init__(self) -> None:
        text(self.target_id, "target_id")
        tenors, values = owned(self.tenors, dtype=float), owned(self.values)
        if (
            tenors.ndim != 1
            or not tenors.size
            or not np.isfinite(tenors).all()
            or (tenors < 0).any()
            or (np.diff(tenors) <= 0).any()
        ):
            raise ForecastError("curve tenors must be ordered finite nonnegative values")
        if (
            values.ndim != 2
            or values.dtype.kind != "f"
            or not values.shape[0]
            or values.shape[1] != tenors.size
            or not np.isfinite(self.time)
            or self.time < 0
        ):
            raise ForecastError("invalid curve axes/time")
        valid = owned(np.isfinite(values) if self.valid is None else self.valid)
        if (
            valid.dtype != np.dtype(bool)
            or valid.shape != values.shape
            or np.any(valid & ~np.isfinite(values))
        ):
            raise ForecastError("invalid curve validity")
        for name, value in (("tenors", tenors), ("values", values), ("valid", valid)):
            object.__setattr__(self, name, value)


@dataclass(frozen=True, slots=True, kw_only=True)
class MarkedEvent:
    time: float
    kind: str
    payload: Any

    def __post_init__(self) -> None:
        if not np.isfinite(self.time) or self.time < 0:
            raise ForecastError("event time must be nonnegative elapsed years")
        text(self.kind, "event kind")
        object.__setattr__(self, "payload", freeze(self.payload))


@dataclass(frozen=True, slots=True, kw_only=True)
class EventBlock:
    target_id: str
    events_by_sample: tuple[tuple[MarkedEvent, ...], ...]

    def __post_init__(self) -> None:
        text(self.target_id, "target_id")
        events = tuple(tuple(row) for row in self.events_by_sample)
        if not events or any(not isinstance(e, MarkedEvent) for row in events for e in row):
            raise ForecastError("events require a sample-aligned collection of typed events")
        if any(any(a.time > b.time for a, b in zip(row, row[1:])) for row in events):
            raise ForecastError("events must be ordered within each sample")
        object.__setattr__(self, "events_by_sample", events)


Block = TrajectoryBlock | MarginalBlock | SurfaceBlock | CurveBlock | EventBlock


def _sample_axis(block: Block) -> int:
    if isinstance(block, EventBlock):
        return len(block.events_by_sample)
    return int((block.iv if isinstance(block, SurfaceBlock) else block.values).shape[0])


def _samples(
    ids: Any, weights: Any, meta: ForecastMeta, *, normalized: bool = False
) -> tuple[Any, Any, float | None]:
    ids = owned(ids)
    if (
        ids.ndim != 1
        or not ids.size
        or ids.dtype.kind not in "US"
        or len(set(ids.tolist())) != ids.size
        or any(not str(x).startswith(meta.draw_set_id + ":") for x in ids)
    ):
        raise ForecastError("sample ids must be unique nonempty text namespaced by draw_set_id")
    if meta.weighting_scheme == "none":
        if weights is not None:
            raise ForecastError("non-probabilistic draws cannot carry weights")
        return ids, None, None
    if meta.weighting_scheme == "equal":
        if weights is not None:
            raise ForecastError(
                "equal weighting uses implicit weights; explicit weights require importance"
            )
        return ids, None, float(ids.size)
    if weights is None:
        raise ForecastError("importance weighting requires weights")
    w = owned(weights, dtype=float)
    mass = float(w.sum())
    if (
        w.shape != ids.shape
        or not np.isfinite(w).all()
        or (w < 0).any()
        or not np.isfinite(mass)
        or mass <= 0
    ):
        raise ForecastError("weights must have positive finite mass and be finite nonnegative")
    if normalized:
        if abs(mass - 1) > 8 * np.finfo(float).eps:
            raise ForecastError("retained normalized weights must sum to one")
        return ids, w, mass
    return ids, owned(w / mass), mass


@dataclass(frozen=True, slots=True, kw_only=True)
class ForecastResult:
    query: ForecastQuery
    targets: tuple[ForecastTarget, ...]
    blocks: Mapping[str, Block]
    sample_ids: Any
    weights: Any
    meta: ForecastMeta
    original_weight_mass: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.query, ForecastQuery) or not isinstance(self.meta, ForecastMeta):
            raise ForecastError("typed query and metadata required")
        targets = tuple(self.targets)
        if (
            any(not isinstance(t, ForecastTarget) for t in targets)
            or tuple(t.target_id for t in targets) != self.query.targets
        ):
            raise ForecastError("target identities/order must match query")
        if self.meta.conditioning_cutoff > self.query.origin:
            raise ForecastError("conditioning cutoff cannot be after forecast origin")
        ids, weights, mass = _samples(
            self.sample_ids,
            self.weights,
            self.meta,
            normalized=self.original_weight_mass is not None,
        )
        if self.original_weight_mass is not None:
            if (
                mass is None
                or not np.isfinite(self.original_weight_mass)
                or self.original_weight_mass <= 0
            ):
                raise ForecastError("original weight mass must be positive and finite")
            mass = float(self.original_weight_mass)
        blocks = dict(self.blocks)
        if set(blocks) != set(self.query.targets):
            raise ForecastError("one typed block per requested target required")
        for target in targets:
            block = blocks[target.target_id]
            if (
                not isinstance(
                    block, (TrajectoryBlock, MarginalBlock, SurfaceBlock, CurveBlock, EventBlock)
                )
                or block.target_id != target.target_id
                or _sample_axis(block) != ids.size
            ):
                raise ForecastError("block target/sample identity mismatch")
            if isinstance(block, TrajectoryBlock):
                if self.query.kind != "path" or not np.array_equal(block.times, self.query.times):
                    raise ForecastError("trajectory grid must match path query")
            elif isinstance(block, EventBlock):
                if self.query.kind != "path" or any(
                    e.time > self.query.times[-1] for row in block.events_by_sample for e in row
                ):
                    raise ForecastError("event paths must lie inside the query horizon")
            elif self.query.kind != "marginal" or self.query.times != (block.time,):
                raise ForecastError("single-date blocks require a matching single marginal query")
            if isinstance(block, SurfaceBlock) != (target.quantity == Quantity.IV_SURFACE):
                raise ForecastError("surface target and block type must agree")
            if isinstance(block, CurveBlock) != (target.quantity == Quantity.DISCOUNT_CURVE):
                raise ForecastError("curve target and block type must agree")
        for name, value in (
            ("targets", targets),
            ("blocks", MappingProxyType(blocks)),
            ("sample_ids", ids),
            ("weights", weights),
            ("original_weight_mass", mass),
        ):
            object.__setattr__(self, name, value)

    def to_scenario_view(
        self, *, target_id: str, channels: tuple[str, ...] = ("spot",)
    ) -> ScenarioView:
        from .adapters.finance import to_scenario_view

        return to_scenario_view(self, target_id=target_id, channels=channels)


@dataclass(frozen=True, slots=True, kw_only=True)
class ScenarioView:
    scenario: Any
    sample_ids: Any
    weights: Any
    valid: Any
    meta: ForecastMeta
    origin: datetime
    convention: str
    target: ForecastTarget
    original_weight_mass: float | None = None

    def __post_init__(self) -> None:
        from ..simulation import Scenario

        if not isinstance(self.scenario, Scenario):
            raise ForecastProjectionError("ScenarioView requires a Scenario")
        if (
            not isinstance(self.target, ForecastTarget)
            or self.target.quantity != Quantity.SPOT
            or self.target.entity != self.scenario.underlier
        ):
            raise ForecastProjectionError("scenario target must identify its spot underlier")
        if self.meta.measure is None:
            raise ForecastProjectionError("financial projection requires a declared measure")
        if "spot" not in self.scenario.state_names:
            raise ForecastProjectionError("financial scenario requires a named spot channel")
        spot = owned(self.scenario.state("spot"))
        ids, weights, mass = _samples(
            self.sample_ids,
            self.weights,
            self.meta,
            normalized=self.original_weight_mass is not None,
        )
        states = owned(self.scenario.states)
        grid = owned(self.scenario.time_grid)
        valid = owned(self.valid)
        if (
            valid.dtype != np.dtype(bool)
            or valid.shape != states.shape[:2]
            or ids.size != states.shape[0]
        ):
            raise ForecastProjectionError("scenario sample/validity axes disagree")
        if np.any(spot[valid] < 0):
            raise ForecastProjectionError("valid spot states cannot be negative")
        if not np.isfinite(states[valid]).all():
            raise ForecastProjectionError("valid scenario entries must be finite")
        scenario = Scenario.from_states(
            self.scenario.underlier,
            time_grid=grid,
            states=states,
            state_names=self.scenario.state_names,
        )
        # Seal the otherwise re-enableable NumPy buffers of the existing Scenario.
        object.__setattr__(scenario, "states", states)
        object.__setattr__(scenario, "time_grid", grid)
        if self.convention not in ("act_365", "act_365_25", "act_360"):
            raise ForecastProjectionError("unsupported scenario time convention")
        object.__setattr__(self, "origin", instant(self.origin))
        if self.meta.conditioning_cutoff > self.origin:
            raise ForecastProjectionError("conditioning cutoff exceeds origin")
        if self.original_weight_mass is not None:
            if (
                mass is None
                or not np.isfinite(self.original_weight_mass)
                or self.original_weight_mass <= 0
            ):
                raise ForecastProjectionError("invalid original weight mass")
            mass = self.original_weight_mass
        for name, value in (
            ("scenario", scenario),
            ("sample_ids", ids),
            ("weights", weights),
            ("valid", valid),
            ("original_weight_mass", mass),
        ):
            object.__setattr__(self, name, value)
