"""Explicit financial projection of typed forecast states; no valuation logic."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable

import numpy as np

from ..errors import ForecastProjectionError
from ..results import ForecastResult, ScenarioView, TrajectoryBlock
from ..targets import ForecastTarget, Quantity


def to_scenario_view(
    result: ForecastResult, *, target_id: str, channels: tuple[str, ...] = ("spot",)
) -> ScenarioView:
    from ...instruments import InstrumentRef
    from ...simulation import Scenario

    targets = {t.target_id: t for t in result.targets}
    target = targets.get(target_id)
    block = result.blocks.get(target_id)
    if (
        target is None
        or target.quantity != Quantity.SPOT
        or not isinstance(target.entity, InstrumentRef)
    ):
        raise ForecastProjectionError(
            "projection requires an identified spot target; convert returns explicitly"
        )
    if not isinstance(block, TrajectoryBlock):
        raise ForecastProjectionError(
            "only retained trajectories project to Scenario; marginals are not paths"
        )
    channels = tuple(channels)
    if (
        not channels
        or len(set(channels)) != len(channels)
        or "spot" not in channels
        or not set(channels).issubset(block.channels)
    ):
        raise ForecastProjectionError("projection needs unique retained channels including spot")
    if result.meta.measure is None:
        raise ForecastProjectionError("financial projection requires a declared measure")
    values = block.values[:, :, [block.channels.index(c) for c in channels]]
    if np.any(values[:, :, channels.index("spot")][block.valid] < 0):
        raise ForecastProjectionError("valid spot states cannot be negative")
    scenario = Scenario.from_states(
        target.entity, time_grid=block.times, states=values, state_names=channels
    )
    return ScenarioView(
        scenario=scenario,
        sample_ids=result.sample_ids,
        weights=result.weights,
        valid=block.valid,
        meta=result.meta,
        origin=result.query.origin,
        convention=result.query.convention,
        target=target,
        original_weight_mass=result.original_weight_mass,
    )


def returns_to_spot(
    result: ForecastResult,
    *,
    target_id: str,
    initial_spot: float,
    underlier: Any,
    dividend_mapping: Callable | None = None,
) -> ForecastResult:
    """Convert an explicitly identified price-return trajectory at its declared base.

    Incremental returns occupy intervals ending at each grid time; the origin row
    must be zero. A total-return series needs a caller dividend mapping returning
    ex-distribution prices, including the same initial spot.
    """
    from ...instruments import InstrumentRef

    target = next((t for t in result.targets if t.target_id == target_id), None)
    block = result.blocks.get(target_id)
    if (
        target is None
        or target.quantity != Quantity.RETURN
        or not isinstance(block, TrajectoryBlock)
    ):
        raise ForecastProjectionError("explicit return target trajectory required")
    if not isinstance(underlier, InstrumentRef) or target.entity != underlier:
        raise ForecastProjectionError("initial-price underlier identity must match the target")
    spec = target.return_spec
    if spec is None or spec.base_time != result.query.origin:
        raise ForecastProjectionError("return base time must explicitly match forecast origin")
    if isinstance(initial_spot, bool) or not np.isfinite(initial_spot) or initial_spot <= 0:
        raise ForecastProjectionError("initial spot must be finite and positive")
    if block.channels != ("return",) or np.any(block.values[:, 0, 0] != 0):
        raise ForecastProjectionError("one return channel with an explicit zero origin required")
    r = block.values[:, :, 0]
    if spec.kind == "log":
        levels = r if spec.cumulative else np.cumsum(r, axis=1)
        with np.errstate(over="ignore", invalid="ignore"):
            prices = initial_spot * np.exp(levels)
    else:
        if np.any((r < -1) & block.valid):
            raise ForecastProjectionError("simple returns cannot be below -1")
        prices = initial_spot * ((1 + r) if spec.cumulative else np.cumprod(1 + r, axis=1))
    if spec.includes_distributions:
        if dividend_mapping is None:
            raise ForecastProjectionError("total returns require an explicit dividend mapping")
        prices = np.asarray(dividend_mapping(prices.copy(), block.times.copy()))
        if prices.shape != r.shape or not np.all(prices[:, 0] == initial_spot):
            raise ForecastProjectionError(
                "dividend mapping must preserve axes and initial-price identity"
            )
    elif dividend_mapping is not None:
        raise ForecastProjectionError("dividend mapping supplied for price-only returns")
    valid = block.valid if spec.cumulative else np.logical_and.accumulate(block.valid, axis=1)
    valid = valid & np.isfinite(prices) & (prices >= 0)
    spot_target = ForecastTarget(
        target_id=target_id,
        entity=underlier,
        quantity=Quantity.SPOT,
        units=target.currency or "price_units",
        resolution=target.resolution,
        currency=target.currency,
    )
    blocks = dict(result.blocks)
    blocks[target_id] = TrajectoryBlock(
        target_id=target_id,
        times=block.times,
        values=prices[:, :, None],
        channels=("spot",),
        valid=valid,
    )
    return replace(
        result,
        blocks=blocks,
        targets=tuple(spot_target if t.target_id == target_id else t for t in result.targets),
    )
