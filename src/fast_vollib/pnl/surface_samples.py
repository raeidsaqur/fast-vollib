"""One joint surface draw on the required union, followed by chunked revaluation."""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
from typing import Any
from uuid import uuid4

import numpy as np

from ..instruments import payoff
from ..observations.records import digest
from ..portfolio._validate import money
from .assumptions import HorizonMarket, SurfaceVolSamples
from .bands import SurfaceSensitivityBand
from .errors import PnLError
from .evaluate import _unit_contract
from .model import _coordinates, _grid, _meta, _model_market, _price_rows, _rows, _sealed_points

DEFAULT_UNION_POINT_LIMIT = 65536


def _sampled(
    snapshot: Any,
    spot: Any,
    *,
    horizons: Any,
    volatility: SurfaceVolSamples,
    market: HorizonMarket,
    model: str,
    backend: str,
    convention: str,
    future_costs: Decimal = Decimal("0"),
    chunk_rows: int | None = None,
    diagnostic: bool = False,
    union_point_limit: int = DEFAULT_UNION_POINT_LIMIT,
    proxy: bool = False,
) -> SurfaceSensitivityBand:
    from ..surface import SurfacePoints, SurfaceSamples
    from .serialization import _encode

    _model_market(market, model)
    if not isinstance(volatility, SurfaceVolSamples) or market.volatility is not None:
        raise PnLError(
            "sampled surface route requires explicit SurfaceVolSamples and market.volatility=None"
        )
    if type(diagnostic) is not bool or backend not in ("auto", "numpy", "torch", "jax", "numba"):
        raise PnLError("explicit supported backend and boolean diagnostic required")
    if type(union_point_limit) is not int or union_point_limit < 1:
        raise PnLError("union_point_limit must be positive")
    grid = _grid(snapshot, spot, horizons, convention, chunk_rows, proxy=proxy)
    costs = money(future_costs, nonnegative=True)
    nrows = int(np.prod(grid.shape))
    point_index = np.full(nrows, -1, dtype=np.int64)
    union: dict[tuple[float, float], int] = {}
    coords: list[tuple[float, float]] = []
    # Only output maps and the deduplicated union persist. Each coordinate batch
    # is capped before creating a Cartesian working buffer.
    for start in range(0, nrows, grid.chunk_rows):
        stop = min(nrows, start + grid.chunk_rows)
        hi, si, li, remaining, intrinsic = _rows(grid, start, stop)
        live = (~intrinsic) & (grid.spot[si] > 0)
        if np.any((~intrinsic) & ~live) and not diagnostic:
            raise PnLError("zero live spot is outside logarithmic surface coordinates")
        if not live.any():
            continue
        points, _, _ = _coordinates(
            volatility, grid, market, model, hi[live], si[live], li[live], remaining[live]
        )
        for row, k, t in zip(start + np.flatnonzero(live), points.k, points.T):
            key = (float(k), float(t))
            if key not in union:
                if len(union) >= union_point_limit:
                    raise PnLError(
                        "required coherent surface union exceeds the declared point limit"
                    )
                union[key] = len(coords)
                coords.append(key)
            point_index[row] = union[key]
    points = SurfacePoints(
        k=[c[0] for c in coords], T=[c[1] for c in coords], point_id=np.arange(len(coords))
    )
    points = _sealed_points(points)
    if coords:
        draws = volatility.distribution.sample(
            points,
            n_samples=volatility.n_samples,
            rng=volatility.rng,
            market=volatility.reference_market,
        )
        if not isinstance(draws, SurfaceSamples) or draws.iv.shape != (
            volatility.n_samples,
            len(coords),
        ):
            raise PnLError("surface sampler changed sample/union axes")
        for name in ("k", "T", "point_id", "surface_id"):
            if not np.array_equal(getattr(draws.points, name), getattr(points, name)):
                raise PnLError("surface sampler changed union coordinates or identity/order")
        iv = np.asarray(draws.iv)
        iv_valid = np.asarray(draws.valid) & np.isfinite(iv) & (iv > 0)
        if not iv_valid.all() and not diagnostic:
            raise PnLError("joint surface draw has invalid required cells")
    else:
        iv = np.empty((volatility.n_samples, 0))
        iv_valid = np.empty_like(iv, dtype=bool)
    shape = (volatility.n_samples, *grid.shape)
    values = np.full(shape, np.nan)
    valid = np.zeros(shape, dtype=bool)
    flat = values.reshape(volatility.n_samples, nrows)
    mask = valid.reshape(volatility.n_samples, nrows)
    units = tuple(
        _unit_contract(x.contract, maturity=0.0, strike=float(x.contract.strike))
        for x in snapshot.position.legs
    )
    for start in range(0, nrows, grid.chunk_rows):
        stop = min(nrows, start + grid.chunk_rows)
        hi, si, li, remaining, intrinsic = _rows(grid, start, stop)
        for leg_index, unit in enumerate(units):
            selected = intrinsic & (li == leg_index)
            if selected.any():
                rows = start + np.flatnonzero(selected)
                flat[:, rows] = (
                    grid.sizes[leg_index]
                    * np.asarray(payoff(unit, grid.spot[si[selected]]))[None, :]
                )
                mask[:, rows] = True
        selected = point_index[start:stop] >= 0
        if not selected.any():
            continue
        row_local = np.flatnonzero(selected)
        union_rows = point_index[start:stop][selected]
        # Chunk across BOTH sample and model-row axes: no pricing call exceeds
        # chunk_rows, including when the sample count exceeds that budget.
        work = volatility.n_samples * row_local.size
        for offset in range(0, work, grid.chunk_rows):
            positions = np.arange(offset, min(work, offset + grid.chunk_rows))
            sample = positions // row_local.size
            local = row_local[positions % row_local.size]
            indices = union_rows[positions % row_local.size]
            sigma = iv[sample, indices]
            ok = iv_valid[sample, indices]
            if ok.any():
                p = _price_rows(
                    grid,
                    market,
                    model,
                    backend,
                    hi[local[ok]],
                    si[local[ok]],
                    li[local[ok]],
                    remaining[local[ok]],
                    sigma[ok],
                )
                finite = np.isfinite(p)
                if not finite.all() and not diagnostic:
                    raise PnLError("sampled model prices are nonfinite")
                flat[sample[ok], start + local[ok]] = np.where(finite, p, np.nan)
                mask[sample[ok], start + local[ok]] = finite
    aggregate = valid.all(axis=-1)
    total = (
        np.full(shape[:3], np.nan)
        if snapshot.cumulative_cash is None
        else np.where(
            aggregate, float(snapshot.cumulative_cash) + values.sum(axis=-1) - float(costs), np.nan
        )
    )
    meta = _meta(grid, market, model, backend, volatility, costs, band=True)
    draw_id = uuid4().hex
    meta = replace(
        meta,
        input_digest=digest(
            {
                "base": meta.input_digest,
                "union": {"k": points.k.tolist(), "T": points.T.tolist()},
                "iv": _encode(iv),
                "valid": iv_valid.tolist(),
                "draw_set_id": draw_id,
            }
        ),
    )
    cls = SurfaceSensitivityBand
    if proxy:
        from .proxy import EuropeanProxySensitivityBand

        cls = EuropeanProxySensitivityBand
    return cls(
        spot=grid.spot,
        horizons=tuple(float(x) for x in grid.horizons),
        samples=values,
        total=total,
        valid=valid,
        sample_ids=tuple(f"{draw_id}:{i}" for i in range(volatility.n_samples)),
        weights=np.ones(volatility.n_samples),
        original_weight_mass=None,
        cumulative_cash=snapshot.cumulative_cash,
        future_costs=costs,
        leg_ids=tuple(x.leg_id for x in snapshot.position.legs),
        contracts=tuple(x.contract for x in snapshot.position.legs),
        signed_sizes=tuple(x.signed_size for x in snapshot.position.legs),
        meta=meta,
    )


def pnl_at_horizon_samples(snapshot: Any, spot: Any, **kwargs: Any) -> SurfaceSensitivityBand:
    """Surface-only sensitivity at fixed spots; never a joint P&L probability law."""
    if "proxy" in kwargs:
        raise PnLError("American legs require the separately typed sampled proxy route")
    return _sampled(snapshot, spot, **kwargs)
