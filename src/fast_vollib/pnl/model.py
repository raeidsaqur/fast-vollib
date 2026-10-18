"""Batched horizon revaluation through the existing unit-option pricing API."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
import hashlib
import json
from typing import Any, cast

import numpy as np

from ..instruments import EuropeanOptionBatch, VanillaMarketInputs, payoff
from ..observations.records import digest
from ..portfolio import PositionSnapshot, year_fraction
from ..portfolio._validate import money
from ..portfolio.contracts import require_standard
from ..portfolio.serialization import snapshot_to_dict
from ..types import BackendLiteral, ModelLiteral
from .assumptions import FlatVol, HorizonMarket, PerLegVol, SurfaceVol, SurfaceVolSamples
from .errors import PnLError, PnLHorizonError
from .evaluate import _unit_contract
from .results import PnLProfile, ResultMeta, _owned

DEFAULT_CHUNK_ROWS = 65536
PROXY_CAVEAT = "European surrogate illustration; not an American valuation or executable close estimate; early exercise, assignment and delivery lifecycle are not modelled"
BAND_CAVEAT = (
    "Surface-sensitivity draws at fixed spots; not joint spot/volatility P&L probabilities."
)


@dataclass(frozen=True, slots=True)
class _Grid:
    snapshot: PositionSnapshot
    spot: Any
    horizons: Any
    maturities: Any
    strikes: Any
    sizes: Any
    flags: Any
    chunk_rows: int
    convention: str
    proxy: bool
    epsilon: float

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.horizons.size, self.spot.size, len(self.snapshot.position.legs)


def repricing_index(flat: Any, *, n_spots: int, n_legs: int) -> tuple[Any, Any, Any]:
    """Flat index order is ((horizon * n_spots) + spot) * n_legs + leg."""
    if type(n_spots) is not int or type(n_legs) is not int or n_spots < 1 or n_legs < 1:
        raise PnLError("positive grid dimensions required")
    flat = np.asarray(flat)
    if flat.dtype.kind not in "iu" or (flat < 0).any():
        raise PnLError("flat row indices must be nonnegative integers")
    return flat // (n_spots * n_legs), (flat // n_legs) % n_spots, flat % n_legs


def _grid(
    snapshot: PositionSnapshot,
    spot: Any,
    horizons: Any,
    convention: str,
    chunk_rows: int | None,
    *,
    proxy: bool,
) -> _Grid:
    from ..simulation.scenario import horizon_tolerance

    if not isinstance(snapshot, PositionSnapshot):
        raise PnLError("typed PositionSnapshot required")
    year_fraction(snapshot.valuation_time, snapshot.valuation_time, convention=convention)
    original = np.asarray(spot)
    if original.dtype.kind in "bc":
        raise PnLError("spot grid must be real and non-boolean")
    raw_horizons = tuple(horizons)
    if any(
        isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, float, np.integer, np.floating))
        for x in raw_horizons
    ):
        raise PnLHorizonError("horizons must be real non-boolean elapsed years")
    spots = np.array(spot, dtype=float, copy=True)
    hs = np.array(raw_horizons, dtype=float)
    if spots.ndim != 1 or not spots.size or not np.isfinite(spots).all() or (spots < 0).any():
        raise PnLError("spot must be a finite nonnegative nonempty vector")
    if (
        hs.ndim != 1
        or not hs.size
        or not np.isfinite(hs).all()
        or (hs < 0).any()
        or (np.diff(hs) <= 0).any()
    ):
        raise PnLHorizonError("horizons must be increasing finite nonnegative elapsed years")
    if chunk_rows is None:
        chunk_rows = DEFAULT_CHUNK_ROWS
    if type(chunk_rows) is not int or chunk_rows < 1:
        raise PnLError("chunk_rows must be positive")
    legs = snapshot.position.legs
    american = any(x.contract.exercise == "american" for x in legs)
    if proxy and not american:
        raise PnLError("European proxy requires at least one American live leg")
    if not proxy and american:
        raise PnLError("standard model valuation refuses American legs; use european_proxy_profile")
    maturities = []
    for leg in legs:
        c = leg.contract
        require_standard(c)
        if c.expiry_instant is None:
            raise PnLHorizonError("every live expiry needs an explicit resolved instant")
        maturity = year_fraction(snapshot.valuation_time, c.expiry_instant, convention=convention)
        if maturity < 0:
            raise PnLHorizonError("valuation is after expiry; reconcile lifecycle evidence")
        maturities.append(maturity)
    epsilon = float(
        np.finfo(original.dtype).eps if original.dtype.kind == "f" else np.finfo(float).eps
    )
    if maturities and hs[-1] > min(maturities) + horizon_tolerance(
        min(maturities), epsilon=epsilon
    ):
        raise PnLHorizonError("horizon crosses the earliest live expiry")
    # Use float64 arithmetic for reports; tolerance still respects the supplied grid precision.
    tolerance = (
        max(horizon_tolerance(t, epsilon=epsilon) for t in maturities) if maturities else 0.0
    )
    for i, t in enumerate(maturities):
        if np.any(hs > t + tolerance):
            raise PnLHorizonError("horizon crosses a live expiry")
    return _Grid(
        snapshot,
        spots,
        hs,
        np.asarray(maturities),
        np.asarray([float(x.contract.strike) for x in legs]),
        np.asarray([float(x.signed_size) for x in legs]),
        np.asarray([x.contract.option_type.value for x in legs]),
        chunk_rows,
        convention,
        proxy,
        epsilon,
    )


def _model_market(market: HorizonMarket, model: str) -> None:
    if not isinstance(market, HorizonMarket) or model not in (
        "black",
        "black_scholes",
        "black_scholes_merton",
    ):
        raise PnLError("explicit supported model and HorizonMarket required")
    if model != "black" and market.forward_mapping is not None:
        raise PnLError(
            "spot models use their declared rate/yield; a Black forward mapping cannot be silently ignored"
        )
    if model == "black_scholes" and market.dividend_yield != 0:
        raise PnLError("Black-Scholes requires explicit zero dividend yield")
    if model == "black_scholes_merton" and market.dividend_yield is None:
        raise PnLError("Black-Scholes-Merton requires explicit dividend yield")
    if model == "black" and (
        market.forward_mapping is None or market.discount_convention != "constant_continuous_rate"
    ):
        raise PnLError(
            "Black-76 spot grids require an explicit forward mapping and constant continuous discount convention"
        )


def _forward(mapping: Any, spot: Any, horizon: Any, remaining: Any) -> Any:
    values = np.asarray(mapping(spot, horizon, remaining), dtype=float)
    try:
        values = np.broadcast_to(values, np.shape(spot))
    except ValueError as exc:
        raise PnLError("forward mapping shape disagrees with repricing rows") from exc
    if not np.isfinite(values).all() or (values <= 0).any():
        raise PnLError("forward mapping must return finite positive forwards")
    return values


def _sealed_points(points: Any) -> Any:
    for name in ("k", "T", "surface_id", "point_id"):
        value = getattr(points, name)
        if value is not None:
            object.__setattr__(points, name, _owned(value, value.dtype))
    return points


def _coordinates(
    vol: Any,
    grid: _Grid,
    market: HorizonMarket,
    model: str,
    hi: Any,
    si: Any,
    li: Any,
    remaining: Any,
) -> tuple[Any, Any, Any]:
    from ..surface import SurfacePoints

    spot = grid.spot[si]
    horizon = grid.horizons[hi]
    forward = _forward(vol.forward_mapping, spot, horizon, remaining)
    expected = (
        _forward(market.forward_mapping, spot, horizon, remaining)
        if model == "black"
        else spot * np.exp((market.rate - (market.dividend_yield or 0)) * remaining)
    )
    if not np.allclose(forward, expected, rtol=1e-12, atol=1e-12):
        raise PnLError("surface scenario forward and pricing state disagree")
    tenor = grid.maturities[li] if vol.time_movement == "fixed_expiry" else remaining
    reference = np.asarray(vol.reference_market.forward_at(tenor), dtype=float)
    if not np.allclose(vol.reference_market.rate_at(tenor), market.rate, rtol=1e-12, atol=1e-12):
        raise PnLError("reference surface discount rates disagree with pricing rates")
    if model != "black":
        assert market.dividend_yield is not None
        if not np.allclose(
            vol.reference_market.carry_at(tenor), market.dividend_yield, rtol=1e-12, atol=1e-12
        ):
            raise PnLError("reference surface carry disagrees with pricing yield")
    if not np.isfinite(reference).all() or (reference <= 0).any():
        raise PnLError("invalid reference forwards")
    at_origin = _forward(
        vol.forward_mapping, np.full_like(tenor, vol.origin_spot), np.zeros_like(tenor), tenor
    )
    if not np.allclose(reference, at_origin, rtol=1e-12, atol=1e-12):
        raise PnLError("reference surface forwards disagree with the declared origin mapping")
    k = np.log(grid.strikes[li] / (reference if vol.spot_movement == "sticky_strike" else forward))
    return _sealed_points(SurfacePoints(k=k, T=tenor)), forward, expected


def _rows(grid: _Grid, start: int, stop: int) -> tuple[Any, Any, Any, Any, Any]:
    from ..simulation.scenario import horizon_tolerance

    hi, si, li = repricing_index(
        np.arange(start, stop, dtype=np.int64), n_spots=grid.spot.size, n_legs=len(grid.sizes)
    )
    remaining = grid.maturities[li] - grid.horizons[hi]
    tolerance = np.asarray([horizon_tolerance(t, epsilon=grid.epsilon) for t in grid.maturities])[
        li
    ]
    intrinsic = remaining <= tolerance
    return hi, si, li, np.maximum(remaining, 0.0), intrinsic


def _volatility(
    vol: Any,
    grid: _Grid,
    market: HorizonMarket,
    model: str,
    hi: Any,
    si: Any,
    li: Any,
    remaining: Any,
) -> tuple[Any, Any]:
    if isinstance(vol, FlatVol):
        return np.full(li.size, vol.sigma), np.ones(li.size, dtype=bool)
    if isinstance(vol, PerLegVol):
        if set(vol.by_leg_id) != {x.leg_id for x in grid.snapshot.position.legs}:
            raise PnLError("per-leg volatility must cover exactly the live leg IDs")
        values = np.asarray([vol.by_leg_id[x.leg_id] for x in grid.snapshot.position.legs])
        return values[li], np.ones(li.size, dtype=bool)
    from ..surface import SurfacePrediction

    points, _, _ = _coordinates(vol, grid, market, model, hi, si, li, remaining)
    prediction = vol.surface.evaluate(points, market=vol.reference_market)
    if (
        not isinstance(prediction, SurfacePrediction)
        or prediction.iv.shape != (li.size,)
        or not np.array_equal(prediction.points.k, points.k)
        or not np.array_equal(prediction.points.T, points.T)
    ):
        raise PnLError("surface prediction changed query coordinates or shape")
    valid = np.asarray(prediction.valid) & np.isfinite(prediction.iv) & (prediction.iv > 0)
    return np.asarray(prediction.iv), valid


def _price_rows(
    grid: _Grid,
    market: HorizonMarket,
    model: str,
    backend: str,
    hi: Any,
    si: Any,
    li: Any,
    remaining: Any,
    sigma: Any,
) -> Any:
    from ..instruments import price_instrument

    underlying = (
        _forward(market.forward_mapping, grid.spot[si], grid.horizons[hi], remaining)
        if model == "black"
        else grid.spot[si]
    )
    batch = EuropeanOptionBatch.from_arrays(
        option_type=grid.flags[li],
        strike=grid.strikes[li],
        maturity=remaining,
        underlier=grid.snapshot.position.underlier,
        notional=1.0,
    )
    inputs = VanillaMarketInputs(
        underlying=underlying,
        rate=market.rate,
        dividend_yield=market.dividend_yield,
        volatility=sigma,
    )
    prices: Any = price_instrument(
        batch,
        inputs,
        model=cast(ModelLiteral, model),
        backend=cast(BackendLiteral, backend),
        return_as="numpy",
    )
    return np.asarray(prices, dtype=np.float64) * grid.sizes[li]


def _vol_meta(vol: Any, market: HorizonMarket, backend: str) -> str:
    data: dict[str, Any] = {
        "backend": backend,
        "discount_convention": market.discount_convention,
        "forward_mapping": market.forward_mapping_id,
        "rate": market.rate,
        "dividend_yield": market.dividend_yield,
    }
    if isinstance(vol, FlatVol):
        data.update(kind="flat", sigma=vol.sigma)
    elif isinstance(vol, PerLegVol):
        data.update(kind="per_leg", by_leg_id=dict(vol.by_leg_id))
    else:
        data.update(
            kind="surface_samples" if isinstance(vol, SurfaceVolSamples) else "surface",
            generator=type(
                vol.distribution if isinstance(vol, SurfaceVolSamples) else vol.surface
            ).__name__,
            spot_movement=vol.spot_movement,
            time_movement=vol.time_movement,
            surface_forward_mapping=vol.forward_mapping_id,
            origin_spot=vol.origin_spot,
            reference_market=vol.reference_market.to_dict()
            if hasattr(vol.reference_market, "to_dict")
            else {
                "T": vol.reference_market.T.tolist(),
                "forward": vol.reference_market.forward.tolist(),
                "rate": vol.reference_market.rate.tolist(),
                "carry": vol.reference_market.carry.tolist(),
            },
        )
    return json.dumps(data, sort_keys=True, allow_nan=False)


def _meta(
    grid: _Grid,
    market: HorizonMarket,
    model: str,
    backend: str,
    vol: Any,
    costs: Decimal,
    *,
    band: bool = False,
    evaluation_digest: str | None = None,
) -> ResultMeta:
    s = grid.snapshot
    caveats = []
    if grid.proxy:
        caveats.append(PROXY_CAVEAT)
    elif any(x.contract.settlement.value == "physical" for x in s.position.legs):
        caveats.append(
            "European model/intrinsic-equivalent illustration; physical delivery and settlement lifecycle are not modelled."
        )
    if band:
        caveats.append(BAND_CAVEAT)
    source_data = json.loads(_vol_meta(vol, market, backend))
    if evaluation_digest is not None:
        source_data["evaluation_sha256"] = evaluation_digest
    source = json.dumps(source_data, sort_keys=True, allow_nan=False)
    return ResultMeta(
        origin=s.origin,
        basis="european_proxy" if grid.proxy else "model",
        valuation_time=s.valuation_time,
        knowledge_cutoff=s.knowledge_cutoff,
        knowledge_label=s.knowledge_label,
        world_id=s.world_id,
        observations_digest=s.observations_digest,
        currency=s.currency,
        caveat=" ".join(caveats) or None,
        terminal_assumption="not_applicable:european_proxy"
        if grid.proxy
        else "not_applicable:model",
        accounting_policy=s.accounting_policy,
        time_policy=grid.convention,
        model=model,
        vol_source=source,
        spot_movement=getattr(vol, "spot_movement", None),
        time_movement=getattr(vol, "time_movement", None),
        expiry_policies=tuple(
            sorted({x.contract.expiry_policy for x in s.position.legs if x.contract.expiry_policy})
        ),
        input_digest=digest(
            {
                "snapshot": snapshot_to_dict(s),
                "spot": grid.spot.tolist(),
                "horizons": grid.horizons.tolist(),
                "model": model,
                "assumptions": source,
                "convention": grid.convention,
                "future_costs": costs,
            }
        ),
        validation_status="surface_sensitivity" if band else "model_assumptions",
        schema_version="pnl.result/2",
    )


def _evaluate(
    snapshot: PositionSnapshot,
    spot: Any,
    *,
    horizons: Any,
    market: HorizonMarket,
    model: str,
    backend: str,
    convention: str,
    future_costs: Decimal = Decimal("0"),
    chunk_rows: int | None = None,
    diagnostic: bool = False,
    proxy: bool = False,
) -> PnLProfile:
    _model_market(market, model)
    if market.volatility is None:
        raise PnLError("deterministic repricing requires an explicit volatility assumption")
    if type(diagnostic) is not bool or backend not in ("auto", "numpy", "torch", "jax", "numba"):
        raise PnLError("explicit supported backend and boolean diagnostic required")
    grid = _grid(snapshot, spot, horizons, convention, chunk_rows, proxy=proxy)
    costs = money(future_costs, nonnegative=True)
    values = np.full(grid.shape, np.nan)
    valid = np.zeros(grid.shape, dtype=bool)
    evaluation = hashlib.sha256(b"pnl.vol_evaluation/1")
    nrows = values.size
    flat = values.reshape(-1)
    mask = valid.reshape(-1)
    if isinstance(market.volatility, PerLegVol) and set(market.volatility.by_leg_id) != {
        x.leg_id for x in snapshot.position.legs
    }:
        raise PnLError("per-leg volatility must cover exactly the live leg IDs")
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
                flat[rows] = grid.sizes[leg_index] * np.asarray(
                    payoff(unit, grid.spot[si[selected]])
                )
                mask[rows] = True
        live = ~intrinsic
        if isinstance(market.volatility, SurfaceVol):
            live &= grid.spot[si] > 0
        elif model == "black":
            live &= grid.spot[si] > 0
        # Surface log coordinates and Black forward mappings require positive
        # live spot. Flat spot models retain the existing kernel boundary behavior.
        zero = (~intrinsic) & (grid.spot[si] == 0) & ~live
        if zero.any() and not diagnostic:
            raise PnLError("zero spot is outside logarithmic surface/forward coordinates")
        if live.any():
            h, s, leg_rows, t = hi[live], si[live], li[live], remaining[live]
            sigma, ok = _volatility(market.volatility, grid, market, model, h, s, leg_rows, t)
            receipt = np.empty(sigma.size, dtype=[("row", "<i8"), ("sigma", "<f8"), ("valid", "?")])
            receipt["row"] = start + np.flatnonzero(live)
            receipt["sigma"] = sigma
            receipt["valid"] = ok
            evaluation.update(receipt.tobytes())
            if not ok.all() and not diagnostic:
                raise PnLError("surface has invalid requested IV cells")
            rows = start + np.flatnonzero(live)
            if ok.any():
                priced = _price_rows(
                    grid, market, model, backend, h[ok], s[ok], leg_rows[ok], t[ok], sigma[ok]
                )
                finite = np.isfinite(priced)
                if not finite.all() and not diagnostic:
                    raise PnLError("model returned nonfinite prices")
                flat[rows[ok]] = np.where(finite, priced, np.nan)
                mask[rows[ok]] = finite
    aggregate = np.asarray(valid.all(axis=-1), dtype=bool)
    if snapshot.cumulative_cash is None:
        total = np.full(grid.shape[:2], np.nan)
        aggregate[:] = False
    else:
        total = float(snapshot.cumulative_cash) + values.sum(axis=-1) - float(costs)
        total = np.where(aggregate, total, np.nan)
    cls = PnLProfile
    if proxy:
        from .proxy import EuropeanProxyProfile

        cls = EuropeanProxyProfile
    return cls(
        horizons=tuple(float(x) for x in grid.horizons),
        target_date=None,
        spot=grid.spot,
        leg_values=values,
        total=total,
        valid=aggregate,
        leg_valid=valid,
        cumulative_cash=snapshot.cumulative_cash,
        paid_fees=snapshot.paid_fees,
        future_costs=costs,
        leg_ids=tuple(x.leg_id for x in snapshot.position.legs),
        meta=_meta(
            grid,
            market,
            model,
            backend,
            market.volatility,
            costs,
            evaluation_digest=evaluation.hexdigest(),
        ),
        metrics=None,
        strikes=tuple(x.contract.strike for x in snapshot.position.legs),
        contracts=tuple(x.contract for x in snapshot.position.legs),
        signed_sizes=tuple(x.signed_size for x in snapshot.position.legs),
        conversion_errors=(),
        conversion_bound=np.full(grid.spot.size, np.nan),
        realized=snapshot.realized,
    )


def pnl_at_horizon(
    snapshot: PositionSnapshot,
    spot: Any,
    *,
    horizons: Any,
    market: HorizonMarket,
    model: str,
    backend: str,
    convention: str,
    future_costs: Decimal = Decimal("0"),
    chunk_rows: int | None = None,
    diagnostic: bool = False,
) -> PnLProfile:
    """European model inventory plus nominal cash, up to the earliest live expiry."""
    return _evaluate(
        snapshot,
        spot,
        horizons=horizons,
        market=market,
        model=model,
        backend=backend,
        convention=convention,
        future_costs=future_costs,
        chunk_rows=chunk_rows,
        diagnostic=diagnostic,
    )
