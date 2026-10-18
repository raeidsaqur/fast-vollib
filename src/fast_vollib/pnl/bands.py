"""Owned surface-sensitivity results; no spot-law probabilities or risk summaries."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
import json
from typing import Any

import numpy as np

from ..portfolio._validate import money
from ..portfolio.contracts import require_standard
from .errors import PnLError
from .results import ResultMeta, _owned


@dataclass(frozen=True, slots=True, kw_only=True)
class SurfaceSensitivityBand:
    spot: Any
    horizons: tuple[float, ...]
    samples: Any
    total: Any
    valid: Any
    sample_ids: tuple[str, ...]
    weights: Any
    original_weight_mass: float | None
    cumulative_cash: Decimal | None
    future_costs: Decimal
    leg_ids: tuple[str, ...]
    contracts: tuple
    signed_sizes: tuple[Decimal, ...]
    meta: ResultMeta

    def __post_init__(self) -> None:
        from .model import BAND_CAVEAT, PROXY_CAVEAT
        from .proxy import EuropeanProxySensitivityBand

        if (
            not isinstance(self.meta, ResultMeta)
            or self.meta.basis not in ("model", "european_proxy")
            or not self.meta.caveat
            or BAND_CAVEAT not in self.meta.caveat
        ):
            raise PnLError("surface bands must retain their non-probability caveat and model basis")
        proxy = isinstance(self, EuropeanProxySensitivityBand)
        if proxy != (self.meta.basis == "european_proxy") or (
            proxy and PROXY_CAVEAT not in self.meta.caveat
        ):
            raise PnLError("sampled proxy type and caveat must be retained")
        for name in ("horizons", "sample_ids", "leg_ids", "contracts", "signed_sizes"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        for name in ("spot", "samples", "total", "weights"):
            object.__setattr__(self, name, _owned(getattr(self, name), float))
        if np.asarray(self.valid).dtype != np.dtype(bool):
            raise PnLError("band validity must be boolean")
        object.__setattr__(self, "valid", _owned(self.valid, bool))
        shape = self.samples.shape
        if (
            len(shape) != 4
            or shape[0] < 1
            or shape[1:] != (len(self.horizons), self.spot.size, len(self.leg_ids))
            or self.spot.ndim != 1
            or not self.spot.size
        ):
            raise PnLError("band axes must be (surface sample, horizon, spot, leg)")
        if self.total.shape != shape[:3] or self.valid.shape != shape:
            raise PnLError("band total/validity axes disagree")
        if (
            len(self.sample_ids) != shape[0]
            or len(set(self.sample_ids)) != shape[0]
            or any(not isinstance(i, str) or not i for i in self.sample_ids)
        ):
            raise PnLError("band sample identities must be unique")
        if (
            not np.isfinite(self.spot).all()
            or (self.spot < 0).any()
            or not self.horizons
            or any(not np.isfinite(h) or h < 0 for h in self.horizons)
            or any(a >= b for a, b in zip(self.horizons, self.horizons[1:]))
        ):
            raise PnLError("invalid band spot/time grid")
        if (
            self.weights.shape != (shape[0],)
            or not np.isfinite(self.weights).all()
            or (self.weights < 0).any()
        ):
            raise PnLError("invalid band weights")
        mass = float(self.weights.sum())
        if not np.isfinite(mass) or mass <= 0:
            raise PnLError("band weights require positive finite mass")
        if self.original_weight_mass is None:
            object.__setattr__(self, "weights", _owned(self.weights / mass, float))
            object.__setattr__(self, "original_weight_mass", mass)
        elif (
            not np.isfinite(self.original_weight_mass)
            or self.original_weight_mass <= 0
            or abs(mass - 1) > 8 * np.finfo(float).eps
        ):
            raise PnLError("retained normalized band weights/mass disagree")
        costs = money(self.future_costs, nonnegative=True)
        object.__setattr__(self, "future_costs", costs)
        if self.cumulative_cash is not None:
            object.__setattr__(self, "cumulative_cash", money(self.cumulative_cash))
        if (
            len(self.contracts) != shape[-1]
            or len(self.signed_sizes) != shape[-1]
            or len(set(self.leg_ids)) != shape[-1]
        ):
            raise PnLError("band contract identities disagree")
        american = False
        for contract, size in zip(self.contracts, self.signed_sizes):
            require_standard(contract)
            money(size)
            if (
                not size
                or contract.currency != self.meta.currency
                or contract.stamp.world_id != self.meta.world_id
            ):
                raise PnLError("band contract world/currency/size disagrees")
            american |= contract.exercise == "american"
        if american != proxy:
            raise PnLError("American surface sensitivity requires a separately typed proxy")
        if not np.isfinite(self.samples[self.valid]).all():
            raise PnLError("valid band cells must be finite")
        if not np.isnan(self.samples[~self.valid]).all():
            raise PnLError("invalid band cells must remain unavailable")
        aggregate = self.valid.all(axis=-1)
        if self.cumulative_cash is None:
            if not np.isnan(self.total).all():
                raise PnLError("unknown entry cash leaves band totals unavailable")
        else:
            expected = (
                float(self.cumulative_cash) + self.samples.sum(axis=-1) - float(self.future_costs)
            )
            scale = (
                1
                + abs(float(self.cumulative_cash))
                + np.abs(self.samples).sum(axis=-1)
                + float(costs)
            )
            if not np.isfinite(self.total[aggregate]).all() or np.any(
                np.abs(self.total - expected)[aggregate]
                > (64 * np.finfo(float).eps * scale)[aggregate]
            ):
                raise PnLError("band violates signed inventory + nominal cash - future costs")
            if not np.isnan(self.total[~aggregate]).all():
                raise PnLError("incomplete band aggregates must remain unavailable")

    def quantiles(self, levels: Any) -> Any:
        """Inverse weighted empirical-CDF sensitivity quantiles; invalid mass stays unavailable."""
        levels = np.asarray(levels, dtype=float)
        if (
            levels.ndim != 1
            or not levels.size
            or not np.isfinite(levels).all()
            or (levels < 0).any()
            or (levels > 1).any()
            or (np.diff(levels) < 0).any()
        ):
            raise PnLError("quantile levels must be sorted and within [0,1]")
        out = np.full((levels.size, *self.total.shape[1:]), np.nan)
        positive = self.weights > 0
        complete = (
            self.valid[positive].all(axis=(0, -1))
            if positive.any()
            else np.zeros(self.total.shape[1:], dtype=bool)
        )
        if self.cumulative_cash is None:
            return _owned(out, float)
        for h, s in zip(*np.nonzero(complete)):
            values = self.total[positive, h, s]
            weights = self.weights[positive]
            order = np.argsort(values, kind="stable")
            cdf = np.cumsum(weights[order])
            cdf[-1] = 1.0
            indices = np.minimum(np.searchsorted(cdf, levels, side="left"), len(order) - 1)
            out[:, h, s] = values[order][indices]
        return _owned(out, float)

    def to_frame(self) -> Any:
        import pandas as pd

        n, h, s = self.total.shape
        rows = {
            "sample_id": np.repeat(self.sample_ids, h * s),
            "weight": np.repeat(self.weights, h * s),
            "horizon": np.tile(np.repeat(self.horizons, s), n),
            "spot": np.tile(self.spot, n * h),
            "total": self.total.reshape(-1).copy(),
            "basis": self.meta.basis,
            "caveat": self.meta.caveat,
            "model": self.meta.model,
            "vol_source": self.meta.vol_source,
            "currency": self.meta.currency,
            "origin": self.meta.origin,
            "world_id": self.meta.world_id,
            "valuation_time": self.meta.valuation_time,
            "knowledge_cutoff": self.meta.knowledge_cutoff,
            "knowledge_label": self.meta.knowledge_label,
            "time_policy": self.meta.time_policy,
            "spot_movement": self.meta.spot_movement,
            "time_movement": self.meta.time_movement,
        }
        for i, key in enumerate(self.leg_ids):
            rows["leg:" + key] = self.samples[:, :, :, i].reshape(-1).copy()
            rows["valid:" + key] = self.valid[:, :, :, i].reshape(-1).copy()
        return pd.DataFrame(rows)

    def to_dict(self) -> dict[str, Any]:
        from .serialization import _encode

        data = _encode(self)
        if self.cumulative_cash is None:
            data["total"] = [
                [[{"status": "missing"} for _ in row] for row in sample] for sample in self.total
            ]
        return {
            "schema": "pnl.surface_sensitivity/1",
            "result_type": "european_proxy_surface_sensitivity"
            if self.meta.basis == "european_proxy"
            else "surface_sensitivity",
            "band": data,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, allow_nan=False)


def band_from_dict(value: Any) -> SurfaceSensitivityBand:
    from ..portfolio.serialization import contract_from_dict
    from .proxy import EuropeanProxySensitivityBand
    from .serialization import _decode, _fields

    if (
        not isinstance(value, dict)
        or set(value) != {"schema", "result_type", "band"}
        or value["schema"] != "pnl.surface_sensitivity/1"
    ):
        raise PnLError("unknown surface-sensitivity schema")
    cls = (
        EuropeanProxySensitivityBand
        if value["result_type"] == "european_proxy_surface_sensitivity"
        else SurfaceSensitivityBand
    )
    if value["result_type"] not in ("surface_sensitivity", "european_proxy_surface_sensitivity"):
        raise PnLError("unknown sensitivity result type")
    data = _decode(value["band"])
    _fields(data, cls)
    _fields(data["meta"], ResultMeta)
    data["meta"] = ResultMeta(**data["meta"])
    data["contracts"] = tuple(contract_from_dict(c) for c in data["contracts"])
    return cls(**data)
