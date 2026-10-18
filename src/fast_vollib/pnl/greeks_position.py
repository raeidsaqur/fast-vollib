"""Unit-batch position Greeks, with explicit legacy or annual/unit report scaling."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
import json
from types import MappingProxyType
from typing import Any, Mapping, cast

import numpy as np

from .._array_api import get_namespace
from ..instruments import EuropeanOptionBatch, VanillaMarketInputs
from ..observations.records import digest
from ..portfolio import PositionSnapshot, year_fraction
from ..portfolio.contracts import require_standard
from ..portfolio.serialization import snapshot_to_dict
from ..types import BackendLiteral, ModelLiteral
from .errors import PnLError, PnLHorizonError
from .results import ResultMeta, _owned

_NAMES = ("delta", "gamma", "theta", "rho", "vega")


@dataclass(frozen=True, slots=True, kw_only=True)
class PositionGreeks:
    leg_ids: tuple[str, ...]
    contracts: tuple
    signed_sizes: tuple[Decimal, ...]
    per_leg: Mapping[str, Any]
    net: Mapping[str, float]
    units: Mapping[str, str]
    scaling_from_existing: Mapping[str, float]
    meta: ResultMeta

    def __post_init__(self) -> None:
        object.__setattr__(self, "leg_ids", tuple(self.leg_ids))
        object.__setattr__(self, "contracts", tuple(self.contracts))
        object.__setattr__(self, "signed_sizes", tuple(self.signed_sizes))
        if len(self.contracts) != len(self.leg_ids) or len(self.signed_sizes) != len(self.leg_ids):
            raise PnLError("Greek contract/size axes disagree")
        for contract in self.contracts:
            require_standard(contract)
            if (
                contract.exercise != "european"
                or contract.currency != self.meta.currency
                or contract.stamp.world_id != self.meta.world_id
            ):
                raise PnLError("Greek reports require matching supported European source contracts")
        if len(set(self.leg_ids)) != len(self.leg_ids) or any(
            set(m) != set(_NAMES)
            for m in (self.per_leg, self.net, self.units, self.scaling_from_existing)
        ):
            raise PnLError("Greek names/leg identities disagree")
        values = {key: _owned(self.per_leg[key], float) for key in _NAMES}
        for key, value in values.items():
            if (
                value.shape != (len(self.leg_ids),)
                or not np.isfinite(value).all()
                or not np.isfinite(self.net[key])
                or not np.isclose(value.sum(), self.net[key], rtol=1e-12, atol=1e-12)
            ):
                raise PnLError("Greek values/net must be finite and agree")
        for key in _NAMES:
            if (
                not isinstance(self.units[key], str)
                or not self.units[key]
                or not np.isfinite(self.scaling_from_existing[key])
                or self.scaling_from_existing[key] <= 0
            ):
                raise PnLError("Greek units/scaling must be explicit")
        if self.meta.basis != "model":
            raise PnLError("position Greeks require supported European model valuation")
        for name, value in (
            ("per_leg", values),
            ("net", dict(self.net)),
            ("units", dict(self.units)),
            ("scaling_from_existing", dict(self.scaling_from_existing)),
        ):
            object.__setattr__(self, name, MappingProxyType(value))

    def to_frame(self) -> Any:
        import pandas as pd

        rows: dict[str, Any] = {"leg_id": [*self.leg_ids, "net"]}
        rows.update({name: [*self.per_leg[name], self.net[name]] for name in _NAMES})
        rows["units"] = json.dumps(dict(self.units), sort_keys=True)
        return pd.DataFrame(rows)

    def to_json(self) -> str:
        from .serialization import _encode

        return json.dumps(
            {"schema": "pnl.position_greeks/1", "greeks": _encode(self)},
            sort_keys=True,
            allow_nan=False,
        )


def position_greeks(
    snapshot: PositionSnapshot,
    market: VanillaMarketInputs,
    *,
    model: str,
    backend: str,
    convention: str,
    report_units: str = "existing",
) -> PositionGreeks:
    from ..instruments import greeks_instrument

    if not isinstance(snapshot, PositionSnapshot) or not isinstance(market, VanillaMarketInputs):
        raise PnLError("typed snapshot and VanillaMarketInputs required")
    if model not in ("black", "black_scholes", "black_scholes_merton") or backend not in (
        "auto",
        "numpy",
        "torch",
        "jax",
        "numba",
    ):
        raise PnLError("explicit supported model/backend required")
    if report_units not in ("existing", "annual_unit"):
        raise PnLError("report_units must be existing or annual_unit")
    year_fraction(snapshot.valuation_time, snapshot.valuation_time, convention=convention)
    legs = snapshot.position.legs
    maturity = []
    for leg in legs:
        require_standard(leg.contract)
        if leg.contract.exercise == "american":
            raise PnLError(
                "American Greeks are unsupported; European proxies do not produce Greeks"
            )
        if leg.contract.expiry_instant is None:
            raise PnLHorizonError("Greek maturity requires a resolved expiry instant")
        t = year_fraction(
            snapshot.valuation_time, leg.contract.expiry_instant, convention=convention
        )
        if t <= 0:
            raise PnLHorizonError("live model Greeks require positive remaining maturity")
        maturity.append(t)
    sizes = np.asarray([float(x.signed_size) for x in legs])
    count = len(legs)
    inputs = {}
    for key in ("underlying", "rate", "volatility", "dividend_yield"):
        value = getattr(market, key)
        if value is not None:
            value = get_namespace(value).to_numpy(value) if hasattr(value, "dtype") else value
            try:
                value = np.broadcast_to(np.asarray(value, dtype=float), (count,)).copy()
            except ValueError as exc:
                raise PnLError("Greek market fields must broadcast to the live leg axis") from exc
            if not np.isfinite(value).all():
                raise PnLError("Greek inputs must be finite")
            inputs[key] = value
    if count:
        if not {"underlying", "rate", "volatility"}.issubset(inputs):
            raise PnLError("Greek market needs underlying, rate and volatility")
        if (inputs["underlying"] <= 0).any() or (inputs["volatility"] < 0).any():
            raise PnLError("Greek state must be positive and volatility nonnegative")
        if model == "black_scholes" and (
            "dividend_yield" not in inputs or (inputs["dividend_yield"] != 0).any()
        ):
            raise PnLError("Black-Scholes Greeks require explicit zero dividend yield")
        if model == "black_scholes_merton" and "dividend_yield" not in inputs:
            raise PnLError("BSM Greeks require explicit dividend yield")
        batch = EuropeanOptionBatch.from_arrays(
            option_type=[x.contract.option_type.value for x in legs],
            strike=[float(x.contract.strike) for x in legs],
            maturity=maturity,
            underlier=snapshot.position.underlier,
            notional=1.0,
        )
        raw = greeks_instrument(
            batch,
            VanillaMarketInputs(**inputs),
            model=cast(ModelLiteral, model),
            backend=cast(BackendLiteral, backend),
            return_as="numpy",
        )
    else:
        raw = {name: np.empty(0) for name in _NAMES}
    scaling = {name: 1.0 for name in _NAMES}
    units = {
        "delta": "per underlying price unit",
        "gamma": "per squared underlying price unit",
        "theta": "per 1/365 model year",
        "vega": "per 0.01 volatility change",
        "rho": "per 0.01 rate change",
    }
    if model == "black":
        units["delta"] = "per forward price unit"
        units["gamma"] = "per squared forward price unit"
    if report_units == "annual_unit":
        scaling.update(theta=365.0, vega=100.0, rho=100.0)
        units.update(
            theta="per model year", vega="per unit volatility change", rho="per unit rate change"
        )
    per_leg = {name: np.asarray(raw[name], dtype=float) * sizes * scaling[name] for name in _NAMES}
    source = json.dumps(
        {
            "backend": backend,
            "market": {k: v.tolist() for k, v in inputs.items()},
            "report_units": report_units,
            "scaling": scaling,
        },
        sort_keys=True,
    )
    meta = ResultMeta(
        origin=snapshot.origin,
        basis="model",
        valuation_time=snapshot.valuation_time,
        knowledge_cutoff=snapshot.knowledge_cutoff,
        knowledge_label=snapshot.knowledge_label,
        world_id=snapshot.world_id,
        observations_digest=snapshot.observations_digest,
        currency=snapshot.currency,
        caveat=None,
        terminal_assumption="not_applicable:model",
        accounting_policy=snapshot.accounting_policy,
        time_policy=convention,
        model=model,
        vol_source=source,
        validation_status="analytic_european_greeks",
        schema_version="pnl.result/2",
        input_digest=digest(
            {
                "snapshot": snapshot_to_dict(snapshot),
                "market": source,
                "model": model,
                "convention": convention,
            }
        ),
    )
    return PositionGreeks(
        leg_ids=tuple(x.leg_id for x in legs),
        contracts=tuple(x.contract for x in legs),
        signed_sizes=tuple(x.signed_size for x in legs),
        per_leg=per_leg,
        net={name: float(value.sum()) for name, value in per_leg.items()},
        units=units,
        scaling_from_existing=scaling,
        meta=meta,
    )


def greeks_from_dict(value: Any) -> PositionGreeks:
    from ..portfolio.serialization import contract_from_dict
    from .serialization import _decode, _fields

    if (
        not isinstance(value, dict)
        or set(value) != {"schema", "greeks"}
        or value["schema"] != "pnl.position_greeks/1"
    ):
        raise PnLError("unknown position Greek schema")
    data = _decode(value["greeks"])
    _fields(data, PositionGreeks)
    _fields(data["meta"], ResultMeta)
    data["meta"] = ResultMeta(**data["meta"])
    data["contracts"] = tuple(contract_from_dict(c) for c in data["contracts"])
    return PositionGreeks(**data)
