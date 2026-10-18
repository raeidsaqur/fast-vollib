"""Explicit volatility, surface movement and spot-to-forward assumptions."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Mapping

import numpy as np

from .errors import PnLError


def finite(value: Any, name: str, *, nonnegative: bool = False, positive: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise PnLError(f"{name} requires a real scalar")
    result = float(value)
    if not np.isfinite(result) or (nonnegative and result < 0) or (positive and result <= 0):
        raise PnLError(f"{name} is outside its finite domain")
    return result


@dataclass(frozen=True, slots=True)
class FlatVol:
    sigma: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "sigma", finite(self.sigma, "sigma", nonnegative=True))


@dataclass(frozen=True, slots=True)
class PerLegVol:
    by_leg_id: Mapping[str, float]

    def __post_init__(self) -> None:
        data = {
            key: finite(value, "leg volatility", nonnegative=True)
            for key, value in self.by_leg_id.items()
        }
        if any(not isinstance(key, str) or not key for key in data):
            raise PnLError("per-leg volatilities require explicit leg IDs")
        object.__setattr__(self, "by_leg_id", MappingProxyType(data))


@dataclass(frozen=True, slots=True, kw_only=True)
class ProportionalCarry:
    """F_h(tau; S) = base_forward(h, tau; S0) * S / S0, explicitly supplied."""

    origin_spot: float
    base_forward: Callable[[Any, Any], Any]
    policy_id: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "origin_spot", finite(self.origin_spot, "origin spot", positive=True)
        )
        if (
            not callable(self.base_forward)
            or not isinstance(self.policy_id, str)
            or not self.policy_id
        ):
            raise PnLError(
                "proportional carry requires an explicit base-forward function and policy ID"
            )

    def __call__(self, spot: Any, horizon: Any, remaining: Any) -> Any:
        return (
            np.asarray(self.base_forward(horizon, remaining), dtype=float)
            * np.asarray(spot, dtype=float)
            / self.origin_spot
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class _SurfaceMovement:
    reference_market: Any
    spot_movement: str
    time_movement: str
    forward_mapping: Callable[[Any, Any, Any], Any]
    forward_mapping_id: str
    origin_spot: float | None = None

    def __post_init__(self) -> None:
        from ..surface import SurfaceMarket

        if not isinstance(self.reference_market, SurfaceMarket):
            raise PnLError("surface volatility requires an explicit reference SurfaceMarket")
        from .results import _owned

        market = self.reference_market
        copied = SurfaceMarket(
            T=market.T,
            forward=market.forward,
            rate=market.rate,
            carry=market.carry,
            interpolation=market.interpolation,
            extrapolation=market.extrapolation,
            source=market.source,
        )
        for name in ("T", "forward", "rate", "carry"):
            object.__setattr__(copied, name, _owned(getattr(copied, name), float))
        object.__setattr__(self, "reference_market", copied)
        if self.spot_movement not in (
            "sticky_strike",
            "sticky_forward_moneyness",
        ) or self.time_movement not in ("fixed_expiry", "constant_tenor"):
            raise PnLError("explicit supported spot/time movement policies required")
        if (
            not callable(self.forward_mapping)
            or not isinstance(self.forward_mapping_id, str)
            or not self.forward_mapping_id
        ):
            raise PnLError(
                "surface movement requires a named spot/horizon/remaining-tenor forward mapping"
            )
        origin = (
            self.origin_spot
            if self.origin_spot is not None
            else getattr(self.forward_mapping, "origin_spot", None)
        )
        object.__setattr__(
            self, "origin_spot", finite(origin, "reference origin spot", positive=True)
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class SurfaceVol(_SurfaceMovement):
    surface: Any

    def __post_init__(self) -> None:
        _SurfaceMovement.__post_init__(self)
        if not callable(getattr(self.surface, "evaluate", None)):
            raise PnLError("surface must expose evaluate")


@dataclass(frozen=True, slots=True, kw_only=True)
class SurfaceVolSamples(_SurfaceMovement):
    distribution: Any
    n_samples: int
    rng: Any

    def __post_init__(self) -> None:
        _SurfaceMovement.__post_init__(self)
        if not callable(getattr(self.distribution, "sample", None)):
            raise PnLError("surface distribution must expose sample")
        if type(self.n_samples) is not int or self.n_samples < 1:
            raise PnLError("n_samples must be a positive integer")


@dataclass(frozen=True, slots=True, kw_only=True)
class HorizonMarket:
    rate: float
    dividend_yield: float | None
    volatility: FlatVol | PerLegVol | SurfaceVol | None
    forward_mapping: Callable[[Any, Any, Any], Any] | None = None
    forward_mapping_id: str | None = None
    discount_convention: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "rate", finite(self.rate, "rate"))
        if self.dividend_yield is not None:
            object.__setattr__(
                self, "dividend_yield", finite(self.dividend_yield, "dividend yield")
            )
        if self.volatility is not None and not isinstance(
            self.volatility, (FlatVol, PerLegVol, SurfaceVol)
        ):
            raise PnLError("market requires explicit supported volatility assumptions")
        if self.forward_mapping is not None and (
            not callable(self.forward_mapping)
            or not isinstance(self.forward_mapping_id, str)
            or not self.forward_mapping_id
        ):
            raise PnLError("a forward mapping must have an explicit reproducibility policy ID")
        if self.forward_mapping is None and self.forward_mapping_id is not None:
            raise PnLError("forward mapping ID has no mapping")
        if self.discount_convention not in (None, "constant_continuous_rate"):
            raise PnLError("unsupported horizon discount convention")
