"""Target identity is separate from channels, model choice and financial projection."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from ..instruments import InstrumentRef
from ._validate import instant, names, text
from .errors import ForecastError


class Quantity(str, Enum):
    SPOT = "spot"
    RETURN = "return"
    VOLUME = "volume"
    RATE = "rate"
    DISCOUNT_CURVE = "discount_curve"
    IV_SURFACE = "iv_surface"
    LOB_STATE = "lob_state"
    LATENT = "latent"
    OTHER = "other"


@dataclass(frozen=True, slots=True, kw_only=True)
class ReturnSpec:
    kind: str
    cumulative: bool
    base_time: datetime | None
    includes_distributions: bool

    def __post_init__(self) -> None:
        if self.kind not in ("simple", "log"):
            raise ForecastError("return kind must be simple or log")
        if type(self.cumulative) is not bool or type(self.includes_distributions) is not bool:
            raise ForecastError("return conventions require explicit booleans")
        if self.base_time is not None:
            object.__setattr__(self, "base_time", instant(self.base_time))


@dataclass(frozen=True, slots=True, kw_only=True)
class ForecastTarget:
    target_id: str
    entity: InstrumentRef | str
    quantity: Quantity
    units: str
    resolution: str
    coordinates: tuple[str, ...] = ()
    return_spec: ReturnSpec | None = None
    currency: str | None = None

    def __post_init__(self) -> None:
        for field in ("target_id", "units", "resolution"):
            text(getattr(self, field), field)
        if isinstance(self.entity, str):
            text(self.entity, "entity")
        elif not isinstance(self.entity, InstrumentRef):
            raise ForecastError("entity must be an InstrumentRef or opaque text")
        object.__setattr__(self, "quantity", Quantity(self.quantity))
        coords = names(self.coordinates, "coordinates") if self.coordinates else ()
        object.__setattr__(self, "coordinates", coords)
        if (self.quantity == Quantity.RETURN) != isinstance(self.return_spec, ReturnSpec):
            raise ForecastError("ReturnSpec is required exactly for return targets")
        if self.currency is not None:
            text(self.currency, "currency")
            if (
                isinstance(self.entity, InstrumentRef)
                and self.entity.currency is not None
                and self.currency != self.entity.currency
            ):
                raise ForecastError("target currency disagrees with entity currency")
