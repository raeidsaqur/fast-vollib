"""Owned canonical observation records, independent of financial payload types."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any

from .errors import ObservationError

CANONICAL_POLICY = "observation.canonical/1"


def utc(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ObservationError("timestamps must be UTC-aware instants")
    return value.astimezone(timezone.utc)


def interval(value: Any) -> Any:
    pair = tuple(value)
    if len(pair) != 2:
        raise ObservationError("an interval needs two endpoints")
    a, b = (utc(x) for x in pair)
    if a > b:
        raise ObservationError("interval endpoints must be ordered")
    return a, b


def text(value: Any, field: Any) -> Any:
    if not isinstance(value, str) or not value.strip():
        raise ObservationError(f"{field} must be a nonempty string")
    return value


def freeze(value: Any) -> Any:
    """Copy a restricted data tree; reject objects, callables and ambiguous tags."""
    if value is None or type(value) in (bool, int, str):
        return value
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ObservationError("nonfinite Decimal payload")
        return value
    if isinstance(value, datetime):
        return utc(value)
    if type(value) is float and math.isfinite(value):
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(k, str) or k.startswith("$") for k in value):
            raise ObservationError("payload keys must be strings without reserved '$' prefix")
        return MappingProxyType({k: freeze(v) for k, v in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(freeze(x) for x in value)
    raise ObservationError(f"unsupported canonical payload type: {type(value).__name__}")


def plain(value: Any) -> Any:
    if isinstance(value, Decimal):
        return {"$decimal": str(value)}
    if isinstance(value, datetime):
        return {"$utc": utc(value).isoformat()}
    if isinstance(value, Mapping):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    return value


def unplain(value: Any) -> Any:
    if isinstance(value, dict):
        if set(value) == {"$decimal"}:
            if not isinstance(value["$decimal"], str):
                raise ObservationError("Decimal tags require text")
            return Decimal(value["$decimal"])
        if set(value) == {"$utc"}:
            return utc(datetime.fromisoformat(value["$utc"]))
        return {k: unplain(v) for k, v in value.items()}
    if isinstance(value, list):
        return tuple(unplain(v) for v in value)
    return value


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        plain(freeze(value)), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def digest(value: Any) -> str:
    return hashlib.sha256(CANONICAL_POLICY.encode() + b"\n" + canonical_bytes(value)).hexdigest()


@dataclass(frozen=True, slots=True, kw_only=True)
class ObservationStamp:
    event_time: datetime | tuple[datetime, datetime]
    available_time: datetime
    recorded_at: datetime
    source: str
    source_record_id: str | None = None
    version: int = 0
    revision_of: str | None = None
    world_id: str = "actual"
    timing_note: str | None = None
    mapping_version: str | None = None

    def __post_init__(self) -> None:
        event = (
            utc(self.event_time)
            if isinstance(self.event_time, datetime)
            else interval(self.event_time)
        )
        object.__setattr__(self, "event_time", event)
        for field in ("available_time", "recorded_at"):
            object.__setattr__(self, field, utc(getattr(self, field)))
        for field in ("source", "world_id"):
            text(getattr(self, field), field)
        for field in ("source_record_id", "revision_of", "timing_note", "mapping_version"):
            if getattr(self, field) is not None:
                text(getattr(self, field), field)
        if type(self.version) is not int or self.version < 0:
            raise ObservationError("version must be a nonnegative integer")
        if self.source == "synthetic" and self.world_id == "actual":
            raise ObservationError("synthetic evidence requires a hypothetical world")


@dataclass(frozen=True, slots=True, kw_only=True)
class ObservationRecord:
    record_id: str
    kind: str
    entities: tuple[str, ...]
    payload: Any
    payload_schema: str
    stamp: ObservationStamp
    payload_digest: str = ""
    coverage: tuple[datetime, datetime] | None = None

    def __post_init__(self) -> None:
        for field in ("record_id", "kind", "payload_schema"):
            text(getattr(self, field), field)
        entities = tuple(self.entities)
        if not entities or len(set(entities)) != len(entities):
            raise ObservationError("entities must be nonempty and unique")
        for entity in entities:
            text(entity, "entity")
        object.__setattr__(self, "entities", entities)
        if not isinstance(self.stamp, ObservationStamp):
            raise ObservationError("stamp must be an ObservationStamp")
        payload = freeze(self.payload)
        calculated = digest(payload)
        if self.payload_digest and self.payload_digest != calculated:
            raise ObservationError("payload digest does not match canonical content")
        object.__setattr__(self, "payload", payload)
        object.__setattr__(self, "payload_digest", calculated)
        if self.coverage is not None:
            object.__setattr__(self, "coverage", interval(self.coverage))
