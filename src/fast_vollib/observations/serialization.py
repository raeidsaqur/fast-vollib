"""Strict, versioned JSON codecs; never execute payload content."""

from __future__ import annotations

from dataclasses import fields
from datetime import datetime
import json
from typing import Any

from .errors import ObservationError
from .records import ObservationRecord, ObservationStamp, plain, unplain


def strict_fields(value: Any, expected: Any) -> Any:
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ObservationError("unknown or missing codec fields")


def stamp_to_dict(stamp: ObservationStamp) -> dict[str, Any]:
    return {f.name: plain(getattr(stamp, f.name)) for f in fields(stamp)}


def stamp_from_dict(value: Any) -> ObservationStamp:
    strict_fields(value, (f.name for f in fields(ObservationStamp)))
    return ObservationStamp(**{k: unplain(v) for k, v in value.items()})


def record_to_dict(record: ObservationRecord) -> dict[str, Any]:
    return {
        "schema": "observation.record/1",
        **{
            f.name: stamp_to_dict(record.stamp)
            if f.name == "stamp"
            else plain(getattr(record, f.name))
            for f in fields(record)
        },
    }


def record_from_dict(value: Any) -> ObservationRecord:
    strict_fields(value, ["schema", *(f.name for f in fields(ObservationRecord))])
    if value["schema"] != "observation.record/1":
        raise ObservationError("unknown observation schema")
    data = {k: unplain(v) for k, v in value.items() if k not in ("schema", "stamp")}
    return ObservationRecord(**data, stamp=stamp_from_dict(value["stamp"]))


def _pairs(pairs: Any) -> Any:
    result = {}
    for k, v in pairs:
        if k in result:
            raise ObservationError("duplicate JSON key")
        result[k] = v
    return result


def load_json(value: Any) -> Any:
    return json.loads(
        value,
        object_pairs_hook=_pairs,
        parse_constant=lambda x: (_ for _ in ()).throw(ObservationError(f"invalid JSON {x}")),
    )


def dumps(record: ObservationRecord) -> str:
    return json.dumps(record_to_dict(record), sort_keys=True, allow_nan=False)


def loads(value: Any) -> Any:
    return record_from_dict(load_json(value))


def snapshot_to_dict(snapshot: Any) -> Any:
    return {
        "schema": "observation.snapshot/1",
        "records": [record_to_dict(r) for r in snapshot.records],
        "knowledge_cutoff": snapshot.knowledge_cutoff.isoformat(),
        "world_id": snapshot.world_id,
        "policy": snapshot.policy,
        "digest": snapshot.digest,
    }


def snapshot_from_dict(value: Any) -> Any:
    from .snapshots import select_as_of

    strict_fields(value, ("schema", "records", "knowledge_cutoff", "world_id", "policy", "digest"))
    if value["schema"] != "observation.snapshot/1":
        raise ObservationError("unknown snapshot schema")
    # Snapshots retain the full admissible chains for independent validation.
    result = select_as_of(
        (record_from_dict(r) for r in value["records"]),
        knowledge_cutoff=datetime.fromisoformat(value["knowledge_cutoff"]),
        world_id=value["world_id"],
        policy=value["policy"],
    )
    if result.digest != value["digest"]:
        raise ObservationError("snapshot digest mismatch")
    return result
