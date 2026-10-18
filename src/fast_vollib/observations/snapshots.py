"""Deterministic available-time archive selection and full replacement revisions."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
import json
from typing import Any

from .errors import ObservationConflictError, ObservationError, RevisionError
from .records import ObservationRecord, digest, text, utc
from .serialization import record_to_dict

POLICY = "available_time/1"


@dataclass(frozen=True, slots=True, kw_only=True)
class ObservationSnapshot:
    records: tuple[ObservationRecord, ...]
    knowledge_cutoff: datetime
    world_id: str
    policy: str
    digest: str
    selected_records: tuple[ObservationRecord, ...] = field(repr=False)

    def __post_init__(self) -> None:
        expected = _select(
            self.records,
            knowledge_cutoff=self.knowledge_cutoff,
            world_id=self.world_id,
            policy=self.policy,
        )
        if (
            self.digest != expected["digest"]
            or tuple(self.selected_records) != expected["selected_records"]
        ):
            raise ObservationError("snapshot identity/selection disagrees with its evidence")
        for name in ("records", "selected_records", "knowledge_cutoff", "policy"):
            object.__setattr__(self, name, expected[name])


def select_as_of(
    records: Iterable[ObservationRecord],
    *,
    knowledge_cutoff: datetime,
    world_id: str = "actual",
    policy: str = "available_time",
) -> ObservationSnapshot:
    """Return an owned archive snapshot, including admissible revision ancestry.

    Only ``selected_records`` are current versions. ``records`` also retains the
    eligible ancestors so a serialized snapshot can validate its own lineage.
    """
    return ObservationSnapshot(
        **_select(records, knowledge_cutoff=knowledge_cutoff, world_id=world_id, policy=policy)
    )


def _select(
    records: Any, *, knowledge_cutoff: Any, world_id: Any = "actual", policy: Any = "available_time"
) -> Any:
    """Select available versions. This is archive availability, not local ingestion replay.

    ``records`` retains eligible ancestors; ``selected_records`` contains only the
    resolved replacements. Future economic events remain here for domain reducers.
    """
    cutoff = utc(knowledge_cutoff)
    text(world_id, "world_id")
    if policy not in ("available_time", POLICY):
        raise ObservationError("unknown as-of policy")
    by_id: dict[str, ObservationRecord] = {}
    by_key: dict[tuple, ObservationRecord] = {}
    for record in records:
        if not isinstance(record, ObservationRecord):
            raise ObservationError("expected observation records")
        if record.stamp.world_id != world_id:
            continue
        stamp = record.stamp
        key = (
            (stamp.source, stamp.source_record_id, stamp.version, stamp.world_id)
            if stamp.source_record_id
            else ("record", record.record_id)
        )
        content = record_to_dict(record)
        for previous in (by_id.get(record.record_id), by_key.get(key)):
            if previous is not None and record_to_dict(previous) != content:
                raise ObservationConflictError("conflicting immutable observation identity")
        by_id[record.record_id] = record
        by_key[key] = record
    children: dict[str, str] = {}
    for record in by_id.values():
        parent_id = record.stamp.revision_of
        if parent_id is None:
            continue
        parent = by_id.get(parent_id)
        if parent is None:
            raise RevisionError("missing revision ancestor")
        if parent_id in children and children[parent_id] != record.record_id:
            raise RevisionError("forked revision chain")
        children[parent_id] = record.record_id
        if (parent.kind, parent.entities, parent.stamp.source, parent.stamp.world_id) != (
            record.kind,
            record.entities,
            record.stamp.source,
            record.stamp.world_id,
        ):
            raise RevisionError("revision changes economic identity or source/world")
        if record.payload_schema != parent.payload_schema:
            raise RevisionError("revision codec changes require an explicit merge policy")
        if record.stamp.source_record_id != parent.stamp.source_record_id:
            raise RevisionError("revision changes source identity")
        if (
            record.stamp.version <= parent.stamp.version
            or record.stamp.available_time < parent.stamp.available_time
        ):
            raise RevisionError("revision version/availability must follow its ancestor")
        seen = {record.record_id}
        node: ObservationRecord | None = parent
        while node is not None:
            if node.record_id in seen:
                raise RevisionError("cyclic revision chain")
            seen.add(node.record_id)
            ancestor_id = node.stamp.revision_of
            node = by_id.get(ancestor_id) if ancestor_id is not None else None
    eligible = {k: r for k, r in by_id.items() if r.stamp.available_time <= cutoff}
    ordered = tuple(eligible[k] for k in sorted(eligible))
    selected = tuple(r for r in ordered if children.get(r.record_id) not in eligible)
    identity = digest(
        {
            "records": [
                json.dumps(record_to_dict(r), sort_keys=True, separators=(",", ":"))
                for r in ordered
            ],
            "cutoff": cutoff,
            "world": world_id,
            "policy": POLICY,
        }
    )
    return dict(
        records=ordered,
        selected_records=selected,
        knowledge_cutoff=cutoff,
        world_id=world_id,
        policy=POLICY,
        digest=identity,
    )
