# Observation records and archive snapshots

`fast_vollib.observations` stores immutable evidence with explicit event,
availability and recording times. It imports no financial payload classes.
All timestamps are UTC-aware; interval events and coverage have ordered endpoints.
Availability is a caller statement. It is never inferred from an event timestamp.
Recording may precede availability when the caller's policy permits that ordering.

An `ObservationRecord` has a stable `record_id`, `kind`, `entities`,
`payload_schema`, `stamp`, optional coverage and a verified `payload_digest`.
Construction copies dictionaries and lists into immutable canonical trees. Accepted
leaves are strings, integers, booleans, finite floats, finite Decimals, UTC instants
and `None`. Arbitrary objects, arrays, callables and reserved `$` mapping keys are
refused. Canonicalization is versioned as `observation.canonical/1`; JSON encodes
Decimals and instants with explicit tags.

`ObservationStamp` includes source, optional source record ID, version, revision
ancestor, world, timing note and mapping version. Synthetic sources require a
non-actual world. Source IDs use `(source, source_record_id, version, world_id)`
for idempotency; otherwise the caller's record ID is the identity. Repeated identical
deliveries deduplicate. Conflicting content under either identity raises
`ObservationConflictError`.

## Available-time selection

```python
from fast_vollib.observations import select_as_of

snapshot = select_as_of(records, knowledge_cutoff=cutoff, world_id="actual")
```

The named policy is `available_time/1` (`available_time` is its input alias).
Selection filters world and availability, then resolves full replacement revisions.
`selected_records` contains the current admissible versions. `records` additionally
retains eligible revision ancestors so serialized snapshots can independently
validate lineage. Both are deterministic tuples owned by the snapshot. The digest
binds records, cutoff, world and policy. A later revision never mutates an older
snapshot.

Revision chains must preserve source, source identity, entities, kind and payload
schema. Versions increase and availability cannot move backwards along the chain.
Missing ancestors, forks, cycles and codec changes raise `RevisionError`. Delta
merges require a separate codec policy and are refused by this replacement route.
A snapshot validates the archive's supplied world before selection; malformed
future chains are refused even when the current cutoff precedes them.

Generic selection retains notices and forecasts about future events. Economic
reducers separately exclude executions after their valuation time. An archive
availability query reconstructs declared knowable information; it does not prove
what a local system had ingested. Local ingestion replay needs recording-time
evidence and a corresponding policy.

## Serialization

`observations.serialization.dumps` / `loads` round-trip individual records.
`record_to_dict` / `record_from_dict` and `snapshot_to_dict` /
`snapshot_from_dict` provide strict versioned data records. Unknown fields,
versions, duplicate JSON keys, invalid tags and digest mismatches are refused.
No executable objects or pickle are accepted.
