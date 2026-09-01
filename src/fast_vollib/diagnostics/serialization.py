"""The ``diagnostics-v1`` wire contract: a strict, deterministic JSON codec.

The contract is deliberately unforgiving, because a diagnostics artifact is
consumed by systems that cannot ask a follow-up question:

* **Closed.** Every object declares ``additionalProperties: false`` and lists
  every property as required.  An unknown field is an error on the way in and
  impossible on the way out.
* **Deterministic.** Field order is fixed by construction, records sort by
  ``(model, split, sample_id)``, and summaries by ``(model, split)``.  The same
  diagnostics render byte-identically on any host.
* **No non-standard floats.** ``allow_nan=False``: an unavailable number is
  ``null``, never ``NaN`` and never a zero standing in for "not measured".
* **Self-checking.** A summary carries both its additive sums and the values
  derived from them; the decoder recomputes the derived values and requires
  exact equality, so a hand-edited or truncated artifact fails to load.

:func:`diagnostics_schema` renders the Draft 2020-12 schema from the same field
tables the encoder uses, so schema and payload cannot drift apart.
"""

from __future__ import annotations

import json
from typing import Any

from .arbitrage import DUPLICATE_POLICIES, GridSummary, RaggedArbitrageConfig, RaggedArbitrageSums
from .fit import ErrorSums, FitDiagnostics, RegionErrors
from .regions import CLOSED_POLICIES, Box
from .report import (
    SCHEMA_VERSION,
    DiagnosticRecord,
    DiagnosticReport,
    DiagnosticsConfig,
    GridGroupSummary,
    GroupSummary,
    RegionSummary,
    SampleDiagnostics,
)
from .spread import SpreadSums

__all__ = [
    "MAX_METADATA_BYTES",
    "diagnostics_schema",
    "dumps",
    "loads",
    "render_schema",
]

#: Upper bound on the serialized size of the free-form ``metadata`` object.
MAX_METADATA_BYTES = 65536
#: Upper bound on the nesting depth of the free-form ``metadata`` object.
MAX_METADATA_DEPTH = 8

_SCHEMA_ID = "https://raeidsaqur.github.io/fast-vollib/schemas/diagnostics-v1.schema.json"
_QUALITY_STATUSES = ("complete", "partial")

# -- field tables: the single source of truth for encoder and schema ---------
_ERROR_SUM_FIELDS: tuple[tuple[str, str], ...] = (
    ("squared_error_sum", "number"),
    ("absolute_error_sum", "number"),
    ("valid_prediction_count", "integer"),
    ("target_count", "integer"),
    ("invalid_prediction_count", "integer"),
    ("max_absolute_error", "number?"),
)
_ERROR_SUM_DERIVED: tuple[tuple[str, str], ...] = (
    ("rmse", "number?"),
    ("mae", "number?"),
    ("coverage", "number?"),
    ("invalid_rate", "number?"),
)
_SPREAD_FIELDS: tuple[tuple[str, str], ...] = (
    ("eligible_quote_count", "integer"),
    ("priced_quote_count", "integer"),
    ("unpriced_quote_count", "integer"),
    ("midpoint_squared_error_sum", "number"),
    ("outside_count", "integer"),
    ("outside_width_sum", "number"),
)
_SPREAD_DERIVED: tuple[tuple[str, str], ...] = (
    ("price_rmse", "number?"),
    ("outside_percentage", "number?"),
    ("mean_miss_width", "number?"),
)
_RAGGED_FIELDS: tuple[tuple[str, str], ...] = (
    ("butterfly_checked_nodes", "integer"),
    ("butterfly_violating_nodes", "integer"),
    ("butterfly_skipped_smiles", "integer"),
    ("butterfly_min_g", "number?"),
    ("calendar_checked_points", "integer"),
    ("calendar_violating_points", "integer"),
    ("calendar_skipped_pairs", "integer"),
    ("smile_count", "integer"),
    ("node_count", "integer"),
    ("duplicate_group_count", "integer"),
    ("collapsed_duplicate_points", "integer"),
    ("max_duplicate_iv_range", "number?"),
    ("nonpositive_total_variance_points", "integer"),
    ("unusable_points", "integer"),
    ("max_maturity_bucket_span", "number?"),
)
_RAGGED_DERIVED: tuple[tuple[str, str], ...] = (
    ("butterfly_percentage", "number?"),
    ("calendar_percentage", "number?"),
)
_GRID_FIELDS: tuple[tuple[str, str], ...] = (
    ("passed", "boolean"),
    ("sas", "number"),
    ("ndm", "number"),
    ("ndm_mean", "number"),
    ("butterfly_fraction", "number"),
    ("calendar_fraction", "number"),
    ("calendar_depth_max", "number"),
    ("vertical_fraction", "number"),
    ("bound_fraction", "number"),
    ("violation_count", "integer"),
    ("native_violation_counts", "counts"),
    ("interpolation_induced_violation_counts", "counts"),
    ("quoted_coverage", "number"),
    ("native_coverage", "number"),
    ("strike_count", "integer"),
    ("maturity_count", "integer"),
    ("tolerance", "number"),
)
_GRID_GROUP_FIELDS: tuple[tuple[str, str], ...] = (
    ("sample_count", "integer"),
    ("passed_count", "integer"),
    ("sas_max", "number"),
    ("sas_mean", "number"),
    ("ndm_max", "number"),
    ("butterfly_fraction_max", "number"),
    ("calendar_fraction_max", "number"),
    ("vertical_fraction_max", "number"),
    ("bound_fraction_max", "number"),
    ("native_coverage_min", "number"),
)
_BOX_FIELDS: tuple[tuple[str, str], ...] = (
    ("kind", "box_kind"),
    ("name", "string"),
    ("complement_name", "string?"),
    ("k_min", "number?"),
    ("k_max", "number?"),
    ("T_min", "number?"),
    ("T_max", "number?"),
    ("closed", "closed_policy"),
)
_RAGGED_CONFIG_FIELDS: tuple[tuple[str, str], ...] = (
    ("maturity_decimals", "integer"),
    ("butterfly_min_unique_strikes", "integer"),
    ("calendar_min_unique_strikes", "integer"),
    ("calendar_grid_points", "integer"),
    ("butterfly_tolerance", "number"),
    ("calendar_tolerance", "number"),
    ("duplicates", "duplicate_policy"),
)


class DiagnosticsDecodeError(ValueError):
    """Raised when a payload does not satisfy the ``diagnostics-v1`` contract."""


# -- encoding ---------------------------------------------------------------
def _number(value: Any) -> float | None:
    if value is None:
        return None
    coerced = float(value)
    if coerced != coerced or coerced in (float("inf"), float("-inf")):
        raise ValueError(
            "diagnostics-v1 cannot carry NaN or Infinity; an unavailable number is null."
        )
    return coerced


def _integer(value: Any) -> int:
    return int(value)


def _encode_fields(source: Any, fields: tuple[tuple[str, str], ...]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, kind in fields:
        value = getattr(source, name)
        if kind == "integer":
            out[name] = _integer(value)
        elif kind == "boolean":
            out[name] = bool(value)
        elif kind == "counts":
            out[name] = {str(k): int(v) for k, v in sorted(value.items())}
        elif kind in ("string", "box_kind", "closed_policy", "duplicate_policy"):
            out[name] = str(value)
        elif kind == "string?":
            out[name] = None if value is None else str(value)
        else:
            out[name] = _number(value)
    return out


def _encode_box(box: Box) -> dict[str, Any]:
    descriptor = box.describe()
    return {
        name: (
            _number(descriptor[name])
            if kind.startswith("number")
            else (None if descriptor[name] is None else str(descriptor[name]))
        )
        for name, kind in _BOX_FIELDS
    }


def _encode_error_sums(sums: ErrorSums, *, derived: bool) -> dict[str, Any]:
    out = _encode_fields(sums, _ERROR_SUM_FIELDS)
    if derived:
        out.update(_encode_fields(sums, _ERROR_SUM_DERIVED))
    return out


def _encode_region_errors(entry: RegionErrors | RegionSummary, *, derived: bool) -> dict[str, Any]:
    if entry.descriptor is None:
        raise ValueError(
            f"Region {entry.name!r} has no serializable descriptor; diagnostics-v1 "
            "refuses a region it cannot describe."
        )
    return {
        "name": str(entry.name),
        "complement_name": None if entry.complement_name is None else str(entry.complement_name),
        "descriptor": _encode_box(Box.from_describe(entry.descriptor)),
        "inside": _encode_error_sums(entry.inside, derived=derived),
        "outside": (
            None if entry.outside is None else _encode_error_sums(entry.outside, derived=derived)
        ),
    }


def _encode_fit(fit: FitDiagnostics, *, derived: bool) -> dict[str, Any]:
    return {
        "overall": _encode_error_sums(fit.overall, derived=derived),
        "regions": [_encode_region_errors(entry, derived=derived) for entry in fit.regions],
    }


def _encode_spread(sums: SpreadSums | None, *, derived: bool) -> dict[str, Any] | None:
    if sums is None:
        return None
    out = _encode_fields(sums, _SPREAD_FIELDS)
    if derived:
        out.update(_encode_fields(sums, _SPREAD_DERIVED))
    return out


def _encode_ragged(sums: RaggedArbitrageSums | None, *, derived: bool) -> dict[str, Any] | None:
    if sums is None:
        return None
    out = _encode_fields(sums, _RAGGED_FIELDS)
    if derived:
        out.update(_encode_fields(sums, _RAGGED_DERIVED))
    return out


def _encode_grid(grid: GridSummary | None) -> dict[str, Any] | None:
    return None if grid is None else _encode_fields(grid, _GRID_FIELDS)


def _encode_record(record: DiagnosticRecord) -> dict[str, Any]:
    sample = record.diagnostics
    return {
        "model": record.model,
        "split": record.split,
        "sample_id": record.sample_id,
        "fit": _encode_fit(sample.fit, derived=False),
        "spread": _encode_spread(sample.spread, derived=False),
        "ragged": _encode_ragged(sample.ragged, derived=False),
        "grid": _encode_grid(sample.grid),
        "quality_status": sample.quality_status,
    }


def _encode_summary(summary: GroupSummary) -> dict[str, Any]:
    return {
        "model": summary.model,
        "split": summary.split,
        "sample_count": int(summary.sample_count),
        "fit": {
            "overall": _encode_error_sums(summary.fit.overall, derived=True),
            "regions": [_encode_region_errors(entry, derived=True) for entry in summary.regions],
        },
        "spread": _encode_spread(summary.spread, derived=True),
        "ragged": _encode_ragged(summary.ragged, derived=True),
        "ragged_butterfly_sample_mean": _number(summary.ragged_butterfly_sample_mean),
        "ragged_calendar_sample_mean": _number(summary.ragged_calendar_sample_mean),
        "quality_status": summary.quality_status,
        "grid": (
            None if summary.grid is None else _encode_fields(summary.grid, _GRID_GROUP_FIELDS)
        ),
    }


def _validate_metadata(metadata: Any) -> dict[str, Any]:
    def walk(node: Any, depth: int) -> Any:
        if depth > MAX_METADATA_DEPTH:
            raise ValueError(f"metadata nests deeper than {MAX_METADATA_DEPTH} levels.")
        if node is None or isinstance(node, (bool, str)):
            return node
        if isinstance(node, int):
            return int(node)
        if isinstance(node, float):
            return _number(node)
        if isinstance(node, dict):
            out = {}
            for key, value in node.items():
                if not isinstance(key, str):
                    raise TypeError(f"metadata keys must be strings; got {key!r}.")
                out[key] = walk(value, depth + 1)
            return out
        if isinstance(node, (list, tuple)):
            return [walk(item, depth + 1) for item in node]
        raise TypeError(f"metadata may only contain JSON values; got {type(node).__name__}.")

    validated = walk(dict(metadata), 0)
    size = len(json.dumps(validated, allow_nan=False, separators=(",", ":")).encode("utf-8"))
    if size > MAX_METADATA_BYTES:
        raise ValueError(f"metadata is {size} bytes; the limit is {MAX_METADATA_BYTES}.")
    return validated


def _encode_config(config: DiagnosticsConfig) -> dict[str, Any]:
    return {
        "regions": [_encode_box(region) for region in config.regions],
        "ragged": _encode_fields(config.ragged, _RAGGED_CONFIG_FIELDS),
        "metadata": _validate_metadata(config.metadata),
    }


def encode(report: DiagnosticReport) -> dict[str, Any]:
    """The canonical ``diagnostics-v1`` object for ``report``."""
    return {
        "schema": SCHEMA_VERSION,
        "config": _encode_config(report.config),
        "record_count": int(report.record_count or 0),
        "records_included": bool(report.records_included),
        "records": [_encode_record(record) for record in report.records],
        "summaries": [_encode_summary(summary) for summary in report.summaries()],
    }


def dumps(report: DiagnosticReport, *, indent: int | None = None) -> str:
    """Render ``report`` as canonical, newline-terminated ``diagnostics-v1`` JSON."""
    separators = (",", ":") if indent is None else (",", ": ")
    return (
        json.dumps(
            encode(report),
            ensure_ascii=False,
            allow_nan=False,
            indent=indent,
            separators=separators,
            sort_keys=False,
        )
        + "\n"
    )


# -- decoding ---------------------------------------------------------------
def _require_object(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise DiagnosticsDecodeError(f"{path} must be an object; got {type(value).__name__}.")
    return value


def _exact_keys(payload: dict[str, Any], expected: tuple[str, ...], path: str) -> None:
    keys = set(payload)
    wanted = set(expected)
    missing = sorted(wanted - keys)
    extra = sorted(keys - wanted)
    if missing or extra:
        raise DiagnosticsDecodeError(
            f"{path} keys mismatch; missing {missing}, unexpected {extra}."
        )


def _decode_fields(
    payload: dict[str, Any], fields: tuple[tuple[str, str], ...], path: str
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, kind in fields:
        value = payload[name]
        where = f"{path}.{name}"
        if kind == "integer":
            if isinstance(value, bool) or not isinstance(value, int):
                raise DiagnosticsDecodeError(f"{where} must be an integer; got {value!r}.")
            out[name] = value
        elif kind == "boolean":
            if not isinstance(value, bool):
                raise DiagnosticsDecodeError(f"{where} must be a boolean; got {value!r}.")
            out[name] = value
        elif kind == "counts":
            table = _require_object(value, where)
            for key, count in table.items():
                if isinstance(count, bool) or not isinstance(count, int):
                    raise DiagnosticsDecodeError(f"{where}.{key} must be an integer.")
            out[name] = {str(k): int(v) for k, v in sorted(table.items())}
        elif kind == "string":
            if not isinstance(value, str):
                raise DiagnosticsDecodeError(f"{where} must be a string; got {value!r}.")
            out[name] = value
        elif kind == "string?":
            if value is not None and not isinstance(value, str):
                raise DiagnosticsDecodeError(f"{where} must be a string or null; got {value!r}.")
            out[name] = value
        elif kind == "box_kind":
            if value != "box":
                raise DiagnosticsDecodeError(f"{where} must be 'box'; got {value!r}.")
            out[name] = value
        elif kind == "closed_policy":
            if value not in CLOSED_POLICIES:
                raise DiagnosticsDecodeError(f"{where} must be one of {CLOSED_POLICIES}.")
            out[name] = value
        elif kind == "duplicate_policy":
            if value not in DUPLICATE_POLICIES:
                raise DiagnosticsDecodeError(f"{where} must be one of {DUPLICATE_POLICIES}.")
            out[name] = value
        elif kind == "number":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise DiagnosticsDecodeError(f"{where} must be a number; got {value!r}.")
            out[name] = float(value)
        elif kind == "number?":
            if value is None:
                out[name] = None
            elif isinstance(value, bool) or not isinstance(value, (int, float)):
                raise DiagnosticsDecodeError(f"{where} must be a number or null; got {value!r}.")
            else:
                out[name] = float(value)
        else:  # pragma: no cover - defensive
            raise AssertionError(f"unknown field kind {kind!r}")
    return out


def _check_derived(instance: Any, payload: dict[str, Any], fields, path: str) -> None:
    for name, _kind in fields:
        expected = getattr(instance, name)
        found = payload[name]
        if expected is None or found is None:
            if expected is not found:
                raise DiagnosticsDecodeError(
                    f"{path}.{name} is {found!r} but its sums imply {expected!r}."
                )
            continue
        if float(expected) != float(found):
            raise DiagnosticsDecodeError(
                f"{path}.{name} is {found!r} but its sums imply {expected!r}."
            )


def _decode_error_sums(payload: Any, path: str, *, derived: bool) -> ErrorSums:
    obj = _require_object(payload, path)
    names = tuple(name for name, _ in _ERROR_SUM_FIELDS)
    if derived:
        names += tuple(name for name, _ in _ERROR_SUM_DERIVED)
    _exact_keys(obj, names, path)
    values = _decode_fields(obj, _ERROR_SUM_FIELDS, path)
    sums = ErrorSums(**values)
    if derived:
        _check_derived(
            sums, _decode_fields(obj, _ERROR_SUM_DERIVED, path), _ERROR_SUM_DERIVED, path
        )
    return sums


def _decode_box(payload: Any, path: str) -> Box:
    obj = _require_object(payload, path)
    _exact_keys(obj, tuple(name for name, _ in _BOX_FIELDS), path)
    values = _decode_fields(obj, _BOX_FIELDS, path)
    return Box.from_describe(values)


def _decode_region(payload: Any, path: str, *, derived: bool) -> RegionErrors:
    obj = _require_object(payload, path)
    _exact_keys(obj, ("name", "complement_name", "descriptor", "inside", "outside"), path)
    box = _decode_box(obj["descriptor"], f"{path}.descriptor")
    if obj["name"] != box.name or obj["complement_name"] != box.complement_name:
        raise DiagnosticsDecodeError(
            f"{path} names disagree with its descriptor: "
            f"{obj['name']!r}/{obj['complement_name']!r} vs {box.name!r}/{box.complement_name!r}."
        )
    outside = obj["outside"]
    if (outside is None) != (box.complement_name is None):
        raise DiagnosticsDecodeError(
            f"{path} must carry outside sums exactly when a complement is named."
        )
    return RegionErrors(
        name=box.name,
        complement_name=box.complement_name,
        descriptor=box.describe(),
        inside=_decode_error_sums(obj["inside"], f"{path}.inside", derived=derived),
        outside=(
            None
            if outside is None
            else _decode_error_sums(outside, f"{path}.outside", derived=derived)
        ),
    )


def _decode_fit(payload: Any, path: str, *, derived: bool) -> FitDiagnostics:
    obj = _require_object(payload, path)
    _exact_keys(obj, ("overall", "regions"), path)
    regions = obj["regions"]
    if not isinstance(regions, list):
        raise DiagnosticsDecodeError(f"{path}.regions must be an array.")
    return FitDiagnostics(
        overall=_decode_error_sums(obj["overall"], f"{path}.overall", derived=derived),
        regions=tuple(
            _decode_region(entry, f"{path}.regions[{index}]", derived=derived)
            for index, entry in enumerate(regions)
        ),
    )


def _decode_optional(payload: Any, path: str, fields, derived_fields, cls, *, derived: bool):
    if payload is None:
        return None
    obj = _require_object(payload, path)
    names = tuple(name for name, _ in fields)
    if derived:
        names += tuple(name for name, _ in derived_fields)
    _exact_keys(obj, names, path)
    instance = cls(**_decode_fields(obj, fields, path))
    if derived:
        _check_derived(instance, _decode_fields(obj, derived_fields, path), derived_fields, path)
    return instance


def _decode_grid(payload: Any, path: str) -> GridSummary | None:
    if payload is None:
        return None
    obj = _require_object(payload, path)
    _exact_keys(obj, tuple(name for name, _ in _GRID_FIELDS), path)
    return GridSummary(**_decode_fields(obj, _GRID_FIELDS, path))


def _decode_sample_id(value: Any, path: str) -> Any:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise DiagnosticsDecodeError(f"{path} must be a string or integer; got {value!r}.")
    return value


def _decode_record(payload: Any, path: str) -> DiagnosticRecord:
    obj = _require_object(payload, path)
    _exact_keys(
        obj,
        ("model", "split", "sample_id", "fit", "spread", "ragged", "grid", "quality_status"),
        path,
    )
    sample = SampleDiagnostics(
        fit=_decode_fit(obj["fit"], f"{path}.fit", derived=False),
        spread=_decode_optional(
            obj["spread"],
            f"{path}.spread",
            _SPREAD_FIELDS,
            _SPREAD_DERIVED,
            SpreadSums,
            derived=False,
        ),
        ragged=_decode_optional(
            obj["ragged"],
            f"{path}.ragged",
            _RAGGED_FIELDS,
            _RAGGED_DERIVED,
            RaggedArbitrageSums,
            derived=False,
        ),
        grid=_decode_grid(obj["grid"], f"{path}.grid"),
    )
    if obj["quality_status"] not in _QUALITY_STATUSES:
        raise DiagnosticsDecodeError(f"{path}.quality_status must be one of {_QUALITY_STATUSES}.")
    if obj["quality_status"] != sample.quality_status:
        raise DiagnosticsDecodeError(
            f"{path}.quality_status is {obj['quality_status']!r} but its counts imply "
            f"{sample.quality_status!r}."
        )
    for label in ("model", "split"):
        if not isinstance(obj[label], str):
            raise DiagnosticsDecodeError(f"{path}.{label} must be a string.")
    return DiagnosticRecord(
        model=obj["model"],
        split=obj["split"],
        sample_id=_decode_sample_id(obj["sample_id"], f"{path}.sample_id"),
        diagnostics=sample,
    )


def _decode_summary(payload: Any, path: str) -> GroupSummary:
    obj = _require_object(payload, path)
    _exact_keys(
        obj,
        (
            "model",
            "split",
            "sample_count",
            "fit",
            "spread",
            "ragged",
            "ragged_butterfly_sample_mean",
            "ragged_calendar_sample_mean",
            "quality_status",
            "grid",
        ),
        path,
    )
    fit = _decode_fit(obj["fit"], f"{path}.fit", derived=True)
    grid_payload = obj["grid"]
    grid = None
    if grid_payload is not None:
        grid_obj = _require_object(grid_payload, f"{path}.grid")
        _exact_keys(grid_obj, tuple(name for name, _ in _GRID_GROUP_FIELDS), f"{path}.grid")
        grid = GridGroupSummary(**_decode_fields(grid_obj, _GRID_GROUP_FIELDS, f"{path}.grid"))
    means = _decode_fields(
        obj,
        (("ragged_butterfly_sample_mean", "number?"), ("ragged_calendar_sample_mean", "number?")),
        path,
    )
    if obj["quality_status"] not in _QUALITY_STATUSES:
        raise DiagnosticsDecodeError(f"{path}.quality_status must be one of {_QUALITY_STATUSES}.")
    return GroupSummary(
        model=obj["model"],
        split=obj["split"],
        sample_count=obj["sample_count"],
        fit=fit,
        regions=tuple(
            RegionSummary(
                name=entry.name,
                complement_name=entry.complement_name,
                descriptor=entry.descriptor,
                inside=entry.inside,
                outside=entry.outside,
            )
            for entry in fit.regions
        ),
        spread=_decode_optional(
            obj["spread"],
            f"{path}.spread",
            _SPREAD_FIELDS,
            _SPREAD_DERIVED,
            SpreadSums,
            derived=True,
        ),
        ragged=_decode_optional(
            obj["ragged"],
            f"{path}.ragged",
            _RAGGED_FIELDS,
            _RAGGED_DERIVED,
            RaggedArbitrageSums,
            derived=True,
        ),
        ragged_butterfly_sample_mean=means["ragged_butterfly_sample_mean"],
        ragged_calendar_sample_mean=means["ragged_calendar_sample_mean"],
        quality_status=obj["quality_status"],
        grid=grid,
    )


def decode(payload: Any) -> DiagnosticReport:
    """Rebuild a :class:`~fast_vollib.diagnostics.DiagnosticReport` from a payload."""
    obj = _require_object(payload, "$")
    _exact_keys(
        obj, ("schema", "config", "record_count", "records_included", "records", "summaries"), "$"
    )
    if obj["schema"] != SCHEMA_VERSION:
        raise DiagnosticsDecodeError(
            f"Unsupported schema {obj['schema']!r}; this codec reads {SCHEMA_VERSION!r} only."
        )
    config_obj = _require_object(obj["config"], "$.config")
    _exact_keys(config_obj, ("regions", "ragged", "metadata"), "$.config")
    if not isinstance(config_obj["regions"], list):
        raise DiagnosticsDecodeError("$.config.regions must be an array.")
    ragged_obj = _require_object(config_obj["ragged"], "$.config.ragged")
    _exact_keys(ragged_obj, tuple(n for n, _ in _RAGGED_CONFIG_FIELDS), "$.config.ragged")
    config = DiagnosticsConfig(
        regions=tuple(
            _decode_box(entry, f"$.config.regions[{index}]")
            for index, entry in enumerate(config_obj["regions"])
        ),
        ragged=RaggedArbitrageConfig(
            **_decode_fields(ragged_obj, _RAGGED_CONFIG_FIELDS, "$.config.ragged")
        ),
        metadata=_validate_metadata(_require_object(config_obj["metadata"], "$.config.metadata")),
    )
    if isinstance(obj["record_count"], bool) or not isinstance(obj["record_count"], int):
        raise DiagnosticsDecodeError("$.record_count must be an integer.")
    if not isinstance(obj["records_included"], bool):
        raise DiagnosticsDecodeError("$.records_included must be a boolean.")
    for label in ("records", "summaries"):
        if not isinstance(obj[label], list):
            raise DiagnosticsDecodeError(f"$.{label} must be an array.")
    records = tuple(
        _decode_record(entry, f"$.records[{index}]") for index, entry in enumerate(obj["records"])
    )
    summaries = tuple(
        _decode_summary(entry, f"$.summaries[{index}]")
        for index, entry in enumerate(obj["summaries"])
    )
    report = DiagnosticReport(
        config=config,
        records=records,
        record_count=obj["record_count"],
        records_included=obj["records_included"],
    )
    if report.records_included:
        recomputed = report.summaries()
        if recomputed != summaries:
            raise DiagnosticsDecodeError(
                "$.summaries do not match the summaries pooled from $.records."
            )
    else:
        object.__setattr__(report, "_summaries", summaries)
    return report


def loads(text: str) -> DiagnosticReport:
    """Parse canonical ``diagnostics-v1`` JSON text, rejecting non-standard floats."""
    return decode(json.loads(text, parse_constant=_reject_constant))


def _reject_constant(name: str) -> Any:
    raise DiagnosticsDecodeError(f"diagnostics-v1 does not permit the JSON constant {name!r}.")


# -- schema -----------------------------------------------------------------
def _type_schema(kind: str) -> dict[str, Any]:
    if kind == "integer":
        return {"type": "integer"}
    if kind == "boolean":
        return {"type": "boolean"}
    if kind == "number":
        return {"type": "number"}
    if kind == "number?":
        return {"type": ["number", "null"]}
    if kind == "string":
        return {"type": "string", "minLength": 1}
    if kind == "string?":
        return {"type": ["string", "null"], "minLength": 1}
    if kind == "counts":
        return {
            "type": "object",
            "additionalProperties": {"type": "integer", "minimum": 0},
        }
    if kind == "box_kind":
        return {"const": "box"}
    if kind == "closed_policy":
        return {"enum": list(CLOSED_POLICIES)}
    if kind == "duplicate_policy":
        return {"enum": list(DUPLICATE_POLICIES)}
    raise AssertionError(f"unknown field kind {kind!r}")  # pragma: no cover


def _object(properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(properties),
        "properties": properties,
    }


def _fields_object(*tables: tuple[tuple[str, str], ...]) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    for table in tables:
        for name, kind in table:
            properties[name] = _type_schema(kind)
    return _object(properties)


def _ref(name: str) -> dict[str, Any]:
    return {"$ref": f"#/$defs/{name}"}


def _nullable_ref(name: str) -> dict[str, Any]:
    return {"oneOf": [_ref(name), {"type": "null"}]}


def diagnostics_schema() -> dict[str, Any]:
    """The closed Draft 2020-12 schema for ``diagnostics-v1``."""
    label = {"type": ["string", "integer"]}
    defs: dict[str, Any] = {
        "box": _fields_object(_BOX_FIELDS),
        "raggedConfig": _fields_object(_RAGGED_CONFIG_FIELDS),
        "errorSums": _fields_object(_ERROR_SUM_FIELDS),
        "errorSumsWithDerived": _fields_object(_ERROR_SUM_FIELDS, _ERROR_SUM_DERIVED),
        "spreadSums": _fields_object(_SPREAD_FIELDS),
        "spreadSumsWithDerived": _fields_object(_SPREAD_FIELDS, _SPREAD_DERIVED),
        "raggedSums": _fields_object(_RAGGED_FIELDS),
        "raggedSumsWithDerived": _fields_object(_RAGGED_FIELDS, _RAGGED_DERIVED),
        "gridSummary": _fields_object(_GRID_FIELDS),
        "gridGroupSummary": _fields_object(_GRID_GROUP_FIELDS),
        "regionErrors": _object(
            {
                "name": _type_schema("string"),
                "complement_name": _type_schema("string?"),
                "descriptor": _ref("box"),
                "inside": _ref("errorSums"),
                "outside": _nullable_ref("errorSums"),
            }
        ),
        "regionSummary": _object(
            {
                "name": _type_schema("string"),
                "complement_name": _type_schema("string?"),
                "descriptor": _ref("box"),
                "inside": _ref("errorSumsWithDerived"),
                "outside": _nullable_ref("errorSumsWithDerived"),
            }
        ),
        "fit": _object(
            {
                "overall": _ref("errorSums"),
                "regions": {"type": "array", "items": _ref("regionErrors")},
            }
        ),
        "fitWithDerived": _object(
            {
                "overall": _ref("errorSumsWithDerived"),
                "regions": {"type": "array", "items": _ref("regionSummary")},
            }
        ),
        "record": _object(
            {
                "model": _type_schema("string"),
                "split": _type_schema("string"),
                "sample_id": label,
                "fit": _ref("fit"),
                "spread": _nullable_ref("spreadSums"),
                "ragged": _nullable_ref("raggedSums"),
                "grid": _nullable_ref("gridSummary"),
                "quality_status": {"enum": list(_QUALITY_STATUSES)},
            }
        ),
        "summary": _object(
            {
                "model": _type_schema("string"),
                "split": _type_schema("string"),
                "sample_count": {"type": "integer", "minimum": 0},
                "fit": _ref("fitWithDerived"),
                "spread": _nullable_ref("spreadSumsWithDerived"),
                "ragged": _nullable_ref("raggedSumsWithDerived"),
                "ragged_butterfly_sample_mean": _type_schema("number?"),
                "ragged_calendar_sample_mean": _type_schema("number?"),
                "quality_status": {"enum": list(_QUALITY_STATUSES)},
                "grid": _nullable_ref("gridGroupSummary"),
            }
        ),
        "config": _object(
            {
                "regions": {"type": "array", "items": _ref("box")},
                "ragged": _ref("raggedConfig"),
                "metadata": {"type": "object"},
            }
        ),
    }
    schema = _object(
        {
            "schema": {"const": SCHEMA_VERSION},
            "config": _ref("config"),
            "record_count": {"type": "integer", "minimum": 0},
            "records_included": {"type": "boolean"},
            "records": {"type": "array", "items": _ref("record")},
            "summaries": {"type": "array", "items": _ref("summary")},
        }
    )
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": _SCHEMA_ID,
        "title": "fast-vollib diagnostics-v1",
        "description": (
            "Reproducible implied-volatility surface diagnostics: fit error, spread "
            "consistency, sampled-smile arbitrage counts, and rectangular-grid summaries."
        ),
        **schema,
        "$defs": defs,
    }


def render_schema(*, indent: int = 2) -> str:
    """The schema as canonical, newline-terminated JSON."""
    return (
        json.dumps(
            diagnostics_schema(),
            ensure_ascii=False,
            allow_nan=False,
            indent=indent,
            sort_keys=False,
        )
        + "\n"
    )
