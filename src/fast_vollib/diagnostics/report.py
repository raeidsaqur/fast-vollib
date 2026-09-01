"""Diagnostic records, their configuration, and deterministic aggregation.

One :class:`DiagnosticRecord` is the diagnostic for one ``(model, split,
sample_id)`` triple: fit error, optional spread consistency, optional sampled
arbitrage counts, and -- only when a genuine rectangular grid was evaluated --
a grid block.  Records carry **sums and counts**, never derived means, so a
:class:`GroupSummary` divides exactly once and is independent of how the
records were partitioned.

Grouping is deterministic: records sort by ``(model, split, sample_id)`` and
summaries by ``(model, split)``, with a type-stable key so integer and string
identifiers never compare against each other.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any

import numpy as np

from .arbitrage import GridSummary, RaggedArbitrageConfig, RaggedArbitrageSums
from .fit import ErrorSums, FitDiagnostics
from .regions import DEFAULT_REGIONS, Box
from .spread import SpreadSums

__all__ = [
    "SCHEMA_VERSION",
    "DiagnosticRecord",
    "DiagnosticReport",
    "DiagnosticsConfig",
    "GridGroupSummary",
    "GroupSummary",
    "RegionSummary",
    "SampleDiagnostics",
    "sort_key",
]

#: The wire-contract version this module produces and consumes.
SCHEMA_VERSION = "diagnostics-v1"


def sort_key(value: Any) -> tuple[int, Any]:
    """A total order over JSON scalar labels that never compares int against str."""
    if isinstance(value, bool):
        raise TypeError("Boolean labels are not valid identifiers.")
    if isinstance(value, int):
        return (0, value)
    return (1, str(value))


@dataclass(frozen=True, slots=True)
class SampleDiagnostics:
    """Everything measured for one evaluated sample."""

    fit: FitDiagnostics
    spread: SpreadSums | None = None
    ragged: RaggedArbitrageSums | None = None
    grid: GridSummary | None = None

    @property
    def quality_status(self) -> str:
        """``"partial"`` when any target went unpredicted, else ``"complete"``."""
        return "partial" if self.fit.overall.invalid_prediction_count else "complete"


@dataclass(frozen=True, slots=True)
class DiagnosticRecord:
    """The diagnostic for one ``(model, split, sample_id)`` triple."""

    model: str
    split: str
    sample_id: Any
    diagnostics: SampleDiagnostics

    def __post_init__(self) -> None:
        for label in ("model", "split"):
            value = getattr(self, label)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{label} must be a non-empty string; got {value!r}.")
        if isinstance(self.sample_id, bool) or not isinstance(self.sample_id, (int, str)):
            raise TypeError(
                f"sample_id must be a non-null string or integer; got {self.sample_id!r}."
            )

    @property
    def group(self) -> tuple[str, str]:
        return (self.model, self.split)

    @property
    def order_key(self) -> tuple[str, str, tuple[int, Any]]:
        return (self.model, self.split, sort_key(self.sample_id))


@dataclass(frozen=True, slots=True)
class DiagnosticsConfig:
    """The evaluation settings a report was produced under."""

    regions: tuple[Box, ...] = DEFAULT_REGIONS
    ragged: RaggedArbitrageConfig = field(default_factory=RaggedArbitrageConfig)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for region in self.regions:
            if region.describe() is None:
                raise ValueError(
                    f"Region {region.name!r} has no serializable descriptor; a report "
                    "cannot describe a predicate it cannot reproduce."
                )
        object.__setattr__(self, "regions", tuple(self.regions))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class RegionSummary:
    """Pooled fit metrics inside a named region and, when named, outside it."""

    name: str
    complement_name: str | None
    descriptor: dict[str, Any] | None
    inside: ErrorSums
    outside: ErrorSums | None


@dataclass(frozen=True, slots=True)
class GridGroupSummary:
    """Aggregate of the rectangular-grid blocks in a group.

    Grid metrics are fractions and composites, not additive sums, so they are
    aggregated as extrema and a mean over the samples that carried a grid --
    never pooled as if they were counts.
    """

    sample_count: int
    passed_count: int
    sas_max: float
    sas_mean: float
    ndm_max: float
    butterfly_fraction_max: float
    calendar_fraction_max: float
    vertical_fraction_max: float
    bound_fraction_max: float
    native_coverage_min: float


@dataclass(frozen=True, slots=True)
class GroupSummary:
    """Pooled diagnostics for one ``(model, split)`` group."""

    model: str
    split: str
    sample_count: int
    fit: FitDiagnostics
    regions: tuple[RegionSummary, ...]
    spread: SpreadSums | None
    ragged: RaggedArbitrageSums | None
    ragged_butterfly_sample_mean: float | None
    ragged_calendar_sample_mean: float | None
    quality_status: str
    grid: GridGroupSummary | None

    @property
    def coverage(self) -> float | None:
        return self.fit.overall.coverage


def _merge_optional(left: Any, right: Any, label: str) -> Any:
    if left is None:
        return right
    if right is None:
        raise ValueError(
            f"Cannot pool a group where some samples carry a {label} block and others do not."
        )
    return left.merge(right)


def _summarize_group(model: str, split: str, records: Sequence[DiagnosticRecord]) -> GroupSummary:
    fit = records[0].diagnostics.fit
    for record in records[1:]:
        fit = fit.merge(record.diagnostics.fit)

    spread: SpreadSums | None = records[0].diagnostics.spread
    ragged: RaggedArbitrageSums | None = records[0].diagnostics.ragged
    for record in records[1:]:
        spread = _merge_optional(spread, record.diagnostics.spread, "spread")
        ragged = _merge_optional(ragged, record.diagnostics.ragged, "sampled arbitrage")

    butterfly_mean = calendar_mean = None
    if ragged is not None:
        # The legacy projection: a record with zero checks contributes 0.0, which is
        # what the historical CSV recorded.  The pooled percentages on `ragged` stay
        # null in that case; only this compatibility column substitutes a zero.
        butterfly_mean = float(
            np.mean(
                [
                    _legacy_percentage(record.diagnostics.ragged, "butterfly_percentage")
                    for record in records
                ]
            )
        )
        calendar_mean = float(
            np.mean(
                [
                    _legacy_percentage(record.diagnostics.ragged, "calendar_percentage")
                    for record in records
                ]
            )
        )

    grids = [r.diagnostics.grid for r in records if r.diagnostics.grid is not None]
    grid_summary = _summarize_grids(grids) if grids else None

    return GroupSummary(
        model=model,
        split=split,
        sample_count=len(records),
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
        spread=spread,
        ragged=ragged,
        ragged_butterfly_sample_mean=butterfly_mean,
        ragged_calendar_sample_mean=calendar_mean,
        quality_status=("partial" if fit.overall.invalid_prediction_count else "complete"),
        grid=grid_summary,
    )


def _legacy_percentage(sums: RaggedArbitrageSums | None, attribute: str) -> float:
    if sums is None:
        return 0.0
    value = getattr(sums, attribute)
    return 0.0 if value is None else float(value)


def _summarize_grids(grids: Sequence[GridSummary]) -> GridGroupSummary:
    return GridGroupSummary(
        sample_count=len(grids),
        passed_count=int(sum(1 for grid in grids if grid.passed)),
        sas_max=float(max(grid.sas for grid in grids)),
        sas_mean=float(np.mean([grid.sas for grid in grids])),
        ndm_max=float(max(grid.ndm for grid in grids)),
        butterfly_fraction_max=float(max(grid.butterfly_fraction for grid in grids)),
        calendar_fraction_max=float(max(grid.calendar_fraction for grid in grids)),
        vertical_fraction_max=float(max(grid.vertical_fraction for grid in grids)),
        bound_fraction_max=float(max(grid.bound_fraction for grid in grids)),
        native_coverage_min=float(min(grid.native_coverage for grid in grids)),
    )


@dataclass(frozen=True, slots=True)
class DiagnosticReport:
    """A configuration, a set of records, and the summaries they pool into.

    ``record_count`` is the number of records the report was *built* from and
    survives :meth:`summary_only`, so a summary-only artifact still states how
    many samples stand behind it while declaring ``records_included=False``.
    """

    config: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)
    records: tuple[DiagnosticRecord, ...] = ()
    record_count: int | None = None
    records_included: bool = True
    _summaries: tuple[GroupSummary, ...] = field(default=(), repr=False, compare=False)

    def __post_init__(self) -> None:
        ordered = tuple(sorted(self.records, key=lambda record: record.order_key))
        object.__setattr__(self, "records", ordered)
        if self.record_count is None:
            object.__setattr__(self, "record_count", len(ordered))
        elif self.records_included and self.record_count != len(ordered):
            raise ValueError(
                f"record_count {self.record_count} does not match the "
                f"{len(ordered)} records carried."
            )
        elif not self.records_included and ordered:
            raise ValueError("records_included=False requires an empty record tuple.")

    @classmethod
    def from_records(
        cls,
        records: Iterable[DiagnosticRecord],
        *,
        config: DiagnosticsConfig | None = None,
    ) -> DiagnosticReport:
        """Build a report and pool its summaries."""
        materialized = tuple(records)
        report = cls(config=config or DiagnosticsConfig(), records=materialized)
        object.__setattr__(report, "_summaries", _pool(report.records))
        return report

    def summaries(self) -> tuple[GroupSummary, ...]:
        """Pooled summaries, one per ``(model, split)``, in deterministic order."""
        if self._summaries:
            return self._summaries
        pooled = _pool(self.records)
        object.__setattr__(self, "_summaries", pooled)
        return pooled

    def summary_only(self) -> DiagnosticReport:
        """The same report with its records dropped and the original count kept."""
        summaries = self.summaries()
        projected = replace(
            self, records=(), record_count=self.record_count, records_included=False
        )
        object.__setattr__(projected, "_summaries", summaries)
        return projected


def _pool(records: Sequence[DiagnosticRecord]) -> tuple[GroupSummary, ...]:
    groups: dict[tuple[str, str], list[DiagnosticRecord]] = {}
    for record in records:
        groups.setdefault(record.group, []).append(record)
    return tuple(
        _summarize_group(model, split, groups[(model, split)]) for model, split in sorted(groups)
    )
