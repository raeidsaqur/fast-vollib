"""Arbitrage diagnostics for scattered smiles and for rectangular grids.

These are two different claims and this module keeps them apart.

**Sampled (ragged) checks** operate on whatever ``(k, T)`` points a pipeline
happened to predict.  They group rows into ``(surface, maturity)`` smiles,
check Durrleman's ``g`` across each smile's interior nodes and total-variance
monotonicity between adjacent maturities, and report how many nodes were
actually checked.  They are a *diagnostic on the sampled points*, never a
certificate about the continuous surface between them: nothing here
interpolates a scattered cloud onto a mesh.

**Rectangular-grid checks** require a genuine mesh.  :func:`quotes_to_surface`
accepts only a complete, duplicate-free Cartesian product of unique
coordinates -- it will not invent the cells a ragged cloud is missing -- and
:func:`diagnose_surface` then runs the full
:func:`~fast_vollib.surface.validate_surface` harness, whose SAS / NDM /
component fractions are meaningful only on such a mesh.

Duplicate observations
----------------------
The same ``(surface, maturity, k)`` node can legitimately be observed more than
once.  Those rows stay separate for fit and spread statistics, but a smile's
*geometry* has one node per strike, so exact duplicates are collapsed here by
**mean total variance**, ``w = mean(iv_i^2 * T_bucket)`` -- equivalent to the
RMS implied volatility at fixed maturity, which is the natural reduction for
diagnostics defined on total variance.  The number of duplicate groups, the
number of points removed, and the widest within-group IV range are all
reported, so the collapse is visible rather than silent.  Set
``duplicates="error"`` to refuse duplicates instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from .._array_api import numpy_namespace
from ..surface.density import durrleman_g

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..surface.grid import IVSurface
    from .fit import PredictionClassification
    from .quotes import SurfaceQuotes

__all__ = [
    "DUPLICATE_POLICIES",
    "GridSummary",
    "RaggedArbitrageConfig",
    "RaggedArbitrageSums",
    "diagnose_surface",
    "grid_arbitrage_summary",
    "quotes_to_surface",
    "sampled_arbitrage",
]

#: How exactly-duplicated smile nodes are handled.
DUPLICATE_POLICIES: tuple[str, ...] = ("mean_total_variance", "error")

#: Minimum width of a maturity pair's log-moneyness overlap before it is checked.
_MIN_OVERLAP = 1e-9


@dataclass(frozen=True, slots=True)
class RaggedArbitrageConfig:
    """Parameters of the sampled-smile arbitrage checks.

    Parameters
    ----------
    maturity_decimals:
        Decimal places ``T`` is rounded to when bucketing rows into smiles.
        Bucketing is by exact equality *after* rounding; strikes are never
        grouped by tolerance.
    butterfly_min_unique_strikes:
        Minimum usable nodes on a smile before Durrleman's ``g`` is evaluated;
        the interior stencil needs three.
    calendar_min_unique_strikes:
        Minimum usable nodes before a maturity bucket takes part in a calendar
        comparison.
    calendar_grid_points:
        Points on the linear-interpolation grid spanning a maturity pair's
        log-moneyness overlap.
    butterfly_tolerance:
        A node violates iff ``g < -butterfly_tolerance``.
    calendar_tolerance:
        A grid point violates iff ``w(T_i) - w(T_{i+1}) > calendar_tolerance``.
    duplicates:
        ``"mean_total_variance"`` collapses exactly-duplicated nodes;
        ``"error"`` refuses them.
    """

    maturity_decimals: int = 6
    butterfly_min_unique_strikes: int = 3
    calendar_min_unique_strikes: int = 2
    calendar_grid_points: int = 40
    butterfly_tolerance: float = 0.0
    calendar_tolerance: float = 1e-10
    duplicates: str = "mean_total_variance"

    def __post_init__(self) -> None:
        if self.duplicates not in DUPLICATE_POLICIES:
            raise ValueError(
                f"duplicates must be one of {DUPLICATE_POLICIES}; got {self.duplicates!r}."
            )
        if not isinstance(self.maturity_decimals, int) or isinstance(self.maturity_decimals, bool):
            raise TypeError("maturity_decimals must be an integer.")
        if self.maturity_decimals < 0:
            raise ValueError("maturity_decimals must be non-negative.")
        if self.butterfly_min_unique_strikes < 3:
            raise ValueError("butterfly_min_unique_strikes must be at least 3.")
        if self.calendar_min_unique_strikes < 2:
            raise ValueError("calendar_min_unique_strikes must be at least 2.")
        if self.calendar_grid_points < 2:
            raise ValueError("calendar_grid_points must be at least 2.")
        for label in ("butterfly_tolerance", "calendar_tolerance"):
            value = float(getattr(self, label))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{label} must be finite and non-negative.")

    def describe(self) -> dict[str, Any]:
        """The JSON-serializable descriptor stored in a diagnostics report."""
        return {
            "maturity_decimals": int(self.maturity_decimals),
            "butterfly_min_unique_strikes": int(self.butterfly_min_unique_strikes),
            "calendar_min_unique_strikes": int(self.calendar_min_unique_strikes),
            "calendar_grid_points": int(self.calendar_grid_points),
            "butterfly_tolerance": float(self.butterfly_tolerance),
            "calendar_tolerance": float(self.calendar_tolerance),
            "duplicates": self.duplicates,
        }


@dataclass(frozen=True, slots=True)
class RaggedArbitrageSums:
    """Additive sampled-arbitrage counts for one group of predicted rows."""

    butterfly_checked_nodes: int = 0
    butterfly_violating_nodes: int = 0
    butterfly_skipped_smiles: int = 0
    butterfly_min_g: float | None = None
    calendar_checked_points: int = 0
    calendar_violating_points: int = 0
    calendar_skipped_pairs: int = 0
    smile_count: int = 0
    node_count: int = 0
    duplicate_group_count: int = 0
    collapsed_duplicate_points: int = 0
    max_duplicate_iv_range: float | None = None
    nonpositive_total_variance_points: int = 0
    unusable_points: int = 0
    max_maturity_bucket_span: float | None = None

    def merge(self, other: RaggedArbitrageSums) -> RaggedArbitrageSums:
        """Pooled counts of two disjoint groups.  Associative and commutative."""
        return RaggedArbitrageSums(
            butterfly_checked_nodes=self.butterfly_checked_nodes + other.butterfly_checked_nodes,
            butterfly_violating_nodes=self.butterfly_violating_nodes
            + other.butterfly_violating_nodes,
            butterfly_skipped_smiles=self.butterfly_skipped_smiles + other.butterfly_skipped_smiles,
            butterfly_min_g=_min_optional(self.butterfly_min_g, other.butterfly_min_g),
            calendar_checked_points=self.calendar_checked_points + other.calendar_checked_points,
            calendar_violating_points=self.calendar_violating_points
            + other.calendar_violating_points,
            calendar_skipped_pairs=self.calendar_skipped_pairs + other.calendar_skipped_pairs,
            smile_count=self.smile_count + other.smile_count,
            node_count=self.node_count + other.node_count,
            duplicate_group_count=self.duplicate_group_count + other.duplicate_group_count,
            collapsed_duplicate_points=self.collapsed_duplicate_points
            + other.collapsed_duplicate_points,
            max_duplicate_iv_range=_max_optional(
                self.max_duplicate_iv_range, other.max_duplicate_iv_range
            ),
            nonpositive_total_variance_points=self.nonpositive_total_variance_points
            + other.nonpositive_total_variance_points,
            unusable_points=self.unusable_points + other.unusable_points,
            max_maturity_bucket_span=_max_optional(
                self.max_maturity_bucket_span, other.max_maturity_bucket_span
            ),
        )

    @property
    def butterfly_percentage(self) -> float | None:
        """Percentage of checked nodes violating, or ``None`` when nothing was checked."""
        if self.butterfly_checked_nodes == 0:
            return None
        return float(100.0 * self.butterfly_violating_nodes / self.butterfly_checked_nodes)

    @property
    def calendar_percentage(self) -> float | None:
        """Percentage of checked points violating, or ``None`` when nothing was checked."""
        if self.calendar_checked_points == 0:
            return None
        return float(100.0 * self.calendar_violating_points / self.calendar_checked_points)


def _min_optional(left: float | None, right: float | None) -> float | None:
    if left is None:
        return right
    if right is None:
        return left
    return min(left, right)


def _max_optional(left: float | None, right: float | None) -> float | None:
    if left is None:
        return right
    if right is None:
        return left
    return max(left, right)


@dataclass(slots=True)
class _Bucket:
    """One ``(surface, rounded maturity)`` smile reduced to unique-strike geometry."""

    maturity: float
    strikes: np.ndarray
    total_variance: np.ndarray
    raw_span: float
    duplicate_groups: int = 0
    collapsed_points: int = 0
    max_iv_range: float | None = None
    nonpositive_points: int = 0


def _build_buckets(
    surface_id: np.ndarray,
    k: np.ndarray,
    T: np.ndarray,
    iv: np.ndarray,
    config: RaggedArbitrageConfig,
) -> dict[Any, list[_Bucket]]:
    """Group usable rows into per-surface, maturity-ordered smile geometries."""
    rounded = np.round(T, config.maturity_decimals)
    grouped: dict[Any, list[_Bucket]] = {}
    for label in dict.fromkeys(surface_id.tolist()):
        in_surface = surface_id == label
        buckets: list[_Bucket] = []
        for maturity in np.unique(rounded[in_surface]):
            rows = in_surface & (rounded == maturity)
            k_rows = k[rows]
            iv_rows = iv[rows]
            T_rows = T[rows]
            order = np.argsort(k_rows, kind="stable")
            k_sorted = k_rows[order]
            iv_sorted = iv_rows[order]
            unique_k, starts, counts = np.unique(k_sorted, return_index=True, return_counts=True)
            duplicate_groups = int(np.count_nonzero(counts > 1))
            if duplicate_groups and config.duplicates == "error":
                raise ValueError(
                    f"Surface {label!r} maturity bucket {float(maturity)!r} has "
                    f"{duplicate_groups} duplicated strike(s); "
                    "pass duplicates='mean_total_variance' to collapse them."
                )
            maturity_value = float(maturity)
            mean_squared = np.empty(unique_k.size, dtype=np.float64)
            widest: float | None = None
            for index, (start, count) in enumerate(zip(starts.tolist(), counts.tolist())):
                block = iv_sorted[start : start + count]
                mean_squared[index] = float(np.mean(block * block))
                if count > 1:
                    span = float(block.max() - block.min())
                    widest = span if widest is None else max(widest, span)
            total_variance = mean_squared * maturity_value
            positive = total_variance > 0.0
            buckets.append(
                _Bucket(
                    maturity=maturity_value,
                    strikes=unique_k[positive],
                    total_variance=total_variance[positive],
                    raw_span=float(T_rows.max() - T_rows.min()) if T_rows.size else 0.0,
                    duplicate_groups=duplicate_groups,
                    collapsed_points=int(k_sorted.size - unique_k.size),
                    max_iv_range=widest,
                    nonpositive_points=int(np.count_nonzero(~positive)),
                )
            )
        grouped[label] = buckets
    return grouped


def sampled_arbitrage(
    pred_iv: Any,
    quotes: "SurfaceQuotes",
    *,
    config: RaggedArbitrageConfig | None = None,
    classification: "PredictionClassification | None" = None,
) -> RaggedArbitrageSums:
    """Butterfly and calendar checks on the sampled points a model predicted.

    Only rows the model actually covered are used -- targets whose prediction
    is finite and non-negative -- so the fit, spread, and arbitrage blocks all
    describe the same rows.  Targets the model could not price are counted in
    ``unusable_points``; usable nodes whose total variance is non-positive are
    excluded from the derivative checks and counted in
    ``nonpositive_total_variance_points`` (Durrleman's ``g`` is singular there,
    and no hidden epsilon is substituted).

    Returns
    -------
    :class:`RaggedArbitrageSums` -- raw counts, not percentages.  With zero
    checks the derived percentages are ``None``, never ``0.0``.
    """
    from .fit import classify_predictions

    config = config or RaggedArbitrageConfig()
    pred = np.asarray(pred_iv, dtype=np.float64)
    if pred.shape != (quotes.n,):
        raise ValueError(f"pred_iv must have shape ({quotes.n},); got {pred.shape}.")
    if classification is None:
        classification = classify_predictions(pred, quotes.iv)
    usable = classification.valid
    unusable = classification.invalid_count
    if not bool(usable.any()):
        return RaggedArbitrageSums(unusable_points=unusable)

    grouped = _build_buckets(
        np.asarray(quotes.surface_id)[usable],
        np.asarray(quotes.k, dtype=np.float64)[usable],
        np.asarray(quotes.T, dtype=np.float64)[usable],
        pred[usable],
        config,
    )

    xp = numpy_namespace()
    totals = RaggedArbitrageSums(unusable_points=unusable)
    for buckets in grouped.values():
        available: list[_Bucket] = []
        for bucket in buckets:
            node_count = int(bucket.strikes.size)
            butterfly_checked = butterfly_violating = 0
            skipped_smile = 0
            min_g: float | None = None
            if node_count >= config.butterfly_min_unique_strikes:
                g = np.asarray(
                    durrleman_g(bucket.strikes, bucket.total_variance[:, None], xp),
                    dtype=np.float64,
                ).ravel()
                finite = np.isfinite(g)
                butterfly_checked = int(np.count_nonzero(finite))
                if butterfly_checked:
                    butterfly_violating = int(
                        np.count_nonzero(g[finite] < -config.butterfly_tolerance)
                    )
                    min_g = float(g[finite].min())
                else:
                    skipped_smile = 1
            else:
                skipped_smile = 1
            totals = totals.merge(
                RaggedArbitrageSums(
                    butterfly_checked_nodes=butterfly_checked,
                    butterfly_violating_nodes=butterfly_violating,
                    butterfly_skipped_smiles=skipped_smile,
                    butterfly_min_g=min_g,
                    smile_count=1,
                    node_count=node_count,
                    duplicate_group_count=bucket.duplicate_groups,
                    collapsed_duplicate_points=bucket.collapsed_points,
                    max_duplicate_iv_range=bucket.max_iv_range,
                    nonpositive_total_variance_points=bucket.nonpositive_points,
                    max_maturity_bucket_span=bucket.raw_span,
                )
            )
            if node_count >= config.calendar_min_unique_strikes:
                available.append(bucket)

        for near, far in zip(available, available[1:]):
            low = max(float(near.strikes.min()), float(far.strikes.min()))
            high = min(float(near.strikes.max()), float(far.strikes.max()))
            if high - low < _MIN_OVERLAP:
                totals = totals.merge(RaggedArbitrageSums(calendar_skipped_pairs=1))
                continue
            grid = np.linspace(low, high, config.calendar_grid_points)
            near_w = np.interp(grid, near.strikes, near.total_variance)
            far_w = np.interp(grid, far.strikes, far.total_variance)
            totals = totals.merge(
                RaggedArbitrageSums(
                    calendar_checked_points=int(grid.size),
                    calendar_violating_points=int(
                        np.count_nonzero(near_w - far_w > config.calendar_tolerance)
                    ),
                )
            )
    return totals


# -- rectangular grid path --------------------------------------------------
@dataclass(frozen=True, slots=True)
class GridSummary:
    """Projection of a rectangular-grid :class:`~fast_vollib.surface.ArbitrageReport`."""

    passed: bool
    sas: float
    ndm: float
    ndm_mean: float
    butterfly_fraction: float
    calendar_fraction: float
    calendar_depth_max: float
    vertical_fraction: float
    bound_fraction: float
    violation_count: int
    native_violation_counts: dict[str, int] = field(default_factory=dict)
    interpolation_induced_violation_counts: dict[str, int] = field(default_factory=dict)
    quoted_coverage: float = 0.0
    native_coverage: float = 0.0
    strike_count: int = 0
    maturity_count: int = 0
    tolerance: float = 0.0


def quotes_to_surface(
    quotes: "SurfaceQuotes",
    *,
    forward: Any = 1.0,
    r: Any = 0.0,
    q: Any = 0.0,
    values: Any = None,
) -> "IVSurface":
    """Build an :class:`~fast_vollib.surface.IVSurface` from a rectangular quote set.

    The rows must form a **complete, duplicate-free Cartesian product** of the
    unique ``k`` and ``T`` values they contain: every ``(k, T)`` cell present
    exactly once.  A ragged cloud, a duplicated coordinate, or a missing cell
    raises -- nothing is interpolated, and no caller may declare a mask that
    would label an invented cell native.  ``native_mask`` is derived here, from
    the rows supplied and which of their values are finite.

    Parameters
    ----------
    quotes:
        Exactly one surface.  Multi-surface input raises; select a surface with
        :meth:`SurfaceQuotes.subset` first.
    forward, r, q:
        Optional grid geometry passed through to the surface.  The quote
        convention is forward log-moneyness, so the defaults describe a
        forward-normalised grid.
    values:
        Optional per-row values to place on the grid instead of ``quotes.iv``
        (a model's predicted surface at the same coordinates).

    Raises
    ------
    ValueError
        On multiple surfaces, duplicate coordinates, or an incomplete product.
    """
    from ..surface.grid import IVSurface

    labels = quotes.surface_ids()
    if len(labels) > 1:
        raise ValueError(
            f"quotes_to_surface accepts exactly one surface; got {len(labels)}: {labels[:5]}."
        )
    iv_rows = quotes.iv if values is None else np.asarray(values, dtype=np.float64)
    if iv_rows.shape != (quotes.n,):
        raise ValueError(f"values must have shape ({quotes.n},); got {iv_rows.shape}.")

    k_values, k_index = np.unique(np.asarray(quotes.k, dtype=np.float64), return_inverse=True)
    T_values, T_index = np.unique(np.asarray(quotes.T, dtype=np.float64), return_inverse=True)
    expected = int(k_values.size) * int(T_values.size)
    if quotes.n != expected:
        raise ValueError(
            f"Rectangular grid requires exactly {expected} rows "
            f"({k_values.size} unique k x {T_values.size} unique T); got {quotes.n}. "
            "A ragged quote set has no rectangular-grid certificate."
        )
    flat = k_index.astype(np.int64) * T_values.size + T_index.astype(np.int64)
    if np.unique(flat).size != expected:
        raise ValueError(
            "Rectangular grid requires each (k, T) cell exactly once; "
            "the quote set has duplicate and/or missing cells."
        )
    grid = np.empty((k_values.size, T_values.size), dtype=np.float64)
    grid.reshape(-1)[flat] = iv_rows
    return IVSurface(
        k=k_values,
        T=T_values,
        iv=grid,
        forward=forward,
        r=r,
        q=q,
        native_mask=np.isfinite(grid),
    )


def grid_arbitrage_summary(report: Any, surface: "IVSurface") -> GridSummary:
    """Project an :class:`~fast_vollib.surface.ArbitrageReport` into a summary block."""
    metrics = report.metrics
    context = report.context
    native = surface.native_mask
    native_coverage = float(np.mean(native)) if native is not None and native.size else 0.0
    return GridSummary(
        passed=bool(report.passed),
        sas=float(report.sas),
        ndm=float(metrics["ndm"]),
        ndm_mean=float(metrics["ndm_mean"]),
        butterfly_fraction=float(metrics["bfly_frac"]),
        calendar_fraction=float(metrics["cal_frac"]),
        calendar_depth_max=float(metrics["cal_depth_max"]),
        vertical_fraction=float(metrics["vert_frac"]),
        bound_fraction=float(metrics["bound_frac"]),
        violation_count=int(report.n_violations),
        native_violation_counts={k: int(v) for k, v in sorted(report.native.items())},
        interpolation_induced_violation_counts={
            k: int(v) for k, v in sorted(report.interpolation_induced.items())
        },
        quoted_coverage=float(context.get("coverage", 0.0)),
        native_coverage=native_coverage,
        strike_count=int(surface.Nk),
        maturity_count=int(surface.Nt),
        tolerance=float(report.tolerance),
    )


def diagnose_surface(
    surface: "IVSurface",
    *,
    reference: "IVSurface | None" = None,
    **validate_kwargs: Any,
) -> GridSummary:
    """Run the rectangular-grid arbitrage harness and project its report.

    Parameters
    ----------
    surface:
        The mesh to certify.
    reference:
        Optional surface whose coordinates ``surface`` must match **exactly**.
        It is a guard, not an interpolation target: it exists so a caller
        cannot claim a grid result aligns with a truth mesh it does not.
    **validate_kwargs:
        Forwarded to :func:`~fast_vollib.surface.validate_surface`.
    """
    from ..surface.metrics import validate_surface

    if reference is not None:
        for name in ("k", "T"):
            left = np.asarray(getattr(surface, name), dtype=np.float64)
            right = np.asarray(getattr(reference, name), dtype=np.float64)
            if left.shape != right.shape or not np.array_equal(left, right):
                raise ValueError(
                    f"surface and reference must share exactly equal {name} coordinates."
                )
    report = validate_surface(surface, **validate_kwargs)
    return grid_arbitrage_summary(report, surface)
