"""Reproducible diagnostics for implied-volatility surface fits.

A host-side, dependency-light layer that turns a model's predicted implied
volatilities into numbers a reader can act on:

``SurfaceQuotes`` / ``align_predictions``
    The validated, immutable long-format quote container, and explicit
    alignment by stable ids or exact unique coordinates -- never by row order
    or float tolerance.
``fit_error`` / ``fit_error_by_region``
    Pooled squared/absolute error sums with target, valid, and invalid counts
    kept separate, so coverage is always visible behind an RMSE.
``spread_consistency``
    Does the predicted implied volatility price inside the quoted bid/ask?
``sampled_arbitrage``
    Butterfly and calendar checks on the points a model actually predicted --
    a diagnostic on sampled smiles, never a continuous-surface certificate.
``quotes_to_surface`` / ``diagnose_surface``
    The strict rectangular-grid path, where SAS / NDM / component fractions
    are meaningful.
``diagnose_fit``
    All of the above for one sample, from one classification of the predictions.
``DiagnosticReport`` and ``serialization``
    Records, deterministic pooling, and the closed ``diagnostics-v1`` JSON
    contract.

Plotting lives in :mod:`fast_vollib.diagnostics.plots` and needs the optional
``[viz]`` extra.  It is **not** imported when this package is imported: the
figure helpers are resolved lazily on first attribute access, so importing
diagnostics on a headless numerics host pulls in no matplotlib.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .arbitrage import (
    DUPLICATE_POLICIES,
    GridSummary,
    RaggedArbitrageConfig,
    RaggedArbitrageSums,
    diagnose_surface,
    grid_arbitrage_summary,
    quotes_to_surface,
    sampled_arbitrage,
)
from .evaluate import diagnose_fit
from .fit import (
    ErrorSums,
    FitDiagnostics,
    PredictionClassification,
    RegionErrors,
    classify_predictions,
    fit_error,
    fit_error_by_region,
)
from .quotes import DEFAULT_COLUMNS, SurfaceQuotes, align_predictions
from .regions import DEFAULT_REGIONS, LIQUID_BOX, Box, NamedRegion, Region, liquid_mask
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
from .spread import SpreadSums, normalized_option_price, spread_consistency

if TYPE_CHECKING:  # pragma: no cover - typing only; no runtime matplotlib import
    from .plots import (
        plot_calendar_map,
        plot_density,
        plot_durrleman_g,
        plot_total_variance_slices,
        plot_trust_map,
        plot_violation_heatmap,
    )

#: Figure helpers resolved lazily from :mod:`fast_vollib.diagnostics.plots`.
_LAZY_PLOTS = frozenset(
    {
        "plot_calendar_map",
        "plot_density",
        "plot_durrleman_g",
        "plot_smile_fit",
        "plot_total_variance_slices",
        "plot_trust_map",
        "plot_violation_heatmap",
    }
)

__all__ = [
    "DEFAULT_COLUMNS",
    "DEFAULT_REGIONS",
    "DUPLICATE_POLICIES",
    "LIQUID_BOX",
    "SCHEMA_VERSION",
    "Box",
    "DiagnosticRecord",
    "DiagnosticReport",
    "DiagnosticsConfig",
    "ErrorSums",
    "FitDiagnostics",
    "GridGroupSummary",
    "GridSummary",
    "GroupSummary",
    "NamedRegion",
    "PredictionClassification",
    "RaggedArbitrageConfig",
    "RaggedArbitrageSums",
    "Region",
    "RegionErrors",
    "RegionSummary",
    "SampleDiagnostics",
    "SpreadSums",
    "SurfaceQuotes",
    "align_predictions",
    "classify_predictions",
    "diagnose_fit",
    "diagnose_surface",
    "fit_error",
    "fit_error_by_region",
    "grid_arbitrage_summary",
    "liquid_mask",
    "normalized_option_price",
    "plot_calendar_map",
    "plot_density",
    "plot_durrleman_g",
    "plot_total_variance_slices",
    "plot_trust_map",
    "plot_violation_heatmap",
    "quotes_to_surface",
    "sampled_arbitrage",
    "spread_consistency",
]


def __getattr__(name: str) -> Any:
    """Resolve the optional plotting helpers on first access."""
    if name in _LAZY_PLOTS:
        from . import plots

        try:
            return getattr(plots, name)
        except AttributeError as error:  # pragma: no cover - defensive
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from error
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(globals()))
