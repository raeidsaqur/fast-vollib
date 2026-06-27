"""Publication-quality diagnostic figures for the arbitrage harness (design §8).

Matplotlib is an **optional** dependency, gated behind the ``[viz]`` extra so
the numerics core stays dependency-light::

    pip install "fast-vollib[viz]"

Each plotting function returns a :class:`matplotlib.figure.Figure` (and is the
static, paper-ready counterpart of the interactive surfaces the Part II UI
renders).  Importing this subpackage without matplotlib installed raises a
clear, actionable error rather than a bare ``ModuleNotFoundError``.
"""

from __future__ import annotations

from .plots import (
    plot_calendar_map,
    plot_density,
    plot_durrleman_g,
    plot_total_variance_slices,
    plot_trust_map,
    plot_violation_heatmap,
)

__all__ = [
    "plot_total_variance_slices",
    "plot_durrleman_g",
    "plot_density",
    "plot_violation_heatmap",
    "plot_calendar_map",
    "plot_trust_map",
]
