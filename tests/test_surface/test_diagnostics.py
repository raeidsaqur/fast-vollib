"""Smoke tests for the diagnostic figures (require the [viz] extra)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("matplotlib")
import matplotlib

matplotlib.use("Agg")  # headless backend for CI

from matplotlib.figure import Figure  # noqa: E402

from fast_vollib.surface import IVSurface, validate_surface  # noqa: E402


@pytest.fixture
def surf(moneyness, maturities, svi_iv):
    return IVSurface.from_logmoneyness(moneyness, maturities, svi_iv)


@pytest.fixture
def dirty_surf(moneyness, maturities, svi_iv):
    iv = svi_iv.copy()
    iv[moneyness.size // 2, 1] *= 0.5
    iv[:, -1] = 0.05
    return IVSurface.from_logmoneyness(moneyness, maturities, iv)


def test_total_variance_slices(surf):
    from fast_vollib.diagnostics import plot_total_variance_slices

    assert isinstance(plot_total_variance_slices(surf), Figure)


def test_durrleman_g(dirty_surf):
    from fast_vollib.diagnostics import plot_durrleman_g

    assert isinstance(plot_durrleman_g(dirty_surf), Figure)


def test_density(dirty_surf):
    from fast_vollib.diagnostics import plot_density

    assert isinstance(plot_density(dirty_surf, t_index=1), Figure)


def test_violation_heatmap(dirty_surf):
    from fast_vollib.diagnostics import plot_violation_heatmap

    rep = validate_surface(dirty_surf, compute_trust=False)
    assert isinstance(plot_violation_heatmap(dirty_surf, rep), Figure)


def test_calendar_map(dirty_surf):
    from fast_vollib.diagnostics import plot_calendar_map

    assert isinstance(plot_calendar_map(dirty_surf), Figure)


def test_trust_map(surf):
    from fast_vollib.diagnostics import plot_trust_map

    fig = plot_trust_map(surf)
    assert isinstance(fig, Figure)


def test_all_figures_distinct_axes(surf):
    from fast_vollib.diagnostics import plot_total_variance_slices

    fig = plot_total_variance_slices(surf)
    assert len(fig.axes) >= 1
    assert np.isfinite(fig.get_size_inches()).all()
