"""The optional smile-fit figure: correct when matplotlib is present, lazy when not."""

from __future__ import annotations

import numpy as np
import pytest

from fast_vollib.diagnostics import SurfaceQuotes

matplotlib = pytest.importorskip("matplotlib", reason="matplotlib not installed")
matplotlib.use("Agg")


def _quotes():
    strikes = np.array([-0.2, -0.1, 0.0, 0.1, 0.2] * 2)
    return SurfaceQuotes(
        k=strikes,
        T=np.array([0.25] * 5 + [0.5] * 5),
        iv=np.array([0.24, 0.22, 0.21, 0.22, 0.24, 0.23, 0.21, 0.20, 0.21, 0.23]),
        surface_id="AAA",
    )


def test_smile_fit_draws_the_first_smile_by_default():
    from fast_vollib.diagnostics import plot_smile_fit

    quotes = _quotes()
    figure = plot_smile_fit(quotes, quotes.iv)
    axes = figure.axes[0]
    assert "T = 0.25" in axes.get_title()
    observed = axes.lines[0].get_xdata()
    assert observed.tolist() == [-0.2, -0.1, 0.0, 0.1, 0.2]


def test_a_specific_smile_can_be_selected():
    from fast_vollib.diagnostics import plot_smile_fit

    quotes = _quotes()
    figure = plot_smile_fit(quotes, quotes.iv, surface_id="AAA", maturity=0.5)
    assert "T = 0.5" in figure.axes[0].get_title()


def test_unpredicted_points_are_drawn_separately_not_interpolated_over():
    from fast_vollib.diagnostics import plot_smile_fit

    quotes = _quotes()
    predictions = np.array(quotes.iv, copy=True)
    predictions[2] = np.nan
    figure = plot_smile_fit(quotes, predictions)
    labels = [line.get_label() for line in figure.axes[0].lines]
    assert labels == ["observed", "predicted", "unpredicted"]
    assert figure.axes[0].lines[1].get_xdata().size == 4


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"surface_id": "ZZZ"}, "No surface"),
        ({"maturity": 9.0}, "No maturity"),
    ],
)
def test_selection_errors_are_explicit(kwargs, message):
    from fast_vollib.diagnostics import plot_smile_fit

    quotes = _quotes()
    with pytest.raises(ValueError, match=message):
        plot_smile_fit(quotes, quotes.iv, **kwargs)


def test_shape_mismatch_is_rejected():
    from fast_vollib.diagnostics import plot_smile_fit

    quotes = _quotes()
    with pytest.raises(ValueError, match="pred_iv must have shape"):
        plot_smile_fit(quotes, np.array([0.2, 0.2]))


def test_an_empty_quote_set_cannot_be_plotted():
    from fast_vollib.diagnostics import plot_smile_fit

    with pytest.raises(ValueError, match="empty quote set"):
        plot_smile_fit(SurfaceQuotes(k=[], T=[], iv=[]), np.array([]))
