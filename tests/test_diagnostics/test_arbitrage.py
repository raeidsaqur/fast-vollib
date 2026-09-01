"""Sampled-smile and rectangular-grid arbitrage checks."""

from __future__ import annotations

import numpy as np
import pytest

from fast_vollib.diagnostics import (
    RaggedArbitrageConfig,
    RaggedArbitrageSums,
    SurfaceQuotes,
    diagnose_surface,
    quotes_to_surface,
    sampled_arbitrage,
)


def _svi_total_variance(k, a=0.04, b=0.4, rho=-0.4, m=0.0, sigma=0.1):
    """Raw SVI: an analytically smooth, butterfly-free slice for these parameters."""
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma**2))


def _svi_smile(maturity: float, strikes: np.ndarray, surface: str = "A") -> SurfaceQuotes:
    iv = np.sqrt(_svi_total_variance(strikes) / maturity)
    return SurfaceQuotes(k=strikes, T=np.full(strikes.size, maturity), iv=iv, surface_id=surface)


# -- butterfly ---------------------------------------------------------------
def test_an_arbitrage_free_svi_slice_reports_no_butterfly_violations():
    strikes = np.linspace(-0.6, 0.6, 41)
    quotes = _svi_smile(1.0, strikes)
    sums = sampled_arbitrage(quotes.iv, quotes)
    assert sums.butterfly_checked_nodes == strikes.size - 2  # interior stencil
    assert sums.butterfly_violating_nodes == 0
    assert sums.butterfly_percentage == 0.0
    assert sums.butterfly_min_g > 0.0


def test_a_deliberately_kinked_smile_is_flagged():
    strikes = np.linspace(-0.3, 0.3, 13)
    total_variance = np.full(13, 0.04)
    total_variance[6] = 0.20  # a spike no density can support
    quotes = SurfaceQuotes(
        k=strikes, T=np.full(13, 1.0), iv=np.sqrt(total_variance), surface_id="A"
    )
    sums = sampled_arbitrage(quotes.iv, quotes)
    assert sums.butterfly_violating_nodes >= 1
    assert sums.butterfly_min_g < 0.0
    assert sums.butterfly_percentage > 0.0


def test_a_smile_below_the_strike_threshold_is_skipped_not_scored():
    quotes = SurfaceQuotes(k=[-0.1, 0.1], T=[1.0, 1.0], iv=[0.2, 0.2], surface_id="A")
    sums = sampled_arbitrage(quotes.iv, quotes)
    assert sums.butterfly_checked_nodes == 0
    assert sums.butterfly_skipped_smiles == 1
    assert sums.butterfly_percentage is None  # zero checks is null, never 0.0


def test_the_butterfly_threshold_is_configurable_upward_only():
    with pytest.raises(ValueError, match="at least 3"):
        RaggedArbitrageConfig(butterfly_min_unique_strikes=2)
    strikes = np.linspace(-0.2, 0.2, 5)
    quotes = _svi_smile(1.0, strikes)
    strict = sampled_arbitrage(
        quotes.iv, quotes, config=RaggedArbitrageConfig(butterfly_min_unique_strikes=6)
    )
    assert strict.butterfly_checked_nodes == 0
    assert strict.butterfly_skipped_smiles == 1


# -- calendar ----------------------------------------------------------------
def test_calendar_is_clean_when_total_variance_increases():
    strikes = np.linspace(-0.3, 0.3, 13)
    near = _svi_smile(0.5, strikes)
    far_iv = np.sqrt(_svi_total_variance(strikes) * 1.5 / 1.0)
    far = SurfaceQuotes(k=strikes, T=np.full(13, 1.0), iv=far_iv, surface_id="A")
    quotes = SurfaceQuotes(
        k=np.concatenate([near.k, far.k]),
        T=np.concatenate([near.T, far.T]),
        iv=np.concatenate([near.iv, far.iv]),
        surface_id="A",
    )
    sums = sampled_arbitrage(quotes.iv, quotes)
    assert sums.calendar_checked_points == 40
    assert sums.calendar_violating_points == 0
    assert sums.calendar_percentage == 0.0


def test_a_decreasing_total_variance_is_flagged_on_every_grid_point():
    strikes = np.linspace(-0.3, 0.3, 13)
    near_w = _svi_total_variance(strikes)
    quotes = SurfaceQuotes(
        k=np.concatenate([strikes, strikes]),
        T=np.concatenate([np.full(13, 0.5), np.full(13, 1.0)]),
        iv=np.concatenate([np.sqrt(near_w / 0.5), np.sqrt(0.5 * near_w / 1.0)]),
        surface_id="A",
    )
    sums = sampled_arbitrage(quotes.iv, quotes)
    assert sums.calendar_checked_points == 40
    assert sums.calendar_violating_points == 40
    assert sums.calendar_percentage == 100.0


def test_a_maturity_pair_without_overlap_is_skipped_and_counted():
    quotes = SurfaceQuotes(
        k=[-0.3, -0.2, 0.2, 0.3],
        T=[0.5, 0.5, 1.0, 1.0],
        iv=[0.2, 0.2, 0.3, 0.3],
        surface_id="A",
    )
    sums = sampled_arbitrage(quotes.iv, quotes)
    assert sums.calendar_checked_points == 0
    assert sums.calendar_skipped_pairs == 1
    assert sums.calendar_percentage is None


def test_a_maturity_bucket_below_the_calendar_threshold_does_not_participate():
    quotes = SurfaceQuotes(
        k=[-0.1, 0.0, 0.1, 0.0],
        T=[0.5, 0.5, 0.5, 1.0],
        iv=[0.2, 0.2, 0.2, 0.3],
        surface_id="A",
    )
    sums = sampled_arbitrage(quotes.iv, quotes)
    assert sums.calendar_checked_points == 0
    assert sums.calendar_skipped_pairs == 0  # the single-node bucket never became a pair


# -- duplicates and zero variance -------------------------------------------
def test_exact_duplicates_collapse_by_mean_total_variance():
    quotes = SurfaceQuotes(
        k=[-0.1, 0.0, 0.0, 0.1],
        T=np.full(4, 0.5),
        iv=[0.20, 0.18, 0.22, 0.20],
        surface_id="A",
    )
    sums = sampled_arbitrage(quotes.iv, quotes)
    assert sums.node_count == 3
    assert sums.duplicate_group_count == 1
    assert sums.collapsed_duplicate_points == 1
    assert sums.max_duplicate_iv_range == pytest.approx(0.04)
    # The collapsed node is the mean of the squared vols, i.e. RMS IV at fixed maturity.
    rms = np.sqrt(np.mean(np.array([0.18, 0.22]) ** 2))
    equivalent = SurfaceQuotes(
        k=[-0.1, 0.0, 0.1], T=np.full(3, 0.5), iv=[0.20, rms, 0.20], surface_id="A"
    )
    assert sums.butterfly_min_g == pytest.approx(
        sampled_arbitrage(equivalent.iv, equivalent).butterfly_min_g
    )


def test_duplicates_can_be_refused_instead():
    quotes = SurfaceQuotes(k=[0.0, 0.0], T=[0.5, 0.5], iv=[0.2, 0.2], surface_id="A")
    with pytest.raises(ValueError, match="duplicated strike"):
        sampled_arbitrage(quotes.iv, quotes, config=RaggedArbitrageConfig(duplicates="error"))


def test_an_unknown_duplicate_policy_is_refused():
    with pytest.raises(ValueError, match="duplicates must be one of"):
        RaggedArbitrageConfig(duplicates="average_iv")


def test_zero_total_variance_nodes_are_excluded_and_counted():
    quotes = SurfaceQuotes(
        k=[-0.2, -0.1, 0.0, 0.1, 0.2],
        T=np.full(5, 0.5),
        iv=[0.2, 0.2, 0.0, 0.2, 0.2],  # a legitimate zero IV: Durrleman is singular there
        surface_id="A",
    )
    sums = sampled_arbitrage(quotes.iv, quotes)
    assert sums.nonpositive_total_variance_points == 1
    assert sums.node_count == 4
    assert sums.butterfly_checked_nodes == 2  # four usable nodes, interior stencil
    assert np.isfinite(sums.butterfly_min_g)


def test_maturity_buckets_record_their_raw_span_for_audit():
    quotes = SurfaceQuotes(
        k=[-0.1, 0.0, 0.1],
        T=[0.5000001, 0.5, 0.4999998],
        iv=[0.2, 0.2, 0.2],
        surface_id="A",
    )
    sums = sampled_arbitrage(quotes.iv, quotes, config=RaggedArbitrageConfig(maturity_decimals=3))
    assert sums.smile_count == 1  # one bucket after rounding
    assert sums.max_maturity_bucket_span == pytest.approx(3e-7, rel=1e-3)


def test_strikes_are_never_grouped_by_tolerance():
    quotes = SurfaceQuotes(
        k=[0.0, 1e-12, 0.1, 0.2],
        T=np.full(4, 0.5),
        iv=[0.2, 0.2, 0.2, 0.2],
        surface_id="A",
    )
    sums = sampled_arbitrage(quotes.iv, quotes)
    assert sums.node_count == 4  # near-equal strikes stay distinct
    assert sums.duplicate_group_count == 0


def test_invalid_predictions_are_excluded_and_counted():
    quotes = SurfaceQuotes(
        k=[-0.2, -0.1, 0.0, 0.1, 0.2], T=np.full(5, 0.5), iv=np.full(5, 0.2), surface_id="A"
    )
    sums = sampled_arbitrage(np.array([0.2, 0.2, -1.0, 0.2, 0.2]), quotes)
    assert sums.unusable_points == 1
    assert sums.node_count == 4


def test_no_usable_rows_yields_only_the_unusable_count():
    quotes = SurfaceQuotes(k=[0.0], T=[0.5], iv=[0.2], surface_id="A")
    sums = sampled_arbitrage(np.array([np.nan]), quotes)
    assert sums == RaggedArbitrageSums(unusable_points=1)


def test_surfaces_are_scored_independently():
    strikes = np.linspace(-0.3, 0.3, 13)
    clean = _svi_smile(1.0, strikes, surface="A")
    broken_w = np.full(13, 0.04)
    broken_w[6] = 0.20
    quotes = SurfaceQuotes(
        k=np.concatenate([clean.k, strikes]),
        T=np.concatenate([clean.T, np.full(13, 1.0)]),
        iv=np.concatenate([clean.iv, np.sqrt(broken_w)]),
        surface_id=["A"] * 13 + ["B"] * 13,
    )
    sums = sampled_arbitrage(quotes.iv, quotes)
    assert sums.smile_count == 2
    assert sums.butterfly_checked_nodes == 22
    assert sums.butterfly_violating_nodes >= 1


def test_merge_is_associative_and_pools_extrema():
    a = RaggedArbitrageSums(butterfly_checked_nodes=3, butterfly_min_g=0.5, smile_count=1)
    b = RaggedArbitrageSums(butterfly_checked_nodes=2, butterfly_min_g=-0.1, smile_count=1)
    c = RaggedArbitrageSums(butterfly_checked_nodes=1, smile_count=1)
    assert a.merge(b).merge(c) == a.merge(b.merge(c))
    assert a.merge(b) == b.merge(a)
    assert a.merge(b).merge(c).butterfly_min_g == pytest.approx(-0.1)


def test_pooled_percentages_equal_the_whole():
    strikes = np.linspace(-0.4, 0.4, 21)
    quotes = _svi_smile(1.0, strikes)
    whole = sampled_arbitrage(quotes.iv, quotes)
    assert whole.butterfly_percentage == pytest.approx(
        100.0 * whole.butterfly_violating_nodes / whole.butterfly_checked_nodes
    )


# -- rectangular grid --------------------------------------------------------
def _rectangular_quotes():
    strikes = np.array([-0.2, -0.1, 0.0, 0.1, 0.2])
    maturities = np.array([0.25, 0.5, 1.0])
    k_grid, T_grid = np.meshgrid(strikes, maturities, indexing="ij")
    iv = np.sqrt(_svi_total_variance(k_grid) / T_grid)
    return SurfaceQuotes(k=k_grid.ravel(), T=T_grid.ravel(), iv=iv.ravel(), surface_id="A")


def test_complete_grid_round_trip_preserves_values_and_derives_the_mask():
    quotes = _rectangular_quotes()
    surface = quotes_to_surface(quotes)
    assert surface.iv.shape == (5, 3)
    assert surface.native_mask.all()
    # Every row lands on its own cell.
    for index in range(quotes.n):
        i = int(np.searchsorted(surface.k, quotes.k[index]))
        j = int(np.searchsorted(surface.T, quotes.T[index]))
        assert surface.iv[i, j] == pytest.approx(quotes.iv[index])


def test_the_native_mask_is_derived_from_finite_values_not_supplied():
    quotes = _rectangular_quotes()
    values = np.array(quotes.iv, copy=True)
    values[4] = np.nan
    surface = quotes_to_surface(quotes, values=values)
    assert surface.native_mask.sum() == quotes.n - 1


def test_a_ragged_quote_set_has_no_grid_certificate():
    quotes = _rectangular_quotes()
    with pytest.raises(ValueError, match="Rectangular grid requires exactly"):
        quotes_to_surface(quotes.subset(np.arange(quotes.n - 1)))


def test_duplicate_cells_are_refused():
    quotes = SurfaceQuotes(
        k=[0.0, 0.0, 0.1, 0.1],
        T=[1.0, 1.0, 1.0, 2.0],
        iv=[0.2, 0.21, 0.2, 0.2],
        surface_id="A",
    )
    with pytest.raises(ValueError, match="exactly once"):
        quotes_to_surface(quotes)


def test_multiple_surfaces_are_refused():
    quotes = SurfaceQuotes(
        k=[0.0, 0.1, 0.0, 0.1],
        T=[1.0, 1.0, 1.0, 1.0],
        iv=[0.2, 0.2, 0.2, 0.2],
        surface_id=["A", "A", "B", "B"],
    )
    with pytest.raises(ValueError, match="exactly one surface"):
        quotes_to_surface(quotes)


def test_grid_summary_projects_the_arbitrage_report():
    surface = quotes_to_surface(_rectangular_quotes())
    summary = diagnose_surface(surface)
    assert summary.strike_count == 5
    assert summary.maturity_count == 3
    assert summary.native_coverage == 1.0
    assert summary.quoted_coverage == 1.0
    assert 0.0 <= summary.sas
    assert set(summary.native_violation_counts) == {
        "bound",
        "butterfly",
        "calendar",
        "vertical",
    }


def test_diagnose_surface_requires_exact_reference_coordinates():
    surface = quotes_to_surface(_rectangular_quotes())
    shifted = quotes_to_surface(_rectangular_quotes())
    shifted.k = np.asarray(shifted.k) + 1e-12
    with pytest.raises(ValueError, match="exactly equal k coordinates"):
        diagnose_surface(surface, reference=shifted)
    assert diagnose_surface(surface, reference=quotes_to_surface(_rectangular_quotes()))
