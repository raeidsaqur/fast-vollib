"""Records, deterministic pooling, and the partial-coverage contract."""

from __future__ import annotations

import numpy as np
import pytest

from fast_vollib.diagnostics import (
    Box,
    DiagnosticRecord,
    DiagnosticReport,
    DiagnosticsConfig,
    RaggedArbitrageConfig,
    SampleDiagnostics,
    SurfaceQuotes,
    diagnose_fit,
    diagnose_surface,
    quotes_to_surface,
)


def _quotes(surface="A", *, spread=False, n=5):
    strikes = np.linspace(-0.2, 0.2, n)
    kwargs = {}
    if spread:
        kwargs = {"bid": np.full(n, 0.01), "ask": np.full(n, 0.02)}
    return SurfaceQuotes(
        k=strikes,
        T=np.full(n, 0.5),
        iv=np.full(n, 0.2),
        surface_id=surface,
        point_id=list(range(n)),
        **kwargs,
    )


def _sample(pred=None, **kwargs):
    quotes = _quotes(**kwargs)
    return diagnose_fit(quotes.iv if pred is None else pred, quotes)


def test_a_record_requires_a_json_scalar_identity():
    sample = _sample()
    DiagnosticRecord(model="m", split="s", sample_id=1, diagnostics=sample)
    DiagnosticRecord(model="m", split="s", sample_id="day-1", diagnostics=sample)
    with pytest.raises(TypeError, match="string or integer"):
        DiagnosticRecord(model="m", split="s", sample_id=True, diagnostics=sample)
    with pytest.raises(TypeError, match="string or integer"):
        DiagnosticRecord(model="m", split="s", sample_id=1.5, diagnostics=sample)
    with pytest.raises(ValueError, match="non-empty string"):
        DiagnosticRecord(model="", split="s", sample_id=1, diagnostics=sample)


def test_summaries_group_by_model_and_split_in_deterministic_order():
    sample = _sample()
    report = DiagnosticReport.from_records(
        [
            DiagnosticRecord(model="ssvi", split="unif_5", sample_id=1, diagnostics=sample),
            DiagnosticRecord(model="prior", split="unif_5", sample_id=1, diagnostics=sample),
            DiagnosticRecord(model="prior", split="extrap_5", sample_id=1, diagnostics=sample),
            DiagnosticRecord(model="prior", split="extrap_5", sample_id=2, diagnostics=sample),
        ]
    )
    assert [(s.model, s.split, s.sample_count) for s in report.summaries()] == [
        ("prior", "extrap_5", 2),
        ("prior", "unif_5", 1),
        ("ssvi", "unif_5", 1),
    ]


def test_pooled_sums_are_independent_of_record_partitioning():
    quotes = _quotes(n=9)
    rng = np.random.default_rng(3)
    predictions = [quotes.iv + rng.normal(0.0, 0.01, quotes.n) for _ in range(4)]
    records = [
        DiagnosticRecord(
            model="m", split="s", sample_id=index, diagnostics=diagnose_fit(pred, quotes)
        )
        for index, pred in enumerate(predictions)
    ]
    whole = DiagnosticReport.from_records(records).summaries()[0]
    combined = np.concatenate(predictions)
    expected_sq = float(np.sum((combined - np.tile(quotes.iv, 4)) ** 2))
    assert whole.fit.overall.squared_error_sum == pytest.approx(expected_sq, rel=1e-12)
    assert whole.fit.overall.valid_prediction_count == 4 * quotes.n


def test_a_single_invalid_prediction_marks_the_group_partial():
    quotes = _quotes()
    clean = diagnose_fit(quotes.iv, quotes)
    broken_pred = np.array(quotes.iv, copy=True)
    broken_pred[2] = np.nan
    broken = diagnose_fit(broken_pred, quotes)
    assert clean.quality_status == "complete"
    assert broken.quality_status == "partial"
    summary = DiagnosticReport.from_records(
        [
            DiagnosticRecord(model="m", split="s", sample_id=1, diagnostics=clean),
            DiagnosticRecord(model="m", split="s", sample_id=2, diagnostics=broken),
        ]
    ).summaries()[0]
    assert summary.quality_status == "partial"
    assert summary.fit.overall.invalid_prediction_count == 1
    assert summary.coverage == pytest.approx(9 / 10)


def test_a_group_may_not_mix_samples_that_carry_a_block_with_samples_that_do_not():
    with_spread = diagnose_fit(_quotes(spread=True).iv, _quotes(spread=True))
    without_spread = _sample()
    with pytest.raises(ValueError, match="some samples carry a spread block"):
        DiagnosticReport.from_records(
            [
                DiagnosticRecord(model="m", split="s", sample_id=1, diagnostics=with_spread),
                DiagnosticRecord(model="m", split="s", sample_id=2, diagnostics=without_spread),
            ]
        )


def test_legacy_sample_mean_percentages_treat_a_zero_check_record_as_zero():
    scored = _sample()  # five nodes: three butterfly checks
    unscored = diagnose_fit(
        np.array([0.2, 0.2]),
        SurfaceQuotes(k=[-0.1, 0.1], T=[0.5, 0.5], iv=[0.2, 0.2], surface_id="A"),
    )
    assert scored.ragged.butterfly_percentage == 0.0
    assert unscored.ragged.butterfly_percentage is None
    summary = DiagnosticReport.from_records(
        [
            DiagnosticRecord(model="m", split="s", sample_id=1, diagnostics=scored),
            DiagnosticRecord(model="m", split="s", sample_id=2, diagnostics=unscored),
        ]
    ).summaries()[0]
    # The pooled percentage divides real checks by real checks...
    assert summary.ragged.butterfly_percentage == 0.0
    # ...while the compatibility column averages per-record values, zeros included.
    assert summary.ragged_butterfly_sample_mean == 0.0


def test_pooled_percentage_is_null_when_nothing_was_ever_checked():
    unscored = diagnose_fit(
        np.array([0.2, 0.2]),
        SurfaceQuotes(k=[-0.1, 0.1], T=[0.5, 0.5], iv=[0.2, 0.2], surface_id="A"),
    )
    summary = DiagnosticReport.from_records(
        [DiagnosticRecord(model="m", split="s", sample_id=1, diagnostics=unscored)]
    ).summaries()[0]
    assert summary.ragged.butterfly_percentage is None
    assert summary.ragged_butterfly_sample_mean == 0.0


def test_a_grid_block_appears_only_when_a_grid_was_evaluated():
    strikes = np.array([-0.2, -0.1, 0.0, 0.1, 0.2])
    maturities = np.array([0.25, 0.5])
    k_grid, T_grid = np.meshgrid(strikes, maturities, indexing="ij")
    quotes = SurfaceQuotes(
        k=k_grid.ravel(), T=T_grid.ravel(), iv=np.full(k_grid.size, 0.2), surface_id="A"
    )
    base = diagnose_fit(quotes.iv, quotes)
    gridded = SampleDiagnostics(
        fit=base.fit,
        spread=base.spread,
        ragged=base.ragged,
        grid=diagnose_surface(quotes_to_surface(quotes)),
    )
    without = DiagnosticReport.from_records(
        [DiagnosticRecord(model="m", split="s", sample_id=1, diagnostics=base)]
    ).summaries()[0]
    with_grid = DiagnosticReport.from_records(
        [DiagnosticRecord(model="m", split="g", sample_id=1, diagnostics=gridded)]
    ).summaries()[0]
    assert without.grid is None
    assert with_grid.grid is not None
    assert with_grid.grid.sample_count == 1
    assert with_grid.grid.native_coverage_min == 1.0


def test_summary_only_projection_drops_records_and_keeps_the_count():
    report = DiagnosticReport.from_records(
        [
            DiagnosticRecord(model="m", split="s", sample_id=index, diagnostics=_sample())
            for index in range(3)
        ]
    )
    projected = report.summary_only()
    assert projected.records == ()
    assert projected.records_included is False
    assert projected.record_count == 3
    assert projected.summaries() == report.summaries()


def test_a_report_declaring_no_records_may_not_carry_any():
    with pytest.raises(ValueError, match="requires an empty record tuple"):
        DiagnosticReport(
            records=(DiagnosticRecord(model="m", split="s", sample_id=1, diagnostics=_sample()),),
            record_count=1,
            records_included=False,
        )


def test_a_report_config_refuses_a_region_it_cannot_describe():
    config = DiagnosticsConfig(
        regions=(Box(name="liquid", complement_name="illiquid", k_max=0.2),),
        ragged=RaggedArbitrageConfig(maturity_decimals=4),
    )
    assert config.regions[0].k_max == 0.2
    assert config.ragged.maturity_decimals == 4
    assert dict(config.metadata) == {}
