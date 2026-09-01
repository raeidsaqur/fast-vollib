"""Fit accumulators: hand-computed values, counts that cannot be collapsed, pooling."""

from __future__ import annotations

import math

import numpy as np
import pytest

from fast_vollib.diagnostics import (
    Box,
    ErrorSums,
    NamedRegion,
    classify_predictions,
    fit_error,
    fit_error_by_region,
)


def test_hand_computed_error_sums():
    sums = fit_error([0.21, 0.18], [0.20, 0.20])
    assert sums.squared_error_sum == pytest.approx(0.0001 + 0.0004)
    assert sums.absolute_error_sum == pytest.approx(0.01 + 0.02)
    assert sums.valid_prediction_count == 2
    assert sums.target_count == 2
    assert sums.invalid_prediction_count == 0
    assert sums.max_absolute_error == pytest.approx(0.02)
    assert sums.rmse == pytest.approx(math.sqrt(0.0005 / 2))
    assert sums.mae == pytest.approx(0.015)
    assert sums.coverage == 1.0
    assert sums.invalid_rate == 0.0


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf, -1e-9])
def test_invalid_predictions_are_counted_not_dropped(bad):
    sums = fit_error([0.20, bad], [0.20, 0.20])
    assert sums.target_count == 2
    assert sums.valid_prediction_count == 1
    assert sums.invalid_prediction_count == 1
    assert sums.coverage == 0.5
    assert sums.invalid_rate == 0.5
    # The error sums describe only the covered half; the count says so.
    assert sums.rmse == pytest.approx(0.0)


def test_zero_is_a_valid_prediction():
    sums = fit_error([0.0], [0.2])
    assert sums.valid_prediction_count == 1
    assert sums.invalid_prediction_count == 0


def test_missing_truth_rows_are_not_targets():
    sums = fit_error([0.2, 0.3], [0.2, np.nan])
    assert sums.target_count == 1
    assert sums.valid_prediction_count == 1
    assert sums.invalid_prediction_count == 0


def test_empty_input_yields_null_derived_values():
    sums = fit_error([], [])
    assert sums.target_count == 0
    assert sums.rmse is None
    assert sums.mae is None
    assert sums.coverage is None
    assert sums.invalid_rate is None
    assert sums.max_absolute_error is None


def test_a_fully_invalid_sample_has_zero_coverage_and_no_rmse():
    sums = fit_error([np.nan, -1.0], [0.2, 0.2])
    assert sums.target_count == 2
    assert sums.valid_prediction_count == 0
    assert sums.coverage == 0.0
    assert sums.rmse is None


def _random_pair(seed: int, size: int):
    rng = np.random.default_rng(seed)
    truth = rng.uniform(0.1, 0.4, size)
    pred = truth + rng.normal(0.0, 0.01, size)
    return pred, truth


def test_merge_is_associative_and_commutative():
    pred, truth = _random_pair(7, 30)
    a = fit_error(pred[:10], truth[:10])
    b = fit_error(pred[10:20], truth[10:20])
    c = fit_error(pred[20:], truth[20:])
    assert a.merge(b).merge(c) == a.merge(b.merge(c))
    assert a.merge(b) == b.merge(a)


def test_pooled_sums_equal_the_concatenation():
    pred, truth = _random_pair(11, 40)
    whole = fit_error(pred, truth)
    pooled = ErrorSums()
    for start in range(0, 40, 7):
        pooled = pooled.merge(fit_error(pred[start : start + 7], truth[start : start + 7]))
    assert pooled.valid_prediction_count == whole.valid_prediction_count
    assert pooled.squared_error_sum == pytest.approx(whole.squared_error_sum, rel=1e-12)
    assert pooled.rmse == pytest.approx(whole.rmse, rel=1e-12)
    assert pooled.max_absolute_error == pytest.approx(whole.max_absolute_error)


def test_merging_an_empty_group_is_the_identity():
    sums = fit_error([0.21], [0.20])
    assert sums.merge(ErrorSums()) == sums
    assert ErrorSums().merge(sums) == sums


# -- regions ----------------------------------------------------------------
def test_regions_and_complements_partition_the_targets():
    pred = np.array([0.21, 0.19, 0.22, 0.20])
    truth = np.full(4, 0.20)
    k = np.array([0.0, 0.1, 0.5, 0.0])
    T = np.array([0.25, 0.25, 0.25, 2.0])
    diagnostics = fit_error_by_region(pred, truth, k, T)
    liquid = diagnostics.region("liquid")
    assert liquid.inside.target_count == 2  # |k| <= 0.2 and T <= 0.5
    assert liquid.outside.target_count == 2
    assert (
        liquid.inside.target_count + liquid.outside.target_count == diagnostics.overall.target_count
    )
    assert liquid.inside.squared_error_sum + liquid.outside.squared_error_sum == pytest.approx(
        diagnostics.overall.squared_error_sum
    )


def test_region_bounds_are_inclusive_by_default_and_respect_closed():
    pred = np.array([0.21])
    truth = np.array([0.20])
    k = np.array([0.2])
    T = np.array([0.5])
    inclusive = fit_error_by_region(pred, truth, k, T)
    assert inclusive.region("liquid").inside.target_count == 1
    half_open = fit_error_by_region(
        pred,
        truth,
        k,
        T,
        (Box(name="liquid", complement_name="illiquid", k_max=0.2, T_max=0.5, closed="left"),),
    )
    assert half_open.region("liquid").inside.target_count == 0


def test_a_region_without_a_complement_reports_only_itself():
    diagnostics = fit_error_by_region(
        [0.21], [0.20], [0.0], [0.25], (Box(name="wings", k_min=-0.1),)
    )
    assert diagnostics.region("wings").outside is None
    assert diagnostics.region("wings").complement_name is None


def test_named_callable_regions_are_scored_but_carry_no_descriptor():
    region = NamedRegion(
        name="short", complement_name="long", predicate=lambda k, T: np.asarray(T) < 1.0
    )
    diagnostics = fit_error_by_region([0.21, 0.19], [0.2, 0.2], [0.0, 0.0], [0.5, 2.0], (region,))
    scored = diagnostics.region("short")
    assert scored.inside.target_count == 1
    assert scored.descriptor is None


def test_region_merge_requires_identical_definitions():
    left = fit_error_by_region([0.21], [0.2], [0.0], [0.25])
    right = fit_error_by_region(
        [0.21], [0.2], [0.0], [0.25], (Box(name="liquid", complement_name="illiquid", k_max=0.5),)
    )
    with pytest.raises(ValueError, match="descriptors must match"):
        left.merge(right)


def test_region_pooling_equals_concatenation():
    pred, truth = _random_pair(3, 24)
    rng = np.random.default_rng(5)
    k = rng.uniform(-0.5, 0.5, 24)
    T = rng.uniform(0.05, 2.0, 24)
    whole = fit_error_by_region(pred, truth, k, T)
    pooled = fit_error_by_region(pred[:8], truth[:8], k[:8], T[:8])
    for start in (8, 16):
        pooled = pooled.merge(
            fit_error_by_region(
                pred[start : start + 8],
                truth[start : start + 8],
                k[start : start + 8],
                T[start : start + 8],
            )
        )
    assert pooled.region("liquid").inside.rmse == pytest.approx(
        whole.region("liquid").inside.rmse, rel=1e-12
    )
    assert pooled.overall.rmse == pytest.approx(whole.overall.rmse, rel=1e-12)


def test_classification_is_shared_and_shape_checked():
    classification = classify_predictions([0.2, np.nan], [0.2, 0.2])
    assert classification.target_count == 2
    assert classification.valid_count == 1
    assert classification.invalid_count == 1
    with pytest.raises(ValueError, match="same shape"):
        classify_predictions([0.2], [0.2, 0.2])


def test_region_shape_mismatch_is_rejected():
    with pytest.raises(ValueError, match="k and T must have shape"):
        fit_error_by_region([0.2, 0.2], [0.2, 0.2], [0.0], [1.0])
