"""Spread consistency: hand-computed prices, honest denominators, absent blocks."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.stats import norm

from fast_vollib.diagnostics import (
    SpreadSums,
    SurfaceQuotes,
    normalized_option_price,
    spread_consistency,
)


def _reference_call(iv: float, k: float, T: float) -> float:
    """An independent Black call, written from the definition."""
    sqrt_w = iv * math.sqrt(T)
    d1 = (-k + 0.5 * iv * iv * T) / sqrt_w
    d2 = d1 - sqrt_w
    return float(norm.cdf(d1) - math.exp(k) * norm.cdf(d2))


def test_normalized_call_matches_an_independent_reference():
    price = normalized_option_price(np.array([0.2]), np.array([-0.1]), np.array([0.5]))
    assert float(price[0]) == pytest.approx(_reference_call(0.2, -0.1, 0.5), rel=1e-12)


def test_puts_follow_put_call_parity():
    k, iv, T = -0.2, 0.25, 0.75
    call = float(normalized_option_price(np.array([iv]), np.array([k]), np.array([T]))[0])
    put = float(
        normalized_option_price(np.array([iv]), np.array([k]), np.array([T]), np.array([False]))[0]
    )
    assert put == pytest.approx(call - (1.0 - math.exp(k)), rel=1e-12)


def test_hand_computed_spread_sums():
    k, T, iv = 0.0, 1.0, 0.2
    price = _reference_call(iv, k, T)
    bid, ask = price - 0.001, price + 0.001  # the model prices exactly at the midpoint
    quotes = SurfaceQuotes(k=[k], T=[T], iv=[iv], bid=[bid], ask=[ask], is_call=[True])
    sums = spread_consistency(np.array([iv]), quotes)
    assert sums.eligible_quote_count == 1
    assert sums.priced_quote_count == 1
    assert sums.unpriced_quote_count == 0
    assert sums.midpoint_squared_error_sum == pytest.approx(0.0, abs=1e-24)
    assert sums.outside_count == 0
    assert sums.price_rmse == pytest.approx(0.0, abs=1e-12)
    assert sums.outside_percentage == 0.0
    assert sums.mean_miss_width is None  # no misses: not a zero


def test_a_miss_is_measured_in_spread_widths():
    k, T = 0.0, 1.0
    price = _reference_call(0.20, k, T)
    width = 0.002
    bid, ask = price - width / 2, price + width / 2
    quotes = SurfaceQuotes(k=[k], T=[T], iv=[0.20], bid=[bid], ask=[ask], is_call=[True])
    high = _reference_call(0.24, k, T)
    sums = spread_consistency(np.array([0.24]), quotes)
    assert sums.outside_count == 1
    assert sums.outside_percentage == 100.0
    assert sums.mean_miss_width == pytest.approx((high - ask) / width, rel=1e-9)


def test_a_price_on_the_bound_is_inside():
    k, T = 0.0, 1.0
    price = _reference_call(0.20, k, T)
    quotes = SurfaceQuotes(k=[k], T=[T], iv=[0.20], bid=[price], ask=[price + 0.01])
    assert spread_consistency(np.array([0.20]), quotes).outside_count == 0


def test_without_bid_ask_there_is_no_spread_block():
    quotes = SurfaceQuotes(k=[0.0], T=[1.0], iv=[0.2])
    assert spread_consistency(np.array([0.2]), quotes) is None


def test_unquoted_rows_are_not_eligible():
    quotes = SurfaceQuotes(
        k=[0.0, 0.1], T=[1.0, 1.0], iv=[0.2, 0.2], bid=[0.01, np.nan], ask=[0.02, np.nan]
    )
    sums = spread_consistency(np.array([0.2, 0.2]), quotes)
    assert sums.eligible_quote_count == 1


def test_an_eligible_quote_the_model_cannot_price_is_unpriced_not_dropped():
    quotes = SurfaceQuotes(
        k=[0.0, 0.1], T=[1.0, 1.0], iv=[0.2, 0.2], bid=[0.01, 0.01], ask=[0.02, 0.02]
    )
    sums = spread_consistency(np.array([0.2, np.nan]), quotes)
    assert sums.eligible_quote_count == 2
    assert sums.priced_quote_count == 1
    assert sums.unpriced_quote_count == 1
    assert sums.eligible_quote_count == sums.priced_quote_count + sums.unpriced_quote_count


def test_nothing_priced_yields_null_metrics_not_zeros():
    quotes = SurfaceQuotes(k=[0.0], T=[1.0], iv=[0.2], bid=[0.01], ask=[0.02])
    sums = spread_consistency(np.array([np.nan]), quotes)
    assert sums.priced_quote_count == 0
    assert sums.price_rmse is None
    assert sums.outside_percentage is None
    assert sums.mean_miss_width is None


def test_merge_is_associative_and_pools_denominators_separately():
    a = SpreadSums(2, 2, 0, 0.5, 1, 3.0)
    b = SpreadSums(3, 1, 2, 0.25, 0, 0.0)
    c = SpreadSums(1, 1, 0, 0.25, 1, 1.0)
    assert a.merge(b).merge(c) == a.merge(b.merge(c))
    assert a.merge(b) == b.merge(a)
    pooled = a.merge(b).merge(c)
    assert pooled.priced_quote_count == 4
    assert pooled.price_rmse == pytest.approx(math.sqrt(1.0 / 4))
    assert pooled.outside_percentage == pytest.approx(50.0)
    assert pooled.mean_miss_width == pytest.approx(2.0)


def test_shape_mismatch_is_rejected():
    quotes = SurfaceQuotes(k=[0.0], T=[1.0], iv=[0.2], bid=[0.01], ask=[0.02])
    with pytest.raises(ValueError, match="pred_iv must have shape"):
        spread_consistency(np.array([0.2, 0.2]), quotes)
