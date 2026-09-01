"""Independent oracles for the statistics an existing scoring harness computes.

The controller these diagnostics replace computes its error split, call-price
spread statistics, and butterfly / calendar percentages inline.  The oracles
below are written **independently from that formulation** -- plain NumPy, no
imports from the diagnostics package -- so the comparison is a real check
rather than the same code twice.

Two of the three agree exactly, by construction:

* the error split is the same sums over the same masks;
* the calendar check is the same 40-point interpolation on the same buckets.

The butterfly check does **not**, and is not made to.  The oracle differentiates
with two ``np.gradient`` passes, which returns a value at every node including
the endpoints and floors the total variance at ``1e-8``; ``fast_vollib`` uses a
local-parabola stencil defined on interior nodes only, with no floor.  Those
are different estimators, so the tests here pin the *structural* difference and
the cases where both must agree, instead of inventing a tolerance that hides it.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from fast_vollib.diagnostics import (
    SurfaceQuotes,
    fit_error_by_region,
    sampled_arbitrage,
    spread_consistency,
)

_T_DECIMALS = 6


# -- oracles (independent re-derivations) -----------------------------------
def oracle_liquid_mask(k, T, k_liq=0.2, T_liq=0.5):
    return (np.abs(np.asarray(k)) <= k_liq) & (np.asarray(T) <= T_liq)


def oracle_split_stats(pred, true, k, T, *, k_liq=0.2, T_liq=0.5) -> dict:
    liquid = oracle_liquid_mask(k, T, k_liq, T_liq)
    out: dict[str, float] = {}
    for tag, mask in (("", np.ones_like(liquid)), ("_liquid", liquid), ("_illiquid", ~liquid)):
        if mask.any():
            error = np.asarray(pred)[mask] - np.asarray(true)[mask]
            out[f"sq{tag}"] = float(np.sum(error**2))
            out[f"n{tag}"] = int(error.size)
            if tag == "":
                out["abs"] = float(np.sum(np.abs(error)))
        else:
            out[f"sq{tag}"] = 0.0
            out[f"n{tag}"] = 0
    return out


def oracle_bs_call_norm(iv, k, T):
    iv = np.asarray(iv, float)
    k = np.asarray(k, float)
    T = np.asarray(T, float)
    sqT = np.sqrt(np.maximum(T, 1e-12))
    d1 = (-k + 0.5 * iv**2 * T) / (iv * sqT + 1e-14)
    d2 = d1 - iv * sqT
    return norm.cdf(d1) - np.exp(k) * norm.cdf(d2)


def oracle_call_price_stats(pred_iv, k, T, bid, ask, is_call=None) -> dict:
    bid = np.asarray(bid, float)
    ask = np.asarray(ask, float)
    quoted = np.isfinite(bid) & np.isfinite(ask)
    if not quoted.any():
        return {"call_sq": 0.0, "call_n": 0, "call_oob": 0, "call_oob_spread_sum": 0.0}
    k_m = np.asarray(k, float)[quoted]
    call = oracle_bs_call_norm(
        np.asarray(pred_iv, float)[quoted], k_m, np.asarray(T, float)[quoted]
    )
    if is_call is None:
        price = call
    else:
        price = np.where(np.asarray(is_call, bool)[quoted], call, call - (1.0 - np.exp(k_m)))
    bid_q, ask_q = bid[quoted], ask[quoted]
    midpoint = 0.5 * (bid_q + ask_q)
    outside = (price < bid_q) | (price > ask_q)
    distance = np.maximum(0.0, np.maximum(bid_q - price, price - ask_q))
    return {
        "call_sq": float(np.sum((price - midpoint) ** 2)),
        "call_n": int(quoted.sum()),
        "call_oob": int(np.sum(outside)),
        "call_oob_spread_sum": float(np.sum(distance / np.maximum(ask_q - bid_q, 1e-12))),
    }


def oracle_durrleman_g(k, w):
    k = np.asarray(k, float)
    w = np.asarray(w, float)
    if k.size < 3:
        return np.full(k.shape, np.nan)
    wp = np.gradient(w, k)
    wpp = np.gradient(wp, k)
    w_safe = np.maximum(w, 1e-8)
    return (1.0 - k * wp / (2.0 * w_safe)) ** 2 - wp**2 / 4.0 * (1.0 / w_safe + 0.25) + wpp / 2.0


def oracle_butterfly_counts(k, T, iv, asset_id) -> tuple[int, int]:
    k, T, iv = (np.asarray(x, float) for x in (k, T, iv))
    asset_id = np.asarray(asset_id)
    bucket = np.round(T, _T_DECIMALS)
    total = violations = 0
    for asset in np.unique(asset_id):
        for maturity in np.unique(bucket[asset_id == asset]):
            mask = (asset_id == asset) & (bucket == maturity)
            if mask.sum() < 3:
                continue
            order = np.argsort(k[mask])
            g = oracle_durrleman_g(k[mask][order], (iv[mask][order] ** 2) * maturity)
            g = g[np.isfinite(g)]
            total += g.size
            violations += int(np.sum(g < 0.0))
    return total, violations


def oracle_calendar_counts(k, T, iv, asset_id, n_grid: int = 40) -> tuple[int, int]:
    k, T, iv = (np.asarray(x, float) for x in (k, T, iv))
    asset_id = np.asarray(asset_id)
    bucket = np.round(T, _T_DECIMALS)
    total = violations = 0
    for asset in np.unique(asset_id):
        maturities = np.sort(np.unique(bucket[asset_id == asset]))
        smiles = {}
        for maturity in maturities:
            mask = (asset_id == asset) & (bucket == maturity)
            if mask.sum() < 2:
                continue
            order = np.argsort(k[mask])
            smiles[maturity] = (k[mask][order], (iv[mask][order] ** 2) * maturity)
        available = [m for m in maturities if m in smiles]
        for near, far in zip(available, available[1:]):
            k1, w1 = smiles[near]
            k2, w2 = smiles[far]
            low, high = max(k1.min(), k2.min()), min(k1.max(), k2.max())
            if high - low < 1e-9:
                continue
            grid = np.linspace(low, high, n_grid)
            total += grid.size
            violations += int(np.sum(np.interp(grid, k1, w1) - np.interp(grid, k2, w2) > 1e-10))
    return total, violations


# -- fixtures ----------------------------------------------------------------
def _scattered_sample(seed: int, *, duplicates: bool = False):
    rng = np.random.default_rng(seed)
    strikes, maturities, assets = [], [], []
    for asset in range(3):
        for maturity in (0.25, 0.5, 1.0):
            nodes = np.sort(rng.uniform(-0.5, 0.5, 9))
            if duplicates:
                nodes = np.sort(np.concatenate([nodes, nodes[:2]]))
            strikes.append(nodes)
            maturities.append(np.full(nodes.size, maturity))
            assets.append(np.full(nodes.size, asset))
    k = np.concatenate(strikes)
    T = np.concatenate(maturities)
    asset_id = np.concatenate(assets)
    truth = 0.20 + 0.08 * k**2 + 0.02 * np.sqrt(T)
    pred = truth + rng.normal(0.0, 0.005, truth.size)
    return k, T, asset_id, truth, pred


# -- fit ---------------------------------------------------------------------
def test_error_split_matches_the_oracle_exactly():
    k, T, asset_id, truth, pred = _scattered_sample(1)
    expected = oracle_split_stats(pred, truth, k, T)
    diagnostics = fit_error_by_region(pred, truth, k, T)
    liquid = diagnostics.region("liquid")
    assert diagnostics.overall.squared_error_sum == expected["sq"]
    assert diagnostics.overall.absolute_error_sum == expected["abs"]
    assert diagnostics.overall.valid_prediction_count == expected["n"]
    assert liquid.inside.squared_error_sum == expected["sq_liquid"]
    assert liquid.inside.valid_prediction_count == expected["n_liquid"]
    assert liquid.outside.squared_error_sum == expected["sq_illiquid"]
    assert liquid.outside.valid_prediction_count == expected["n_illiquid"]


def test_error_split_matches_the_oracle_with_duplicate_coordinates_retained():
    k, T, asset_id, truth, pred = _scattered_sample(2, duplicates=True)
    expected = oracle_split_stats(pred, truth, k, T)
    diagnostics = fit_error_by_region(pred, truth, k, T)
    # Duplicate rows stay in the denominator: the counts match row-for-row.
    assert diagnostics.overall.valid_prediction_count == expected["n"] == k.size
    assert diagnostics.overall.squared_error_sum == expected["sq"]


# -- spread ------------------------------------------------------------------
def test_call_price_statistics_match_the_oracle():
    rng = np.random.default_rng(4)
    k, T, asset_id, truth, pred = _scattered_sample(4)
    price = oracle_bs_call_norm(truth, k, T)
    half = rng.uniform(0.0005, 0.002, price.size)
    # Deep-OTM normalized calls can be worth less than half a spread; a quoted bid
    # is never negative, so clamp it the way a real book would.
    bid, ask = np.maximum(price - half, 0.0), price + half
    is_call = k >= 0.0
    quotes = SurfaceQuotes(
        k=k, T=T, iv=truth, surface_id=asset_id, bid=bid, ask=ask, is_call=is_call
    )
    expected = oracle_call_price_stats(pred, k, T, bid, ask, is_call)
    sums = spread_consistency(pred, quotes)
    assert sums.priced_quote_count == expected["call_n"]
    assert sums.outside_count == expected["call_oob"]
    # The oracle carries a 1e-14 additive guard in its d1 denominator; the shared
    # fast-vollib kernel floors the square root instead.  Same formula, different
    # guard, so the sums agree to well inside the 1e-9 relative parity budget.
    assert sums.midpoint_squared_error_sum == pytest.approx(expected["call_sq"], rel=1e-9)
    assert sums.outside_width_sum == pytest.approx(expected["call_oob_spread_sum"], rel=1e-9)


# -- calendar ----------------------------------------------------------------
def test_calendar_counts_match_the_oracle_exactly_without_duplicates():
    k, T, asset_id, truth, pred = _scattered_sample(5)
    quotes = SurfaceQuotes(k=k, T=T, iv=truth, surface_id=asset_id)
    sums = sampled_arbitrage(pred, quotes)
    total, violations = oracle_calendar_counts(k, T, pred, asset_id)
    assert sums.calendar_checked_points == total
    assert sums.calendar_violating_points == violations
    assert sums.calendar_percentage == pytest.approx(100.0 * violations / total)


def test_calendar_counts_still_match_when_duplicates_collapse_the_geometry():
    # The oracle keeps duplicate nodes; collapsing exact duplicates by mean total
    # variance changes the interpolant only where the duplicates disagreed.  Here
    # the repeated rows carry identical predictions, so both must agree exactly.
    k = np.array([-0.2, -0.1, -0.1, 0.0, 0.1, -0.2, 0.0, 0.1])
    T = np.array([0.5] * 5 + [1.0] * 3)
    iv = np.array([0.22, 0.21, 0.21, 0.20, 0.21, 0.26, 0.25, 0.26])
    asset_id = np.zeros(8, dtype=int)
    quotes = SurfaceQuotes(k=k, T=T, iv=iv, surface_id=asset_id)
    sums = sampled_arbitrage(iv, quotes)
    total, violations = oracle_calendar_counts(k, T, iv, asset_id)
    assert sums.duplicate_group_count == 1
    assert (sums.calendar_checked_points, sums.calendar_violating_points) == (total, violations)


# -- butterfly: a measured, structural difference ----------------------------
def test_butterfly_stencils_differ_by_exactly_the_two_endpoints_per_smile():
    k, T, asset_id, truth, pred = _scattered_sample(6)
    quotes = SurfaceQuotes(k=k, T=T, iv=truth, surface_id=asset_id)
    sums = sampled_arbitrage(pred, quotes)
    oracle_total, _ = oracle_butterfly_counts(k, T, pred, asset_id)
    # Nine smiles, each losing both endpoints to the interior stencil.
    assert sums.smile_count == 9
    assert oracle_total - sums.butterfly_checked_nodes == 2 * sums.smile_count


def test_both_estimators_agree_that_a_clean_svi_slice_is_arbitrage_free():
    strikes = np.linspace(-0.6, 0.6, 41)
    total_variance = 0.04 + 0.4 * (-0.4 * strikes + np.sqrt(strikes**2 + 0.01))
    iv = np.sqrt(total_variance / 1.0)
    asset_id = np.zeros(41, dtype=int)
    quotes = SurfaceQuotes(k=strikes, T=np.ones(41), iv=iv, surface_id=asset_id)
    sums = sampled_arbitrage(iv, quotes)
    _, oracle_violations = oracle_butterfly_counts(strikes, np.ones(41), iv, asset_id)
    assert sums.butterfly_violating_nodes == 0
    assert oracle_violations == 0


def test_both_estimators_flag_a_deliberately_broken_slice():
    strikes = np.linspace(-0.3, 0.3, 13)
    total_variance = np.full(13, 0.04)
    total_variance[6] = 0.20
    iv = np.sqrt(total_variance)
    asset_id = np.zeros(13, dtype=int)
    quotes = SurfaceQuotes(k=strikes, T=np.ones(13), iv=iv, surface_id=asset_id)
    _, oracle_violations = oracle_butterfly_counts(strikes, np.ones(13), iv, asset_id)
    assert sampled_arbitrage(iv, quotes).butterfly_violating_nodes >= 1
    assert oracle_violations >= 1
