"""
Experiment A: Jäckel parity stress test.

Compares fast-vollib IV recovery against py_lets_be_rational (the Jäckel
reference implementation) over a wide grid of (log-moneyness, volatility) inputs.

The tolerance tightens as experiments B–D land:
  - Experiment A baseline:   max rel error ~1e-4  (current Halley × 8)
  - After Experiment B (Householder × 2):    assert < 1e-5
  - After Experiment C (3-branch g(σ)):      assert < 1e-8
  - After Experiment D (4-branch guess):     assert < 1e-10

py_lets_be_rational is a transitive venv dep of py_vollib — no new install needed.
"""

from __future__ import annotations

import numpy as np
import pytest

from fast_vollib.jackel.jackel_iv import jackel_iv_black as _jackel_iv_black

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _lbr_prices_and_ivs(
    F: float, K_arr: np.ndarray, sv_arr: np.ndarray, T: float
) -> tuple[np.ndarray, np.ndarray]:
    """Compute reference prices and IV using py_lets_be_rational (Jäckel)."""
    from py_lets_be_rational import (
        black as lbr_black,
        implied_volatility_from_a_transformed_rational_guess as lbr_iv,
    )

    prices = np.array([lbr_black(F, k, s, T, 1) for k, s in zip(K_arr, sv_arr)])
    # lbr_iv signature: (price, F, K, T, flag)  flag: +1=call, -1=put
    lbr_recovered = np.array([lbr_iv(p, F, k, T, 1) for p, k in zip(prices, K_arr)])
    return prices, lbr_recovered


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def wide_grid():
    """Random (K, σ) grid spanning OTM, ATM, near-ITM — typical trading range.

    K ∈ [0.5F, 2.0F], σ ∈ [10%, 150%].  Excludes pathological deep-ITM + low-vol
    cases (where any σ prices to the same intrinsic — ill-conditioned by design).
    Those are handled by the deep_itm_grid fixture and require Experiment D.
    """
    rng = np.random.default_rng(42)
    N = 2000
    F = 100.0
    T = 1.0
    K_arr = F * np.exp(rng.uniform(-0.7, 0.7, N))  # moneyness [0.50F, 2.01F]
    sv_arr = rng.uniform(0.10, 1.50, N)  # σ in [10%, 150%]
    return F, T, K_arr, sv_arr


@pytest.fixture(scope="module")
def extreme_grid():
    """Stress OTM calls (K > F) with wide vol range — tests the upper/lower branches."""
    rng = np.random.default_rng(99)
    N = 500
    F = 100.0
    T = 1.0
    # OTM calls: K > F, i.e. x = ln(F/K) < 0
    K_otm_low = F * np.exp(rng.uniform(0.01, 3.0, N // 2))  # OTM, low vol
    K_otm_high = F * np.exp(rng.uniform(0.01, 2.0, N // 2))  # OTM, high vol
    K_arr = np.concatenate([K_otm_low, K_otm_high])
    sv_arr = np.concatenate(
        [
            rng.uniform(0.05, 0.20, N // 2),  # low vol OTM (small price)
            rng.uniform(1.0, 3.0, N // 2),  # high vol OTM
        ]
    )
    return F, T, K_arr, sv_arr


@pytest.fixture(scope="module")
def deep_itm_grid():
    """Deep ITM calls + low vol: the pathological case Experiment D targets.

    These cases fail with the current Halley solver (max error ~5x) because any σ
    prices to approximately the same intrinsic value — IV inversion is ill-conditioned.
    Jäckel's 4-branch initial guess (normalised space with put-call reduction) handles
    these correctly. This test is expected to FAIL until Experiment D lands.
    """
    rng = np.random.default_rng(77)
    N = 300
    F = 100.0
    T = 1.0
    K_arr = F * np.exp(rng.uniform(-3.0, -1.0, N))  # very ITM: K ∈ [5, 37]
    sv_arr = rng.uniform(0.05, 0.15, N)  # low vol — tiny time value
    return F, T, K_arr, sv_arr


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _run_parity(F, T, K_arr, sv_arr, max_rel_err_threshold: float) -> dict:
    """Core parity check: jackel_iv_black vs lbr_iv oracle."""
    prices, lbr_sv = _lbr_prices_and_ivs(F, K_arr, sv_arr, T)

    valid_price = np.isfinite(prices) & (prices > 0)
    prices_v = prices[valid_price]
    K_v = K_arr[valid_price]
    sv_in_v = sv_arr[valid_price]

    # Call Jäckel directly — undiscounted Black-76 (r=0 → disc_factor=1)
    is_call = np.ones(len(prices_v), dtype=bool)
    fast_sv = _jackel_iv_black(prices_v, F, K_v, T, is_call)

    valid = np.isfinite(fast_sv) & np.isfinite(sv_in_v) & (sv_in_v > 0)
    rel_err = np.abs(fast_sv[valid] - sv_in_v[valid]) / sv_in_v[valid]

    stats = {
        "n_total": len(K_arr),
        "n_valid": int(valid.sum()),
        "max_rel_err": float(rel_err.max()) if len(rel_err) else 0.0,
        "median_rel_err": float(np.median(rel_err)) if len(rel_err) else 0.0,
        "p99_rel_err": float(np.percentile(rel_err, 99)) if len(rel_err) else 0.0,
        "n_nan": int((~np.isfinite(fast_sv)).sum()),
    }

    print(
        f"\n  [jackel] max={stats['max_rel_err']:.2e}  "
        f"median={stats['median_rel_err']:.2e}  "
        f"p99={stats['p99_rel_err']:.2e}  "
        f"NaN={stats['n_nan']}/{stats['n_total']}"
    )

    assert stats["max_rel_err"] < max_rel_err_threshold, (
        f"[jackel] max relative error {stats['max_rel_err']:.2e} "
        f"exceeds threshold {max_rel_err_threshold:.2e}"
    )
    return stats


# Jäckel Householder(3) × 2 — machine precision (lbr oracle itself has ~7e-9 error)
_THRESHOLD = 1e-8


class TestJackelParity:
    """Parity tests for jackel_iv_black vs py_lets_be_rational oracle."""

    def test_wide_grid(self, wide_grid):
        F, T, K_arr, sv_arr = wide_grid
        _run_parity(F, T, K_arr, sv_arr, max_rel_err_threshold=_THRESHOLD)

    def test_extreme_grid(self, extreme_grid):
        F, T, K_arr, sv_arr = extreme_grid
        _run_parity(F, T, K_arr, sv_arr, max_rel_err_threshold=_THRESHOLD)


class TestJackelSelfConsistency:
    """Verify py_lets_be_rational round-trips to machine precision (sanity check)."""

    def test_lbr_round_trip(self, wide_grid):
        F, T, K_arr, sv_arr = wide_grid
        prices, lbr_sv = _lbr_prices_and_ivs(F, K_arr, sv_arr, T)
        # lbr returns 0.0 for edge cases (above b_max, subnormal) — exclude those
        valid = np.isfinite(lbr_sv) & np.isfinite(sv_arr) & (sv_arr > 0) & (lbr_sv > 0)
        rel_err = np.abs(lbr_sv[valid] - sv_arr[valid]) / sv_arr[valid]
        print(f"\n  [lbr oracle self-check] n={valid.sum()}/{len(sv_arr)} max={rel_err.max():.2e}")
        assert rel_err.max() < 1e-8, (
            f"py_lets_be_rational oracle self-consistency failed: {rel_err.max():.2e}"
        )
