"""
Jäckel parity: the host ``jackel_iv_black`` against py_lets_be_rational.

Two separate checks on the same quotes:

(a) round trip: the quote is priced by py_lets_be_rational at a generating volatility, and the
    recovered volatility must match it (only meaningful where the quote is well conditioned);
(b) direct agreement: both solvers invert the *same* quote and must agree, which is what an
    oracle comparison means.

Calls and puts, a wide grid, an out-of-the-money stress grid and a deep in-the-money low-volatility
grid; every test also asserts its coverage (how many rows were finite on both sides).
The independent high-precision reference and the scoring protocol live in
``test_iv_adjudicated.py`` / ``test_iv_domain.py``.

Requires py-lets-be-rational (dev dependency group); skipped when absent.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip(
    "py_lets_be_rational",
    reason="py-lets-be-rational not installed — install dev dependencies: uv sync",
)

from fast_vollib.jackel.jackel_iv import jackel_iv_black as _jackel_iv_black

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _lbr_prices_and_ivs(
    F: float, K_arr: np.ndarray, sv_arr: np.ndarray, T: float, flag: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """Prices from py_lets_be_rational at the generating volatility, and its own inversion."""
    from py_lets_be_rational import (
        black as lbr_black,
        implied_volatility_from_a_transformed_rational_guess as lbr_iv,
    )

    prices = np.array([lbr_black(F, k, s, T, flag) for k, s in zip(K_arr, sv_arr)])
    # lbr_iv signature: (price, F, K, T, flag)  flag: +1=call, -1=put
    lbr_recovered = np.array([lbr_iv(p, F, k, T, flag) for p, k in zip(prices, K_arr)])
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
    """Deep in-the-money calls at low volatility: tiny time value, ill-conditioned quotes.

    The generating volatility is not recoverable to full precision from the rounded quote, so
    only direct agreement with py_lets_be_rational on the same quote is asserted here.
    """
    rng = np.random.default_rng(77)
    N = 300
    F = 100.0
    T = 1.0
    K_arr = F * np.exp(rng.uniform(-1.2, -0.4, N))  # deep ITM calls: K in [30, 67]
    sv_arr = rng.uniform(0.15, 0.35, N)  # low vol: small but resolvable time value
    return F, T, K_arr, sv_arr


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _run_parity(
    F, T, K_arr, sv_arr, flag: int, round_trip_tol: float | None, agree_tol: float
) -> dict:
    """(a) round trip vs the generating volatility, (b) agreement with LBR on the same quote."""
    prices, lbr_sv = _lbr_prices_and_ivs(F, K_arr, sv_arr, T, flag)
    # Only quotes that carry volatility information are compared: a time value above 2**-40 of
    # the price (otherwise the quote rounds to intrinsic) and a normalised out-of-the-money price
    # that is a normal double (the Jäckel domain). Excluded rows are counted, not hidden; the
    # public wrappers refuse them with a signal (see test_iv_domain.py).
    intrinsic = np.maximum(flag * (F - K_arr), 0.0)
    time_value = prices - intrinsic
    beta = time_value / np.sqrt(F * K_arr)
    valid_price = (
        np.isfinite(prices) & (prices > 0) & (time_value > 2.0**-40 * prices) & (beta >= _DBL_MIN)
    )
    is_call = np.full(int(valid_price.sum()), flag == 1)
    fast_sv = _jackel_iv_black(prices[valid_price], F, K_arr[valid_price], T, is_call)
    sv_in = sv_arr[valid_price]
    lbr = lbr_sv[valid_price]

    finite = np.isfinite(fast_sv) & (fast_sv > 0)
    both = finite & np.isfinite(lbr) & (lbr > 0)
    stats = {
        "n_total": len(K_arr),
        "n_priced": int(valid_price.sum()),
        "n_finite": int(finite.sum()),
        "n_both": int(both.sum()),
    }
    # Coverage: enough eligible rows, and every eligible quote is inverted by both solvers.
    assert stats["n_priced"] >= _MIN_ELIGIBLE * stats["n_total"], stats
    assert stats["n_finite"] == stats["n_priced"], stats
    assert stats["n_both"] == stats["n_priced"], stats

    agree = np.abs(fast_sv[both] - lbr[both]) / lbr[both]
    stats["max_rel_disagreement"] = float(agree.max())
    assert agree.max() < agree_tol, stats
    if round_trip_tol is not None:
        rt = np.abs(fast_sv[finite] - sv_in[finite]) / sv_in[finite]
        stats["max_rel_round_trip"] = float(rt.max())
        assert rt.max() < round_trip_tol, stats
    return stats


_DBL_MIN = 2.2250738585072014e-308
#: Each grid must keep at least this share of its rows after the eligibility filter.
_MIN_ELIGIBLE = 0.5
#: Round trip vs the generating volatility (well-conditioned grids only).
_ROUND_TRIP = 1e-8
#: Direct agreement with py_lets_be_rational on the same quote.
_AGREE = 1e-10
#: Deep in-the-money quotes are ill-conditioned for both solvers; they must still agree closely.
_AGREE_ILL = 1e-6


@pytest.mark.parametrize("flag", [1, -1], ids=["call", "put"])
class TestJackelParity:
    """jackel_iv_black vs py_lets_be_rational, calls and puts."""

    def test_wide_grid(self, wide_grid, flag):
        F, T, K_arr, sv_arr = wide_grid
        _run_parity(F, T, K_arr, sv_arr, flag, _ROUND_TRIP, _AGREE)

    def test_extreme_grid(self, extreme_grid, flag):
        F, T, K_arr, sv_arr = extreme_grid
        # K > F: out-of-the-money calls; as puts these are deep in the money, where the rounded
        # quote does not determine the generating volatility to 1e-8 (agreement still holds).
        _run_parity(F, T, K_arr, sv_arr, flag, _ROUND_TRIP if flag == 1 else None, _AGREE)

    def test_deep_itm_grid(self, deep_itm_grid, flag):
        F, T, K_arr, sv_arr = deep_itm_grid
        _run_parity(F, T, K_arr, sv_arr, flag, None, _AGREE_ILL)


class TestJackelSelfConsistency:
    """py_lets_be_rational round-trips on the wide grid (sanity check of the comparator)."""

    def test_lbr_round_trip(self, wide_grid):
        F, T, K_arr, sv_arr = wide_grid
        prices, lbr_sv = _lbr_prices_and_ivs(F, K_arr, sv_arr, T)
        valid = np.isfinite(lbr_sv) & (lbr_sv > 0)
        assert valid.sum() == len(sv_arr)
        rel_err = np.abs(lbr_sv - sv_arr) / sv_arr
        assert rel_err.max() < 1e-8, rel_err.max()
