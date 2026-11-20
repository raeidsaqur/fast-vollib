# Author: Raeid Saqur
# Date: 2026-11-19
"""Cross-backend consistency: prices, Greeks and implied volatility (both solvers), float64.

Every available backend is compared with NumPy on one coarse batch (calls and puts, r < 0, = 0,
> 0, q != 0, all three models), and the output dtype is checked. float32 bounds are provisional
in the IV protocol and are not asserted here.
"""

from __future__ import annotations

import numpy as np
import pytest

from fast_vollib import backends, fast_implied_volatility
from fast_vollib.api import get_all_greeks
from fast_vollib.models import fast_black, fast_black_scholes, fast_black_scholes_merton

BACKENDS = [b for b in ("numba", "torch", "jax") if b in backends.available_backends()]
MODELS = ("black", "black_scholes", "black_scholes_merton")


def _batch():
    rng = np.random.default_rng(20261119)
    n = 400
    flag = np.where(rng.random(n) < 0.5, "c", "p")
    s = rng.uniform(50.0, 150.0, n)
    k = s * np.exp(rng.uniform(-0.8, 0.8, n))
    t = np.exp(rng.uniform(np.log(0.02), np.log(5.0), n))
    r = rng.choice([-0.01, 0.0, 0.04], n)
    q = rng.uniform(0.0, 0.05, n)
    sigma = rng.uniform(0.05, 1.5, n)
    return flag, s, k, t, r, q, sigma


def _price(model, backend, flag, s, k, t, r, q, sigma):
    kw = dict(return_as="numpy", backend=backend)
    if model == "black":
        return fast_black(flag, s, k, t, r, sigma, **kw)
    if model == "black_scholes":
        return fast_black_scholes(flag, s, k, t, r, sigma, **kw)
    return fast_black_scholes_merton(flag, s, k, t, r, sigma, q, **kw)


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("backend", BACKENDS)
def test_prices_and_greeks_match_numpy(model: str, backend: str) -> None:
    flag, s, k, t, r, q, sigma = _batch()
    base = _price(model, "numpy", flag, s, k, t, r, q, sigma)
    other = _price(model, backend, flag, s, k, t, r, q, sigma)
    assert np.asarray(other).dtype == np.float64
    np.testing.assert_allclose(other, base, rtol=1e-11, atol=1e-12)
    qq = q if model == "black_scholes_merton" else None
    g0 = get_all_greeks(flag, s, k, t, r, sigma, qq, model=model, return_as="dict", backend="numpy")
    g1 = get_all_greeks(flag, s, k, t, r, sigma, qq, model=model, return_as="dict", backend=backend)
    for name in g0:
        np.testing.assert_allclose(g1[name], g0[name], rtol=1e-10, atol=1e-12, err_msg=name)


@pytest.mark.parametrize("solver", ["halley", "jackel"])
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("backend", BACKENDS)
def test_implied_volatility_matches_numpy(model: str, backend: str, solver: str) -> None:
    if solver == "jackel" and backend == "numba":
        pytest.skip("the Jäckel solver has no numba implementation (UnsupportedSolverError)")
    flag, s, k, t, r, q, sigma = _batch()
    price = _price(model, "numpy", flag, s, k, t, r, q, sigma)
    qq = q if model == "black_scholes_merton" else None
    kw = dict(model=model, solver=solver, return_as="numpy", on_error="ignore")
    base = fast_implied_volatility(price, s, k, t, r, flag, qq, backend="numpy", **kw)
    other = fast_implied_volatility(price, s, k, t, r, flag, qq, backend=backend, **kw)
    assert np.asarray(other).dtype == np.float64
    np.testing.assert_array_equal(np.isnan(other), np.isnan(base))
    # Agreement is bounded by the quote's own conditioning: one ulp of the price moves sigma by
    # ulp(price) / vega, which on ill-conditioned rows is far above 1e-9 relative.
    raw_vega = (
        100.0
        * get_all_greeks(
            flag,
            s,
            k,
            t,
            r,
            np.where(np.isnan(base), 0.2, base),
            qq,
            model=model,
            return_as="dict",
            backend="numpy",
        )["vega"]
    )
    with np.errstate(divide="ignore"):
        tol = 1e-9 * np.abs(base) + 4.0 * np.spacing(price) / np.abs(raw_vega)
    ok = np.isnan(base) | (np.abs(other - base) <= tol)
    assert ok.all(), np.flatnonzero(~ok)
