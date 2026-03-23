from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
UPSTREAM_ROOT = PROJECT_ROOT.parent / "py_vollib_vectorized"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(UPSTREAM_ROOT) not in sys.path:
    sys.path.insert(0, str(UPSTREAM_ROOT))

from fast_vollib import vectorized_black_scholes, vectorized_implied_volatility


def main() -> None:
    try:
        from py_vollib_vectorized.implied_volatility import vectorized_implied_volatility as upstream_iv
        from py_vollib_vectorized.models import vectorized_black_scholes as upstream_bs
    except Exception as exc:
        print(f"upstream_import_error={type(exc).__name__}: {exc}")
        return

    flag = np.array(["c", "p", "c", "p"])
    s = np.array([100.0, 100.0, 95.0, 105.0])
    k = np.array([90.0, 110.0, 100.0, 100.0])
    t = np.array([0.25, 0.25, 0.5, 0.5])
    r = np.array([0.01, 0.01, 0.03, 0.03])
    sigma = np.array([0.2, 0.2, 0.35, 0.15])

    fast_prices = vectorized_black_scholes(flag, s, k, t, r, sigma, return_as="numpy")
    upstream_prices = upstream_bs(flag, s, k, t, r, sigma, return_as="numpy")

    fast_ivs = vectorized_implied_volatility(fast_prices, s, k, t, r, flag, return_as="numpy")
    upstream_ivs = upstream_iv(upstream_prices, s, k, t, r, flag, return_as="numpy")

    print(f"max_price_diff={np.max(np.abs(fast_prices - upstream_prices)):.8f}")
    print(f"max_iv_diff={np.nanmax(np.abs(fast_ivs - upstream_ivs)):.8f}")


if __name__ == "__main__":
    main()
