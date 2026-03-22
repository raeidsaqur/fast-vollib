from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fastiv import get_all_greeks, vectorized_black_scholes, vectorized_implied_volatility


def main() -> None:
    n = 10_000
    flag = np.where(np.arange(n) % 2 == 0, "c", "p")
    s = np.full(n, 100.0)
    k = np.linspace(80.0, 120.0, n)
    t = np.full(n, 30.0 / 365.0)
    r = np.full(n, 0.03)
    sigma = np.full(n, 0.2)

    start = time.perf_counter()
    prices = vectorized_black_scholes(flag, s, k, t, r, sigma, return_as="numpy")
    price_seconds = time.perf_counter() - start

    start = time.perf_counter()
    ivs = vectorized_implied_volatility(prices, s, k, t, r, flag, return_as="numpy")
    iv_seconds = time.perf_counter() - start

    start = time.perf_counter()
    greeks = get_all_greeks(flag, s, k, t, r, sigma, return_as="dict")
    greek_seconds = time.perf_counter() - start

    print(f"pricing_seconds={price_seconds:.6f}")
    print(f"iv_seconds={iv_seconds:.6f}")
    print(f"greeks_seconds={greek_seconds:.6f}")
    print(f"sample_iv={ivs[0]:.6f}")
    print(f"sample_delta={greeks['delta'][0]:.6f}")


if __name__ == "__main__":
    main()
