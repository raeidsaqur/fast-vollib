from __future__ import annotations

from pathlib import Path
import sys
import time

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fast_vollib import get_all_greeks, vectorized_black_scholes, vectorized_implied_volatility
from fast_vollib.config import get_backend


def _sync():
    """Synchronize GPU ops if a CUDA backend is active."""
    backend = get_backend()
    if backend == "torch":
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except ImportError:
            pass
    elif backend == "jax":
        try:
            import jax

            jax.effects_barrier()
        except (ImportError, AttributeError):
            pass


def _make_inputs(n: int):
    flag = np.where(np.arange(n) % 2 == 0, "c", "p")
    s = np.full(n, 100.0)
    k = np.linspace(80.0, 120.0, n)
    t = np.full(n, 30.0 / 365.0)
    r = np.full(n, 0.03)
    sigma = np.full(n, 0.2)
    return flag, s, k, t, r, sigma


def main() -> None:
    active_backend = get_backend()
    print(f"active_backend={active_backend}")

    n = 10_000
    flag, s, k, t, r, sigma = _make_inputs(n)

    # --- warmup (3 iterations to fully JIT-compile torch.compile / Triton kernels) ---
    for _ in range(3):
        prices_warm = vectorized_black_scholes(flag, s, k, t, r, sigma, return_as="numpy")
        _ = vectorized_implied_volatility(prices_warm, s, k, t, r, flag, return_as="numpy")
        _ = get_all_greeks(flag, s, k, t, r, sigma, return_as="dict")
        _sync()

    # --- timed runs (n=10k) ---
    start = time.perf_counter()
    prices = vectorized_black_scholes(flag, s, k, t, r, sigma, return_as="numpy")
    _sync()
    price_seconds = time.perf_counter() - start

    start = time.perf_counter()
    ivs = vectorized_implied_volatility(prices, s, k, t, r, flag, return_as="numpy")
    _sync()
    iv_seconds = time.perf_counter() - start

    start = time.perf_counter()
    greeks = get_all_greeks(flag, s, k, t, r, sigma, return_as="dict")
    _sync()
    greek_seconds = time.perf_counter() - start

    print(f"pricing_seconds={price_seconds:.6f}")
    print(f"iv_seconds={iv_seconds:.6f}")
    print(f"greeks_seconds={greek_seconds:.6f}")
    print(f"sample_iv={ivs[0]:.6f}")
    print(f"sample_delta={greeks['delta'][0]:.6f}")

    # --- numpy reference (always runs) ---
    flag2, s2, k2, t2, r2, sigma2 = _make_inputs(n)
    prices_np = vectorized_black_scholes(
        flag2, s2, k2, t2, r2, sigma2, backend="numpy", return_as="numpy"
    )
    start = time.perf_counter()
    vectorized_black_scholes(flag2, s2, k2, t2, r2, sigma2, backend="numpy", return_as="numpy")
    numpy_price_seconds = time.perf_counter() - start
    start = time.perf_counter()
    vectorized_implied_volatility(
        prices_np, s2, k2, t2, r2, flag2, backend="numpy", return_as="numpy"
    )
    numpy_iv_seconds = time.perf_counter() - start
    start = time.perf_counter()
    get_all_greeks(flag2, s2, k2, t2, r2, sigma2, backend="numpy", return_as="dict")
    numpy_greeks_seconds = time.perf_counter() - start

    print(f"numpy_pricing_seconds={numpy_price_seconds:.6f}")
    print(f"numpy_iv_seconds={numpy_iv_seconds:.6f}")
    print(f"numpy_greeks_seconds={numpy_greeks_seconds:.6f}")

    # --- large batch (100k) for GPU crossover ---
    n_large = 100_000
    flag_l, s_l, k_l, t_l, r_l, sigma_l = _make_inputs(n_large)
    prices_l_warm = vectorized_black_scholes(flag_l, s_l, k_l, t_l, r_l, sigma_l, return_as="numpy")
    _sync()
    start = time.perf_counter()
    prices_l = vectorized_black_scholes(flag_l, s_l, k_l, t_l, r_l, sigma_l, return_as="numpy")
    _sync()
    large_price_seconds = time.perf_counter() - start
    start = time.perf_counter()
    vectorized_implied_volatility(prices_l, s_l, k_l, t_l, r_l, flag_l, return_as="numpy")
    _sync()
    large_iv_seconds = time.perf_counter() - start
    print(
        f"large_n={n_large},pricing_seconds={large_price_seconds:.6f},iv_seconds={large_iv_seconds:.6f}"
    )


if __name__ == "__main__":
    main()
