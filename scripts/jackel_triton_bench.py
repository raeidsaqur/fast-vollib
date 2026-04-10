"""I-7 Triton kernel correctness + benchmark.

Canonical 100k near-ATM grid. Checks max_rel_err vs py_lets_be_rational,
then measures wall-clock and GPU time with CUDA events.
"""

from __future__ import annotations

import math
from pathlib import Path
import sys
import time

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fast_vollib.jackel.triton_kernels import jackel_iv_triton

# ---------------------------------------------------------------------------
# Canonical benchmark grid (same as prior experiments)
# ---------------------------------------------------------------------------


def make_canonical_grid(N: int = 100_000, seed: int = 42):
    rng = np.random.default_rng(seed)
    F = 100.0
    T = 1.0
    K = F * np.exp(rng.uniform(-0.3, 0.3, N))
    iv_true = rng.uniform(0.10, 0.60, N)
    return F, T, K, iv_true


def compute_black_prices(F, K, T, sigma, is_call=True):
    """Undiscounted Black-76 call prices via py_lets_be_rational."""
    from py_lets_be_rational import black as lbr_black

    flag = 1 if is_call else -1
    return np.array([lbr_black(F, k, s, T, flag) for k, s in zip(K, sigma)])


# ---------------------------------------------------------------------------
# Correctness check
# ---------------------------------------------------------------------------


def check_correctness(F, T, K, iv_true, prices_np, device):
    price_t = torch.as_tensor(prices_np, dtype=torch.float64, device=device)
    K_t = torch.as_tensor(K, dtype=torch.float64, device=device)

    # Warmup (compile)
    _ = jackel_iv_triton(price_t[:512], F, K_t[:512], T, is_call=True)
    torch.cuda.synchronize()

    sigma_t = jackel_iv_triton(price_t, F, K_t, T, is_call=True)
    torch.cuda.synchronize()
    sigma = sigma_t.cpu().numpy()

    valid = np.isfinite(sigma) & np.isfinite(iv_true) & (iv_true > 0)
    rel_err = np.abs(sigma[valid] - iv_true[valid]) / iv_true[valid]
    n_nan = int(np.sum(~np.isfinite(sigma)))

    print(f"  n_valid={valid.sum()}/{len(iv_true)}, n_nan={n_nan}")
    print(f"  max_rel_err={rel_err.max():.3e}  median={np.median(rel_err):.3e}")
    return float(rel_err.max()), n_nan


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


def benchmark(F, T, K, prices_np, device, n_warmup=5, n_timed=20):
    price_t = torch.as_tensor(prices_np, dtype=torch.float64, device=device)
    K_t = torch.as_tensor(K, dtype=torch.float64, device=device)

    # Warmup
    for _ in range(n_warmup):
        _ = jackel_iv_triton(price_t, F, K_t, T, is_call=True)
    torch.cuda.synchronize()

    # Wall-clock
    wall_times = []
    for _ in range(n_timed):
        t0 = time.perf_counter()
        _ = jackel_iv_triton(price_t, F, K_t, T, is_call=True)
        torch.cuda.synchronize()
        wall_times.append(time.perf_counter() - t0)

    # CUDA events
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    gpu_times = []
    for _ in range(n_timed):
        start_evt.record()
        _ = jackel_iv_triton(price_t, F, K_t, T, is_call=True)
        end_evt.record()
        torch.cuda.synchronize()
        gpu_times.append(start_evt.elapsed_time(end_evt))

    wall_ms = np.median(wall_times) * 1e3
    gpu_ms = np.median(gpu_times)
    print(f"  wall_clock_ms={wall_ms:.3f}  gpu_kernel_ms={gpu_ms:.3f}")
    return wall_ms, gpu_ms


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    device = torch.device("cuda")
    N = 100_000

    print(f"[I-7] Triton Jäckel IV benchmark — N={N} on {torch.cuda.get_device_name(0)}")

    print("\n-- Building canonical grid --")
    F, T, K, iv_true = make_canonical_grid(N)
    print("  Computing reference prices via py_lets_be_rational ...")
    prices_np = compute_black_prices(F, K, T, iv_true, is_call=True)

    print("\n-- Correctness --")
    max_rel_err, n_nan = check_correctness(F, T, K, iv_true, prices_np, device)

    threshold = 1e-8
    if max_rel_err < threshold and n_nan == 0:
        print(f"  PASS  max_rel_err={max_rel_err:.3e} < {threshold:.0e}, n_nan=0")
    else:
        print(f"  FAIL  max_rel_err={max_rel_err:.3e}, n_nan={n_nan}")
        sys.exit(1)

    print("\n-- Benchmark --")
    wall_ms, gpu_ms = benchmark(F, T, K, prices_np, device)
    print(
        f"\n  SUMMARY: 100k_iv_wall={wall_ms:.2f}ms  gpu_compute={gpu_ms:.2f}ms  max_rel_err={max_rel_err:.2e}"
    )

    target_ms = 0.636
    if gpu_ms <= target_ms:
        print(f"  TARGET MET: {gpu_ms:.3f}ms ≤ {target_ms}ms")
    else:
        print(f"  Target {target_ms}ms NOT met (gpu_compute={gpu_ms:.3f}ms)")


if __name__ == "__main__":
    main()
