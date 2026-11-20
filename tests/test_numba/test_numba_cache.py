# Author: Raeid Saqur
# Date: 2026-11-19
"""The numba kernels are served from numba's on-disk cache in a new process.

Kernels nested inside a builder function are closures over each other: every process recompiled
them (seconds on the first call) and wrote further cache entries. At module scope, a second
process loads them and the cache does not grow.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest

pytest.importorskip("numba")

ROOT = Path(__file__).resolve().parents[2]

SNIPPET = """
import warnings
import numpy as np
warnings.simplefilter("ignore")
from fast_vollib import fast_black_scholes, fast_implied_volatility, get_all_greeks
k = np.array([90.0, 100.0, 110.0])
p = fast_black_scholes("c", 100.0, k, 1.0, 0.02, 0.2, backend="numba", return_as="numpy")
fast_implied_volatility(p, 100.0, k, 1.0, 0.02, "c", backend="numba", return_as="numpy")
get_all_greeks("c", 100.0, k, 1.0, 0.02, 0.2, backend="numba", return_as="dict")
"""


def _entries(cache: Path) -> set[str]:
    return {str(p.relative_to(cache)) for p in cache.rglob("*") if p.suffix in (".nbi", ".nbc")}


def test_second_process_reuses_the_kernel_cache(tmp_path: Path) -> None:
    env = {
        **os.environ,
        "NUMBA_CACHE_DIR": str(tmp_path),
        "PYTHONPATH": str(ROOT / "src"),
        "CUDA_VISIBLE_DEVICES": "",
        "JAX_PLATFORMS": "cpu",
    }
    for _ in range(2):
        subprocess.run([sys.executable, "-c", SNIPPET], env=env, check=True, timeout=600)
        if _ == 0:
            first = _entries(tmp_path)
    assert first, "numba wrote no cache entries"
    assert _entries(tmp_path) == first
