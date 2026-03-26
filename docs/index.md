# fast-vollib

**fast-vollib** is a fast, modern Python library for Black, Black-Scholes, and Black-Scholes-Merton option pricing, implied volatility solving, and Greeks — with pluggable NumPy, PyTorch, and JAX backends.

Stable tagged releases are published to PyPI. Development snapshots from each
`main` commit are published to TestPyPI with VCS-derived `.devN` versions.

[![PyPI version](https://img.shields.io/pypi/v/fast-vollib.svg)](https://pypi.org/project/fast-vollib/)
[![Python](https://img.shields.io/pypi/pyversions/fast-vollib.svg)](https://pypi.org/project/fast-vollib/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/raeidsaqur/fast-vollib/actions/workflows/tests.yml/badge.svg)](https://github.com/raeidsaqur/fast-vollib/actions/workflows/tests.yml)
[![Docs](https://github.com/raeidsaqur/fast-vollib/actions/workflows/docs.yml/badge.svg)](https://raeidsaqur.github.io/fast-vollib/)

---

## Features

- **Three pricing models** — Black-76, Black-Scholes, Black-Scholes-Merton
- **Vectorized IV solver** — Newton-Raphson with bisection fallback; handles large option chains efficiently
- **Full Greeks** — delta, gamma, theta, rho, vega, and `get_all_greeks` in one call
- **Pluggable backends** — NumPy (default), PyTorch (GPU-accelerated), JAX
- **DataFrame-native** — `price_dataframe` works directly on pandas DataFrames
- **Drop-in compatibility** — `patch_py_vollib()` replaces `py_vollib` / `py_vollib_vectorized` at runtime
- **Automatic backend selection** — prefers CUDA-capable PyTorch > JAX > NumPy

---

## Quick example

```python
import numpy as np
import fast_vollib

# Price a batch of European calls with Black-Scholes
prices = fast_vollib.fast_black_scholes(
    flag=["c", "c", "p"],
    S=[100, 105, 95],
    K=[100, 100, 100],
    t=[0.25, 0.25, 0.25],
    r=[0.05, 0.05, 0.05],
    sigma=[0.20, 0.20, 0.20],
    return_as="numpy",
)

# Recover implied volatility
iv = fast_vollib.fast_implied_volatility(
    price=prices,
    S=[100, 105, 95],
    K=[100, 100, 100],
    t=[0.25, 0.25, 0.25],
    r=[0.05, 0.05, 0.05],
    flag=["c", "c", "p"],
    return_as="numpy",
)

# Compute all Greeks at once
greeks = fast_vollib.get_all_greeks(
    flag=["c", "c", "p"],
    S=[100, 105, 95],
    K=[100, 100, 100],
    t=[0.25, 0.25, 0.25],
    r=[0.05, 0.05, 0.05],
    sigma=[0.20, 0.20, 0.20],
)
# returns a pandas DataFrame with columns: delta, gamma, theta, rho, vega
```

---

## Navigation

| Section | Description |
|---|---|
| [Installation](installation.md) | Install fast-vollib with pip, uv, or conda |
| [Quick Start](quickstart.md) | More complete worked examples |
| [Backend Selection](backends.md) | How to choose between NumPy, PyTorch, and JAX |
| [Compatibility](compatibility.md) | Drop-in `py_vollib` replacement guide |
| [Benchmarks](benchmarks.md) | Performance numbers and how to reproduce them |
| [API Reference](api.md) | Complete function signatures and parameters |
| [Changelog](changelog.md) | Version history |
