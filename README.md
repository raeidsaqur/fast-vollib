<p align="center">
  <img
    src="https://raeidsaqur.github.io/fast-vollib/assets/fast-vollib-icon.png"
    alt="fast-vollib icon"
    width="144"
  />
</p>

<h1 align="center">fast-vollib</h1>

<p align="center">
  Accelerated Black-Scholes pricing, implied volatility, and Greeks library with pluggable
  NumPy, PyTorch, and JAX backends.
</p>

<p align="center">
  <a href="https://pypi.org/project/fast-vollib/">
    <img src="https://img.shields.io/pypi/v/fast-vollib.svg" alt="PyPI version" />
  </a>
  <a href="https://pypi.org/project/fast-vollib/">
    <img src="https://img.shields.io/pypi/pyversions/fast-vollib.svg" alt="Python versions" />
  </a>
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT" />
  </a>
  <a href="https://github.com/raeidsaqur/fast-vollib/actions/workflows/tests.yml">
    <img src="https://github.com/raeidsaqur/fast-vollib/actions/workflows/tests.yml/badge.svg" alt="Tests" />
  </a>
  <a href="https://raeidsaqur.github.io/fast-vollib/">
    <img src="https://github.com/raeidsaqur/fast-vollib/actions/workflows/docs.yml/badge.svg" alt="Docs" />
  </a>
</p>

**fast-vollib** is an accelerated --- kernel-fused, optimized --- Python library for Black, Black-Scholes, and
Black-Scholes-Merton option pricing, implied volatility solving, and Greeks —
with pluggable NumPy, PyTorch, and JAX backends and a compatibility-first API
modeled on `py_vollib_vectorized`.

---

## Features

- **Three pricing models** — Black-76, Black-Scholes, Black-Scholes-Merton
- **Vectorized IV solver** — Halley's method with compiled bisection fallback
- **Full Greeks** — delta, gamma, theta, rho, vega; all five in one `get_all_greeks` call
- **Pluggable backends** — NumPy (default), PyTorch (CUDA), JAX (JIT)
- **Automatic backend selection** — prefers CUDA > JAX > NumPy
- **DataFrame-native** — `price_dataframe` works directly on a `pandas.DataFrame`
- **Drop-in compatibility** — `patch_py_vollib()` and `patch_py_vollib_vectorized()` patch the scalar and vectorized upstream namespaces
- **Surface arbitrage harness** — `fast_vollib.surface` scores generated IV surfaces for static arbitrage with normalized, cross-model metrics and a differentiable training penalty ([guide](https://raeidsaqur.github.io/fast-vollib/surface/))

---

## Install

```bash
pip install fast-vollib
```

**Optional extras:**

```bash
pip install "fast-vollib[torch]"       # PyTorch backend
pip install "fast-vollib[jax]"         # JAX backend
pip install "fast-vollib[torch,jax]"   # both backends
```

### Development snapshots from TestPyPI

Stable releases are published from Git tags to PyPI. Development snapshots are
available via to TestPyPI versions such as `0.1.2.dev3`.

```bash
pip install --pre \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  fast-vollib
```

Use the dev TestPyPI channel only if you want nightly or dev builds only.

---

## Quick start

```python
import numpy as np
import fast_vollib

# Price a batch of European options
prices = fast_vollib.fast_black_scholes(
    flag=np.array(["c", "c", "p"]),
    S=100.0,
    K=np.array([95, 100, 105]),
    t=0.25,
    r=0.05,
    sigma=0.20,
    return_as="numpy",
)

# Recover implied volatility
iv = fast_vollib.fast_implied_volatility(
    price=prices,
    S=100.0,
    K=np.array([95, 100, 105]),
    t=0.25,
    r=0.05,
    flag=np.array(["c", "c", "p"]),
    return_as="numpy",
)

# All Greeks in one call (returns a pandas DataFrame)
greeks = fast_vollib.get_all_greeks(
    flag=np.array(["c", "p"]),
    S=100.0, K=100.0, t=0.25, r=0.05, sigma=0.20,
)
```

### DataFrame helper

```python
import pandas as pd

df = pd.DataFrame({
    "flag": ["c", "p"],
    "S": [100, 100],
    "K": [100, 100],
    "t": [0.25, 0.25],
    "r": [0.05, 0.05],
    "sigma": [0.20, 0.20],
})

result = fast_vollib.price_dataframe(
    df,
    flag_col="flag",
    underlying_price_col="S",
    strike_col="K",
    annualized_tte_col="t",
    riskfree_rate_col="r",
    sigma_col="sigma",
)
# Columns: Price, delta, gamma, theta, rho, vega
```

### Drop-in `py_vollib_vectorized` replacement

The [`py_vollib_vectorized`](https://github.com/marcdemers/py_vollib_vectorized)
API can be kept intact in your codebase via the included monkey-patching helper.

```python
import fast_vollib
fast_vollib.patch_py_vollib_vectorized()

# All py_vollib_vectorized imports now use fast_vollib transparently
from py_vollib_vectorized import vectorized_black_scholes
```

---

## Backend selection

```python
# Automatic (CUDA > JAX > NumPy)
fast_vollib.get_backend()        # e.g. "torch"

# Set for the session
fast_vollib.set_backend("numpy")

# Override per call
price = fast_vollib.fast_black_scholes(..., backend="jax")
```

`backend="auto"` resolution order:
1. Explicit `backend=` kwarg
2. `fast_vollib.set_backend()` override
3. `FAST_VOLLIB_BACKEND` environment variable
4. `torch` when `torch.cuda.is_available()`
5. `jax` when installed
6. `numpy`

---

## Public API

```python
from fast_vollib import (
    # Pricing
    fast_black,
    fast_black_scholes,
    fast_black_scholes_merton,
    # Implied volatility
    fast_implied_volatility,
    fast_implied_volatility_black,
    # Greeks (compatibility aliases)
    vectorized_delta,
    vectorized_gamma,
    vectorized_rho,
    vectorized_theta,
    vectorized_vega,
    get_all_greeks,
    # Utilities
    price_dataframe,
    patch_py_vollib,
    patch_py_vollib_vectorized,
    get_backend,
    set_backend,
)
```

Full documentation: **[raeidsaqur.github.io/fast-vollib](https://raeidsaqur.github.io/fast-vollib/)**

---

## Development

```bash
git clone https://github.com/raeidsaqur/fast-vollib.git
cd fast-vollib

uv sync --all-groups --extra torch --extra jax   # all deps + both backends
uv run pytest               # run tests
ruff check . --fix          # lint
ruff format .               # format
uv run mkdocs serve         # local docs server → http://localhost:8000
```

### Release model

- Tagged releases like `v0.1.2` publish stable builds to PyPI.
- PRs on `main` publish development snapshots to TestPyPI.
- The package version is derived from Git tags with `hatch-vcs`, so version
  strings are no longer maintained manually in source files for each release

---

## Contributing

Contributions are welcome. Please open an issue before sending a large pull
request to discuss the change. See [CONTRIBUTING.md](CONTRIBUTING.md) if
present, or follow the standard fork-and-PR workflow.

---

## License

MIT — see [LICENSE](LICENSE).
