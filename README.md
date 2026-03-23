# fast-vollib

[![PyPI version](https://img.shields.io/pypi/v/fast-vollib.svg)](https://pypi.org/project/fast-vollib/)
[![Python](https://img.shields.io/pypi/pyversions/fast-vollib.svg)](https://pypi.org/project/fast-vollib/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/raeid-saqur/fast-vollib/actions/workflows/tests.yml/badge.svg)](https://github.com/raeid-saqur/fast-vollib/actions/workflows/tests.yml)
[![Docs](https://github.com/raeid-saqur/fast-vollib/actions/workflows/docs.yml/badge.svg)](https://raeid-saqur.github.io/fast-vollib/)

**fast-vollib** is a modern Python library for Black, Black-Scholes, and
Black-Scholes-Merton option pricing, implied volatility solving, and Greeks —
with pluggable NumPy, PyTorch, and JAX backends and a compatibility-first API
modeled on `py_vollib_vectorized`.

---

## Features

- **Three pricing models** — Black-76, Black-Scholes, Black-Scholes-Merton
- **Vectorized IV solver** — Newton-Raphson with compiled bisection fallback (~10 M solves/s on CPU)
- **Full Greeks** — delta, gamma, theta, rho, vega; all five in one `get_all_greeks` call
- **Pluggable backends** — NumPy (default), PyTorch (CUDA), JAX (JIT)
- **Automatic backend selection** — prefers CUDA > JAX > NumPy
- **DataFrame-native** — `price_dataframe` works directly on a `pandas.DataFrame`
- **Drop-in replacement** — `patch_py_vollib()` replaces `py_vollib` at runtime with no code changes

---

## Install

```bash
pip install fast-vollib
```

**Optional extras:**

```bash
pip install "fast-vollib[torch]"   # PyTorch backend
pip install "fast-vollib[jax]"     # JAX backend
```

---

## Quick start

```python
import numpy as np
import fast_vollib

# Price a batch of European options
prices = fast_vollib.vectorized_black_scholes(
    flag=np.array(["c", "c", "p"]),
    S=100.0,
    K=np.array([95, 100, 105]),
    t=0.25,
    r=0.05,
    sigma=0.20,
    return_as="numpy",
)

# Recover implied volatility
iv = fast_vollib.vectorized_implied_volatility(
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

### Drop-in `py_vollib` replacement

```python
import fast_vollib
fast_vollib.patch_py_vollib()

# All py_vollib imports now use fast_vollib transparently
from py_vollib.black_scholes import black_scholes
```

---

## Backend selection

```python
# Automatic (CUDA > JAX > NumPy)
fast_vollib.get_backend()        # e.g. "torch"

# Set for the session
fast_vollib.set_backend("numpy")

# Override per call
price = fast_vollib.vectorized_black_scholes(..., backend="jax")
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
    vectorized_black,
    vectorized_black_scholes,
    vectorized_black_scholes_merton,
    # Implied volatility
    vectorized_implied_volatility,
    vectorized_implied_volatility_black,
    # Greeks
    vectorized_delta,
    vectorized_gamma,
    vectorized_rho,
    vectorized_theta,
    vectorized_vega,
    get_all_greeks,
    # Utilities
    price_dataframe,
    patch_py_vollib,
    get_backend,
    set_backend,
)
```

Full documentation: **[raeid-saqur.github.io/fast-vollib](https://raeid-saqur.github.io/fast-vollib/)**

---

## Development

```bash
git clone https://github.com/raeid-saqur/fast-vollib.git
cd fast-vollib

uv sync --all-groups        # install all deps (CPU)
uv run pytest               # run tests
ruff check . --fix          # lint
ruff format .               # format
uv run mkdocs serve         # local docs server → http://localhost:8000
```

---

## Contributing

Contributions are welcome. Please open an issue before sending a large pull
request to discuss the change. See [CONTRIBUTING.md](CONTRIBUTING.md) if
present, or follow the standard fork-and-PR workflow.

---

## License

MIT — see [LICENSE](LICENSE).
