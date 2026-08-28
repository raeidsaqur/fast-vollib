<p align="center">
  <img
    src="https://raeidsaqur.github.io/fast-vollib/assets/fast-vollib-icon.png"
    alt="fast-vollib icon"
    width="144"
  />
</p>

<h1 align="center">fast-vollib</h1>

<p align="center">
  Accelerated derivatives pricing, implied volatility, typed instruments, and Monte Carlo
  simulation with NumPy, PyTorch, and JAX backends.
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

**fast-vollib** is an accelerated --- kernel-fused, optimized --- Python library
for Black, Black-Scholes, and Black-Scholes-Merton pricing, implied volatility,
Greeks, typed derivative contracts, and explicit Monte Carlo workflows. It has
pluggable NumPy, PyTorch, and JAX backends and a compatibility-first functional
API modeled on `py_vollib_vectorized`.

---

## What's New?

**v0.2.0 — Processes, simulation, and instruments.** This release extends the
library from pricing kernels to typed contracts and Monte Carlo workflows. It
is the next stable release after v0.1.8; no v0.1.9 release was cut. Three major
public modules are included:

- **`fast_vollib.processes`** provides stateless stochastic-process dynamics,
  beginning with exact geometric Brownian motion on regular or irregular time
  grids.
- **`fast_vollib.simulation`** provides validated, backend-native scenarios and
  an explicit Monte Carlo engine with standard errors, antithetic sampling, and
  PyTorch/JAX automatic differentiation.
- **`fast_vollib.instruments`** provides immutable contract objects, columnar
  option batches, market-input adapters, payoffs, serialization, and capability
  discovery while keeping the existing functional pricing API canonical.

See the [v0.2.0 changelog](https://raeidsaqur.github.io/fast-vollib/changelog/#020-2026-08-28),
[instrument guide](https://raeidsaqur.github.io/fast-vollib/instruments/), and
[simulation guide](https://raeidsaqur.github.io/fast-vollib/simulation/) for the
complete contracts and examples.

**v0.1.8 — Differentiable Jäckel implied volatility.** `fast_vollib.jackel` now
ships autograd wrappers around the machine-precision "Let's Be Rational" solver
for PyTorch and JAX:

- **`implied_volatility_autograd`** (PyTorch) and
  **`implied_volatility_autograd_jax`** (JAX `custom_vjp`) — the forward pass
  runs the full Jäckel solver (~2e-11 relative error); the backward pass applies
  the **implicit function theorem** to the discounted pricing equation
  (`∂σ/∂price = 1/ν`, `∂σ/∂θ = −(∂price/∂θ)/ν` with `ν` = vega), giving exact
  gradients w.r.t. price, spot, strike, maturity, rate, and dividend yield
  without back-propagating through the branch-heavy Householder iterations.
- **Well-defined edge behavior** — invalid domain (below-intrinsic, non-positive
  price / spot / strike, zero maturity) yields `NaN` in both forward and
  backward; an upstream-aware low-vega guard returns a zero gradient when the
  upstream cotangent is exactly zero, so `0 × NaN` chain-rule poisoning cannot
  contaminate valid rows.
- **Training-loop guide** — a new
  [Differentiable Jäckel IV page](https://raeidsaqur.github.io/fast-vollib/differentiable_iv/)
  with PyTorch IV-loss and hybrid price + IV-roundtrip examples, a JAX
  equivalent, and an expanded tutorial notebook.
- Dev-environment: CUDA wheels are now selected by CPU architecture (cu130 on
  x86_64, cu126 on aarch64 / GH200), and an in-tree `testcapi-compat` shim keeps
  `py_lets_be_rational` 1.0.x importable on `python-build-standalone`
  interpreters.

```python
import torch
from fast_vollib.jackel import implied_volatility_autograd

price = price.requires_grad_(True)
sigma = implied_volatility_autograd(price, S, K, t, r, is_call, q=q, model="black_scholes")
sigma.sum().backward()   # exact ∂σ/∂price = 1/vega via the implicit function theorem
```

**v0.1.7 — IV-surface arbitrage evaluation harness.** `fast_vollib.surface` is a
generator-agnostic, backend-pluggable evaluator for implied-volatility surfaces:

- **`IVSurface` / `SurfaceSequence`** containers — build from log-moneyness,
  strikes, total variance, or call prices; numpy / torch / jax arrays are
  preserved with dtype and device.
- **`validate_surface()` → `ArbitrageReport`** — price-space checks (convexity,
  slope, box, calendar) and total-variance checks (`∂_T w ≥ 0`, Durrleman
  `g ≥ 0`) with normalized, cross-model metrics, localized violations, an
  interpolation-artifact vs model-arbitrage split, and a `σ→C→σ'` round-trip
  trust mask.
- **`arbitrage_penalty()`** — a differentiable soft form of the same checks that
  never leaves the input tensor's namespace, so it is autograd-traceable on
  torch/jax and drops directly into a surface generator's training loss.
- **`fast_vollib.diagnostics`** — six publication-quality figures (total-variance
  slices, Durrleman `g`, risk-neutral density, violation heatmap, calendar map,
  trust map) behind the optional `[viz]` extra: `pip install "fast-vollib[viz]"`.

```python
from fast_vollib.surface import IVSurface, validate_surface, arbitrage_penalty

surf = IVSurface.from_logmoneyness(k, T, iv)   # numpy, torch, or jax iv grid
report = surf.validate()                       # → ArbitrageReport (passed, metrics, violations)
loss = pricing_loss + arbitrage_penalty(iv, k, T, forward)  # differentiable soft constraint
```

See the [surface harness guide](https://raeidsaqur.github.io/fast-vollib/surface/)
and the [changelog](https://raeidsaqur.github.io/fast-vollib/changelog/) for details.

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
- **Typed instruments** — immutable vanilla, digital, Asian, barrier, lookback, and variance-swap contracts with strict serialization and columnar option batches
- **Processes and scenarios** — backend-native GBM paths on regular or irregular grids, preserving NumPy, PyTorch, and JAX semantics
- **Explicit Monte Carlo** — opt-in valuation with standard errors, antithetic sampling, capability discovery, and no silent analytic/Monte Carlo fallback

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

Stable releases are published from Git tags to PyPI. After v0.2.0, development
snapshots from `main` use TestPyPI versions such as `0.2.1.devN`.

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

- Tagged releases like `v0.2.0` publish stable builds to PyPI.
- Pushes to `main` publish development snapshots to TestPyPI.
- The package version is derived from Git tags with `hatch-vcs`, so version
  strings are no longer maintained manually in source files for each release.

---

## Contributing

Contributions are welcome. Please open an issue before sending a large pull
request to discuss the change. See [CONTRIBUTING.md](CONTRIBUTING.md) if
present, or follow the standard fork-and-PR workflow.

---

## Citation

If you use **fast-vollib** in your work, please cite:

```bibtex
@misc{saqur2026fastvollibfastimpliedvolatility,
      title={Fast-Vollib: A Fast Implied Volatility Library for Python with PyTorch, JAX, and CUDA Fused-Kernel Backends},
      author={Raeid Saqur},
      year={2026},
      eprint={2604.27210},
      archivePrefix={arXiv},
      primaryClass={q-fin.CP},
      url={https://arxiv.org/abs/2604.27210},
}
```

---

## License

MIT — see [LICENSE](LICENSE).
