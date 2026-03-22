# fastiv

`fastiv` is a modern open-source Black, Black-Scholes, and Black-Scholes-Merton
pricing and implied-volatility library with a compatibility-first API modeled on
`py_vollib_vectorized`.

The repository is intentionally scaffolded for iterative development:

- a functional NumPy baseline
- explicit backend routing for `numpy`, `torch`, and `jax`
- `py_vollib` monkeypatch compatibility via `patch_py_vollib()`
- dataframe helpers for pricing, IV, and greeks
- benchmark and compatibility scripts against the local `py_vollib_vectorized`

## Install

```bash
uv sync
```

Optional extras:

```bash
uv sync --extra torch
uv sync --extra jax
uv sync --extra cuda
uv sync --group docs --group bench
```

Phoenix GPU install:

```bash
uv sync --extra cuda --group bench
```

On Linux/Phoenix, the `cuda` extra is the combined GPU path:

- PyTorch from the CUDA 13.0 wheel index
- JAX with CUDA 13 support
- optional RAPIDS benchmarking packages

## Public API

```python
from fastiv import (
    vectorized_black,
    vectorized_black_scholes,
    vectorized_black_scholes_merton,
    vectorized_implied_volatility,
    vectorized_implied_volatility_black,
    vectorized_delta,
    vectorized_gamma,
    vectorized_rho,
    vectorized_theta,
    vectorized_vega,
    get_all_greeks,
    price_dataframe,
    patch_py_vollib,
)
```

All compatibility functions preserve `py_vollib_vectorized` argument order and
return-shape semantics. New optional kwargs:

- `backend="auto"`
- `return_native=False`

`backend="auto"` resolves in this order:

1. explicit kwarg
2. `FASTIV_BACKEND`
3. `torch` when CUDA is available
4. `jax` when installed
5. `numpy`

## Notes

This first pass provides a serious scaffold and a working NumPy baseline. The
Torch and JAX layers are intentionally thin compatibility backends so
`autoresearch-iv` has a concrete target for iterative improvement.
