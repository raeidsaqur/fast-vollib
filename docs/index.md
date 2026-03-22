# fastiv

`fastiv` is a modern compatibility-focused implied volatility library.

## Goals

- match the high-usage `py_vollib_vectorized` API surface
- provide explicit backend routing for `numpy`, `torch`, and `jax`
- keep outputs compatible with NumPy and pandas by default
- support a clean research loop for iterative backend improvement

## Current status

- working NumPy baseline for pricing, IV, and greeks
- thin Torch and JAX backend adapters
- dataframe helper and `py_vollib` monkeypatch compatibility
- tests and comparison scripts against the local upstream clone
