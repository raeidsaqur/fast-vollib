# Changelog

All notable changes to this project will be documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Only tagged public releases are recorded here. Development snapshots published
from `main` to TestPyPI use VCS-derived `.devN` versions and are not tracked as
separate changelog entries.

---

## [0.1.3] — 2026-04-04
- Added backend_parity tests for torch
- Updated tutorial notebook with Mac MPS backend (for Apple silicon chips).


## [Unreleased]

### Added

- **Opt-in shape-aware runtime type checking** — pure-annotation layer
  (`jaxtyping` + `beartype`) applied to the public API (`fast_black`,
  `fast_black_scholes`, `fast_black_scholes_merton`, `fast_implied_volatility`,
  `fast_implied_volatility_black`, `get_all_greeks`, `price_dataframe`,
  the `vectorized_*` Greeks) and to the four backend dispatch entry points
  (`price_*`, `greeks`, `implied_volatility`).
    - Annotations are stored as PEP 563 strings (every annotated module uses
      `from __future__ import annotations`) — **zero runtime cost** when not
      enabled.
    - Runtime checking is scoped to the public dispatch layer only via
      `fast_vollib._typing.enable_runtime_checks()`.  Inner `torch.compile`
      closures, Triton kernels, Numba `@njit` factories, and JAX
      `@jax.jit`-traced functions are **never decorated or rewritten**, so the
      hot paths are bit-identical to the un-annotated build (verified with
      sha256 fingerprints of the `jackel_iv` numpy / torch / triton outputs
      before and after).
    - Install via the new `[typecheck]` extra:
      `pip install "fast-vollib[typecheck]"` (adds `jaxtyping>=0.2` and
      `beartype>=0.18`).  Default installs do **not** pull either package
      into `sys.modules`.
- **`fast_vollib.jackel` module** — full implementation of Peter Jäckel's
  *"Let's Be Rational"* (2016) algorithm with four backends:
    - `jackel_iv_black` — NumPy + Numba (six parallel kernels; ~8.5 ms / 100k)
    - `jackel_iv_black_torch` — PyTorch with `torch.compile(dynamic=True)` (~2.7 ms GPU compute)
    - `jackel_iv_black_jax` — JAX `lax.fori_loop` + `@jax.jit` (~2.4 ms GPU compute)
    - `jackel_iv_triton` — single-pass Triton kernel; entire pipeline in registers (**0.056 ms GPU compute / 100k**)
- Dedicated test package `tests/test_jackel/` with parity tests against
  `py_lets_be_rational` (oracle); max relative error < 10⁻⁸.
- `py-lets-be-rational` added to the `dev` dependency group so CI installs
  the oracle automatically.
- `scripts/jackel_triton_bench.py` — correctness + CUDA-event timing script
  for the Triton kernel.

- **Numba backend** (`backend="numba"`): JIT-compiled CPU kernels via
  `@numba.njit(parallel=True)`.  Pricing, Greeks, and the full
  Halley+bisection IV solver run as a single native-code dispatch per batch.
  Enabled by `pip install "fast-vollib[numba]"` (requires `numba>=0.60.0`).
  Kernels are compiled on first call and cached to `__pycache__` for
  subsequent runs.
- Isolated numba test suite under `tests/numba/` (skipped automatically when
  numba is not installed).


### Fixed

- `get_all_greeks(..., return_native=True)` now returns native torch/JAX arrays
  instead of formatting the result back into pandas containers.
- Below-intrinsic IV handling now honors `on_error=` consistently across the
  NumPy, PyTorch, and JAX backends.
- The `compare_against_py_vollib_vectorized.py` helper now imports the current
  upstream `vectorized_*` entry points correctly.

---

## [0.1.2] — 2026-03-26

Release focused on packaging automation, public release channels, and broader
compatibility coverage.

### Added

- Development and nightly-style build publishing to TestPyPI from `main`
  using trusted publishing via GitHub Actions OIDC.
- Additional test coverage for backend parity, packaging consistency, and
  release workflow support.
- Monkey-patching support for baseline replacement workflows, including
  `py_vollib_vectorized` compatibility-oriented patch helpers.

### Changed

- Versioning is now derived from Git tags via VCS-based build metadata, so
  stable PyPI releases are tag-driven and development snapshots use `.devN`
  versions automatically.

---

## [0.1.1] — 2026-03-26

First public release after the initial beta version. This release improves
runtime correctness, tightens packaging and typing metadata, and aligns backend
behaviour across NumPy, PyTorch, and JAX.

### Fixed

- `price_dataframe` now raises explicit `ValueError` exceptions instead of
  relying on bare `assert` guards.
- The JAX implied-volatility backend now returns `NaN` for below-intrinsic and
  zero-price inputs, matching NumPy and PyTorch behaviour.
- Backend configuration is now standardized on `FAST_VOLLIB_BACKEND`;

### Added

- `py.typed` marker for PEP 561-compatible downstream type-checking.
- Packaging metadata improvements, including explicit mypy configuration and
  updated Python version support metadata.

---

## [0.1.0] — 2026-03-22

### Features

- **Pricing** — `fast_black`, `fast_black_scholes`,
  `fast_black_scholes_merton` with full NumPy vectorization and
  broadcasting.
- **Implied Volatility** — `fast_implied_volatility` and
  `fast_implied_volatility_black` using Halley's method with a compiled
  bisection fallback (~10 M solves / s on CPU).
- **Greeks** — `vectorized_delta`, `vectorized_gamma`, `vectorized_theta`,
  `vectorized_rho`, `vectorized_vega`, and `get_all_greeks`.
- **Backend routing** — pluggable NumPy, PyTorch, and JAX backends with
  automatic resolution (`FAST_VOLLIB_BACKEND` env var, `set_backend()`,
  per-call `backend=` kwarg).
- **DataFrame helper** — `price_dataframe` for end-to-end pricing, IV
  solving, and Greek computation on a `pandas.DataFrame`.
- **Compatibility** — patch helpers for `py_vollib` and
  `py_vollib_vectorized` namespaces at runtime.

### Fixed

- Corrected Black-76 forward pricing formula (`q = r`, not `q = 0`).
- Added below-intrinsic NaN guard in the PyTorch IV solver.

### Performance

- Pre-computed CDF symmetry (`N(-x) = 1 - N(x)`) eliminating 5 redundant
  CDF evaluations per option in the Greeks hot path.
- Reduced CDF calls in pricing hot path using the same symmetry identity.
- Compiled bisection fallback yields a **16× throughput improvement** on large
  WRDS-scale datasets compared to the pure Python fallback.

[0.1.2]: https://github.com/raeidsaqur/fast-vollib/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/raeidsaqur/fast-vollib/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/raeidsaqur/fast-vollib/releases/tag/v0.1.0
