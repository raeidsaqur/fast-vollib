# Changelog

All notable changes to this project will be documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

## [0.1.1] — 2026-03-23

### Fixed

- Replaced bare `assert` guards in `price_dataframe` with proper `ValueError`
  raises (asserts are suppressed under `python -O`).
- JAX IV backend: return `NaN` (not `0.0`) for below-intrinsic and zero-price
  options, matching the NumPy and PyTorch backend behaviour.
- Removed `FASTIV_BACKEND` legacy env-var alias (leftover from the `fastiv`
  rename); the canonical name `FAST_VOLLIB_BACKEND` is now the only recognised
  environment variable.  Removed the corresponding test and notebook cell.

### Added

- `py.typed` PEP 561 marker — downstream type-checkers now see the package
  as typed.
- Python 3.13 classifier in `pyproject.toml`.
- `[tool.mypy]` configuration block in `pyproject.toml`.

---

## [0.1.0] — 2026-03-22

### Added

- Initial public release.
- **Pricing** — `fast_black`, `fast_black_scholes`,
  `fast_black_scholes_merton` with full NumPy vectorization and
  broadcasting.
- **Implied Volatility** — `fast_implied_volatility` and
  `fast_implied_volatility_black` using Newton-Raphson with a compiled
  bisection fallback (~10 M solves / s on CPU).
- **Greeks** — `vectorized_delta`, `vectorized_gamma`, `vectorized_theta`,
  `vectorized_rho`, `vectorized_vega`, and `get_all_greeks`.
- **Backend routing** — pluggable NumPy, PyTorch, and JAX backends with
  automatic resolution (`FAST_VOLLIB_BACKEND` env var, `set_backend()`,
  per-call `backend=` kwarg).
- **DataFrame helper** — `price_dataframe` for end-to-end pricing, IV
  solving, and Greek computation on a `pandas.DataFrame`.
- **Compatibility** — `patch_py_vollib()` monkey-patches `py_vollib` and
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

[Unreleased]: https://github.com/raeidsaqur/fast-vollib/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/raeidsaqur/fast-vollib/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/raeidsaqur/fast-vollib/releases/tag/v0.1.0
