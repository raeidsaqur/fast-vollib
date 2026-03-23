# Changelog

All notable changes to this project will be documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

## [0.1.0] — 2026-03-22

### Added

- Initial public release.
- **Pricing** — `vectorized_black`, `vectorized_black_scholes`,
  `vectorized_black_scholes_merton` with full NumPy vectorization and
  broadcasting.
- **Implied Volatility** — `vectorized_implied_volatility` and
  `vectorized_implied_volatility_black` using Newton-Raphson with a compiled
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

[Unreleased]: https://github.com/raeid-saqur/fast-vollib/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/raeid-saqur/fast-vollib/releases/tag/v0.1.0
