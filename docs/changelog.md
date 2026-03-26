# Changelog

All notable changes to this project will be documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Only tagged public releases are recorded here. Development snapshots published
from `main` to TestPyPI use VCS-derived `.devN` versions and are not tracked as
separate changelog entries.

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

[0.1.2]: https://github.com/raeidsaqur/fast-vollib/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/raeidsaqur/fast-vollib/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/raeidsaqur/fast-vollib/releases/tag/v0.1.0
