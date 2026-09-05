# Changelog

All notable changes to this project will be documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Only tagged public releases are recorded here. Development snapshots published
from `main` to TestPyPI use VCS-derived `.devN` versions and are not tracked as
separate changelog entries.

---

## [Unreleased]

### Added

- Fixed-income contracts: `Cashflow`, `FixedIncomeSecurity`, `ZeroCouponBond`,
  and `FixedRateBond`, with explicit payment times and accrual fractions.
- Cashflow present values against structural discount curves, including flat,
  CIR, and log-linear interpolated discount-factor curves.
- Risk-neutral CIR bond prices, zero and forward rates, and a complex
  integrated-rate transform, with NumPy, PyTorch and JAX array support.
- CIR short-rate simulation using full-truncation Euler, quadratic-exponential,
  or exact transitions. Exact transitions support NumPy and JAX; PyTorch
  is refused because a public generator-bound gamma sampler is unavailable.
- Composable Bates and BCC97 processes with Heston or constant variance,
  Merton compound-Poisson lognormal jumps or no jumps, and CIR or constant rates.
- Bates and BCC97 European-option pricing through Lewis and Gatheral Fourier
  inversion. BCC97 uses independent equity/variance and short-rate drivers.
- Explicit `initial_state` and `discounting` arguments on `MonteCarloEngine.price`,
  with constant-rate and pathwise short-rate discounting rules.
- Additive version-1 instrument schemas, public examples, independent analytical
  checks, and seeded reference fixtures for the new models.

### Fixed

- Stabilize the Heston characteristic function at small positive vol-of-vol
  without substituting deterministic variance; retain ordinary-parameter
  arithmetic and handle the removable martingale-argument singularity.
- Keep jump counts and jump sizes on independent JAX random keys.
- Validate custom Monte Carlo discount factors before multiplying payoffs.

### Compatibility

- Existing equity instrument roots, pricing-model selectors, and default
  Monte Carlo discounting are unchanged. Fixed-income securities use
  `present_value`, not the option-pricing or terminal-payoff routes.
- Stored numerical references allow documented cross-platform rounding;
  same-environment seeded reproducibility is checked separately.

---



## [0.2.2] — 2026-09-01

### Added

- **`fast_vollib.surface` model layer** — the counterpart to the existing
  arbitrage harness: what a *model* produces, rather than how a produced surface
  is scored. A model joins the ecosystem by returning a **`DefiniteIVSurface`**
  (an evaluable map from points to implied volatilities) or a
  **`SurfaceDistribution`** over them, and everything downstream consumes only
  those two.
    - **Coordinates and adapters** — `SurfacePoints` in canonical forward
      log-moneyness `k = log(K / F(T))` and year-fraction `T`, with
      `points_from_strikes`, `points_from_spot_moneyness`, and
      `points_from_forward_delta`. Each records a `CoordinateConvention` naming
      the source coordinate, maturity convention, and market state consumed, so
      a conversion stays reproducible.
    - **Market state** — `SurfaceMarket`, a forward / discount / carry term
      structure with a declared interpolation policy (`log_linear` or `exact`)
      and provenance. Never inferred: a computation that needs one and lacks it
      raises `MissingMarketStateError` rather than assuming `F = 1`.
    - **Value objects** — `SurfaceObservations` (promoted from
      `diagnostics.SurfaceQuotes`, which remains an alias), `SurfacePrediction`,
      `SurfaceSamples`, and `SurfaceGridSpec`. All arrays are owned, read-only
      copies; predictions accept and flag non-finite or non-positive implied
      volatilities rather than dropping them.
    - **Four protocols** — `SurfaceCalibrator`, `ConditionalSurfaceEstimator`,
      `SurfaceForecaster`, and `GenerativeSurfaceModel`, plus `ForecastHorizon`.
      Four rather than one hierarchy because training, conditioning,
      forecasting, and sampling are different lifecycles; no protocol means
      "train on a corpus".
    - **Materialization** — `materialize_surface`, `materialize_samples`, and
      `GridIVSurface`, which reads a grid at off-node points under a named
      `policy` (`total_variance`, `implied_volatility`, `nearest`) and a named
      `extrapolation`. `IVSurface` deliberately gains no `evaluate()`.
    - **Capability registry** — `list_algorithms`, `get_algorithm`,
      `build_algorithm`, and `capabilities_document`, built once from a fixed
      list (no `register()`), reporting availability with machine-readable
      reasons and validating configurations against per-algorithm closed
      schemas.
    - **Evaluation** — `evaluate_prediction` and `SurfaceEvaluation`, reporting
      coverage as part of the error (target, valid, and invalid counts are never
      collapsed) alongside implied-volatility, vega-weighted, price-space, and
      spread metrics, split `by_region` and `by_maturity`. An attached arbitrage
      report always carries a `VerificationLevel`.
    - **Fitting algorithms** — flat, SVI and SVI-JW, SSVI, penalized splines,
      PCA factor bases, Tikhonov-regularized least squares, a Kalman
      state-space forecaster, a persistence baseline, and Heston.
    - **Generative evaluation** — `fast_vollib.surface.generative` with
      `GaussianFieldSurfaceDistribution` and `evaluate_samples`, which
      materializes and checks **every** draw rather than a mean surface, and
      reports each probability with a Monte Carlo standard error and a Wilson
      interval.
- **Heston** — `fast_vollib.processes.Heston` (QE and full-truncation-Euler
  schemes, neither exact; the Feller condition is reported, never enforced),
  `fast_vollib.pricing.heston_price` (Fourier inversion in two independent
  formulations, `lewis` and `gatheral`, on a fixed Gauss-Legendre node set), and
  `HestonIVSurface` / `HestonCalibrator`. The Fourier price carries absolute
  rather than relative accuracy, so the surface declines low-vega wings instead
  of inverting noise.
- **Schemas** — `docs/schemas/fast-vollib-surface-capabilities-v1.schema.json`,
  `fast-vollib-surface-evaluation-v1.schema.json`, and
  `fast-vollib-generative-arbitrage-v1.schema.json`, all closed and
  self-checking.
- **Documentation** — a new [Surface Models](surface_models.md) user guide, a
  pointer to it from the arbitrage-harness page, and API-reference entries for
  the new public surface and Heston namespaces.

## [0.2.1] — 2026-08-28

Packaging and citation metadata only. No API, behaviour, or numerical changes.

### Added

- `CITATION.cff` — machine-readable citation metadata (CFF 1.2.0), which also
  enables GitHub's **Cite this repository** button.
- `.zenodo.json` — deposit metadata for the Zenodo–GitHub integration, so each
  tagged release is archived and assigned a DOI.
- `.github/FUNDING.yml`.

### Changed

- `README.md` — citation section now points at `CITATION.cff`; placeholder for
  the Zenodo concept-DOI badge added to the badge row.
- `.gitignore` — expanded macOS Finder entries (`.DS_Store`, `._*`, `Icon?`,
  and friends).

## [0.2.0] — 2026-08-28

This is the next stable release after v0.1.8. No v0.1.9 release was cut: the
new public instrument, process, and simulation layers warrant a minor-version
boundary.

### Added

- **`fast_vollib.instruments`** — a nominal layer over the functional API:
  contract objects, columnar batches, market inputs, backend-native payoffs, and
  thin adapters onto the existing kernels. The functional API is unchanged and
  remains canonical.
    - **Contracts** — `Asset`, `Forward`, `Future`, `EuropeanOption`, and
      `InstrumentRef` as frozen, slotted, keyword-only dataclasses. They hold
      contract terms and nothing else: no arrays, market state, devices, RNG, or
      attached engines, so they stay hashable, comparable, and serializable, and
      importing the package pulls in neither torch, jax, numba, nor triton.
      `maturity` is a year fraction, never a date.
    - **Batches** — `EuropeanOptionBatch` with three constructors sharing one
      validation core: `from_arrays` (canonical, allocates no per-row Python
      object), `from_frame` (explicit column mapping; market columns refused by
      name), and `from_instruments`. One batch is one vectorized kernel call.
      Pure coordinate functions `moneyness`, `log_moneyness`,
      `time_to_maturity`, and `forward_price` in the `k = log(K/F)` convention
      `fast_vollib.surface` uses.
    - **Payoffs** — `payoff(instrument, terminal_state)` evaluates in the
      caller's own array namespace, preserving dtype, device, and the autograd
      tape; no path stages through host memory.
    - **Adapters** — `price_instrument`, `greeks_instrument`, and
      `implied_volatility_instrument`, wrapping only the existing public
      kernels. Tests verify exact array equality against direct kernel calls.
      Notional scales price and Greeks; an observed quote is divided by it
      before implied-volatility inversion.
    - **Solver-aware IV** — `solver="jackel"` (default, machine-precision, the
      only route with gradient support) or `"halley"`. Jäckel routes through the
      full-model wrappers for numpy, torch, and jax; native torch/jax input with
      `return_native=True` reaches the differentiable implicit-function-theorem
      wrappers with their invalid-domain contract intact.
    - **Discovery and fail-closed errors** — a read-only `instrument_types()`
      registry, `capabilities()` recording implied volatility per solver and
      gradient-preserving `(model, solver, backend)` triples separately, and a
      typed error hierarchy under `InstrumentError`. No request ever falls back
      to a different model, engine, solver, or backend.
    - **Serialization** — a strict versioned dict/JSON codec rejecting unknown
      schema versions, types, and fields, plus a checked-in JSON Schema at
      `docs/schemas/instrument-v1.schema.json` generated from the same field
      table and regenerated byte-for-byte by a test. The schema remains
      experimental until explicitly declared stable. `jsonschema` is a
      development dependency only.
- **`fast_vollib._array_api`** — the `ArrayNS` namespace adapter, promoted from
  `fast_vollib.surface._xp`, which keeps working as a re-export shim.
- **Documentation** — `docs/instruments.md`, including a capability table
  generated from the registry and the differentiability table.
- **`fast_vollib.processes`** — stochastic dynamics holding parameters and
  nothing else: no random state, no path buffer, no device, no contract.
    - `GBM(drift, volatility)` and `GBM.risk_neutral(rate, volatility,
      dividend_yield=0.0)`, sampling the closed-form log transition on an
      arbitrary, possibly irregular grid rather than stepping the SDE, so the
      grid controls only how often a path is observed and contributes no
      discretization bias. The first state is the initial state bit for bit and
      zero volatility gives the deterministic path exactly. Parameters are
      stored as the objects the caller passed, so gradients reach them.
    - `StochasticProcess`, the structural protocol `simulate()` drives.
- **`fast_vollib.simulation`** — scenarios and explicit Monte Carlo valuation.
    - `simulate(underlier, process, *, initial_state, time_grid, n_paths, rng,
      antithetic=False)` returning a `Scenario`. Pure, contract-agnostic, and
      native: it attaches nothing to the underlier or the process and performs
      no maturity check, because a scenario may be evaluated against several
      contracts.
    - `Scenario` and `Scenario.from_states` — an execution value, frozen,
      identity-hashed, and deliberately not serializable. NumPy buffers a
      caller supplies are copied and marked read-only; buffers `simulate()`
      allocates are frozen in place. Native state arrays are stored undetached,
      so the tape survives and the arrays stay mutable by whoever else holds
      them; the small grid is converted when needed to match the state dtype.
    - `MonteCarloEngine` and `MCResult`. Concrete and asked for by name: no
      registry selects it, no analytic adapter falls back to it, and it never
      substitutes for a closed form. `market.rate` discounts and never rewrites
      a drift, and `market.volatility` is not read at all, because the process
      owns volatility. Everything is validated before a path is drawn, the RNG
      included. `supports(type)` reports a route; `supports(instance)` also
      applies positive-maturity eligibility.
    - Estimator reporting the sample mean and the standard error of that mean,
      with antithetic sampling averaging each matched pair first and dividing
      by the number of pairs, which `effective_samples` reports.
    - Typed errors `SimulationError`, `SimulationValidationError`,
      `UnsupportedProcessError`, `ScenarioMismatchError`.
- **Path-dependent and digital contracts** — `BinaryOption`, `AsianOption`,
  `BarrierOption`, `LookbackOption`, and `VarianceSwap`, with the vocabularies
  `BarrierType`, `AveragingMethod`, and `StrikeConvention` and the new
  `PayoffRequirement.PATH`. Conventions that change what a contract is worth
  are contract fields: averaging method, strike convention, barrier direction
  and knock sense. Monitoring is discrete and inclusive at the scenario's own
  observation times; Asian fixings exclude the valuation date while barrier and
  lookback monitoring includes both ends; a binary pays nothing at the strike
  exactly; realized variance is the sum of squared log returns over the year
  fraction, with no sample-mean subtraction and no factor of 252.
- **Path payoff dispatch** — `payoff(instrument, scenario)` for path-dependent
  contracts, with the scenario checking underlier and horizon before any
  arithmetic. A bare array is refused rather than interpreted, and a terminal
  contract handed a scenario is pointed at `Scenario.payoff`.
- **Simulation capabilities** — `CapabilitySet.simulate` and
  `CapabilitySet.simulation_autodiff`, the latter recording tape retention on
  installed backends rather than promising useful Greeks for discontinuous
  payoffs.
- **`FV_REQUIRE_BACKENDS`** — naming a backend makes a test session refuse to
  start when it is not installed and turns a skip attributed to it into a
  failure, so a CI job that installed an optional backend can no longer report
  green while skipping the tests it exists to run.
- **Documentation** — `docs/simulation.md`, covering measure, randomness and
  reproducibility per backend, scenario ownership and mutation, terminal
  versus path dispatch, every payoff convention, the estimator, and a
  differentiability table that separates tape retention from useful gradients.

### Changed

- `fast_vollib.instruments` resolves through a module-level `__getattr__`, so a
  bare `import fast_vollib` does not pay for it; `fast_vollib.processes` and
  `fast_vollib.simulation` resolve the same way.
- `fast_vollib.types` gains `InstrumentKindLiteral`, `ExerciseLiteral`, and
  `IVSolverLiteral`, pinned to the instrument enums by test.
- `fast_vollib._array_api` gains reductions, cumulative sum, stacking,
  dtype-aware scalar construction, and tracer-safe concrete-value readers, each
  tested against NumPy's answer in every installed namespace. Existing
  operations are unchanged.

### Fixed

- Binary-option payoff scaling is constructed in the caller's array namespace,
  preserving float32 dtype, device placement, and the autodiff graph for
  fractional cash and notional terms.

### Deprecated

- The Halley-with-bisection implied-volatility route (`solver="halley"`,
  `fast_implied_volatility`) in favour of the Jäckel solver, which is more
  accurate at comparable cost and is the only route with gradient support. The
  Halley route remains available, tested, and unchanged, serves every backend,
  and raises no runtime warning; it is for reproducing existing results rather
  than for new work.

---

## [0.1.8] — 2026-08-15

### Added

- **Differentiable Jäckel implied volatility** (`fast_vollib.jackel`):
    - `implied_volatility_autograd` — PyTorch `autograd.Function` wrapper around
      the machine-precision "Let's Be Rational" solver. Forward runs the full
      Jäckel solver; backward applies the implicit function theorem to the
      discounted pricing equation (`∂σ/∂price = 1/ν`,
      `∂σ/∂θ = −(∂price/∂θ)/ν`), giving exact gradients w.r.t. price, spot,
      strike, maturity, rate, and dividend yield without differentiating
      through the Householder iterations. Also exported at the top level
      (`fast_vollib.implied_volatility_autograd`; `None` when torch is not
      installed).
    - `implied_volatility_autograd_jax` — JAX `custom_vjp` equivalent.
    - Contract: invalid domain (below-intrinsic, non-positive price / spot /
      strike, zero maturity) produces `NaN` in forward and backward; a low-vega
      upstream-aware guard returns a zero gradient when the upstream cotangent
      is exactly zero, preventing `0 × NaN` chain-rule contamination of valid
      rows.
    - Test suites for both wrappers (`tests/test_jackel/test_autograd.py`,
      `test_autograd_jax.py`) covering gradient correctness against finite
      differences, the NaN domain contract, and the low-vega guard.
- **Differentiable IV documentation** — `docs/differentiable_iv.md` training-loop
  guide (PyTorch IV-loss, hybrid price + IV-roundtrip with the caller-side
  low-vega filter, JAX example) wired into the mkdocs nav; expanded tutorial
  notebook.

### Changed

- Development/CI environments select CUDA wheels by CPU architecture: cu130 on
  x86_64 Linux, cu126 on aarch64 Linux (Grace-Hopper). Dev-only `[tool.uv]`
  resolution constraints — not published in wheel metadata; installed package
  requirements are unchanged.
- Dev dependency `py-lets-be-rational` pinned `<1.1` (1.1.x breaks
  `py_vollib_vectorized`'s numba type inference), with an in-tree
  `testcapi-compat` shim supplying CPython's private `_testcapi` module on
  interpreters that omit it (e.g. `python-build-standalone` via
  `uv python install`).

---

## [0.1.7] — 2026-07-03

### Added

- **Surface arbitrage-evaluation harness** (`fast_vollib.surface`) — a
  generator-agnostic, backend-pluggable, differentiable evaluator for implied
  -volatility surfaces. Takes an arbitrary surface on an arbitrary
  `(log-moneyness × maturity)` mesh and returns calibrated, dimensionless
  arbitrage diagnostics.
    - `IVSurface` / `SurfaceSequence` containers with `from_logmoneyness`,
      `from_strikes`, `from_total_variance`, and `from_call_prices` constructors;
      numpy / torch / jax arrays preserved with dtype and device.
    - `validate_surface()` → `ArbitrageReport`: price-space discrete checks
      (convexity / slope / box / calendar; Davis–Hobson 2007) **and**
      total-variance checks (`∂_T w ≥ 0`, Durrleman `g ≥ 0`;
      Gatheral–Jacquier 2014), with **normalized** metrics
      (`ndm`, `bfly_frac`, `cal_depth_max`, `cal_frac`, `vert_frac`,
      `bound_frac`) and the `SAS` composite (reported only alongside its
      components).
    - **Artifact-vs-arbitrage separation**: violations whose stencil touches an
      interpolated node are bucketed as `interpolation_induced` rather than
      counted as model arbitrage.
    - **Round-trip trust mask**: per-node `σ→C→σ'` Jäckel LBR fixed-point
      residual, machine-tight where the quote is well-posed.
    - Butterfly violations gate on the **per-slice-normalized** density magnitude
      vs the dimensionless tolerance (never raw `density < 0`), so O(h²)
      truncation noise at near-degenerate wings cannot manufacture spurious
      violations on an arbitrage-free surface. Severity bands key off the
      normalized magnitude; an empty / all-NaN surface reports `passed=False`
      (`context["coverage"]`).
    - `arbitrage_penalty()` — a differentiable soft form of the same checks that
      stays in the input tensor's namespace (no host round-trip), so it is
      autograd-traceable on torch/jax and matches the numpy report to machine
      precision. A reusable replacement for the inline VolGAN / deep-smoothing
      penalty functions.
    - Backend parity verified numpy == torch == jax to fp tolerance; SVI
      closed-form oracles validate the non-uniform divided-difference stencils
      (second-order convergence) and Durrleman `g`; `models.fast_black` validates
      the surface's own normalized-Black pricing to machine epsilon.
- **`fast_vollib.diagnostics`** — six publication-quality figures
  (total-variance slices, Durrleman `g`, risk-neutral density, violation
  heatmap, calendar map, round-trip trust map), gated behind a new `[viz]`
  extra (`pip install "fast-vollib[viz]"`). Matplotlib stays out of the
  numerics core dependencies.

---

## [0.1.6] — 2026-06-27

### Fixed

- **CUDA tensor inputs to `fast_implied_volatility`** — passing a CUDA-resident
  `torch.Tensor` (or any CPU tensor with `requires_grad=True`) raised
  `TypeError: can't convert cuda:0 device type tensor to numpy` because
  `to_numpy()` fell through to `np.asarray(value)`, which invoked
  `Tensor.__array__()` → `.numpy()` — illegal for both cases.
  `to_numpy()` now detects torch tensors via `type(value).__module__` and
  calls `.detach().cpu().numpy()` before the conversion.  All other input types
  (numpy arrays, pandas, scalars, lists, JAX arrays) are unaffected.
  Note: `.detach()` means gradients do **not** flow through IV inversion; the
  compute still round-trips through host numpy.  A fully differentiable GPU-
  resident IV path remains a separate feature request.

---

## [0.1.5] — 2026-05-29

### Added

- **Python 3.10 support** — lowered `requires-python` from `>=3.11` to `>=3.10`,
  added the `Programming Language :: Python :: 3.10` classifier, and extended the
  CI test matrix to cover 3.10 alongside 3.11–3.13.
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


## [0.1.4] — 2026-04-10

### Added

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


## [0.1.3] — 2026-04-04
- Added backend_parity tests for torch
- Updated tutorial notebook with Mac MPS backend (for Apple silicon chips).

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

[Unreleased]: https://github.com/raeidsaqur/fast-vollib/compare/v0.2.2...HEAD
[0.2.2]: https://github.com/raeidsaqur/fast-vollib/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/raeidsaqur/fast-vollib/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/raeidsaqur/fast-vollib/compare/v0.1.8...v0.2.0
[0.1.8]: https://github.com/raeidsaqur/fast-vollib/compare/v0.1.7...v0.1.8
[0.1.7]: https://github.com/raeidsaqur/fast-vollib/compare/v0.1.6...v0.1.7
[0.1.6]: https://github.com/raeidsaqur/fast-vollib/compare/v0.1.5...v0.1.6
[0.1.5]: https://github.com/raeidsaqur/fast-vollib/compare/v0.1.4...v0.1.5
[0.1.4]: https://github.com/raeidsaqur/fast-vollib/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/raeidsaqur/fast-vollib/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/raeidsaqur/fast-vollib/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/raeidsaqur/fast-vollib/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/raeidsaqur/fast-vollib/releases/tag/v0.1.0
