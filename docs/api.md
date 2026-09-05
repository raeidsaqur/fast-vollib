# API Reference

All public symbols are importable directly from `fast_vollib`:

```python
from fast_vollib import fast_black_scholes, fast_implied_volatility, ...
```

---

## Common parameters

The following parameters appear across pricing, IV, and Greek functions:

| Parameter | Type | Description |
|---|---|---|
| `flag` | `str \| array-like` | `"c"` for call, `"p"` for put |
| `S` | `float \| array-like` | Underlying spot price |
| `F` | `float \| array-like` | Forward price (Black-76 only) |
| `K` | `float \| array-like` | Strike price |
| `t` | `float \| array-like` | Time to expiry in years |
| `r` | `float \| array-like` | Risk-free rate (continuous, annualized) |
| `sigma` | `float \| array-like` | Volatility (annualized) |
| `q` | `float \| array-like` | Continuous dividend yield (BSM only) |
| `model` | `str` | `"black"`, `"black_scholes"` (default), or `"black_scholes_merton"` |
| `return_as` | `str` | Output container; see each function below |
| `dtype` | `numpy.dtype` | Input coercion dtype; default `numpy.float64` |
| `backend` | `str` | `"auto"` (default), `"numpy"`, `"torch"`, or `"jax"` |
| `return_native` | `bool` | Return backend-native arrays for torch/jax instead of formatted pandas/NumPy output |

All array-like parameters are broadcast against each other using NumPy
broadcasting rules.

!!! tip "Shape-aware type hints"
    Every public entry point carries [`jaxtyping`](https://docs.kidger.site/jaxtyping/)
    shape annotations under a `TYPE_CHECKING` guard.  Static checkers see
    `Float[np.ndarray, "n"]` / `Bool[np.ndarray, "n"]` at the backend
    dispatch layer and a permissive `ArrayLike | FlagLike` union on the
    user-facing signatures.  At runtime the annotations are stored as
    PEP 563 strings and are never evaluated — there is no call-site cost.
    See [Runtime type checking](#runtime-type-checking) to turn them into
    enforced checks in tests or debug sessions.

---

## Pricing

Pricing functions default to `return_as="dataframe"`. Supported container
formats are:

- `return_as="dataframe"`: `pandas.DataFrame`
- `return_as="series"`: `pandas.Series`
- `return_as="numpy"`: `numpy.ndarray`
- `return_native=True` with `backend="torch"` or `backend="jax"`: native tensor/array

### `fast_black`

Black-76 model for options on futures.

```python
fast_vollib.fast_black(
    flag,
    F,
    K,
    t,
    r,
    sigma,
    *,
    return_as="dataframe",
    dtype=numpy.float64,
    backend="auto",
    return_native=False,
)
```

**Returns:** Option price(s) shaped by broadcasting rules.

### `fast_black_scholes`

Black-Scholes model for European equity options without dividends.

```python
fast_vollib.fast_black_scholes(
    flag,
    S,
    K,
    t,
    r,
    sigma,
    *,
    return_as="dataframe",
    dtype=numpy.float64,
    backend="auto",
    return_native=False,
)
```

**Returns:** Option price(s) shaped by broadcasting rules.

### `fast_black_scholes_merton`

Black-Scholes-Merton model with a continuous dividend yield.

```python
fast_vollib.fast_black_scholes_merton(
    flag,
    S,
    K,
    t,
    r,
    sigma,
    q,
    *,
    return_as="dataframe",
    dtype=numpy.float64,
    backend="auto",
    return_native=False,
)
```

**Returns:** Option price(s) shaped by broadcasting rules.

---

## Implied Volatility

IV functions share the same output conventions as pricing functions:
`"dataframe"`, `"series"`, `"numpy"`, or native torch/jax arrays via
`return_native=True`.

### `fast_implied_volatility`

Solve for implied volatility given a market price.

```python
fast_vollib.fast_implied_volatility(
    price,
    S,
    K,
    t,
    r,
    flag,
    q=None,
    *,
    on_error="warn",
    model="black_scholes",
    return_as="dataframe",
    dtype=numpy.float64,
    backend="auto",
    return_native=False,
)
```

**Returns:** Implied volatility shaped by broadcasting rules. Below-intrinsic
inputs return `NaN` when `on_error` is `"warn"` or `"ignore"`.

!!! note "`on_error`"
    - `"raise"`: raise `ValueError` for below-intrinsic inputs
    - `"warn"`: emit a warning and return `NaN`
    - `"ignore"`: silently return `NaN`

### `fast_implied_volatility_black`

Convenience wrapper for Black-76 IV. The positional argument order matches
`py_vollib`: `price, F, K, r, t, flag`.

```python
fast_vollib.fast_implied_volatility_black(
    price,
    F,
    K,
    r,
    t,
    flag,
    *,
    on_error="warn",
    return_as="dataframe",
    dtype=numpy.float64,
    backend="auto",
    return_native=False,
)
```

**Returns:** Implied volatility under Black-76.

---

## Greeks

The individual Greek functions are exported under the compatibility aliases
`vectorized_delta`, `vectorized_gamma`, `vectorized_theta`, `vectorized_rho`,
and `vectorized_vega`.

Their signature is:

```python
fast_vollib.vectorized_<greek>(
    flag,
    S,
    K,
    t,
    r,
    sigma,
    q=None,
    *,
    model="black_scholes",
    return_as="dataframe",
    dtype=numpy.float64,
    backend="auto",
    return_native=False,
)
```

Supported outputs:

- `return_as="dataframe"`: `pandas.DataFrame`
- `return_as="series"`: `pandas.Series`
- `return_as="numpy"`: `numpy.ndarray`
- `return_native=True` with `backend="torch"` or `backend="jax"`: native tensor/array

### `vectorized_delta`

First derivative of price with respect to the underlying (∂V/∂S).

### `vectorized_gamma`

Second derivative of price with respect to the underlying (∂²V/∂S²).

### `vectorized_theta`

Rate of change of price with respect to time (∂V/∂t), expressed as daily decay.

### `vectorized_rho`

Sensitivity to the risk-free rate (∂V/∂r).

### `vectorized_vega`

Sensitivity to implied volatility (∂V/∂σ), expressed per 1% move in vol.

---

## `get_all_greeks`

Compute all five Greeks in a single vectorized call.

```python
fast_vollib.get_all_greeks(
    flag,
    S,
    K,
    t,
    r,
    sigma,
    q=None,
    *,
    model="black_scholes",
    return_as="dataframe",
    dtype=numpy.float64,
    backend="auto",
    return_native=False,
)
```

**Returns:**

- `return_as="dataframe"` (default): `pandas.DataFrame` with columns `delta`, `gamma`, `theta`, `rho`, `vega`
- `return_as="json"`: JSON string mapping Greek name to values
- `return_as="dict"`: Python `dict[str, numpy.ndarray]`
- `return_native=True` with `backend="torch"` or `backend="jax"`: `dict[str, native array]`

---

## DataFrame helper

### `price_dataframe`

Price, solve IV, and compute Greeks for every row of a DataFrame in one call.

```python
fast_vollib.price_dataframe(
    df,
    *,
    flag_col,
    underlying_price_col,
    strike_col,
    annualized_tte_col,
    riskfree_rate_col,
    sigma_col=None,
    price_col=None,
    dividend_col=None,
    model="black_scholes",
    inplace=False,
    dtype=numpy.float64,
    backend="auto",
    return_native=False,
)
```

You must supply at least one of `sigma_col` or `price_col`:

- `sigma_col`: compute `Price` plus Greeks
- `price_col`: compute `IV` plus Greeks
- both: use the supplied price and volatility data as inputs for the Greek calculation; the helper does not duplicate those input columns into the output

**Returns:** A new `pandas.DataFrame`, or `None` when `inplace=True`. Added
columns are:

- `Price` when `sigma_col` is provided
- `IV` when `price_col` is provided
- `delta`, `gamma`, `theta`, `rho`, `vega`

`price_dataframe` always materializes pandas output; `return_native` only
affects intermediate backend execution.

---

## Backend management

### `get_backend`

```python
fast_vollib.get_backend(explicit=None) -> str
```

Return the backend that would be used for a given explicit choice, or the
auto-resolved backend when called without arguments.

### `set_backend`

```python
fast_vollib.set_backend(name: str) -> None
```

Set a process-level backend override. Valid values are `"auto"`, `"numpy"`,
`"torch"`, and `"jax"`.

---

## Jäckel IV — machine-precision solver

The `fast_vollib.jackel` module provides a standalone implementation of Peter
Jäckel's *"Let's Be Rational"* algorithm.  These functions are called directly
(not routed through `backend=`).  See [Jäckel IV](jackel.md) for full
documentation.

### `jackel_iv_black` (CPU — NumPy + Numba)

```python
from fast_vollib.jackel.jackel_iv import jackel_iv_black

jackel_iv_black(price, F, K, T, is_call=True) -> np.ndarray
```

| Parameter | Type | Description |
|---|---|---|
| `price` | `float \| ndarray` | Undiscounted option price |
| `F` | `float \| ndarray` | Forward price |
| `K` | `float \| ndarray` | Strike |
| `T` | `float \| ndarray` | Time to expiry (years) |
| `is_call` | `bool \| ndarray` | `True` = call, `False` = put |

**Returns:** Annualised implied volatility. `NaN` for degenerate inputs
(zero price, below intrinsic, zero expiry).

### `jackel_iv_black_torch` (GPU — PyTorch)

```python
from fast_vollib.jackel.torch_backend import jackel_iv_black_torch

jackel_iv_black_torch(price, F, K, T, is_call=True) -> torch.Tensor
```

Inputs must be `torch.float64` tensors on the same device.  The Householder
loop is compiled with `torch.compile(dynamic=True)`.

### `jackel_iv_black_jax` (GPU — JAX)

```python
from fast_vollib.jackel.jax_backend import jackel_iv_black_jax

jackel_iv_black_jax(price, F, K, T, is_call=True) -> jax.Array
```

Uses `jax.lax.fori_loop` inside a `@jax.jit`-compiled function.
Float64 mode is enabled automatically at import.

### `jackel_iv_triton` (GPU — Triton, fastest)

```python
from fast_vollib.jackel.triton_kernels import jackel_iv_triton

jackel_iv_triton(price, F, K, T, is_call=True) -> torch.Tensor
```

Single-pass `@triton.jit` kernel.  All pipeline stages (preproc, boundary,
Hermite guess, Householder×3) execute in registers with one HBM read and one
HBM write per element.

---

## Surface arbitrage-evaluation harness

The `fast_vollib.surface` subpackage scores generated IV surfaces for static
arbitrage and provides a differentiable training penalty. See
[Surface Arbitrage Harness](surface.md) for the full guide.

```python
from fast_vollib.surface import (
    IVSurface, SurfaceSequence, validate_surface, arbitrage_penalty,
)
```

### `IVSurface`

Backend-agnostic surface container, parametrized in forward log-moneyness
`k = log(K/F)` × maturity `T`. Constructors:

```python
IVSurface.from_logmoneyness(k, T, iv, *, forward=1.0, r=0.0, q=0.0, native_mask=None)
IVSurface.from_strikes(K, T, iv, *, spot, r=0.0, q=0.0, native_mask=None)
IVSurface.from_total_variance(k, T, w, *, forward=1.0, r=0.0, q=0.0)
IVSurface.from_call_prices(K, T, call_prices, *, spot, r=0.0, q=0.0, discounted=True)
```

Accepts numpy / torch / jax arrays and preserves dtype and device.

### `validate_surface`

```python
validate_surface(
    surf,
    *,
    tolerance=1e-6,
    trust_tolerance=1e-6,
    weights=None,
    max_violations=2000,
    compute_trust=True,
    return_as="report",      # "report" | "dict" | "json"
) -> ArbitrageReport
```

**Returns:** an `ArbitrageReport` with `passed`, normalized `metrics`, the
`sas` composite, localized `violations`, `by_condition` counts, the
`native` / `interpolation_induced` artifact buckets, and the round-trip
`trust_mask`.

### `arbitrage_penalty`

```python
arbitrage_penalty(
    iv, k, T, forward, r=0.0,
    *, weights=None, reduction="mean", shared_k=True,
) -> scalar
```

Differentiable scalar arbitrage penalty in the namespace of `iv` (≥ 0; 0 for an
arbitrage-free surface). Drop into a generator's training loss; gradients flow
to `iv`. `penalty_from_surface(surf)` is the `IVSurface` convenience wrapper.

### Diagnostic figures

`fast_vollib.diagnostics` (requires the `[viz]` extra) returns matplotlib
`Figure`s: `plot_total_variance_slices`, `plot_durrleman_g`, `plot_density`,
`plot_violation_heatmap`, `plot_calendar_map`, `plot_trust_map`.

---

## Surface models

The model-facing half of `fast_vollib.surface`: the value objects a model
exchanges, the protocols it satisfies, and the machinery that scores it. See
[Surface Models](surface_models.md) for the design and worked examples.

```python
from fast_vollib.surface import (
    SurfacePoints, SurfaceObservations, SurfacePrediction, SurfaceSamples,
    SurfaceMarket, SurfaceGridSpec, GridIVSurface,
    materialize_surface, materialize_samples,
    evaluate_prediction, SurfaceEvaluation, VerificationLevel,
    list_algorithms, get_algorithm, build_algorithm, capabilities_document,
)
```

| Group | Names |
|---|---|
| Coordinates | `SurfacePoints`, `CoordinateConvention`, `points_from_strikes`, `points_from_spot_moneyness`, `points_from_forward_delta` |
| Data | `SurfaceObservations`, `align_predictions`, `SurfacePrediction`, `SurfaceSamples` |
| Market and mesh | `SurfaceMarket`, `SurfaceGridSpec` |
| Protocols | `DefiniteIVSurface`, `SurfaceCalibrator`, `ConditionalSurfaceEstimator`, `SurfaceForecaster`, `SurfaceDistribution`, `GenerativeSurfaceModel`, `ForecastHorizon` |
| Materialization | `materialize_surface`, `materialize_samples`, `GridIVSurface` |
| Capabilities | `list_algorithms`, `get_algorithm`, `build_algorithm`, `capabilities_document`, `AlgorithmAvailability`, `SurfaceAlgorithmSpec`, `BackendSupport` |
| Evaluation | `evaluate_prediction`, `SurfaceEvaluation`, `RegionEvaluation`, `MaturityEvaluation`, `VerificationLevel` |
| Errors | `SurfaceError`, `SurfaceValidationError`, `SurfaceTypeError`, `SurfaceDomainError`, `SurfaceCalibrationError`, `MissingMarketStateError`, `SurfaceAlgorithmUnavailableError` |

### Value objects

```python
SurfacePoints(k, T, surface_id=None, point_id=None, convention=CoordinateConvention())
SurfaceObservations(
    k, T, iv, surface_id=None, point_id=None,
    bid=None, ask=None, is_call=None, weight=None, price=None,
    convention=CoordinateConvention(),
)
SurfacePrediction(points, iv, sd=None, quantiles=None, quantile_levels=None, valid=None)
SurfaceSamples(points, iv, valid=None, rng_policy=None)
```

All arrays are copied and marked read-only. `SurfacePrediction.valid` defaults to
`isfinite(iv) & (iv > 0)`; `iv` is never sanitized. `SurfaceObservations` is the
class `fast_vollib.diagnostics` exports as `SurfaceQuotes`, and carries
`.points`, `.subset`, `.surfaces()`, `.smiles()`, `.has_spread`, `.to_dataframe()`,
and the `.from_dataframe()` / `.from_points()` / `.from_strikes()` constructors.
`SurfaceSamples` carries `.sample(i)`, `.mean_prediction()`, and
`.median_prediction()`.

### Coordinate adapters

```python
points_from_strikes(K, T, *, forward, surface_id=None, point_id=None,
                    maturity="year_fraction", market_source=None)
points_from_spot_moneyness(m, T, *, forward, spot, ...)
points_from_forward_delta(delta, T, iv, *, is_call=True, ...)
```

Each returns `SurfacePoints` in canonical `k = log(K / F(T))` with a
`CoordinateConvention` recording the source coordinate, maturity convention, and
`market_source`.

### `SurfaceMarket`

```python
SurfaceMarket(T, forward, rate=0.0, carry=0.0,
              interpolation="log_linear", extrapolation="flat", source=None)
SurfaceMarket.flat(*, forward, rate=0.0, carry=0.0, source=None)
SurfaceMarket.from_spot(*, spot, T, rate=0.0, carry=0.0, source=None)
```

Lookups: `forward_at(T)`, `rate_at(T)`, `carry_at(T)`, `discount_at(T)`,
`strikes_at(k, T)`, `to_dict()`. `interpolation` is `"log_linear"` or `"exact"`;
`extrapolation` is `"flat"` or `"error"`. Never inferred — a computation needing
a market and given none raises `MissingMarketStateError`.

### `SurfaceGridSpec`

```python
SurfaceGridSpec(k, T, market=None, topology="shared_moneyness",
                native_mask=None, name=None)
SurfaceGridSpec.uniform(*, k_min, k_max, n_k, T, market=None, name=None)
SurfaceGridSpec.from_strikes(K, T, *, market, name=None)
grid.to_points(*, surface_id=None, point_ids=True)   # -> SurfacePoints
```

`topology` is `"shared_moneyness"` or `"fixed_strike"`. Also `Nk`, `Nt`, `shape`,
`n_nodes`, `shared_k`, `k2d()`, `T2d()`, `to_dict()`, `require_market(what)`.

### Protocols

```python
DefiniteIVSurface.evaluate(points, *, market=None) -> SurfacePrediction
SurfaceCalibrator.fit(observations, *, rng=None) -> DefiniteIVSurface
ConditionalSurfaceEstimator.condition(context, *, rng=None) -> DefiniteIVSurface
SurfaceForecaster.forecast(history, horizon, *, rng=None)
    -> DefiniteIVSurface | SurfaceDistribution
SurfaceDistribution.sample(points, *, n_samples, rng, market=None) -> SurfaceSamples
GenerativeSurfaceModel.distribution(context, *, horizon=None) -> SurfaceDistribution
ForecastHorizon(steps=1, step_years=None)
```

`runtime_checkable` structural protocols — `isinstance` checks that the member
exists, not that it behaves.

### Materialization

```python
materialize_surface(surface, grid, *, market=None, surface_id=None, t_index=None) -> IVSurface
materialize_samples(samples, grid, *, market=None) -> Iterator[IVSurface]
GridIVSurface(surface, policy="total_variance", extrapolation="invalid")
```

`policy` is one of `"total_variance"`, `"implied_volatility"`, `"nearest"`;
`extrapolation` one of `"invalid"`, `"error"`, `"clamp"`. `GridIVSurface` is a
`DefiniteIVSurface` and adds `native_mask_for(grid, *, decimals=12)`.
`IVSurface` deliberately has **no** `evaluate()`.

### Capability registry

```python
list_algorithms(*, family=None, available_only=False) -> tuple[AlgorithmAvailability, ...]
get_algorithm(public_id) -> AlgorithmAvailability
build_algorithm(public_id, config=None)      # config validated against a closed schema
capabilities_document() -> dict
```

`family` is one of `"calibrator"`, `"conditional"`, `"forecaster"`,
`"generative"`; `output` one of `"definite"`, `"distribution"`. Unavailable
entries carry `unavailable_code` in `"optional_dependency"`, `"backend"`,
`"checkpoint"`. The document serializes as
`fast-vollib-surface-capabilities-v1`.

### `evaluate_prediction`

```python
evaluate_prediction(
    prediction, observations, *,
    market=None, require_prices=False,
    grid=None, materialized=None, verification=None,
    regions=None, maturity_decimals=6,
) -> SurfaceEvaluation
```

`SurfaceEvaluation` carries `target_count` / `valid_count` / `invalid_count`,
`coverage`, `invalid_rate`, `iv_rmse`, `iv_mae`, `max_absolute_iv_error`,
`weighted_iv_rmse`, `vega_weighted_iv_rmse`, `price_rmse`, `price_mae`,
`inside_spread_fraction`, `by_region`, `by_maturity`, `arbitrage`,
`verification`, `native_node_fraction`, `grid_shape`, `market_source`, plus
`to_dict()` / `to_json()` under `fast-vollib-surface-evaluation-v1`. Supplying
`grid` also requires `materialized`. An unavailable number is `None`, never a
zero.

`VerificationLevel` is `EMPIRICAL_FINITE_GRID`, `TRAINING_PENALTY`,
`MATHEMATICAL_GUARANTEE`, `EXTERNAL_CLAIM_UNVERIFIED`.

### Fitting algorithms

`fast_vollib.surface.fitting` — every entry consumes `SurfaceObservations` and
returns a `DefiniteIVSurface`.

| Public id | Constructor |
|---|---|
| `flat` | `FlatVolatilityCalibrator(objective="implied_volatility", use_weights=True)` |
| `svi` | `SVICalibrator(objective="total_variance", butterfly_penalty=0.0, n_starts=3, max_iterations=2000, maturity_interpolation="total_variance_linear")` |
| `ssvi` | `SSVICalibrator(phi_family="power_law", enforce_no_butterfly=True, n_starts=3, max_iterations=4000)` |
| `spline` | `SplineSurfaceCalibrator(degree_k=3, degree_t=3, n_interior_knots_k=None, n_interior_knots_t=None, smoothing_k=1e-6, smoothing_t=1e-6, use_weights=True)` |
| `heston` | `HestonCalibrator(objective="implied_volatility", n_starts=4, max_iterations=800, n_nodes=768, diff_step=1e-4)` |
| `factor-pca` | `FactorSurfaceCalibrator(basis, policy=None, extrapolation="invalid", use_weights=True)` |
| `persistence` | `PersistenceForecaster(calibrator=FlatVolatilityCalibrator())` |
| `state-space` | `StateSpaceForecaster(calibrator=..., transition_variance=1e-4, observation_variance=1e-6, initial_variance=1.0)` |
| `gaussian-field` | `GaussianFieldSurfaceGenerator(calibrator=..., volatility=0.05, length_scale_k=0.25, length_scale_T=1.0)` |

Also exported: `FlatIVSurface`, `SVISurface`, `SVIParameters`, `SVIJumpWings`,
`SVISmile`, `SVISliceFit`, `SSVISurface`, `PowerLawPhi`, `HestonLikePhi`,
`SplineIVSurface`, `SplineSmileCalibrator`, `FactorIVSurface`,
`SurfaceFactorBasis`, `fit_factor_basis`, `FactorPCARecipe`, `HestonIVSurface`,
`HestonParameters`, the Kalman primitives (`kalman_filter`, `kalman_predict`,
`kalman_update`, `kalman_smooth`, `LinearGaussianModel`, `GaussianState`,
`FilteredPath`), the regularization helpers (`solve_penalized_least_squares`,
`difference_matrix`, `TikhonovPenalty`, `couple_parameter_sequence`), and
`fit_each`.

### Generative evaluation

```python
from fast_vollib.surface.generative import (
    GaussianFieldSurfaceDistribution, GaussianFieldSurfaceGenerator,
    evaluate_samples, GenerativeArbitrageReport, wilson_interval,
)

GaussianFieldSurfaceDistribution(base, volatility=0.05,
                                 length_scale_k=0.25, length_scale_T=1.0)
evaluate_samples(
    source, grid, *, n_samples=None, rng=None, market=None,
    severity_quantiles=(0.5, 0.9, 0.99), summary_surfaces=True,
    verification=None, tolerance=None,
) -> GenerativeArbitrageReport
```

`source` is a `SurfaceDistribution` (then `n_samples` and `rng` are required) or
already-drawn `SurfaceSamples`. The report carries `any_violation_probability`
with its Wilson `any_violation_interval` and `any_violation_stderr`,
`condition_probability` and `condition_expected_fraction` over
`("butterfly", "calendar", "vertical", "bound")`, `expected_severity`,
`severity_quantiles`, `worst_severity`, `valid_sample_fraction`,
`point_coverage`, `mean_surface_metrics`, `median_surface_metrics`, and
`verification`; `to_dict()` / `to_json()` render
`fast-vollib-generative-arbitrage-v1`.

---

## Compatibility

### `patch_py_vollib`

```python
fast_vollib.patch_py_vollib() -> None
```

Monkey-patch the scalar `py_vollib` namespace with fast-vollib implementations.
Requires `py_vollib` to be installed.

### `patch_py_vollib_vectorized`

```python
fast_vollib.patch_py_vollib_vectorized() -> None
```

Monkey-patch the `py_vollib_vectorized` namespace with fast-vollib
implementations. Requires `py_vollib_vectorized` to be installed.

See [Compatibility](compatibility.md) for examples and caveats.

---

## Runtime type checking

fast-vollib ships pure shape annotations (no decorators) on the public
API and on the four backend dispatch entry points.  When combined with
`jaxtyping` + `beartype`, the annotations become enforced runtime
checks that reject wrong-shape or wrong-dtype inputs at the boundary —
without touching any inner hot path.

### Install the extra

```bash
pip install "fast-vollib[typecheck]"
```

This pulls in `jaxtyping` and `beartype`.  The base install never loads
either package; you can verify with:

```bash
python -c "import fast_vollib, sys; assert 'jaxtyping' not in sys.modules"
```

### Enable checks at import time

```python
from fast_vollib._typing import enable_runtime_checks
enable_runtime_checks()          # install before importing fast_vollib
import fast_vollib                 # public signatures now enforced
```

`enable_runtime_checks()` installs a `jaxtyping` import hook scoped to:

- `fast_vollib.api`
- `fast_vollib.models`
- `fast_vollib.greeks`
- `fast_vollib.implied_volatility`
- `fast_vollib.backends.{numpy,torch,jax,numba}_backend`

Everything under `fast_vollib.jackel`, all `@triton.jit` kernels,
`@numba.njit` factories, `torch.compile` closures, and `@jax.jit`-traced
functions are **deliberately excluded** from the hook so that compiled
pipelines see exactly the same bytecode with or without the extra.

### Customising the scope

Pass an explicit tuple of module names to narrow or widen the hook
(for example, only the backend dispatch layer):

```python
enable_runtime_checks((
    "fast_vollib.backends.numpy_backend",
    "fast_vollib.backends.torch_backend",
))
```

### Performance impact

None, when the hook is not installed.  The annotations live inside a
`TYPE_CHECKING` block and as PEP 563 strings, so:

- `jaxtyping` / `beartype` are not imported by default
- No decorators are ever applied at module load
- Call-site dispatch is unchanged byte-for-byte

When the hook **is** installed, `beartype` performs an O(1) isinstance +
shape + dtype check per public-API call (microseconds, independent of
array size).  This is intended for tests and development; production
code typically leaves the hook off.

---

## Instruments

`fast_vollib.instruments` is a separate, optional namespace: contract objects
and columnar batches with thin adapters onto the kernels above. The namespace
is exposed lazily as `fast_vollib.instruments`, but its individual APIs are not
re-exported from `fast_vollib`. Importing `fast_vollib` alone does not load the
instruments package.

```python
from fast_vollib.instruments import EuropeanOption, VanillaMarketInputs, price_instrument
```

| Group | Names |
|---|---|
| Contracts | `Asset`, `Forward`, `Future`, `EuropeanOption`, `InstrumentRef`, `Instrument`, `Derivative` |
| Digital and path-dependent contracts | `BinaryOption`, `AsianOption`, `BarrierOption`, `LookbackOption`, `VarianceSwap` |
| Batches and coordinates | `EuropeanOptionBatch`, `moneyness`, `log_moneyness`, `time_to_maturity`, `forward_price` |
| Market inputs | `VanillaMarketInputs` |
| Payoffs and adapters | `payoff`, `payoff_requirement`, `price_instrument`, `greeks_instrument`, `implied_volatility_instrument` |
| Discovery | `instrument_types`, `instrument_type`, `InstrumentTypeInfo`, `capabilities`, `CapabilitySet` |
| Serialization | `instrument_to_dict`, `instrument_from_dict`, `instrument_to_json`, `instrument_from_json` |
| Vocabularies | `AssetClass`, `InstrumentKind`, `OptionType`, `ExerciseStyle`, `SettlementType`, `PricingModel`, `IVSolver`, `PayoffRequirement`, `BarrierType`, `AveragingMethod`, `StrikeConvention` |
| Errors | `InstrumentError`, `InstrumentValidationError`, `UnsupportedInstrumentError`, `UnsupportedModelError`, `UnsupportedSolverError`, `MissingMarketInputError`, `SerializationError` |

The adapters take `model` as a required keyword and never infer it, and they
never fall back to a different model, engine, solver, or backend. `price_instrument`
and `greeks_instrument` are **not** differentiable; see
[Instruments](instruments.md#differentiability) for the full table.

`BinaryOption` has a terminal payoff; the other four need a whole trajectory
and are evaluated on a `Scenario`. None of the five has an analytic kernel
here: they are valued by simulation, explicitly requested. See
[Simulation](simulation.md).

---

## Processes and simulation

`fast_vollib.processes` and `fast_vollib.simulation` are two more lazily
exposed, optional namespaces. Importing `fast_vollib` alone loads neither, and
importing either loads no torch, jax, numba, or triton.

```python
from fast_vollib.processes import GBM
from fast_vollib.simulation import MonteCarloEngine, Scenario, simulate
```

| Group | Names |
|---|---|
| Processes | `GBM`, `GBM.risk_neutral`, `Heston`, `Bates`, `BCC97`, `CIRShortRate`, `StochasticProcess` |
| Components | `HestonVariance`, `ConstantVariance`, `LognormalJumps`, `NoJumps`, `CIRShortRate`, `ConstantShortRate` |
| Discounting | `DiscountingRule`, `ConstantRateDiscounting`, `PathwiseShortRateDiscounting` |
| Scenarios | `Scenario`, `Scenario.from_states`, `simulate` |
| Pricing | `MonteCarloEngine`, `MonteCarloEngine.supports`, `MonteCarloEngine.price`, `MCResult` |
| Errors | `SimulationError`, `SimulationValidationError`, `UnsupportedProcessError`, `ScenarioMismatchError` |

### `simulate`

```python
simulate(underlier, process, *, initial_state, time_grid, n_paths, rng, antithetic=False)
```

Returns a `Scenario`. Knows no contract and checks no maturity. Pure: nothing
is attached to the underlier or the process.

### `MonteCarloEngine.price`

```python
MonteCarloEngine(antithetic=False).price(
    instrument, market, *, process, n_paths, rng,
    time_grid=None, n_steps=None,
    initial_state=None, discounting=None, return_native=False,
)
```

Returns an `MCResult` carrying `price`, `stderr`, `n_paths`, and
`effective_samples`. `market.underlying` is the initial spot;
`market.volatility` is not read, because the process owns it. Supply exactly one
of `time_grid` and `n_steps`. The engine never rewrites a drift: a risk-neutral
price needs a process you made risk-neutral.

The process's first state must be `"spot"`. Any others — a variance, a short
rate — are supplied by name through `initial_state`, which may not carry
`"spot"` and must carry every other state the process declares.

`discounting` is a `DiscountingRule`. Omitted, the payoff is discounted at
`market.rate` over the contract's maturity, bit for bit as before. Supplied,
`market.rate` is **not read at all** and the factor is applied path by path,
which is what a stochastic-rate model needs. See
[Simulation](simulation.md) for the conventions, the estimator, and the
differentiability table.

### Heston

```python
from fast_vollib.processes import Heston
from fast_vollib.pricing import (
    heston_call_price, heston_characteristic_function, heston_price,
)

Heston(kappa, theta, vol_of_vol, rho, drift=0.0)
Heston.risk_neutral(*, rate, kappa, theta, vol_of_vol, rho, dividend_yield=0.0)
Heston.sample(*, initial_state, time_grid, n_paths, rng,
              antithetic=False, scheme="quadratic_exponential")
```

Two state variables, `('spot', 'variance')`, so a sample is
`(n_paths, n_times, 2)`. `scheme` is `"quadratic_exponential"` (Andersen's QE) or
`"full_truncation_euler"`; **neither is exact**, and the bias shrinks with the
step size without vanishing at any finite one. `feller_ratio` and
`satisfies_feller` report the Feller condition; it is never enforced. Priced
through `MonteCarloEngine` it needs `initial_state={"variance": v0}`.

```python
heston_price(*, forward, strike, maturity, v0, kappa, theta, vol_of_vol, rho,
             is_call=True, discount=1.0, formulation="lewis", n_nodes=768)
heston_call_price(...)                      # heston_price with is_call=True
heston_characteristic_function(u, *, maturity, v0, kappa, theta, vol_of_vol, rho)
```

Host-side float64 Fourier inversion, Gauss-Legendre on a fixed node set, so two
runs give bitwise identical prices. `formulation` is `"lewis"` (one regular
integral) or `"gatheral"` (the two-probability decomposition), kept as an
independent cross-check. The price carries **absolute**, not relative, accuracy;
`HestonIVSurface` therefore declines low-vega wings rather than inverting noise.
See [Surface Models](surface_models.md#heston).

### Bates and BCC97

```python
from fast_vollib.processes import (
    BCC97, Bates, CIRShortRate, ConstantShortRate, ConstantVariance,
    HestonVariance, LognormalJumps, NoJumps,
)
from fast_vollib.pricing import bates_price, bcc97_price

Bates(variance, jumps, drift=0.0)                     # ('spot', 'variance')
Bates.risk_neutral(*, rate, variance, jumps, dividend_yield=0.0)
BCC97(variance, jumps, rates, dividend_yield=0.0)     # + 'short_rate'
BCC97.sample(*, initial_state, time_grid, n_paths, rng, antithetic=False,
             scheme="quadratic_exponential", rate_scheme="quadratic_exponential")
```

One configurable lattice, not several models: switching a component off is an
exact reduction, and the ones the arithmetic allows are **bitwise**. `drift`
means `r - q` *before* jump compensation, which the sampler subtracts itself;
`BCC97` is risk-neutral by construction and has no `drift` at all, because the
drift is a state. `scheme` and `rate_scheme` are separate because
`exact_transition` exists for the square-root rate and not for the variance.

```python
bates_price(*, forward, strike, maturity, v0, kappa, theta, vol_of_vol, rho,
            jump_intensity=0.0, mean_log_jump=0.0, jump_volatility=0.0,
            is_call=True, discount=1.0, formulation="lewis", n_nodes=768)

bcc97_price(*, spot, strike, maturity, v0, kappa, theta, vol_of_vol, rho,
            jump_intensity=0.0, mean_log_jump=0.0, jump_volatility=0.0,
            rate_kappa, rate_theta, rate_volatility, initial_rate,
            dividend_yield=0.0, is_call=True, formulation="lewis", n_nodes=768)
```

`bcc97_price` takes a **spot** rather than a forward: under a stochastic rate
the forward and the discount factor are outputs of the model, and
`bcc97_forward_measure` returns the pair it used. `rate_volatility=0.0` with
`rate_theta=initial_rate` is a flat deterministic rate, and the price is then
bitwise `bates_price` at that forward and discount.
