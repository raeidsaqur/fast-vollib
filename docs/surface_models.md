# Surface models

The [arbitrage harness](surface.md) scores a surface somebody else produced. This
page is the other half: how a *model* enters that machinery, and why a calibrated
SVI slice, a Heston parameter set, a forecaster, and one draw of a generative
model are all scored by identical code.

## A model joins by producing one of two things

**A model joins this ecosystem by producing a definite surface, or a distribution
over definite surfaces.** Everything downstream — the arbitrage checks, the fit
and price metrics, the differentiable penalty, the report — consumes those two
and nothing model-specific.

A definite surface is a *map*, not a grid and not a training algorithm. Its whole
contract is `evaluate(points, *, market=None)`, returning one implied volatility
per queried row, in row order. `DefiniteIVSurface` is a structural protocol:
nothing to inherit, nothing to register.

```python
import numpy as np
from fast_vollib.surface import (
    DefiniteIVSurface, SurfaceGridSpec, SurfacePrediction, materialize_surface,
)

class ParabolicSmile:
    """w(k, T) = (a + b k^2) T -- a surface, and nothing else."""

    def __init__(self, level, curvature):
        self.level, self.curvature = level, curvature

    def evaluate(self, points, *, market=None):
        w = (self.level + self.curvature * points.k**2) * points.T
        return SurfacePrediction(points=points, iv=np.sqrt(w / points.T))

model = ParabolicSmile(level=0.04, curvature=0.05)
isinstance(model, DefiniteIVSurface)       # True

grid = SurfaceGridSpec(k=np.linspace(-0.4, 0.4, 17), T=[0.25, 0.5, 1.0, 2.0])
report = materialize_surface(model, grid).validate()
report.passed, report.sas                  # (True, 0.0)
```

That class shares no ancestry with `SVISurface`, `HestonIVSurface`, or a sampled
draw, and the harness cannot tell the four apart. A definite surface may also
*decline* a point — a prediction whose `valid` entry is `False` says "outside
what I will speak for", and `SurfaceDomainError` is the stricter alternative.
What it must not do is return a plausible number it does not stand behind; see
[Heston's wings](#the-wings-are-declined-on-purpose).

## Canonical coordinates and the adapters into them

The canonical coordinates are forward log-moneyness `k = log(K / F(T))` and
maturity `T` in years, with total variance `w = iv² · T` as the dependent
variable. They are canonical because the no-arbitrage conditions are stated in
them — `k` is what Durrleman's condition differentiates against. Strikes,
spot-moneyness, deltas, and day-count maturities are *inputs*, each converted by
an explicit adapter.

```python
import numpy as np
from fast_vollib.surface import (
    SurfaceMarket, points_from_forward_delta, points_from_spot_moneyness,
    points_from_strikes,
)

market = SurfaceMarket.from_spot(spot=100.0, T=[0.25, 1.0, 2.0], rate=0.04, carry=0.01)
strikes, T = np.array([90.0, 100.0, 110.0]), np.full(3, 1.0)
forward = market.forward_at(T)

by_strike = points_from_strikes(strikes, T, forward=forward)
np.round(by_strike.k, 6).tolist()            # [-0.135361, -0.03, 0.06531]

by_moneyness = points_from_spot_moneyness(
    np.log(strikes / 100.0), T, forward=forward, spot=100.0)
float(np.max(np.abs(by_moneyness.k - by_strike.k)))   # 1.8041124150158794e-16

by_delta = points_from_forward_delta([0.25, 0.5, 0.75], T, iv=[0.22, 0.20, 0.19])
np.round(by_delta.k, 6).tolist()             # [0.172588, 0.02, -0.110103]

by_strike.convention.to_dict()
# {'source': 'strike', 'maturity': 'year_fraction', 'market_source': None,
#  'notes': None}
```

Note the at-the-money strike: `k = -0.03`, not zero. The two moneyness measures
differ by the carry over the option's life — small at a week, not at two years —
and converting needs *both* spot and the forward. The delta adapter takes an
implied volatility as an input because the coordinate is defined in terms of the
quantity being observed (`k = w/2 − √w · N⁻¹(Δ_call)`), which leaves that
circularity with the caller. A conversion that is not recorded is not
reproducible, so every adapter attaches a `CoordinateConvention`, and points
built directly in canonical coordinates carry the identity one.

!!! note "The market is never inferred"
    Nothing substitutes `F = 1` or `r = 0` when a market is missing: a
    computation that needs one and lacks it raises `MissingMarketStateError`,
    because a price against an invented forward looks exactly like a price
    against the real one. `SurfaceMarket` *stores* the forward rather than
    deriving it — a real forward curve is quoted, not computed — and
    interpolation between pillars is a declared field, not a hidden convention.

## The value objects

| Object | Holds |
|---|---|
| `SurfacePoints` | `(k, T)` plus `surface_id` / `point_id`, and the convention that produced them |
| `SurfaceObservations` | points plus `iv`, and optional `bid`/`ask`, `weight`, `price`, `is_call` |
| `SurfacePrediction` | one `iv` per point, optional `sd` / `quantiles`, and a `valid` mask |
| `SurfaceSamples` | a sample axis of draws, each row a whole surface on the points |
| `SurfaceMarket` | forward, discount, and carry term structure, with provenance |
| `SurfaceGridSpec` | a declared `(Nk, Nt)` mesh — a question, never an answer |

Every stored array is an owned, read-only copy, so a caller mutating its own array
cannot change an already-computed number. Duplicate coordinates are legitimate —
two exchanges quoting the same strike and expiry are two observations — which is
why `point_id`, not `(k, T)`, is a row's identity. `SurfaceObservations` is the
promoted form of the diagnostics package's `SurfaceQuotes`; the old name still
resolves to this class.

Predictions are permissive where observations are strict: `SurfacePrediction`
accepts negative and non-finite implied volatilities, because that is what an
unconstrained network or a diverged optimizer emits. They are flagged invalid
rather than dropped — an evaluation that discarded them would report the error of
the points a model happened to get right.

```python
import numpy as np
from fast_vollib.surface import (
    SurfaceGridSpec, SurfaceMarket, SurfaceObservations,
    evaluate_prediction, materialize_surface,
)
from fast_vollib.surface.fitting import SVICalibrator

k = np.array([-0.30, -0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20, 0.30])
iv = np.array([0.286, 0.259, 0.235, 0.225, 0.216, 0.209, 0.204, 0.199, 0.203])
observations = SurfaceObservations(
    k=k, T=np.full(k.size, 1.0), iv=iv, surface_id="SPX-2026-08-31",
)

surface = SVICalibrator().fit(observations)          # a DefiniteIVSurface
prediction = surface.evaluate(observations.points)

market = SurfaceMarket.flat(forward=5000.0, rate=0.04, source="synthetic flat curve")
grid = SurfaceGridSpec(k=np.linspace(-0.35, 0.35, 15), T=[0.5, 1.0, 2.0], market=market)
mesh = materialize_surface(surface, grid)

evaluation = evaluate_prediction(
    prediction, observations, market=market, grid=grid, materialized=mesh)
round(evaluation.iv_rmse, 6)           # 0.000133
evaluation.coverage                    # 1.0
round(evaluation.price_rmse, 6)        # 0.222923  (index points, on F = 5000)
evaluation.arbitrage.passed            # True
evaluation.verification.value          # 'empirical_finite_grid'
```

A calibrator holds configuration only, never the last fit: a stateful one would
make the second of two otherwise identical runs depend on the order they ran in.

## Four protocols, not one hierarchy

| Protocol | Method | Returns |
|---|---|---|
| `SurfaceCalibrator` | `fit(observations, *, rng=None)` | a definite surface |
| `ConditionalSurfaceEstimator` | `condition(context, *, rng=None)` | a definite surface |
| `SurfaceForecaster` | `forecast(history, horizon, *, rng=None)` | a definite surface **or** a distribution |
| `GenerativeSurfaceModel` | `distribution(context, *, horizon=None)` | a `SurfaceDistribution` |

One hierarchy would have been the wrong shape. Calibrating a slice to today's
quotes, conditioning an already-trained network on a context set, forecasting
from a history, and sampling from a generative model are four lifecycles, and a
single `fit(data) -> surface` makes three of them lie: a trained network's `fit`
would mean "train", a forecaster's would silently drop the horizon, and a
generative model would collapse a distribution to a point estimate before anyone
asked it to.

**Training is not conditioning**, and no protocol here means "learn parameters
from a corpus". `fit` fits *one* surface to *one* set of observations;
`condition` applies weights learned elsewhere, which is cheap and changes
nothing — calling that `fit` would make "how long does fitting take" unanswerable
across families. Training lifecycles differ too much to have a useful common
contract yet, so dataset selection, schedules, and checkpoint policy live in
whatever runs the experiment.

```python
import numpy as np
from fast_vollib.surface import (
    DefiniteIVSurface, ForecastHorizon, SurfaceDistribution, SurfaceForecaster,
    SurfaceObservations, SurfacePoints,
)
from fast_vollib.surface.fitting import FlatVolatilityCalibrator, PersistenceForecaster
from fast_vollib.surface.generative import GaussianFieldSurfaceDistribution

k = np.array([-0.2, -0.1, 0.0, 0.1, 0.2])
history = [
    SurfaceObservations(k=k, T=np.full(5, 1.0), iv=0.22 + 0.05 * k**2 - 0.04 * k),
    SurfaceObservations(k=k, T=np.full(5, 1.0), iv=0.20 + 0.05 * k**2 - 0.04 * k),
]

forecaster = PersistenceForecaster(calibrator=FlatVolatilityCalibrator())
tomorrow = forecaster.forecast(history, ForecastHorizon(steps=1, step_years=1 / 252))
isinstance(forecaster, SurfaceForecaster)        # True
isinstance(tomorrow, DefiniteIVSurface)          # True

distribution = GaussianFieldSurfaceDistribution(base=tomorrow, volatility=0.05)
isinstance(distribution, SurfaceDistribution)    # True
samples = distribution.sample(
    SurfacePoints(k=k, T=np.full(5, 1.0)), n_samples=8, rng=20260831,
)
samples.n_samples, samples.n_points              # (8, 5)
```

`ForecastHorizon` counts observation steps and carries an optional `step_years`,
supplied when a forecaster needs it to age maturities — a one-day-ahead forecast
of a 30-day option is a forecast of a 29-day option — and never guessed. `rng` is
*required* on `sample`: an evaluation that cannot be replayed is not evidence.

## Materialization, and the interpolation policy

Arbitrage conditions are *discrete* statements about neighbouring nodes: a
butterfly needs three strikes, a calendar needs two maturities. Which nodes those
are is part of the result, and two models compared on different meshes have not
been compared. `materialize_surface(surface, grid)` asks the model for exactly
the grid's nodes, in C order over `(Nk, Nt)`, and reshapes the answer back;
declined nodes become `NaN`, and a surface answering on different points than it
was asked about is caught here rather than producing a transposed mesh.

The reverse conversion is `GridIVSurface`, which wraps a grid so it can be asked
about points that are not nodes. `IVSurface` itself gains no `evaluate()` method,
so a grid is never *silently* treated as a continuous surface: reading one is a
named object carrying a named `policy`. The policy matters, because it changes
the answer and the report attributes the difference to the model:

```python
import numpy as np
from fast_vollib.surface import (
    GridIVSurface, IVSurface, SurfaceGridSpec, materialize_surface,
)

k = np.linspace(-0.2, 0.2, 9)
coarse = IVSurface.from_logmoneyness(
    k, np.array([0.25, 1.0]),
    np.column_stack([np.full(k.size, 0.40), np.full(k.size, 0.20)]),
)
# Total variance is flat in T: 0.40**2 * 0.25 == 0.20**2 * 1.0 == 0.04.

fine = SurfaceGridSpec(k=k, T=[0.25, 0.5, 1.0])
for policy in ("total_variance", "implied_volatility"):
    mesh = materialize_surface(GridIVSurface(coarse, policy=policy), fine)
    w = np.asarray(mesh.iv[4]) ** 2 * np.asarray(fine.T)
    print(policy, round(float(mesh.iv[4, 1]), 6), np.round(w, 6).tolist(),
          bool(mesh.validate().passed))
```

```text
total_variance 0.282843 [0.04, 0.04, 0.04] True
implied_volatility 0.333333 [0.04, 0.055556, 0.04] False
```

Interpolating implied volatility linearly across maturities manufactured a
calendar violation out of a surface whose total variance is exactly flat.
Interpolating total variance did not, which is why `'total_variance'` is the
default of the three (`'nearest'` is the third) — but the field is named on the
object rather than assumed at the call site, because the report will attribute
the difference to the model. Interpolated nodes are therefore labelled — a grid
built by
interpolation carries a `native_mask` (`GridIVSurface.native_mask_for(grid)`
computes it) and the report separates `native` violations from
`interpolation_induced` ones on that basis — and outside the mesh
`extrapolation` defaults to `'invalid'`, reporting the point unanswered rather
than inventing a value.

## Capability discovery

A benchmark runner, a web API, or a notebook needs to know which algorithms
exist, what they produce, and which cannot run here. Reading that off a source
tree is how a client ends up advertising a model deleted two releases ago.

```python
from fast_vollib.surface import list_algorithms

for entry in list_algorithms():
    spec = entry.spec
    mark = "ok " if entry.available else "-- "
    print(f"{mark}{spec.public_id:<14} {spec.family:<12} -> {spec.output}")
```

```text
ok flat           calibrator   -> definite
ok svi            calibrator   -> definite
ok ssvi           calibrator   -> definite
ok spline         calibrator   -> definite
ok heston         calibrator   -> definite
ok factor-pca     calibrator   -> definite
ok state-space    forecaster   -> definite
ok gaussian-field generative   -> distribution
ok persistence    forecaster   -> definite
```

Two distinctions carry weight. **A family is not an output kind**: `family` is
the lifecycle, `output` is what a caller gets back, and a forecaster may return
either, so a client dispatching on family alone would have to inspect the
returned object to learn what it asked for. **Unavailable is not absent**: an
algorithm is listed only when implemented, and reported unavailable with a
machine-readable code (`optional_dependency`, `backend`, `checkpoint`) when it
cannot run — a model never written is not listed, because listing it would be
advertising. The registry is built once from a fixed list and cannot be mutated,
so `'svi'` cannot change meaning with import order, and describing an algorithm
never imports it.

```python
from fast_vollib.surface import build_algorithm, capabilities_document, get_algorithm

get_algorithm("ssvi").spec.support.to_dict()
# {'backends': ['numpy'], 'dtypes': ['float64'], 'devices': ['host'],
#  'gradients': False}

type(build_algorithm("svi", {"n_starts": 1})).__name__      # 'SVICalibrator'

sorted(capabilities_document())
# ['algorithms', 'library', 'library_version', 'schema']
```

Each entry carries a closed JSON Schema for its constructor keywords and
`build_algorithm` validates against it first, so a typo in a stored configuration
is an error naming the field: `build_algorithm("svi", {"nstarts": 1})` raises
`SurfaceValidationError: Unknown configuration field(s) 'nstarts' for 'svi'.` The
support matrix is declared honestly rather than optimistically — claiming a torch
backend because the *surface* SSVI returns evaluates on tensors would make "does
this fit on GPU" unanswerable. The document renders under the closed
`fast-vollib-surface-capabilities-v1` schema.

## Evaluation: coverage is part of the error

A model that answered a fifth of the points and fitted those perfectly has an
RMSE of zero, and reporting that number alone is how a partially-predicted
surface comes to look like the best one.

```python
import numpy as np
from fast_vollib.surface import (
    SurfaceObservations, SurfacePrediction, evaluate_prediction,
)

k = np.array([-0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4])
observations = SurfaceObservations(k=k, T=np.full(k.size, 1.0), iv=0.20 + 0.05 * k**2)

# Answers only the five points nearest the money -- and nails them.
narrow_iv = np.where(np.abs(k) <= 0.2, observations.iv, np.nan)
narrow = evaluate_prediction(
    SurfacePrediction(points=observations.points, iv=narrow_iv), observations,
)
narrow.target_count, narrow.valid_count, narrow.invalid_count   # (9, 5, 4)
narrow.iv_rmse, round(narrow.coverage, 4)                       # (0.0, 0.5556)

# Answers every point, to within a tenth of a volatility point.
wide = evaluate_prediction(
    SurfacePrediction(points=observations.points, iv=observations.iv + 0.001),
    observations,
)
wide.target_count, wide.valid_count, wide.invalid_count         # (9, 9, 0)
round(wide.iv_rmse, 12), wide.coverage                          # (0.001, 1.0)
```

Target, valid, and invalid counts are three integers and are never collapsed into
one. An unmeasured error is `None`, not zero: `price_rmse` without a market and
`inside_spread_fraction` without bid/ask are both `None`, because a zero standing
in for "not measured" would read as perfect agreement. The same accounting
repeats `by_region` and `by_maturity`, and a prediction must answer the
observations' own points, compared exactly — align with `align_predictions`
first if the rows are not already matched.

**A finite grid is not a certificate**, so an attached arbitrage report always
carries a `VerificationLevel` saying what kind of statement it is:

| Level | Means |
|---|---|
| `empirical_finite_grid` | Checked on a stated finite mesh and held there. Says nothing about the continuum between nodes, or about a different mesh. |
| `training_penalty` | A penalty on the violations entered the training objective. A penalty changes what a model is likely to do; it decides nothing. |
| `mathematical_guarantee` | The parameterization admits no violation, so no grid can find one — SSVI's non-decreasing `theta` is of this kind for the calendar condition. |
| `external_claim_unverified` | Somebody asserted the property and this library did not check it. It is a level so the claim can be carried without being promoted. |

Those four are different, and a report that does not distinguish them will
eventually be read as the strongest of them. The default is
`empirical_finite_grid`, because that is what was measured.

!!! note "A grid needs its materialized mesh"
    `evaluate_prediction(..., grid=...)` also requires `materialized=`. Passing a
    grid alone raises rather than re-evaluating the model behind the caller's
    back, so the arbitrage report describes the same evaluation the metrics do.

## Scoring a distribution: every draw, not the mean

Average a hundred arbitrageable surfaces and the butterfly violations, which
point in different directions on different draws, can cancel: the mean comes back
convex and every draw the model would actually produce is inadmissible. Averaging
can also work the other way. Either way the number describes the average, and
nobody trades the average.

So `evaluate_samples` materializes **every** draw as its own definite surface,
checks it by the same hard conditions a deterministic model faces, and only then
aggregates. `GaussianFieldSurfaceDistribution` multiplies a base surface's total
variance by a log-normal random field with a squared-exponential covariance in
`(k, T)`, drawn jointly from one Cholesky factor so a draw is a smooth surface
rather than a point cloud, and corrected so the mean total variance is the base
surface's. Set the correlation length in `k` to the grid spacing and the draws
turn rough:

```python
import numpy as np
from fast_vollib.surface import SurfaceGridSpec
from fast_vollib.surface.fitting import FlatIVSurface
from fast_vollib.surface.generative import (
    GaussianFieldSurfaceDistribution, evaluate_samples,
)

grid = SurfaceGridSpec(k=np.linspace(-0.3, 0.3, 13), T=[0.25, 0.5, 1.0])
rough = GaussianFieldSurfaceDistribution(
    base=FlatIVSurface(level=0.2),
    volatility=0.10,
    length_scale_k=0.05,        # the grid spacing itself
    length_scale_T=1.0,
)
report = evaluate_samples(rough, grid, n_samples=200, rng=20260831)

report.n_samples, report.n_nodes, report.grid_shape   # (200, 39, (13, 3))
report.any_violation_probability                      # 0.93
tuple(round(b, 4) for b in report.any_violation_interval)
# (0.8859, 0.9578)
report.condition_probability
# (('butterfly', 0.925), ('calendar', 0.0), ('vertical', 0.005), ('bound', 0.0))

report.mean_surface_metrics["passed"]                 # 1.0
report.mean_surface_metrics["sas"]                    # 0.0
report.median_surface_metrics["passed"]               # 1.0
```

**Ninety-three percent of the draws violate a no-arbitrage condition, and the
mean surface passes every check with a Static-Arbitrage Score of exactly zero.**
Reporting the mean alone would have called this model clean. The mean and median
metrics *are* reported, and are labelled as summaries of the cloud rather than as
members of it.

Lengthen the correlation and the same field stops violating:

```python
smooth = GaussianFieldSurfaceDistribution(
    base=FlatIVSurface(level=0.2), volatility=0.10,
    length_scale_k=1.0, length_scale_T=1.0,
)
smooth_report = evaluate_samples(smooth, grid, n_samples=200, rng=20260831)
smooth_report.any_violation_probability     # 0.0
tuple(round(b, 4) for b in smooth_report.any_violation_interval)
# (0.0, 0.0188)
smooth_report.worst_severity                # 0.0
```

Note the interval on that zero: two hundred clean draws do not establish that
violations are impossible, only that the rate is below about two percent. Every
probability carries a Monte Carlo standard error and a Wilson interval, which
stays sensible at zero and one where the textbook normal interval collapses to a
point — and zero and one are where such a report lands most often. It renders
under the closed `fast-vollib-generative-arbitrage-v1` schema.

## Heston

Heston arrives in three pieces that know nothing about each other: the dynamics
(`fast_vollib.processes.Heston`), the Fourier pricer
(`fast_vollib.pricing.heston_price`), and the surface and calibrator
(`fast_vollib.surface.fitting`).

```python
from fast_vollib.pricing import heston_price
from fast_vollib.processes import Heston

params = dict(kappa=2.0, theta=0.04, vol_of_vol=0.3, rho=-0.7)
process = Heston.risk_neutral(rate=0.0, **params)
process.state_names                      # ('spot', 'variance')
round(process.feller_ratio, 6)           # 1.777778

lewis = heston_price(forward=100.0, strike=90.0, maturity=1.0, v0=0.04, **params)
gatheral = heston_price(
    forward=100.0, strike=90.0, maturity=1.0, v0=0.04, **params,
    formulation="gatheral",
)
float(lewis)                             # 13.80289575338324
bool(abs(lewis - gatheral) < 1e-10)      # True
```

The two formulations are mathematically identical and numerically independent —
`'lewis'` is one integral with a regular integrand, `'gatheral'` assembles
`F·P₁ − K·P₂` from two whose integrands have a removable singularity at the
origin — and their agreement stands in for a published reference table. The
characteristic function uses the "little trap" branch of the square root; the
textbook form is algebraically identical and numerically wrong past a maturity of
a year or two, where the complex logarithm wraps. Quadrature is Gauss-Legendre on
a fixed node set, so two runs give bitwise identical prices. The Feller condition
is reported, never enforced: real calibrations violate it routinely.

### There is no exact sampling scheme

`GBM` samples its closed-form log transition, so its only discretization is how
often a path is *observed*. The square-root variance has no elementary exact
transition, so every scheme here carries a bias that shrinks with the step size
and vanishes at no finite one. Two are provided and the choice is a parameter,
because a Monte Carlo price disagreeing with the Fourier price by a tenth of a
volatility point is a discretization artifact and a caller has to be able to tell
that from a bug.

```python
import numpy as np
from fast_vollib.pricing import heston_price
from fast_vollib.processes import Heston

params = dict(kappa=0.5, theta=0.04, vol_of_vol=0.9, rho=-0.9)
reference = float(heston_price(forward=100.0, strike=100.0, maturity=1.0, v0=0.04, **params))
process = Heston.risk_neutral(rate=0.0, **params)
round(process.feller_ratio, 6)                             # 0.049383 -- violated

for scheme in ("quadratic_exponential", "full_truncation_euler"):
    for n_steps in (12, 48, 192):
        paths = process.sample(
            initial_state={"spot": 100.0, "variance": 0.04},
            time_grid=np.linspace(0.0, 1.0, n_steps + 1),
            n_paths=100_000, rng=20260831, scheme=scheme,
        )
        payoff = np.maximum(paths[:, -1, 0] - 100.0, 0.0)
        stderr = payoff.std(ddof=1) / np.sqrt(payoff.size)
        print(f"{scheme:<22} {n_steps:>4}  {payoff.mean():7.4f}  "
              f"{payoff.mean() - reference:+.4f}  +/- {stderr:.4f}")
```

```text
quadratic_exponential    12   4.6888  -0.0027  +/- 0.0139
quadratic_exponential    48   4.6897  -0.0019  +/- 0.0139
quadratic_exponential   192   4.7075  +0.0160  +/- 0.0138
full_truncation_euler    12   5.9917  +1.3001  +/- 0.0190
full_truncation_euler    48   4.9706  +0.2791  +/- 0.0148
full_truncation_euler   192   4.7605  +0.0690  +/- 0.0140
```

The Fourier price is `4.6915`. At a Feller ratio of 0.05 the naive scheme is off
by 1.30 at twelve steps — twenty-eight percent, and sixty-eight standard errors —
and still off by 0.069 at a hundred and ninety-two. Andersen's QE scheme, which
matches the first two moments of the exact non-central chi-squared transition and
switches to an exponential-with-atom approximation where that distribution is
nearly degenerate, sits within about one standard error at every step count.
Neither is exact, and this table is why the choice is a parameter rather than a
default nobody reads.

### Calibration

`HestonCalibrator` fits five parameters to a whole surface at once, measuring its
residual in implied volatility by default — a price residual is dominated by
at-the-money options, where the vega is largest.

```python
import numpy as np
from fast_vollib.surface import SurfaceObservations
from fast_vollib.surface.fitting import HestonCalibrator, HestonIVSurface, HestonParameters

truth = HestonParameters(v0=0.04, kappa=2.0, theta=0.045, vol_of_vol=0.35, rho=-0.65)
k = np.tile(np.linspace(-0.25, 0.25, 9), 3)
T = np.repeat([0.25, 1.0, 2.0], 9)
observations = SurfaceObservations(
    k=k, T=T, iv=HestonIVSurface(parameters=truth).implied_volatility(k, T),
    surface_id="synthetic-heston",
)

fitted = HestonCalibrator().fit(observations)          # about 0.4 s here
p = fitted.parameters
tuple(round(v, 9) for v in (p.v0, p.kappa, p.theta, p.vol_of_vol, p.rho))
# (0.04, 2.0, 0.045, 0.35, -0.65)

residual = fitted.evaluate(observations.points).iv - observations.iv
f"{np.sqrt(np.mean(residual ** 2)):.2e}"               # '7.57e-14'
```

Starting points are fixed rather than random, so two runs on the same quotes agree
exactly, and there are four because the objective has well-known local minima — a
low-vol-of-vol, high-mean-reversion fit and a high-vol-of-vol, low-mean-reversion
one describe the same smile almost equally well. The finite-difference step is
`1e-4`, not SciPy's ~`1.5e-8`: each residual is a quadrature followed by a
root-find and carries a noise floor near `1e-10`, so the default differences two
values whose difference *is* that noise.

`HestonIVSurface` needs no market state, which is a property of the coordinates:
in forward log-moneyness the undiscounted call per unit forward depends on no
level and no rate, so it prices at `F = 1`, `K = e^k`, and inverts. A `market`
argument is accepted and ignored.

### The wings are declined, on purpose

The Fourier price computes a call as `F` minus a quantity approaching `F`, so it
has **absolute**, not relative, accuracy — however many quadrature nodes are used.
Measured across maturities from a month to three years and log-moneyness out to
±1.2, that noise is about `3e-12` per unit forward. Divided by Black vega it
becomes an uncertainty on the implied volatility, and where that exceeds `1e-6`
the surface reports the point invalid:

```python
import numpy as np
from fast_vollib.surface import SurfacePoints
from fast_vollib.surface.fitting import HestonIVSurface, HestonParameters

surface = HestonIVSurface(
    parameters=HestonParameters(v0=0.04, kappa=2.0, theta=0.04, vol_of_vol=0.3, rho=-0.7)
)
k = np.array([-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5])
T = np.full(k.size, 0.25)

prediction = surface.evaluate(SurfacePoints(k=k, T=T))
prediction.valid.tolist()
# [False, False, True, True, False, False, False]
np.round(prediction.iv, 6).tolist()
# [nan, nan, 0.286987, 0.195729, nan, nan, nan]
[f"{value:.2e}" for value in surface.inversion_uncertainty(k, T)]
# ['nan', '2.83e-04', '8.38e-09', '1.51e-11', '1.07e-02', 'nan', 'nan']
```

Returning the last resolvable value would put a fabricated number exactly on the
wing where every surface fit is judged. `inversion_uncertainty` is exposed so a
caller can see *why* a point was declined rather than inferring it from a `NaN`.

A calibrated Heston surface is arbitrage-free *in the model*, because its prices
come from a genuine martingale measure — which is not the same as passing a
discrete check on a finite mesh, where the wings can trip a second divided
difference dominated by inversion noise. The report says which of the two it
measured, and the model's guarantee is never promoted into a claim about the grid.

## What is not here yet

Production learned surface models — amortized conditional estimators, neural
operators, diffusion models over surfaces — are future work. The
`ConditionalSurfaceEstimator` protocol and the `conditional` family exist for
them, and no algorithm currently declares either: the registry lists what is
implemented, and listing an unwritten model would be advertising.
`GaussianFieldSurfaceDistribution` is the only distribution that ships, and it is
deliberately not a *good* model — no dynamics, no conditioning, no state.

Also absent: a common training contract; a term structure of Heston parameters,
which is a different model and is not silently accepted; an exact Heston sampler;
and gradients through any calibration — `support.gradients` is `False` for every
calibrator here.
