# Retained forecasts

`fast_vollib.forecasting` describes targets and retained sample sets without
portfolio, pricing, broker or training dependencies. Numerical adapters load the
existing simulation and surface APIs only when called. No model is trained here.

## Draw once, query stored values

```python
from datetime import datetime, timezone
from fast_vollib.forecasting import ForecastQuery, ProcessDistribution
from fast_vollib.instruments import InstrumentRef
from fast_vollib.processes import GBM

origin = datetime(2026, 1, 1, tzinfo=timezone.utc)
query = ForecastQuery(origin=origin, times=(0.0, 0.25, 1.0),
    convention="act_365", targets=("asset",), kind="path")
law = ProcessDistribution(process=GBM(drift=0.05, volatility=0.2),
    initial_state=100.0, underlier=InstrumentRef(identifier="ACME", currency="USD"),
    query=query, measure="physical", backend_metadata={"backend": "numpy"},
    conditioning_cutoff=origin, observations_digest="example:observations")
draws = law.draw(n_samples=100, rng=17)
result = draws.query(query)
view = result.to_scenario_view(target_id="asset")
assert view.scenario.states.shape == (100, 3, 1)
```

The process adapter delegates to `simulation.simulate`; it never changes drift
because of a measure label. Backend, device and dtype follow the existing
simulation inputs. Its `backend_metadata` is caller-declared provenance, not a
backend selector. JAX needs an explicit PRNG key; NumPy and Torch accept their
existing RNG inputs. Same seed comparisons apply to a fixed backend, version and
grid, and do not establish consistency on arbitrary grids.

`FixedGridDrawSet.query()` only selects retained targets and stored times. It
never consumes randomness, interpolates, refines, extends or replays a sampler.
Repeated and subset queries preserve draw-set identity, sample IDs, values and
weights. Queries cannot change origin or day count. Each independent draw gets a
fresh `draw_set_id`, even if its numbers replay exactly from the same seed. Sample
IDs are text namespaced by that ID, rather than unqualified row numbers.

A path query requires at least two strictly increasing times beginning at zero.
Aware datetime inputs normalize to elapsed years using the required `act_365`,
`act_365_25` or `act_360` convention. Negative, nonfinite, mixed and duplicate times
are refused. A marginal query can request a single stored date without an origin
row. Single-date blocks answer one marginal time per query; make separate queries
for multiple marginal dates. A marginal cannot become a path by supplying an
initial spot later.

## Targets and blocks

Every `ForecastTarget` has a unique `target_id`, entity, `Quantity`, units and
resolution. Coordinates describe additional axes. `ReturnSpec` is required exactly
for return targets, naming simple/log, cumulative/incremental, base time and
whether distributions are included. A target contains no implicit conversion to
spot. Channel names are unique within a target; the same channel on two targets
remains distinct.

`ForecastResult` carries the requested target descriptors as well as its query.
Blocks are keyed by target ID, never by channel name. Each block shares the same
sample axis and the query's time convention:

| Block | Values and validity |
|---|---|
| `TrajectoryBlock` | `(sample, time, channel)`; validity `(sample, time)` |
| `MarginalBlock` | One date, `(sample, channel)`; validity `(sample,)` |
| `SurfaceBlock` | One date, canonical `SurfacePoints`, IV and validity `(sample, point)` |
| `CurveBlock` | One date, increasing tenors, values and validity `(sample, tenor)` |
| `EventBlock` | Sample-aligned tuples of ordered `MarkedEvent` records |

Missing or invalid cells retain their explicit masks. Valid entries must be finite;
valid IV is strictly positive. Spot projection consumes only its own target's
validity, so an invalid volume target does not invalidate a spot trajectory.
Reporting blocks own sealed host NumPy copies. Changing an input buffer or an
external native `Scenario` cannot change a retained report. This boundary detaches
native tensors and does not promise gradients or device retention. Use native
simulation directly when those are needed.

`ForecastMeta` declares measure, weighting, sampling dependence, target coupling,
point/distributed initial state, generator, draw-set identity, conditioning cutoff,
observation digest and optional checkpoint/seed/backend metadata. Conditioning
cannot use observations available after the forecast origin. Matching sample
counts alone never establishes joint target coupling.

Equal weighting uses `weights=None`; explicit importance weights must be finite,
nonnegative and have positive finite mass. They are normalized once, retaining
`original_weight_mass`. Stress sets use `measure="stress_only"` and
`weighting_scheme="none"`, with no probability weights. Generic nonfinancial
targets may have no measure. Financial projection requires one. These containers
provide no probability-of-profit or risk-summary methods.

## Financial projection

`ForecastResult.to_scenario_view(target_id=..., channels=("spot",))` requires an
identified spot `TrajectoryBlock`, an `InstrumentRef` underlier, a declared measure
and a grid starting at zero. `ScenarioView` owns the resulting scenario, IDs,
weights, validity, origin, convention, target identity and metadata. It has no
contract marks, inventory or P&L fields. Marginals, surfaces, curves and events
cannot masquerade as scenarios.

`returns_to_spot` is an explicit adapter requiring a matching underlier and positive
initial price. The declared return base must equal the forecast origin and the
origin return must be zero. It accumulates incremental log returns or compounds
incremental simple returns; cumulative returns are applied directly. Total returns
need a caller-supplied dividend mapping to ex-distribution prices. No initial row,
dividend convention or settlement interval is fabricated.

## Surface draws

`SurfaceDistributionAdapter(surface_distribution=..., points=..., date=...,
query=..., target=..., meta=..., market=...)` wraps the existing joint surface
sampling protocol for one aware date and fixed canonical `(k, T)` points. The date
must agree with the single-date marginal query. Drawing calls `sample()` once over
all points and checks sample count, coordinates and point identities/order.
Retained queries never resample. This adapter supplies neither temporal coupling
nor an arbitrary evaluable surface. `SurfaceBlock.surfaces` is currently required
to be `None`: retaining live mutable model objects has no ownership guarantee.

Forecast snapshots, context updates, external learned-model adapters, evaluation
and portfolio readouts are separate capabilities. A retained fixed-grid draw alone
does not establish calibrated probabilities or a continuous-time path law.
