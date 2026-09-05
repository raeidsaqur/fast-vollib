# Simulation

The functional API and the instruments layer both answer "what is this worth
right now, in closed form". `fast_vollib.processes` and
`fast_vollib.simulation` answer a different question: what does this underlier
*do*, and what does a contract pay along the way.

The split is deliberate and it is four objects, not one:

| Layer | Holds | Never holds |
|---|---|---|
| `processes` | Dynamics and parameters — `GBM(drift, volatility)` | Random state, paths, devices, contracts |
| `simulation` | `Scenario`: the paths one call produced | Terms, engines, market snapshots |
| `instruments` | Contract terms | Arrays, paths, RNG, engines |
| `MonteCarloEngine` | The initial state, the discount factor, the estimator | A measure — see below |

Nothing in that chain infers anything about another link. An asset class does
not choose a process, a process does not choose a measure, and no analytic
route ever silently becomes a simulated one.

```python
import fast_vollib.simulation as sim   # imports no torch, jax, numba, or triton
```

## Quick start

```python
import numpy as np
from fast_vollib.instruments import EuropeanOption, VanillaMarketInputs
from fast_vollib.processes import GBM
from fast_vollib.simulation import MonteCarloEngine

option = EuropeanOption(
    underlier="SPX", option_type="call", strike=5000.0, maturity=0.75,
)
market = VanillaMarketInputs(underlying=5100.0, rate=0.04)

result = MonteCarloEngine().price(
    option, market,
    process=GBM.risk_neutral(rate=0.04, volatility=0.18),
    n_paths=100_000, n_steps=64, rng=0,
)
result.price, result.stderr, result.effective_samples
```

`market.volatility` is not read. Volatility is a *process* parameter here, and
reading both would let two disagreeing values silently pick one.

## Measure is yours, not the engine's

`market.rate` discounts. It is never used to rewrite a drift.

```python
risk_neutral = GBM.risk_neutral(rate=0.04, volatility=0.18)   # drift = r - q
stress       = GBM(drift=0.25, volatility=0.40)               # a physical view
```

Both simulate. Both discount at `market.rate` if you price with them. Only the
first produces a risk-neutral price, and the library will not tell you which
one you used — discounting a physical-measure simulation gives a number, and
the number is not a price. `GBM.risk_neutral` exists so that the modelling
decision is recorded in the code rather than assumed by an engine.

## Processes

`GBM` samples the closed-form log transition

```text
log S(t+dt) = log S(t) + (drift - volatility**2 / 2) * dt
              + volatility * sqrt(dt) * Z
```

directly, on an arbitrary and possibly irregular grid. It is not an Euler step:
there is no discretization bias to attribute a pricing difference to, and the
grid controls only how often the path is *observed* — which is exactly what a
discretely monitored contract needs. Two consequences are tested rather than
asserted: the first state is the initial state bit for bit, and zero volatility
gives the deterministic path exactly.

Parameters are stored as the objects you passed. A torch tensor an optimizer is
stepping stays that tensor, so gradients reach it.

`StochasticProcess` is a structural protocol — `state_names`, `params()`,
`sample()` — so your own dynamics can be driven by `simulate()`.

### The model lattice

`Bates(variance, jumps, drift)` is not one model but four, and switching a
component off is a *reduction* rather than a different implementation:

| `variance` | `jumps` | The model this is |
|---|---|---|
| `HestonVariance` | `NoJumps` | Heston (1993) |
| `HestonVariance` | `LognormalJumps` | Bates (1996), SVJ |
| `ConstantVariance` | `NoJumps` | Black–Scholes–Merton |
| `ConstantVariance` | `LognormalJumps` | Merton (1976) |

`BCC97(variance, jumps, rates, dividend_yield)` is the same lattice with the
short rate promoted from a number to a state, so the table gains a column and
two rows:

| `variance` | `jumps` | `rates` | The model this is |
|---|---|---|---|
| any of the four above | | `ConstantShortRate` | that model, unchanged — **bitwise** |
| `HestonVariance` | `NoJumps` | `CIRShortRate` | SVSI |
| `HestonVariance` | `LognormalJumps` | `CIRShortRate` | BCC97, SVSI-J |

`BCC97` is risk-neutral by construction and has no `drift` field: the drift *is*
a state, `r_t − q − λμ_J`. `dividend_yield` is the only level a caller supplies.
`rho` correlates the spot and the variance; the rate driver is independent of
both, which is the assumption Bakshi, Cao and Chen make — they call it severe,
and report that relaxing it did not improve empirical performance.

**The rate quadrature is the discounting quadrature.** The spot step uses
`0.5 * (r_k + r_{k+1}) * dt`, which is exactly what
`PathwiseShortRateDiscounting(rule="trapezoid")` accumulates, so the rate
cancels between the drift and the discount factor path by path rather than only
in expectation. That is what makes `exp(-int r) * S_T` a martingale in the
*discretized* model and not merely in its limit.

`scheme` and `rate_scheme` are separate arguments because the two state
variables do not offer the same set: `exact_transition` exists for the
square-root rate and not for the variance, whose spot coupling has no exact
joint transition.

```python
from fast_vollib.processes import Bates, HestonVariance, LognormalJumps

process = Bates.risk_neutral(
    rate=0.03, dividend_yield=0.0,
    variance=HestonVariance(kappa=2.0, theta=0.04, vol_of_vol=0.3, rho=-0.7),
    jumps=LognormalJumps(jump_intensity=0.5, mean_log_jump=-0.05, jump_volatility=0.2),
)
```

A **constant marker stores no level**: `ConstantVariance()` does not hold the
variance, which comes from `initial_state` like every other state. An API able
to hold two disagreeing values for one quantity eventually will.

The unions are **closed** — `HestonVariance | ConstantVariance`,
`LognormalJumps | NoJumps`, `CIRShortRate | ConstantShortRate` — and membership
is checked before any arithmetic or randomness. A `Heston` process is not
accepted as a variance component: it carries a drift, and a facade that ignored
it would price a different model from the one you described.

`drift` means what `Heston.drift` means, `r − q`, **before** jump compensation.
The sampler subtracts `jumps.drift_compensator` itself, so switching jumps on
does not require remembering to adjust the drift, and the *same* expression is
the one the characteristic function carries.

`params()` is flat with dotted keys — `"variance.kappa"`,
`"jumps.jump_intensity"`, `"drift"` — because the engine validates each value as
a scalar, and a bare `kappa` would not say which component it came from.

#### The draw-order contract

What makes a reduction checkable is that the randomness does not move when a
component is switched. Blocks are drawn in a fixed order into fixed **slots**:

| Slot | Block | Drawn when |
|---|---|---|
| 0 | diffusion normals `(n_paths, n_steps, 2)` | always — the identical call `Heston` makes |
| 1 | Poisson counts, then jump normals | `LognormalJumps` only |
| 2 | the short-rate path | `CIRShortRate` only |

`ConstantVariance` leaves column 0 of block 1 unused rather than drawing a
narrower block: a block whose shape depended on the configuration would make
every reduction a different stream.

Slots are assigned by *role*, not by which components are active, so switching
one off does not renumber the ones after it. On JAX that rests on
`jax.random.split(k, n)[i]` not depending on `n` — verified for typed and legacy
keys — which is also what lets a third slot be added without moving these.

The consequences are tested, and they are **bitwise**: `NoJumps` reproduces
`Heston` exactly under both schemes with and without antithetic pairing,
`LognormalJumps(jump_intensity=0)` reproduces `NoJumps` exactly, and `BCC97`
with `ConstantShortRate` reproduces the corresponding `Bates` configuration
exactly on NumPy, torch and JAX — nothing is drawn from slot 2.

One limit is worth stating, because the JAX guarantee is stronger than the
other two. On JAX a slot is a derived key, so switching a component off leaves
*every* other slot untouched: `LognormalJumps(jump_intensity=0)` and `NoJumps`
give identical rate paths. On NumPy and torch a slot is a position in one
advancing stream, so skipping a block moves the ones after it — the variance,
driven by slot 0 alone, is unchanged; the rate path is not.

### The CIR short rate

`CIRShortRate(kappa, theta, volatility)` evolves one state, `"short_rate"`,
under

```text
dr = kappa * (theta - r) dt + volatility * sqrt(r) dW
```

the same square-root diffusion Heston's variance follows, with the same
risk-neutral parameters as
[`CIRDiscountCurve`](fixed_income.md#the-cir-term-structure). It exists because
the analytic curve cannot carry a path, and a payoff whose discount factor and
whose value depend on the *same* realized rate needs one.

| `scheme` | Bias at the grid points | Backends |
|---|---|---|
| `quadratic_exponential` (default) | discretization, small | all |
| `full_truncation_euler` | discretization, larger — a transparent comparison, not a recommendation | all |
| `exact_transition` | none | NumPy, JAX |

`exact_transition` draws the non-central chi-square transition law directly
(Glasserman 2004, §3.4). It is exact **at the grid points** and says nothing
about the path between them, which is the distinction that lets the two error
sources be measured apart: run it, and whatever error remains in a
pathwise-discounted price is quadrature.

It is refused on torch, before any draw, because torch publishes no
generator-bound gamma sampler — `torch.distributions.Gamma.sample` takes no
generator and would read the global torch stream, so a "seeded" run would not
reproduce. The two discretizations need only normals and run everywhere.

`feller_ratio` is reported and never enforced, as on the curve. `volatility`
must be strictly positive: at zero there is nothing to simulate, and the limit
is exact in closed form via `process.discount_curve(initial_rate=...)`.

The numerical claims here are about `GBM`, `Heston` and `CIRShortRate`, the
processes this library implements.

## Randomness

Random number generation is the one part that cannot be written against a
single array namespace, so it is not pretended to be:

| Backend | Accepted `rng` | What it does |
|---|---|---|
| NumPy | `numpy.random.Generator`, or a non-negative integer seed | A seed builds a *local* generator; a supplied one advances |
| torch | `torch.Generator`, or a non-negative integer seed | A seed builds a local generator on the inferred device; a supplied one must already be on it |
| JAX | An explicit PRNG key from `jax.random.key` or `jax.random.PRNGKey` | Keys are immutable, so the same key reproduces a draw. Split it yourself for an independent stream; an integer seed is refused rather than silently turned into one |

**Backend selection.** Native arrays and native generators or keys *select* a
backend. Python numbers, lists, tuples, and integer seeds are neutral, and an
all-neutral call runs on NumPy. Two inputs selecting different backends is an
error raised before any draw — promoting one would move your data off its
device without saying so.

**Precision.** A NumPy scalar is neutral for selection and is not neutral for
promotion: NumPy and JAX treat one as a strong type and torch treats it as a
weak number, and dtype resolution follows each backend rather than ignoring the
scalar. A scenario therefore carries one precision throughout: single-precision
input stays single-precision all the way to `MCResult`.

**Reproducibility** is promised within one backend and library version. It is
never promised across backends: three generators with the same seed produce
three different streams, and nothing here claims otherwise.

**Antithetic sampling** draws `n_paths / 2` normals and places their exact
negatives in the second half, in order, so path `i` and path `i + n/2` are a
matched pair. A stateful generator therefore advances by the half-sample count.

For a **non-Gaussian** draw the second half is a *duplicate* rather than a
mirror: a negated Poisson count is not a Poisson count. That is not a weaker
version of the same idea — in a mixed draw such as the exact CIR transition,
a pair that shares its chi-square component while mirroring its normal is a
genuine antithetic pair, and duplicating is what makes the sharing exact. Where
a construction contains no normal at all, antithetic is refused rather than
silently producing two identical halves.

**JAX keys are split, never reused.** A sampler that needs one block of draws
uses the key you handed it. A sampler that needs more calls `split` once and
takes one child per block, and never draws from the parent afterwards — the
children are derived from the parent's own output, so a parent draw alongside
them would reuse the randomness they were built from. `GBM`, `Heston` and the
two discretized CIR schemes draw one block; `exact_transition` splits.

## Discounting

Under a deterministic rate the discount factor is a constant, so it commutes
with the expectation and `MonteCarloEngine` applies it to the average at the
end. Under a stochastic one it does not commute, because the payoff and the
factor are functions of the same path:

```text
E[exp(-int r) * X_T]  !=  E[exp(-int r)] * E[X_T]
```

A `P(0,T)`-times-expected-payoff shortcut is therefore a different quantity
rather than an approximation, and it is not offered. `DiscountingRule` is a
structural protocol with one method, and the two implementations are the two
things a caller might mean:

| Rule | Factor |
|---|---|
| `ConstantRateDiscounting(rate)` | `exp(-rate * time_grid[-1])` — today's behaviour, named |
| `PathwiseShortRateDiscounting(state_name, rule)` | `exp(-int_0^T r_u du)` from a named simulated state |

```python
import numpy as np
from fast_vollib.processes import CIRShortRate
from fast_vollib.simulation import PathwiseShortRateDiscounting

process = CIRShortRate(kappa=0.3, theta=0.04, volatility=0.1)
grid = np.linspace(0.0, 2.0, 17)
paths = process.sample(
    initial_state={"short_rate": 0.05}, time_grid=grid, n_paths=100_000, rng=0,
    scheme="exact_transition",
)
factors = PathwiseShortRateDiscounting().discount_factors(
    states=paths, time_grid=grid, state_names=process.state_names,
)
factors.mean()          # the zero-coupon bond price, to within its standard error
```

`rule` is `"trapezoid"` or `"left_riemann"` and is never inferred. The two
converge at different orders, so a caller comparing runs needs to know which
one produced a number; `left_riemann` is there to make the remaining
discretization error measurable, not as a recommendation.

The state is found **by name** in the `state_names` it is handed, never by
position. A name that is not there is an error listing the ones that are —
discounting a payoff by a spot instead of a rate produces a number, and the
number is not a price.

`MonteCarloEngine.price` takes one as the keyword-only `discounting`. Omitted,
it discounts at `market.rate` over the contract's maturity — bit for bit the
behaviour of every call written before the argument existed, and semantically
`ConstantRateDiscounting(market.rate)`. Supplied,
`market.rate` is **not read at all** — for the reason `market.volatility` is
never read: two values for one quantity would silently pick one.

## More than one state

A process whose first state is `"spot"` may evolve others, and the engine takes
them from the keyword-only `initial_state`:

```python
from fast_vollib.instruments import EuropeanOption, VanillaMarketInputs
from fast_vollib.processes import BCC97, CIRShortRate, HestonVariance, LognormalJumps
from fast_vollib.simulation import MonteCarloEngine, PathwiseShortRateDiscounting

option = EuropeanOption(
    underlier="ACME", option_type="call", strike=100.0, maturity=1.0,
)
process = BCC97(
    variance=HestonVariance(kappa=2.0, theta=0.04, vol_of_vol=0.3, rho=-0.7),
    jumps=LognormalJumps(jump_intensity=0.5, mean_log_jump=-0.05, jump_volatility=0.2),
    rates=CIRShortRate(kappa=0.5, theta=0.05, volatility=0.2),
    dividend_yield=0.01,
)
result = MonteCarloEngine().price(
    option,
    VanillaMarketInputs(underlying=100.0, rate=None),   # the model supplies the curve
    process=process,
    initial_state={"variance": 0.04, "short_rate": 0.03},
    discounting=PathwiseShortRateDiscounting(),
    n_paths=200_000, n_steps=64, rng=0,
)
```

`initial_state` may not carry `"spot"` — `market.underlying` is where that comes
from — and may not carry a name the process does not evolve. Every name it
*does* evolve is required: an engine that defaulted a missing variance would be
choosing a model on the caller's behalf. A one-state process takes nothing here.

The rate path above drives the drift **and** the discount factor, and it is the
same path in both roles: `BCC97`'s step integrates the rate with the trapezoid
that `PathwiseShortRateDiscounting`'s default accumulates. Discounting those
paths with `"left_riemann"` is a different quantity, not a coarser reading of
this one.

## Scenarios

```python
from fast_vollib.processes import GBM
from fast_vollib.simulation import simulate

scenario = simulate(
    "SPX",
    GBM.risk_neutral(rate=0.04, volatility=0.18),
    initial_state=5100.0,                 # mandatory; a bare number for a one-state process
    time_grid=np.linspace(0.0, 0.75, 65),
    n_paths=100_000,
    rng=0,
)
scenario.n_paths, scenario.n_steps, scenario.spot.shape
scenario.terminal("spot")                 # the value at the horizon, one per path
```

`simulate()` knows no contract and checks no maturity. A scenario is worth
generating on its own and may be evaluated against several contracts; matching
a horizon to a maturity is the payoff's concern. It is pure: it returns a new
scenario every call and attaches nothing to the asset, the reference, or the
process, so re-running a simulation cannot change what any of them mean.

The grid must be one-dimensional, finite, start at exactly zero, and have at
least two strictly increasing points. Irregular spacing is fine.

A scenario is an *execution value*, not an instrument. It is passed alongside a
contract the way market inputs are, nothing stores it, and it is **not
serializable**: a hundred thousand paths with a device and an autograd graph
attached are not a record anyone can read back, and encoding one raises.

### Ownership and mutation

NumPy buffers you hand to `Scenario.from_states` are **copied** and then marked
read-only, so a later edit to your array cannot change what a scenario means —
and freezing does not reach back into the array you still hold. Buffers
`simulate()` allocates are frozen in place, so a simulation pays for one
allocation rather than two.

Native state buffers are stored **as they arrive**, undetached and uncopied,
because detaching would cut the tape that is the reason for using them. The
small time grid is converted when needed to match the states' dtype, so a
scenario carries one precision. A torch state tensor therefore remains mutable
by whoever else holds it. The scenario's own attributes are frozen and it has
no mutating methods; that is the guarantee, and it is not the same as the array
being immutable.

Scenarios compare and hash by identity. Two drawn from the same seed hold equal
numbers but are different execution values, and a container of arrays has no
single defensible `==`.

## Terminal and path payoffs

`payoff_requirement(instrument)` reports what a contract's payoff needs, and
dispatch is explicit rather than inferred:

```python
from fast_vollib.instruments import payoff

payoff(european, scenario.terminal("spot"))   # TERMINAL: an array of last values
payoff(asian, scenario)                       # PATH: the whole scenario
scenario.payoff(instrument)                   # routes on the requirement for you
```

A bare array handed to a path payoff is refused before any arithmetic. It
carries no underlier, no observation times, and no way to tell whether its
horizon is the contract's maturity, so evaluating one would produce a number
for an unknown instrument. Conversely a terminal contract handed a scenario is
refused and pointed at `scenario.payoff(...)`; nothing is guessed.

Before any path payoff runs, the scenario checks that it describes the
contract: the same underlier — with asset class and currency compared wherever
both sides state one — and a final grid point equal to the contract's maturity
within `max(1e-12, 1e-12 * |maturity|)`. No payoff is ever evaluated on a
truncated or overlong horizon.

That tolerance widens to a few units in the last place when the grid is stored
in a coarser dtype than float64. A maturity of 0.1 is not representable in
binary32, so a single-precision grid ends about 1.5e-9 away from it — a
property of the storage, not of the contract, and single precision would
otherwise be unusable for every maturity that is not a dyadic rational. Double
precision is unaffected: four of its ulps are smaller than the floor, so the
rule reduces to the one above.

## Conventions that change what a contract is worth

These are contract meaning, not numerical settings. They are stated once, here
and in the contract types, and never defaulted inside an evaluator.

| Contract | Unit payoff, before notional |
|---|---|
| Binary call / put | `cash_amount` if `S_n > K` / `S_n < K`, else zero |
| Barrier | European intrinsic at maturity, times the in/out indicator |
| Fixed-strike Asian call / put | `max(A - K, 0)` / `max(K - A, 0)` |
| Floating-strike Asian call / put | `max(S_n - A, 0)` / `max(A - S_n, 0)` |
| Fixed-strike lookback call / put | `max(max(S) - K, 0)` / `max(K - min(S), 0)` |
| Floating-strike lookback call / put | `S_n - min(S)` / `max(S) - S_n` |
| Variance swap | `RV - strike_variance`, `RV = sum(log(S_i / S_{i-1})**2) / T` |

- **Asian fixings are `S_1 … S_n`** — the valuation date is excluded, because
  `S_0` is known when the contract is written. `A` is their arithmetic or
  geometric mean, as the contract's `averaging_method` says.
- **Barrier and lookback monitoring includes both endpoints**, and a barrier
  hit is **inclusive**: `max(S) >= barrier` for an up barrier,
  `min(S) <= barrier` for a down one.
- **Monitoring is discrete**, at the scenario's own observation times. A
  coarser grid prices a differently monitored contract; it does not approximate
  a continuously monitored one, and no Brownian-bridge or other continuity
  correction is applied. There are no barrier rebates.
- **At the strike exactly a binary pays nothing**, call and put alike: the
  comparison is strict on both sides, so the two can never both pay at a level
  the market treats as unresolved.
- **Realized variance has no sample-mean subtraction and no factor of 252.**
  The contract pays on the sum of squared log returns, and dividing by the year
  fraction already annualizes: for daily observations `T = n/252`, so `1/T`
  *is* the familiar `252/n`.
- Geometric averaging and realized variance take logarithms, so both refuse a
  non-positive path rather than returning a `NaN` that becomes an all-`NaN`
  price with no indication of which path caused it.

Payoffs are **undiscounted** and scaled by `notional`, which may be negative
for a short position.

## The estimator

```text
ordinary:    price = mean(X)
             stderr = std(X, ddof=1) / sqrt(n_paths)
             effective_samples = n_paths

antithetic:  Y = (X[:m] + X[m:]) / 2          with n_paths = 2m
             price = mean(Y)
             stderr = std(Y, ddof=1) / sqrt(m)
             effective_samples = m
```

`X` is the discounted payoff, path by path. Under antithetic sampling the two
halves are **one** sample rather than two: path `i` and path `i + m` share
their randomness by construction, so averaging each pair first is what makes
the remaining values independent, and the standard error divides by the number
of pairs.

`stderr` is the standard error *of the mean*, so `price ± 2 * stderr` is
roughly a 95% interval for the value the estimator converges to. It says
nothing about model error.

Whether antithetic sampling reduces variance depends on the payoff. The
estimator is correct either way, and nothing here claims it always helps —
`scripts/benchmark_monte_carlo.py` measures the ratio per product so you can
check rather than assume. It is worth roughly 1.2–1.7× on the directional
payoffs and it *hurts* on a variance swap, whose payoff is close to even in the
Brownian increment, so mirroring a path produces a near-duplicate of it rather
than an offsetting draw.

## Grids, path counts, and what is refused

Exactly one of `time_grid` and `n_steps` is supplied. Both would need a rule
for which wins; neither leaves the observation schedule undefined — and for a
path-dependent payoff the schedule is part of the contract's meaning. An
explicit grid must end at the contract's maturity; the engine does not extend
or truncate one.

`n_paths` is at least 2, or an even number of at least 4 with `antithetic`: a
standard error needs more than one sample, and an antithetic standard error
needs more than one pair. (`simulate()` itself permits a single path — it is
not estimating anything.)

Everything is validated before a single path is drawn, the RNG included. An
unsupported type, a zero maturity, a state the process evolves and
`initial_state` does not supply, a grid that does not end at maturity, a
non-scalar market input, a mixed array namespace, or an unusable generator each
raise with the sampling budget untouched.

### Zero maturity and futures

A contract expiring now has a payoff but no path: a strictly increasing grid
with at least two points cannot both start and end at zero. Rather than
simulate a fake step, the engine refuses and points at `payoff()`, which
answers the question exactly. The contract itself stays perfectly valid.

`Future` is refused for a different reason. Its terminal formula coincides with
a forward's, so pricing it here would return a plausible number — but a
future's economics are a stream of daily variation margin, which this library
does not model.

## Native output and differentiability

`return_native=True` returns `price` and `stderr` as backend arrays with dtype,
device, and autograd graph intact. The default extracts Python floats only at
the final formatting boundary. Numerical work never stages through host
memory; eager validation may read scalar inputs or a reduced domain flag to
decide whether to raise, and those values never enter the result or its graph.

| Route | Gradients | Why |
|---|---|---|
| Forward, variance swap | **Useful**, tested | Smooth in spot, rate, drift, and volatility on a positive path |
| European, Asian, fixed lookback | **Useful away from the kink**, tested | A single kink at the strike; gradcheck passes on configurations that do not straddle it |
| Floating lookback | Retained | Non-smooth where the extreme is attained at more than one point |
| Binary, barrier | **Retained, and zero almost everywhere** | The indicator's pathwise derivative is zero away from the boundary and undefined at it. The graph survives so the answer is `0.0` rather than `None`, which is the correct pathwise derivative and *not* a Greek to act on |
| Any route with `return_native=False` | No | Extraction ends the graph, by design |

`capabilities(Type).simulation_autodiff` records the backends on which a
simulated valuation keeps a graph, computed against what is installed. It means
tape retention and only that; the table above is the whole claim about which
gradients are worth using. A smoothed or likelihood-ratio estimator for the
discontinuous payoffs is not implemented, so no useful Greek is claimed for
them.

Contract terms are Python scalars and are not differentiable inputs; their
sensitivities are covered by finite differences rather than advertised as
autograd.

For a reproducible finite difference or a `gradcheck`, use an integer seed or a
fixed JAX key so that every function evaluation draws the *same* numbers.
Without common random numbers you are differentiating sampling noise.

## Errors

The layer fails closed, like the instruments package beside it. Each error also
subclasses the builtin you would reach for anyway, and `SimulationError`
catches the whole layer at once.

| Error | Raised when |
|---|---|
| `SimulationValidationError` | An input to sampling is malformed or outside its domain: a bad time grid, a path count, an initial state, a non-scalar market input, a mixed array namespace, a device mismatch, or an RNG the backend cannot use. |
| `UnsupportedProcessError` | The process is not served or violates the structural contract — a process whose first state is not `"spot"` handed to `MonteCarloEngine`, a state it evolves that `initial_state` does not supply, missing or invalid state metadata, a missing `sample()`, or a sample with the wrong shape, namespace, dtype, or device. |
| `ScenarioMismatchError` | The scenario does not describe the contract: a different underlier, or a horizon that is not the contract's maturity. |
| `UnsupportedInstrumentError` | The contract type has no route, or the instance is not eligible — a `Future`, or a contract maturing now. |
| `MissingMarketInputError` | A market field the request needed was not supplied; the message names it. |

```python
from fast_vollib.simulation import (
    ScenarioMismatchError, SimulationError, SimulationValidationError,
    UnsupportedProcessError,
)
```

## Serialization

Contracts serialize; scenarios do not. Every instrument type on this page
round-trips through the same strict versioned codec and the same checked-in
JSON Schema as the rest of the [instruments layer](instruments.md), and
`instrument_to_dict(scenario)` raises.

## Scope

American and Bermudan exercise, longstaff-schwartz or any other regression
method, calibration, control variates, quasi-random sequences, importance
sampling, streaming or chunked simulation, barrier rebates, and contract-level
fixing calendars are not implemented. The registry and
`MonteCarloEngine.supports()` list what is.

Stochastic volatility, jumps and stochastic rates *are* implemented, as
components of one configurable lattice — `Heston`, `Bates`, `BCC97`, and every
reduction between them; see [the process table](#processes) and
`fast_vollib.pricing.bcc97_price` for the transform each one is checked
against.
