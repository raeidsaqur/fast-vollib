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
| `MonteCarloEngine` | The initial spot, the discount factor, the estimator | A measure — see below |

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
`sample()` — so your own dynamics can be driven by `simulate()`. The numerical
claims here are about `GBM`, the process this library implements.

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
unsupported type, a zero maturity, a multi-state process, a grid that does not
end at maturity, a non-scalar market input, a mixed array namespace, or an
unusable generator each raise with the sampling budget untouched.

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
| `UnsupportedProcessError` | The process is not served or violates the structural contract — a multi-state process handed to `MonteCarloEngine`, missing or invalid state metadata, a missing `sample()`, or a sample with the wrong shape, namespace, dtype, or device. |
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

American and Bermudan exercise, stochastic-volatility processes, longstaff-
schwartz or any other regression method, calibration, control variates,
quasi-random sequences, importance sampling, streaming or chunked simulation,
barrier rebates, and contract-level fixing calendars are not implemented. The
registry and `MonteCarloEngine.supports()` list what is.
