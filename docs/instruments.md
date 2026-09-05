# Instruments

fast-vollib's core is functional. `fast_black_scholes(flag, S, K, t, r, sigma)`
and its neighbours take arrays and return arrays, and that stays the canonical,
fastest API. What it has no vocabulary for is *what* is being priced: nothing in
it **is** an option, an underlier, or a forward.

`fast_vollib.instruments` adds that nominal layer without moving any
mathematics into it. Every number an adapter returns comes from the same kernel
you would have called yourself — the test suite asserts exact array equality
against direct calls — so using the instruments layer costs you nothing numerically
and cannot drift from the functional API as either side changes.

```python
import fast_vollib.instruments as inst   # imports no torch, jax, numba, or triton
```

## Quick start

```python
import numpy as np
from fast_vollib.instruments import (
    Asset, AssetClass, EuropeanOption, VanillaMarketInputs, price_instrument,
)

spx = Asset(identifier="SPX", asset_class=AssetClass.INDEX, currency="USD")
option = EuropeanOption(
    instrument_id="SPX-C-5000",
    underlier=spx.ref(),
    option_type="call",       # "c" and "p" work too
    strike=5000.0,
    maturity=0.75,            # a year fraction, never a date
    notional=100.0,
)

market = VanillaMarketInputs(underlying=5100.0, rate=0.04, volatility=0.18)
price = price_instrument(option, market, model="black_scholes")
```

The contract holds terms. The market holds observations. They meet at the call
site and nowhere else — a contract never stores a spot price, so it is never
stale, and two identical options are equal whenever they have identical terms.

### A book at a time

A batch is the execution unit: one batch is one fused kernel call, never a
Python loop over contracts.

```python
from fast_vollib.instruments import EuropeanOptionBatch, greeks_instrument

chain = EuropeanOptionBatch.from_arrays(
    option_type=np.where(np.arange(20_000) % 2 == 0, "c", "p"),
    strike=np.linspace(4_000.0, 6_000.0, 20_000),
    maturity=0.75,
    underlier="SPX",
)
market = VanillaMarketInputs(underlying=5100.0, rate=0.04, volatility=0.18)

prices = price_instrument(chain, market, model="black_scholes")
greeks = greeks_instrument(chain, market, model="black_scholes")   # dict of arrays
```

`from_arrays` builds no per-row Python object — not a contract, not even an
underlier reference. At 100k rows it is roughly 9x faster than building the
scalar objects and handing them to `from_instruments`, before you count the
cost of building them.

### Implied volatility, with the solver named

```python
from fast_vollib.instruments import implied_volatility_instrument

quotes = VanillaMarketInputs(underlying=5100.0, rate=0.04, price=prices)
iv = implied_volatility_instrument(chain, quotes, model="black_scholes")          # Jäckel
iv = implied_volatility_instrument(chain, quotes, model="black_scholes",
                                   solver="halley")
```

`market.price` is the price of the *position*: it is divided by the contract's
notional before inversion, so a quote for 100 contracts inverts to the same
volatility as a quote for one.

### Path-dependent contracts

An average-rate, barrier, lookback, or variance contract needs the whole
trajectory rather than its last point, so its payoff takes a
[`Scenario`](simulation.md):

```python
from fast_vollib.instruments import AsianOption
from fast_vollib.processes import GBM
from fast_vollib.simulation import MonteCarloEngine, simulate

asian = AsianOption(
    underlier="SPX", option_type="call", strike=5000.0,
    averaging_method="arithmetic", strike_convention="fixed", maturity=0.75,
)

result = MonteCarloEngine().price(
    asian, VanillaMarketInputs(underlying=5100.0, rate=0.04),
    process=GBM.risk_neutral(rate=0.04, volatility=0.18),
    n_paths=100_000, n_steps=64, rng=0,
)
result.price, result.stderr
```

`payoff_requirement(instrument)` says which kind of state a contract needs, and
a bare array handed to a path payoff is refused rather than interpreted — it
carries no underlier, no observation times, and no way to tell whether its
horizon is the contract's maturity. See [Simulation](simulation.md) for the
monitoring, fixing, and variance conventions, which are contract meaning rather
than numerical settings.

### Payoffs keep your array type

```python
import torch
from fast_vollib.instruments import payoff

terminal = torch.tensor([4_900.0, 5_200.0], dtype=torch.float64, requires_grad=True)
cashflow = payoff(option, terminal)      # torch tensor, same device, tape intact
cashflow.sum().backward()
```

## The five layers

| Layer | What it holds | Key names |
|---|---|---|
| Contracts | Terms only: frozen, slotted, keyword-only dataclasses | `Asset`, `Forward`, `Future`, `EuropeanOption`, `BinaryOption`, `AsianOption`, `BarrierOption`, `LookbackOption`, `VarianceSwap`, `InstrumentRef` |
| Batches | Columnar host arrays; the unit of execution | `EuropeanOptionBatch`, `moneyness`, `log_moneyness`, `time_to_maturity` |
| Market inputs | Valuation-time observations, never stored on a contract | `VanillaMarketInputs` |
| Payoffs & adapters | Backend-native payoffs; thin wrappers over the kernels | `payoff`, `price_instrument`, `greeks_instrument`, `implied_volatility_instrument` |
| Discovery | What exists and what can be done with it | `instrument_types`, `capabilities`, the error hierarchy |

## What each type supports

Generated from `instrument_types()`, so it cannot claim a capability the
registry does not have.

<!-- BEGIN generated: instrument capability table -->

| Type | `type_id` | Payoff | Payoff needs | Price | Greeks | Implied volatility | Monte Carlo | Cashflows | Present value |
|---|---|---|---|---|---|---|---|---|---|
| `Asset` | `asset` | no | — | — | — | — | no | no | no |
| `Forward` | `forward` | yes | terminal | — | — | — | yes | no | no |
| `Future` | `future` | yes | terminal | — | — | — | no | no | no |
| `EuropeanOption` | `european_option` | yes | terminal | black, bs, bsm | black, bs, bsm | black: halley, jackel; bs: halley, jackel; bsm: halley, jackel | yes | no | no |
| `BinaryOption` | `binary_option` | yes | terminal | — | — | — | yes | no | no |
| `AsianOption` | `asian_option` | yes | path | — | — | — | yes | no | no |
| `BarrierOption` | `barrier_option` | yes | path | — | — | — | yes | no | no |
| `LookbackOption` | `lookback_option` | yes | path | — | — | — | yes | no | no |
| `VarianceSwap` | `variance_swap` | yes | path | — | — | — | yes | no | no |
| `FixedRateBond` | `fixed_rate_bond` | no | — | — | — | — | no | yes | yes |
| `ZeroCouponBond` | `zero_coupon_bond` | no | — | — | — | — | no | yes | yes |

Model abbreviations: `black` = Black-76, `bs` = Black-Scholes, `bsm` = Black-Scholes-Merton. A dash means the operation is not available for that type, and asking for it raises rather than returning an approximation. **Monte Carlo** is a type-level answer: an individual contract is additionally eligible only with a strictly positive maturity, which `MonteCarloEngine.supports(instrument)` applies and is authoritative for an actual request. **Cashflows** and **present value** are the fixed-income route: a security with dated payments has no payoff and no option-pricing model, so `cashflows()` reads its schedule and `fast_vollib.pricing.present_value()` values it against a `DiscountCurve`.

<!-- END generated: instrument capability table -->

Regenerate with `uv run python scripts/generate_instrument_docs.py`; a test
fails if the table is stale.

`Forward` and `Future` are recognized and have terminal payoffs, but the
instruments layer provides no analytic pricing adapter for them — pricing a
forward requires discounting rather than a Black formula, and the adapters add
no mathematics of their own.
Asking for a price returns an error that says exactly that, rather than an
option value.

The **Monte Carlo** column is a different question from **Price**. `Price` means
an analytic kernel; Monte Carlo means
[`MonteCarloEngine`](simulation.md) can simulate the underlier and average the
payoff. The two never stand in for each other: an analytic request for a type
with no kernel raises rather than quietly simulating, and a simulated price is
only ever produced by asking for one by name.

`Future` is the one type with a payoff and no valuation route at all. Its
terminal formula coincides with a forward's, so simulating it would return a
plausible number — but a future's economics are a stream of daily variation
margin, which this library does not model.

American and Bermudan exercise are not represented. The registry contains only
instrument types with supported behaviour.

## Differentiability

A native tensor return is **not** evidence that gradients survived. The table
below is the whole claim; `capabilities(...).native_autodiff` records the same
thing as machine-readable `(model, solver, backend)` triples, computed against
the backends you actually have installed.

| Path | Differentiable? | Why |
|---|---|---|
| `payoff()` on torch/jax input | **Yes**, tested | Pure array-namespace operations; the tape is preserved and never staged through host memory. A digital or barrier indicator keeps the graph and differentiates to exactly zero, which is the correct pathwise answer rather than a useful Greek |
| `MonteCarloEngine.price(..., return_native=True)` | **Yes**, tested | Gradients reach spot, discount rate, process drift, and process volatility on smooth routes; see [Simulation](simulation.md) for which routes those are |
| `implied_volatility_instrument(solver="jackel")`, native torch/jax input, `return_native=True` | **Yes**, tested | Routes to `jackel.implied_volatility_autograd[_jax]`; implicit-function-theorem gradients w.r.t. price, spot or forward, strike, maturity, rate, and (BSM) dividend yield |
| The same request with host or formatted output | No | Formatting terminates the gradient path — a documented boundary, not an accident |
| `implied_volatility_instrument(solver="halley")`, any form | No claim | Inherits the existing backend's host-staging contract |
| `price_instrument`, `greeks_instrument` | **No — do not rely on it** | The fused kernels stage host-to-device by design; `return_native=True` changes the container, not the tape |

The differentiable route preserves the Jäckel wrappers' own edge behaviour:
below-intrinsic and other invalid-domain inputs return `NaN` in both the forward
and the backward pass, and low-vega rows follow the wrappers' documented guard.
That behaviour is not normalized to another solver's policy, and `on_error` is
not consulted on that route.

## Two rules

**Nothing is inferred.** `model` is always an explicit argument. An equity
underlier does not imply Black-Scholes over Black-Scholes-Merton — that turns on
whether a dividend yield is being modelled, which is a modelling decision, not a
property of the asset. The asset class may validate a choice; it never makes
one. The same holds for the IV solver and the backend.

Consequently `market.underlying` is neutrally named: under `"black"` it is read
as a forward, under the other two models as a spot. The model at the call site
fixes which.

**Nothing falls back.** A request that cannot be served exactly raises.

| Error | Raised when |
|---|---|
| `InstrumentValidationError` | A contract term is missing, malformed, or out of range. Batch messages name the row. |
| `UnsupportedInstrumentError` | The operation has no implementation for this type. The message says whether the type is recognized at all. |
| `UnsupportedModelError` | The model is unknown or unavailable here; the message lists the ones that work. |
| `UnsupportedSolverError` | The solver cannot serve the requested backend, or a differentiable route's optional dependency is missing. |
| `MissingMarketInputError` | A needed market field was not supplied; the message names it. |
| `SerializationError` | A record is structurally undecodable — unknown version, type, or field. |

All six subclass `InstrumentError`, and each also subclasses the builtin you
would reach for anyway (`ValueError` or `NotImplementedError`).

There is no case in which requesting machine-precision gradients yields a
host-staged approximation, or in which a barrier is priced as a European.

## Implied-volatility routing

`solver="jackel"` is the default: it is machine-precision and the only route
with gradient support. It reaches the **full-model** Jäckel wrappers for the
`numpy`, `torch`, and `jax` backends. Requesting it with `numba` raises and
names `solver="halley"` as the alternative.

`solver="halley"` is **deprecated**. It remains available, tested, and
unchanged, it serves every backend, and it raises no runtime warning — but it
carries no gradient support and is less accurate than the Jäckel route at
comparable cost. Reach for it to reproduce existing results, not for new work.

The raw `jackel_iv_black` solver consumes an *undiscounted* Black-76 price and a
forward. Market prices are discounted, so handing one to the raw solver produces
a plausible and wrong volatility. The adapter never reaches it: the full-model
wrappers perform the discount and forward conversion for the model actually
requested, and a test spies on the raw solver to confirm what it receives.

## Maturity is a year fraction

`maturity: float` is measured in years from valuation and is never a date, and
never a date-or-float union. Calendars, day counts, and schedule resolution
belong to a layer above this one; admitting them here would make every
contract's meaning depend on an evaluation date that contracts do not carry.

Convert before constructing:

```python
maturity = (expiry_date - valuation_date).days / 365.0
```

## Serialization

```python
from fast_vollib.instruments import instrument_to_json, instrument_from_json

record = instrument_to_json(option)
assert instrument_from_json(record) == option
```

```json
{
  "schema_version": 1,
  "instrument_type": "european_option",
  "instrument_id": "SPX-C-5000",
  "underlier": {"identifier": "SPX", "asset_class": "index", "currency": "USD"},
  "option_type": "call",
  "strike": 5000.0,
  "maturity": 0.75,
  "settlement": "cash",
  "notional": 100.0
}
```

The codec is strict: unknown schema versions, unknown instrument types, unknown
or missing fields, and non-canonical enum spellings are all errors. The
constructors accept `"c"` and `"p"`; the wire format has one spelling per value.
Records carry no engines, arrays, market data, or model parameters, and pickle
is not a wire format for these objects.

A JSON Schema is checked in at
[`docs/schemas/instrument-v1.schema.json`](schemas/instrument-v1.schema.json).
It is generated from the same field table the codec reads
(`uv run python scripts/generate_instrument_schema.py`), regenerated
byte-for-byte by a test, and closed (`additionalProperties: false`) with the
runtime numeric constraints mirrored. Nothing validates against it at runtime;
`jsonschema` is not a dependency of the library.

!!! warning "Experimental format"
    Backward compatibility is not guaranteed until the format is declared
    stable. Incompatible record-shape changes increment `schema_version`.

## The DataFrame boundary

`from_frame` maps every column explicitly. Nothing is inferred from column
names — guessing is how a book gets priced against the wrong column and nobody
finds out until the P&L is wrong.

```python
batch = EuropeanOptionBatch.from_frame(
    chain_df,
    option_type_col="cp", strike_col="K",
    maturity_col="T", underlier_col="symbol",
    notional_col="contracts",
)
```

A contract frame holds contract terms only. Spot, rate, volatility, and observed
prices are market observations; naming one of them as a contract column is
refused by name, and they go to the pricing call as `VanillaMarketInputs`.
Batches are homogeneous, so a frame mixing instrument types is rejected rather
than silently split — group it first. `to_frame()` round-trips, preserving row
order.

Coordinates come from pure functions rather than attributes, and take an
explicit model, because the forward is `S`, `S·e^{rT}`, or `S·e^{(r−q)T}`
depending on it:

```python
from fast_vollib.instruments import log_moneyness, time_to_maturity

k = log_moneyness(chain, market, model="black_scholes")   # log(K/F)
T = time_to_maturity(chain)
```

`log_moneyness` is the coordinate
[`fast_vollib.surface`](surface.md) is parametrized in, so a batch feeds a
surface without a sign flip.

## Batch equality

Batches deliberately have no `__eq__`. For a container of arrays, `a == b` has
two defensible meanings — one bool, or an elementwise mask — and silently
picking either produces wrong control flow in code that expected the other. Use
`batch.equals(other)` for the whole-batch question. Batches are
identity-hashable, and their columns are read-only.

## Scope

The instruments layer does not represent exercise descriptors beyond European
exercise, calibration, or direct construction of an `IVSurface` from an
instrument batch. Stochastic processes and simulation live in
[`fast_vollib.processes`](simulation.md) and
[`fast_vollib.simulation`](simulation.md) rather than here: a contract holds
terms, and neither a process nor a scenario is ever attached to one. The
registry lists only the operations that are supported.
