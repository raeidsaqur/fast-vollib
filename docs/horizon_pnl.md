# Horizon valuation and terminal laws

`fast_vollib.pnl` adds European model revaluation, explicit volatility assumptions,
position Greeks and analytic profit probability under a caller-specified terminal
law. These are separate from observed quotes and executable fills. Models reuse
`instruments.price_instrument` and `greeks_instrument`; no pricing formula is
reimplemented here. Reports own read-only host arrays and do not promise gradients.
See `examples/horizon_pnl.py` for an executable synthetic example.

## Model horizons

```python
from fast_vollib.pnl import FlatVol, HorizonMarket, pnl_at_horizon
market = HorizonMarket(rate=0.03, dividend_yield=0.01, volatility=FlatVol(0.20))
profile = pnl_at_horizon(snapshot, spot_grid, horizons=(0.0, 0.01),
    market=market, model="black_scholes_merton", backend="numpy",
    convention="act_365", future_costs=Decimal("4"), chunk_rows=65536)
```

All horizons are elapsed years from the snapshot valuation time. Every live leg
needs an explicitly resolved expiry instant and a named actual-day convention.
Remaining maturity is the valuation-origin maturity minus the horizon. Mixed
expiries are supported through the earliest live expiry inclusive; crossing that
boundary or reconstructing already-expired inventory is refused. Near zero,
the existing intrinsic payoff is used with a tolerance respecting the input grid
precision. Zero spot is supported for intrinsic-only rows; logarithmic surface and
Black forward routes refuse zero live spot or mark it invalid in diagnostic mode.

`model` and `backend` are required. Black-Scholes requires explicit zero dividend
yield; Black-Scholes-Merton requires an explicit yield. Black-76 requires a named
mapping `(spot, horizon, remaining_tenor) -> forward` and
`discount_convention="constant_continuous_rate"`. Spot models use their declared
rate/yield and refuse a separate Black mapping that would be ignored. Cash remains
nominal: only option values discount from the target horizon to expiry. Paid fees
remain in cash; future close costs are subtracted once.

`FlatVol(sigma)` uses decimal annualized volatility. `PerLegVol(by_leg_id)` must
cover exactly the live leg IDs. `SurfaceVol` requires all of:

- An evaluable surface and explicit reference `SurfaceMarket`.
- `spot_movement="sticky_strike"` or `"sticky_forward_moneyness"`.
- `time_movement="fixed_expiry"` or `"constant_tenor"`.
- A forward mapping, reproducibility policy ID and origin spot.

Reference tenor is the origin maturity for fixed expiry, and remaining maturity
for constant tenor. Sticky strike queries `log(K / F_ref(reference_tenor))`;
sticky forward-moneyness queries `log(K / F_h(remaining_tenor; spot))`. The supplied
reference market must agree with the origin forward mapping and pricing rate/carry.
Reference points are immutable and no interpolation/extrapolation policy is inferred.

`ProportionalCarry(origin_spot=S0, base_forward=..., policy_id=...)` explicitly
implements `F_h(tau; S) = base_forward(h, tau) * S / S0`. The base forward can depend
on horizon. It is never silently replaced with an origin curve at `h + tau`.
A proportional mapping provides its origin spot to surface assumptions; other
callables require `origin_spot` separately. Callables are used only for evaluation
and are never serialized. Metadata retains named policies and evaluated-input
digests, not executable models.

Pricing rows flatten as `((horizon * n_spots) + spot) * n_legs + leg`.
`repricing_index` exposes that map. Compatible rows use unit-notional columnar
batches, then signed quantity times multiplier exactly once. `chunk_rows` defaults
to 65,536 and bounds working pricing rows before Cartesian allocation. Returned
full arrays still require `(horizon, spot, leg)` output memory. Invalid surface
cells raise in complete mode; diagnostic mode retains per-leg/aggregate masks and
unavailable totals. It never fills an invalid leg with zero.

## Coherent surface sensitivity

`pnl_at_horizon_samples(..., volatility=SurfaceVolSamples(...), market=...)`
returns `SurfaceSensitivityBand`. For this route, pass a `HorizonMarket` with
`volatility=None`: the sampled assumption is supplied separately rather than
silently overriding a second volatility choice.

The adapter deduplicates the required union of reference `(k, T)` points and calls
`SurfaceDistribution.sample` once over that union. Model pricing is then chunked
across both sample and valuation axes. No chunk, horizon or leg is resampled.
`union_point_limit` defaults to 65,536; a generator's own coherent-domain limit
still applies and may be lower. If the required union cannot be supplied, the
request is refused. Intrinsic-only rows need no volatility queries.

`samples` has axes `(surface_sample, horizon, spot, leg)` and contains signed
inventory values; `total` adds the same nominal cash and subtracts future costs.
Sample IDs, normalized weights and original weight mass are retained. Sensitivity
quantiles use the inverse weighted empirical CDF. A cell with invalid positive
weight remains unavailable rather than silently dropping or conditioning on it.
These vary surface assumptions at **fixed spots**. They are not joint spot/volatility
P&L probabilities and expose no probability of profit, VaR, ES or wealth-risk summary.

`plot_pnl_band` returns a Figure with that qualification. `export_surface_band`
writes JSON/CSV and an optional supplied figure. JSON uses the strict
`pnl.surface_sensitivity/1` schema; values, masks, identities, weights, listed terms
and caveats survive export. Unknown fields or result types are refused.

## American contracts and the separate proxy

Standard `pnl_at_horizon`, sampled model valuation and `position_greeks` refuse
American live legs. `european_proxy_profile` requires at least one American leg and
returns `EuropeanProxyProfile`, retaining the source designation and this caveat:

> European surrogate illustration; not an American valuation or executable close
> estimate; early exercise, assignment and delivery lifecycle are not modelled

`european_proxy_samples` returns `EuropeanProxySensitivityBand`. Figures distinguish
proxy curves; JSON/CSV retain the proxy type and caveat. Relabelling a serialized
proxy as a standard result is refused. Proxy curves do not produce American Greeks,
profit probabilities or validated wealth-risk measures. There is no exercise,
assignment, settlement or funding lifecycle implementation. Observed quote marks
remain a separate supported route for American options.

## Position Greeks

`position_greeks(snapshot, VanillaMarketInputs(...), model=..., backend=...,
convention=..., report_units="existing")` returns per-leg and net `PositionGreeks`.
It uses one unit-option batch and scales by signed inventory once. Expiries must
be resolved and live; closed complete inventory has zero Greeks. Under Black,
`VanillaMarketInputs.underlying` explicitly means forward and delta/gamma retain
that meaning.

The existing kernels report theta per **1/365 model year**, and vega/rho per **0.01
change** in volatility/rate. The default preserves those outputs. Choosing
`report_units="annual_unit"` explicitly multiplies theta by 365 and vega/rho by 100,
with delta/gamma unchanged. Units and conversion factors are stored in the report.
The maturity day count does not silently change this reporting convention. Neither
route introduces a notional or multiplier a second time. The strict Greek codec is
`pnl.position_greeks/1`.

## Profit under a specified terminal law

`LognormalTerminal(spot=..., drift=..., sigma=..., T=..., measure=...,
convention=...)` declares `S_T = S0 * exp((drift - sigma**2 / 2)*T + sigma*sqrt(T)*Z)`.
Drift is required; choosing `r - q` is the caller's explicit risk-neutral assumption.
Spot must be positive; horizon/volatility nonnegative; all parameters finite.
A physical or risk-neutral measure is required. Stress sets are not probabilities.

`probability_of_profit(expiry_profile, law)` integrates CDF mass over exact
strictly-profitable payoff intervals, independently of the plotting grid. The law
must match the resolved payoff horizon and day count. At zero horizon or zero
volatility it evaluates the deterministic atom, with equality at breakeven counting
as **no profit**. Initial float parameters retain their decimal representation when
evaluating degenerate atoms against exact payoff roots.

The result is titled **Under the specified terminal law** and retains the law,
measure, origin/cutoff and terminal exercise/delivery assumptions. It uses strict
`pnl.terminal_probability/1` JSON. American terminal intrinsic illustrations retain
the hold-to-expiry caveat; European proxies are refused. This is not calibration
or evaluation of a learned forecaster and is not a probability derived from surface
sensitivity bands. Path risk, paired comparisons and lifecycle modelling are separate
capabilities.
