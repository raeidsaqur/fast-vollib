# Terminal P&L curves

`fast_vollib.pnl` evaluates signed listed-option inventory plus nominal entry cash
and explicit future costs. It provides native terminal arrays, owned host reports,
exact grid-independent roots/extrema and optional static plots.

An expiry curve describes an intrinsic-equivalent terminal function. For American
or physically delivered options, callers must explicitly choose
`terminal_assumption="hold_to_expiry_no_early_exercise"`. The resulting report,
JSON and plot retain the caveat: early exercise, assignment, physical delivery,
funding, ex-dividend exercise and pin risk are not modelled. Standard European
cash-equivalent contracts also accept `"standard_intrinsic_equivalent"`.
This API does not price options before expiry or estimate executable closes.

## Offline example

Contract setup is explicit. The entry, snapshot and plotted profile then need
three calls. Install the existing `viz` extra to produce the figure.

```python
--8<-- "examples/terminal_pnl.py"
```

The example's credit is USD 190, maximum P&L is 190 on `[580, 620]`, minimum P&L
is −810, and breakevens are 578.10 and 621.90. Its American/physical terms are
retained even though the terminal intrinsic shape uses the existing unit-option
payoff implementation.

## Native evaluation

```python
from fast_vollib.pnl import prepare_terminal, terminal_pnl

prepared = prepare_terminal(
    snap, terminal_assumption="hold_to_expiry_no_early_exercise", dtype="float64"
)
values = terminal_pnl(prepared, spot)
```

`prepare_terminal` converts Decimal strikes, signed sizes, cash and future costs
once, recording exact rational conversion errors per coefficient. `future_costs`
is a nonnegative Decimal amount, defaulting to zero. Paid fees are already in cash
and are not deducted again. Preparation refuses nonfinite conversions, underflowed
strikes, incomplete entry basis, unsupported delivery terms and mixed live expiries.

`terminal_pnl` preserves floating spot shape, namespace, dtype and device, including
non-contiguous NumPy/Torch input and spot gradients. Spots must be nonnegative,
finite arrays with exactly the prepared dtype (`float64` by default; `float32`
requires explicit preparation). Integer, object and mismatched floating dtypes are
refused. It uses existing native payoff dispatch once per strategy leg, with no
per-spot loop. Gradients away from strikes equal the piecewise slope; at strikes,
each backend's `clip` subgradient applies. A closed inventory returns a constant
connected to the input by zero slope. Concrete invalid spots raise `PnLError`;
under JAX tracing, shape/dtype checks remain static and invalid dynamic entries
produce NaN.

`prepared.conversion_error_bound(host_spot)` bounds coefficient conversion error
pointwise on a finite supplied grid. It excludes payoff/summation arithmetic
rounding. No global error bound is asserted over unbounded spots. Float32 has
weaker precision than the default host reports.

## Scenario bridge and time

`terminal_pnl(prepared, scenario)` accepts an existing validated
`simulation.Scenario` and returns native `(n_paths,)` values. Live contracts require
resolved expiry instants and an explicit preparation day count. Unit payoffs use
the actual elapsed maturity from snapshot valuation to expiry, so the existing
Scenario underlier and horizon guards apply. A positive scenario horizon is never
matched against a fabricated zero maturity. No terminal intrinsic value is used
as an intermediate option mark.

An expiry profile needs only a legal date when valuation precedes that date.
Its `target_date` is the legal expiry. `horizons` contains an elapsed year fraction
only when an instant is resolved and a day count supplied; otherwise it contains
`None`. Valuation on an unresolved expiry date, after a resolved expiry, or across
an unsupported lifecycle boundary raises `PnLHorizonError`. A fully closed complete
trade has no fabricated target date or horizon.

## Reports, exact metrics and plots

`expiry_profile(snapshot, spot, terminal_assumption=..., future_costs=...,
convention=...)` returns `PnLProfile` with read-only owned float64 NumPy arrays:

| Field | Shape / meaning |
|---|---|
| `spot` | `(n_spot,)` |
| `leg_values` | `(1, n_spot, n_leg)`, signed inventory value |
| `total`, `valid` | `(1, n_spot)` |
| `leg_valid` | `(1, n_spot, n_leg)` |
| `cumulative_cash`, `paid_fees`, `future_costs` | Decimal accounting amounts |
| `contracts`, `signed_sizes`, `leg_ids` | retained listed terms, sizes and identities |
| `meta` | origin, basis, world, cutoff, knowledge label, evidence/input digests and policies |
| `conversion_errors`, `conversion_bound` | coefficient conversion receipt and pointwise bound |

The accounting identity is `total = cumulative_cash + sum(leg_values) - future_costs`.
Arrays use immutable storage; caller buffer mutations and edits to fresh
`to_frame()` tables cannot change a saved report.

`profile.metrics` analyzes `{0} ∪ strikes` and the analytic right tail over all
`S >= 0`. Metrics are independent of plotting-grid limits. `breakevens` are isolated
exact `Fraction` roots; `zero_intervals` separately describes intervals of equality,
with `None` as an unbounded upper endpoint. Signed `minimum_pnl` and `maximum_pnl`
include Decimal signed infinities for unbounded tails. `minimum_regions` and
`maximum_regions` record finite extremizers, including flat regions.
`maximum_loss` and `maximum_profit` are nonnegative magnitudes. Equality at zero
is not profit. No probability distribution is inferred from the curve or grid.

`plot_pnl([profile], ax=None, show_legs=False, show_breakevens=True)` returns a
Matplotlib `Figure`; it never calls `show`. Strikes, exact isolated breakevens, the
zero line, signed extrema and persistent caveat captions accompany the curve.
Plotting imports Matplotlib lazily. The caller owns saving/displaying the figure.

`pnl.serialization.dumps` / `loads` and `profile.to_dict()` / `to_json()` use
`pnl.profile/1` and `pnl.result/1`. Decimal and exact rational values have explicit
tags; missing, invalid and signed unbounded values use distinct status tags.
Unknown versions, fields and duplicate JSON keys are refused. Contracts in exports
retain their source exercise, settlement and delivery descriptors. No live model,
callable, RNG, tensor or pickle is serialized.

## Quote indications

`mark_position(snapshot, quotes, policy=..., future_costs=Decimal("0"))` values
signed inventory from observed bid/ask inputs. It can mark American options from
quotes without claiming American model-pricing support. It refuses unreconciled
post-expiry inventory. Paid fees already belong to cumulative cash; future close
costs are deducted separately once. Fixed realized P&L is retained, and unrealized
P&L changes with inventory value. A positions-only snapshot has an inventory mark
but no total trade P&L or invented zero entry cash.

Construct an `OptionQuote` for each contract, and a `QuoteSnapshot` with explicit
valuation time and knowledge cutoff matching the position snapshot. The snapshot's
`stamp` describes the **underlying quote**, while each option has its own source
event and availability times. A wrapper timestamp never makes stale legs fresh.
`quotes_as_of` selects admissible quote observations after revision resolution,
then excludes quote events after valuation. Conflicting same-time deliveries need
an explicit revision rather than input-order selection. `quote_record` supplies
the immutable `pnl.option_quote/1` observation payload codec.

`QuotePolicy` requires a name, `mark_rule`, maximum option/underlying ages in seconds,
permitted delay in seconds and an explicit `allow_zero_size` choice:

```python
from fast_vollib.pnl import QuotePolicy
policy = QuotePolicy(name="current_midpoints", mark_rule="midpoint",
    max_quote_age_s=60.0, max_underlier_age_s=60.0,
    permitted_delay_s=0, allow_zero_size=False)
```

- `midpoint` needs both bid and ask. `last` is never a substitute.
- `close_estimate` marks longs at bid and shorts at ask. It is a sum of per-leg
  estimates, **not an executable multi-leg fill**. One side suffices only when it
  is the side that leg needs.
- Missing sides, crossed, halted, source-stale or policy-stale quotes are refused.
  Quote events must be no later than valuation and availability no later than cutoff.
- Delayed and zero-size indications need named policy permission and retain their
  flags. Required-side size is checked; missing size is unknown, never zero.
  `allow_unknown_size`, `allow_unknown_delay` and `allow_unknown_quote_time` default
  to false. Explicitly allowing unknown source time does not establish freshness.
- If the source quote timestamp is unknown, set `quote_time_known=False` and retain
  a timing note explaining the stamp's event-time assumption. Do not replace it
  with last-trade time. `underlying_time_known` handles the analogous underlying case.
- Per-option `delay_s` overrides the explicitly declared feed-level delay. An
  unquantified delayed flag cannot be passed off as zero delay.

Complete mode raises when a live leg cannot be priced. `diagnostic=True` retains
per-leg `LegQuoteStatus` with flags, reasons, bid/ask, size, age, delay, source and
availability times, and optional Decimal mark/signed value. Any missing leg leaves
the aggregate unavailable. No leg is silently priced at zero. Unknown entry basis
also leaves totals unavailable, even with complete inventory prices.

`QuoteFlags` includes `MISSING`, `ONE_SIDED`, `CROSSED`, `STALE`, `HALTED`,
`ZERO_SIZE` and `DELAYED`. The original source flags are retained alongside flags
derived from values and policy. Broker-reported IV/Greeks may be retained in
`reported_estimates`, labelled by their quote's source stamp; they never become
model inputs automatically.

The fee fixture's signed midpoint inventory is `80 - 140 - 120 + 70 = -110` USD.
Its cumulative entry cash is USD 156, already net of USD 4 paid fees, so midpoint
P&L is USD 46. Deducting a separately estimated USD 4 close cost gives USD 42.
Changing to side-specific estimates changes the inventory values as well.

## Tables, figures and files

`plot_pnl([expiry_profile], mark=quote_profile)` overlays a current-spot indication.
It returns a Figure without displaying it, retains exercise/delivery and quote
caveats, and labels missing totals as unavailable. `show_legs=True` displays signed
inventory values before cash and future costs. Quote marks have no expiry roots or
grid-independent extrema; those properties are unavailable, not inferred from one
point. Coefficient conversion bounds are inapplicable to quote marks.

`profile.to_frame()` returns fresh rows with basis, currency, caveats, time/cutoff,
quote policy, per-leg values/masks and quote status. `export_pnl(profile, prefix,
figure=fig)` writes `.json`, `.csv` and the supplied `.png`; omit the figure for
JSON/CSV only. JSON preserves Decimal text, UTC instants, listed terms and distinct
missing/invalid/unbounded values. CSV rows retain basis/caveats; JSON carries full
metadata. Terminal reports retain the original `pnl.profile/1` wire fields; quote
reports use strict `pnl.profile/2` / `pnl.result/2` schemas. Unknown fields or versions
are refused. See `examples/quote_marks.py` for a synthetic offline figure.

For batched European horizon repricing, coherent IV-sensitivity bands, explicit
Greek report units and terminal-law profit probability, see
[Horizon valuation and terminal laws](horizon_pnl.md). American pre-expiry model
valuation is refused; the European proxy is a separate, qualified result type.
