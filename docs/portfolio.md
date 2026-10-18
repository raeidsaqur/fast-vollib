# Listed options and Decimal accounting

`fast_vollib.portfolio` keeps listed contracts, executions and hypothetical entries
separate. A `ListedContract` preserves the broker-neutral key, underlier reference,
call/put type, Decimal strike, legal expiry date, source exercise designation,
settlement, premium multiplier, deliverable, currency and observation stamp.
It is a descriptor, not an `instruments.Instrument` or a pricing model.

Supported terminal terms are a standard 100-share equity/ETF deliverable
(`deliverable="100 SPY"` for SPY, `multiplier=Decimal("100")`) or explicitly
specified standard cash-equivalent terms
(`settlement=SettlementType.CASH`, `deliverable="standard_cash_equivalent:100"`,
`multiplier=Decimal("100")`). Descriptors can retain other terms, but valuation
refuses adjusted or mixed deliverables and unsupported multipliers.

## Explicit hypothetical strategies

`Position.from_legs` accepts signed `OptionLeg` records. `allocated_entry` is the
per-unit premium; `allocated_fees` is that leg's total paid fee allocation.
`signed_size` is signed quantity times the contract's premium multiplier.
A position validates common underlier, currency and world, plus unique leg IDs.

Builders accept explicit listed contracts, a positive integer `quantity`, a typed
price record and total entry `fees`. They return hypothetical positions without
creating execution observations.

| Builder | Contract role keywords | Price record |
|---|---|---|
| `iron_condor` | `put_long`, `put_short`, `call_short`, `call_long` | `IronCondorPrices` |
| `iron_butterfly` | same roles; equal short strikes | `IronButterflyPrices` |
| `butterfly` | `lower`, `middle`, `upper` | `ButterflyPrices` |
| `vertical_spread` | `long`, `short` | `VerticalPrices` |
| `straddle` | `put`, `call`; equal strikes | `StraddlePrices` |
| `strangle` | `put`, `call`; ordered strikes | `StranglePrices` |
| `calendar_spread` | `long`, `short`; later long, same strike | `CalendarPrices` |
| `diagonal_spread` | `long`, `short`; later long, distinct strikes | `DiagonalPrices` |

Price-record fields have the same names as contract roles. Butterflies use
quantities `+1, -2, +1`; other builders use one contract per role with the named
long/short signs. Unequal wings retain their actual arithmetic. Calendars and
diagonals can be constructed, but a terminal expiry profile refuses mixed expiries.
Reversed strategies use explicit signed legs.

`hypothetical_snapshot(position, valuation_time=..., knowledge_cutoff=...,
world_id=...)` requires an explicit non-actual world matching the position's stamps.
It derives entry cash from supplied quantities, premiums and fees. Unknown premiums
are refused. It never invents actual broker fills.

## Executed snapshots and FIFO

An `Execution` has a signed nonzero integer quantity, nonnegative Decimal premium,
execution ID, instant event stamp, optional source sequence and explicit
`FeeRecord` values. `Trade` groups chosen execution identities and owns its contract
mapping; grouping is never inferred from nearby timestamps.

Use `portfolio.serialization.contract_record` and `execution_record` to encode
financial records into generic observation payloads. Then select an observation
snapshot and call:

```python
snapshot = position_snapshot(
    trade, observations=observations, valuation_time=valuation_time, policy="fifo"
)
```

The factory resolves admissible financial records from `selected_records` before
including executions with event time through valuation time. Detached caller trade
values cannot override selected evidence. A complete trade with missing referenced
admissible evidence raises `IncompleteBasisError`. Events unavailable at the cutoff
and future economic events contribute no cash. A cutoff after valuation is labelled
`latest_corrected`; other queries are `as_known`. Hypothetical worlds cannot enter
this actual-evidence route.

`match_lots(executions, contracts=..., policy="fifo", tie_policy=None)` deduplicates
identical executions and refuses conflicting deliveries. Equal-time executions for
the same contract require unique sequences from one source or the explicitly named
`tie_policy="execution_id"`. Iteration order never chooses realized P&L.

The ledger uses a local Decimal context of precision 50. Money requires finite
Decimal values or canonical decimal text; binary floats are refused. The accepted
input domain is at most 50 significant digits, adjusted exponent magnitude at most
100, and integer quantity magnitude at most `10**12`. Computation uses the stated
precision; display rounding does not change accounting or roots. Allocation is
proportional to absolute quantities, with the final portion receiving the residual.
A zero-crossing fill splits into closing and opening inventory. Remaining signed
basis includes allocated opening fees. Paid fees are already included in cash.

Before estimated future costs:

\[
R = C + B_{open},\qquad U = V - B_{open},\qquad R + U = C + V.
\]

`PositionSnapshot.accounting(inventory_value)` returns `(realized, unrealized,
total)` using that identity. Inventory value is supplied by the caller; this method
does not select a quote or price a live American option. Closed complete trades
have constant realized cash. Partial evidence keeps cash and realized P&L unavailable.
A `positions_only` trade references selected `position_record` evidence using
`inventory_record_id`; it retains inventory but refuses trade P&L and breakevens.
Cash is nominal trade-date cash; funding, FX, tax, settlement and assignment ledgers
are outside this accounting contract.

## Dates and codecs

A legal expiry date does not imply an instant.
`resolve_expiry_instant(contract, policy=(time(16), "America/New_York"))` returns
a new descriptor with its policy identifier. This is a caller wall-time assumption;
the caller must verify holidays and settlement sessions first. It is not an exchange
calendar. Ambiguous and nonexistent local times are refused. Instants are stored in
UTC. `year_fraction(start, end, convention=...)` requires `act_365`, `act_365_25`
or `act_360`, with fractional days for aware datetimes. Mixed date/instant arguments
and negative elapsed time are refused.

`portfolio.serialization.dumps` / `loads` handle contracts, executions, positions,
trades and position snapshots. Each has its own versioned schema. Money is decimal
text, timestamps are aware ISO instants, mappings are copied, and unknown fields or
versions are refused. Existing instrument wire formats are unchanged.
