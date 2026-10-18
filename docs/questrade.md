# Offline Questrade normalization

`fast_vollib.integrations.questrade` normalizes **saved** option symbol details,
executions, option quotes, positions and activities. It performs no network access,
imports no broker client and takes no credentials or account identifier. The
[synthetic notebook](https://github.com/raeidsaqur/fast-vollib/blob/main/notebooks/offline_questrade.ipynb)
and `examples/offline_questrade.py` show a complete offline chain/fee/mark example.

## Capture context and lossless inputs

Use `load_saved_json(text)` to decode JSON fractional numbers directly into Decimal,
with duplicate keys and nonfinite values refused. Passing an ordinary decoded
binary-float payload is refused. Money uses Decimal/canonical text, not magnitude
heuristics. Option quantities must be integral; side and quantity sign must agree.

`NormalizationContext` requires an opaque UUID namespace, capture time, explicit
availability time/note and world ID. Keep any account-to-namespace association
privately outside the library. If only retrieval establishes availability, provide
that instant and state the assumption. Synthetic examples use a hypothetical world.
Contract/position event times may be assumed retrieval and are labelled accordingly.
No historical local-ingestion replay is established by that assumption.

Retain the original context with each saved capture. Re-importing the same capture
with the same context and identities is deterministic. Preserve first-seen evidence
when processing refetches. A correction needs an explicitly supplied `revisions`
mapping from source identity to `(version, ancestor_record_id)`; a new retrieval
alone does not establish a correction version. Activities lack documented source
row IDs, so their normalizer requires stable, unique caller UUIDs.

## Named variants and source terms

Choose one of `questrade.saved/1:properties-percent`, `properties-decimal`,
`examples-percent` or `examples-decimal` (each with the same full prefix). The
properties variant accepts `quotes` and position `closedPnL`/`openPnL`; the examples
variant accepts `optionQuotes` and `closedPnl`/`openPnl`. These are explicit input
contracts. IV percent/decimal units are a caller-selected mapping assumption, not
inferred from a value. Unknown envelopes, fields and mixed variants are refused.
The source docs disagree on the option envelope and contain inconsistent position
sample types. See [option quotes](https://www.questrade.com/api/documentation/rest-operations/market-calls/markets-quotes-options)
and [positions](https://www.questrade.com/api/documentation/rest-operations/account-calls/accounts-id-positions).

`normalize_symbols` requires explicit `ContractTerms` per source symbol ID: stable
contract key, underlier reference, premium multiplier, settlement and assumption
note. It preserves source American/European designation, strike, currency, legal
expiry date and deliverables. It does not invent an expiry instant or infer terms
from a ticker. Share deliverables cannot be relabelled cash settlement. Adjusted or
mixed delivery is retained as unsupported inspection data and refused by valuation.
See the [symbol endpoint](https://www.questrade.com/api/documentation/rest-operations/market-calls/symbols-id).

The supported option-quote payload has last-trade time but no source quote timestamp.
`normalize_option_quotes` records unknown quote time explicitly; last trade is never
used for quote freshness. The default mark policy therefore refuses it. An explicit
unknown-time indication policy can permit it without claiming freshness. Delay,
size and halt fields remain labelled. Reported broker IV/Greeks retain their source
and selected units; they do not become model inputs.

## Fees, inventory and cash comparisons

`normalize_executions` requires all declared fee fields and an explicit composition:
`separate_components` adds distinct components; `commission_includes_components`
splits the supplied total without adding included fees again. Missing/negative or
inconsistent totals are refused. The named saved mapping treats source `totalCost`
as positive premium times unsigned quantity, without the option multiplier; it
checks that identity. A different source convention needs its own reviewed mapping.
Source order IDs are retained as hints, never an automatic strategy grouping.
See [executions](https://invest.questrade.com/api/documentation/rest-operations/account-calls/accounts-id-executions).

`normalize_positions` produces inventory evidence with source-reported values.
It never turns average entry, reported closed P&L or a positions-only endpoint into
a complete execution history. Bool-valued money and string-valued real-time flags
are refused. `position_evidence` groups **caller-selected** rows into unknown-basis
inventory, with no invented fills. Component revision numbers do not infer group
revision numbers; group corrections use the core explicit record/revision APIs.
`broker_position_record` preserves source-labelled audit data.

`normalize_activities` requires a commission sign policy, signed option quantities
for trades and caller row IDs. Other cash activities remain audit evidence. Exercise,
assignment, corporate actions, funding and settlement lifecycle are not implemented.
`activity_record` does not make activity cash enter the execution ledger. Broker
notes/descriptions and account identifiers are not copied into normalized records.
See [activities](https://www.questrade.com/api/documentation/rest-operations/account-calls/accounts-id-activities).

`reconcile_executions_activities` takes explicit activity-to-execution links and a
trade-date timezone. It compares quantity, currency, contract, date, gross, fees and
net cash, including multiple fills per activity. It refuses reused links and
reports unmatched trade rows and other activities separately. Execution and activity
cash are **compared, never added**. A `consistent` result only describes supplied
rows/links; it does not prove complete account reconciliation. Real-payload
reconciliation remains pending until a private comparison with suitable real saved
payloads is performed. Synthetic tests establish internal mapping consistency only.
