# Surface-fit diagnostics

`fast_vollib.diagnostics` turns a model's predicted implied volatilities into
numbers a reader can act on: how large the error is and *how much of the
surface it describes*, whether the prediction would have priced inside the
quoted spread, and whether the predicted smiles violate static no-arbitrage
conditions.

It is a host-side layer built on `fast_vollib.surface`. It adds no runtime
dependency, and importing it pulls in no plotting stack — the figure helpers
resolve lazily and need the optional `viz` extra.

```python
import numpy as np
from fast_vollib.diagnostics import SurfaceQuotes, diagnose_fit

truth = SurfaceQuotes(
    k=[-0.2, -0.1, 0.0, 0.1, 0.2],
    T=[0.5, 0.5, 0.5, 0.5, 0.5],
    iv=[0.24, 0.22, 0.21, 0.22, 0.24],
    surface_id="AAPL-2026-03-20",
    point_id=[0, 1, 2, 3, 4],
)
predicted = np.array([0.243, 0.221, 0.210, 0.219, 0.238])

sample = diagnose_fit(predicted, truth)
sample.fit.overall.rmse        # pooled RMSE over the covered rows
sample.fit.overall.coverage    # how much of the target set that RMSE describes
sample.ragged.butterfly_percentage
```

## Coordinates and conventions

| Symbol | Meaning |
|---|---|
| `k` | forward log-moneyness, `log(K / F)` |
| `T` | maturity in years, strictly positive |
| `w` | total variance, `iv² · T` |
| `bid` / `ask` | forward-normalised prices, `quote / (F · e^{−rT})` |

Prices are compared on the forward-normalised scale, so a spread diagnostic is
independent of the price level of the underlying.

## Quotes are owned and validated

[`SurfaceQuotes`][quotes] is the long-format container every entry point takes.
It copies each input array and marks the copy read-only, so a caller mutating
its own array afterwards cannot change an already-computed diagnostic.

Validation is deliberate rather than permissive:

- every `k` is finite and every `T` is finite and strictly positive — including
  on rows whose observation is missing;
- `iv` is either `NaN` (missing) or a finite value `>= 0`; an infinity is an
  error, not a large number;
- `surface_id` and `point_id` are strings or integers, never booleans or
  floats, so a label survives a JSON round trip unchanged;
- `bid`/`ask` are supplied together, are missing together on a row, and satisfy
  `bid <= ask`;
- `is_call` accepts boolean dtype, numeric `{0, 1}`, or the canonical strings
  `{true, false, call, put, c, p}` case-insensitively. Arbitrary truthiness is
  rejected: `"yes"` is not a call.

Repeated observations at the same coordinate are legitimate and are kept as
separate rows.

## Alignment is explicit

A prediction is matched to a truth row by a key you choose, never by row order
or float proximity:

```python
from fast_vollib.diagnostics import align_predictions

aligned = align_predictions(truth, model_rows, values=raw_predictions)
```

`align_predictions` defaults to the stable `(surface_id, point_id)` key.
`on="coordinates"` uses exact `(surface_id, T, k)` equality, and is permitted
only when those keys are unique on both sides. Missing, extra, or duplicated
keys raise; there is no tolerance-based join and no occurrence-order fallback.

It returns a plain array of **raw** values. A model may emit a negative or
non-finite implied volatility, which `SurfaceQuotes` will not hold as an
observation; pass those through `values=` and the evaluator classifies and
counts them instead of losing them during alignment.

## Coverage is reported, never assumed

`diagnose_fit` classifies the predictions once and shares that classification
with the fit, spread, and arbitrage paths, so all three describe the same rows.

| Count | Meaning |
|---|---|
| `target_count` | rows whose truth is a finite observation |
| `valid_prediction_count` | targets whose prediction is finite and `>= 0` |
| `invalid_prediction_count` | targets the model could not usefully predict |
| `coverage` | `valid_prediction_count / target_count` |

Invalid predictions are excluded from the error sums **and** counted. Without
that split, a model that declines to predict the hard part of a surface would
score better than one that tries — so any sample with an invalid prediction is
marked `partial`.

Records store additive sums and counts; a summary divides once. Pooled RMSE
over a set of samples therefore equals the RMSE of their concatenation, in any
merge order.

Fit error is also reported per region. A region is a named, serializable
[`Box`][regions] with explicit bounds and an explicit inclusion policy; the
default is `liquid` (`|k| <= 0.2`, `T <= 0.5`) against its `illiquid`
complement.

## Spread consistency

When quotes carry bid/ask, the predicted implied volatility is converted to a
forward-normalised price and compared to the book:

- `price_rmse` — RMS distance to the quote midpoint;
- `outside_percentage` — share of priced quotes falling strictly outside
  `[bid, ask]`;
- `mean_miss_width` — how far outside, in spread-widths, averaged over the
  misses alone.

The denominators differ on purpose, and each is reported alongside its metric.
A dataset with no bid/ask carries **no** spread block — `None`, never a zero
that reads as perfect agreement.

## Sampled smiles are not a surface certificate

Two different claims are kept apart.

**Sampled (ragged) checks** run on whatever points a pipeline predicted. Rows
are grouped into `(surface, maturity)` smiles, Durrleman's `g` is evaluated on
each smile's interior nodes, and total variance is compared between adjacent
maturities on their overlap. The report says how many nodes were actually
checked, how many smiles were skipped, and how many maturity pairs had no
usable overlap. Nothing is interpolated onto a mesh.

Two details matter for reading those numbers:

- **Duplicates.** The same `(surface, maturity, k)` node can be observed more
  than once. Those rows stay separate for fit and spread, but a smile's
  geometry has one node per strike, so exact duplicates collapse by *mean total
  variance*, `w = mean(iv_i² · T)` — equivalent to the RMS implied volatility at
  fixed maturity, the natural reduction for a diagnostic defined on total
  variance. The number of duplicate groups, points removed, and the widest
  within-group IV range are all reported. `duplicates="error"` refuses them
  instead.
- **Zero total variance.** Durrleman's `g` is singular at `w = 0`. A zero
  implied volatility remains a valid price and a valid fit value, but such
  nodes are excluded from the derivative checks and counted in
  `nonpositive_total_variance_points`. No epsilon is substituted.

With zero checks, the percentages are `null` — not `0.0`. A smile that was
never examined is not a smile that passed.

**Rectangular-grid checks** require a real mesh. `quotes_to_surface` accepts
only a complete, duplicate-free Cartesian product of the unique coordinates
present, derives the native mask from the rows and their finite values, and
refuses a ragged cloud outright. Only then does `diagnose_surface` produce the
composite scores — SAS, negative-density mass, and the per-condition fractions
— which are meaningful on a grid and misleading on scattered quotes.

## Reports and the `diagnostics-v1` contract

A `DiagnosticRecord` is one `(model, split, sample_id)` diagnostic. A
`DiagnosticReport` pools records into one `GroupSummary` per `(model, split)`,
in deterministic order.

`fast_vollib.diagnostics.serialization` renders a report as
`diagnostics-v1` JSON:

```python
from fast_vollib.diagnostics import serialization

text = serialization.dumps(report)          # canonical, newline-terminated
restored = serialization.loads(text)        # strict: unknown fields are errors
summary = serialization.dumps(report.summary_only())
```

The contract is closed and self-checking. Every object declares
`additionalProperties: false` and lists every property as required; field and
record order are fixed by construction, so the same diagnostics render
byte-identically on any host; unavailable numbers are `null`, and `NaN` and
`Infinity` are refused on both the way in and the way out. Summaries carry both
their additive sums and the values derived from them, and the decoder recomputes
the derived values and requires exact equality — a truncated or hand-edited
artifact fails to load rather than loading wrong.

`report.summary_only()` drops the records while keeping `record_count`, so a
summary artifact still states how many samples stand behind it.

The schema and a golden fixture are build products, regenerated with:

```bash
python scripts/generate_diagnostics_schema.py          # write
python scripts/generate_diagnostics_schema.py --check  # verify bytes
```

The schema is published at `docs/schemas/diagnostics-v1.schema.json`.

## Figures

Plotting helpers live in `fast_vollib.diagnostics.plots` and need the optional
extra:

```bash
pip install "fast-vollib[viz]"
```

They are resolved lazily, so importing `fast_vollib.diagnostics` on a headless
host imports no plotting stack. The first *call* to a figure helper raises an
actionable error if the extra is missing.

[quotes]: #quotes-are-owned-and-validated
[regions]: #coverage-is-reported-never-assumed
