# Implied-volatility adjudication fixtures

These fixtures contain the exact inputs and reference results for implied-volatility tests.
`scripts/reference_inverse.py` calculates the reference with `mpmath` and a bracketed inverse.
It uses an independent implementation.

| File | Contents |
|---|---|
| `protocol.json` | Frozen row classes, consistency-set scoring rule and float64 tolerance protocol (`fv-iv-protocol/1`) |
| `fixtures.json` | Development, control and boundary fixtures (see below) |
| `heldout_chains.csv.gz` | Varied-parameter sample from development. It has 20 option chains of 100 rows each (seed 20261120). F, T and r vary, with r ≠ 0. The quotes are the exact model prices rounded to float64. |
| `sweep81_reference.csv.gz` | The canonical grid: x = ln(F/K) ∈ [−4, 4] (81 points) × s = σ√T ∈ [1e−3, 4] (81, log-spaced) × {call, put}; F = 100, T = 1, r = 0. Quotes from the NumPy Black-76 generator. Inputs and reference for all 13,122 rows. |

## Conventions

- **Inputs are exact binary values**, written with `float.hex`. The model is discounted Black-76: `P = D·Black(F, K, σ√T)` with `D = exp(−rT)`.
  - Forward-form rows give `F_hex`.
  - Spot-form rows (Black-Scholes and Black-Scholes-Merton) give `S_hex` and `q_hex` instead. The reference calculates the forward `S·exp((r − q)T)` in high precision.
- **The reference is the point inverse of the exact quote.** `ref_lo_hex` and `ref_hi_hex` give the bisection bounds after outward rounding. `ref_dps` is the working precision at which the float64 midpoint stopped changing.
- **The rounding interval** contains the prices that round to the quote under round-half-even. The reference limits this interval to admissible prices. It converts the endpoints to σ and records `sigma_int_lo_hex` and `sigma_int_hi_hex`.
  - An endpoint status of `touches_lower_bound` means every σ down to 0 is consistent with the quote.
  - The point inverse tests a solver. The interval measures the ambiguity caused by representing the price in floating point.
- **`cls`** is the point class of the exact quote, as defined in `protocol.json`.
- **`scoring_class`** identifies the rule for each row. The rule uses the consistency set, which contains the σ values possible within the rounding interval.
  - `strict`: the set has two finite bounds. The result must be within the reference tolerance for rows in the solver domain.
  - `open_at_zero` / `open_at_infinity`: the set reaches a price bound, so any value in the set passes, or NaN with the `on_error` signal.
  - `zero_quote`: an out-of-the-money quote of exactly 0. The scoring rule accepts 0, a value in the set or `NaN` with or without a signal. The public row policy returns `NaN` and uses `on_error` for the signal.
  - `inconsistent`: the set is empty, so the result must be NaN with the signal.
  - `invalid_input`, `expiry`.
- **`h_quote_hex`, `h_forward_hex`, `h_discount_hex`** are the σ half-widths attributable to unavoidable input roundings:
  - `h_quote`: the quote itself;
  - `h_forward`: one rounding of a derived forward (spot form only);
  - `h_discount`: one rounding of the discount factor (r ≠ 0 only).
- **The tolerance** is `max(abs_floor, rel_tol·ref) + M·(h_quote + h_forward + h_discount)`.
- **The declared domain** defaults to every strict row. A strict row outside a declared domain passes only with NaN and the `on_error` signal. The development data establish the limits of the domain. The varied-parameter sample checks for false refusals.
  - Revision 3: the Jäckel solver's domain is `beta >= DBL_MIN` (the normalised out-of-the-money price is a normal double). The Halley solver has no narrowing.
- **`generator_quote_matches`** says whether the generating quote equals the exact price at the generating σ, correctly rounded. When it does not, `generator_quote_ulps` gives the difference. The generating σ is therefore not a valid reference in general.

## `fixtures.json`

Each record carries:

- **id, role and quote source**, where the role is one of:
  - `dev`: a case that exposed a defect;
  - `control`: a matched neighbour that every route passes;
  - `boundary`: a boundary-class or boundary-neighbour case;
  - `wrapper`: a spot-form Black-Scholes or Black-Scholes-Merton case with r ≠ 0 or q ≠ 0, reference only.
- **The exact inputs and the reference.**
- **Execution-path diagnostics:**
  - the Jäckel initial-guess zone, computed independently;
  - the Halley seed, whether it was clipped, the iteration count, and whether the bisection fallback triggered.
- **The observed v0.2.4 output of every implied-volatility route**, with its verdict under the protocol. Routes are keyed `<lane>:<route>_<device>`, and the `routes` map records the terminal function that actually ran.

The tests record expiry rows without a score. See [the API reference](../../../docs/api.md) for the public row policy.

## Regenerating

```bash
python scripts/reference_inverse.py csv rows.csv out.csv --jobs 8      # flag, F_hex | S_hex [q_hex], K_hex, T_hex, [r_hex], price_hex, [sigma_gen_hex]
python scripts/reference_inverse.py point --F 100 --K 90.48374180359595 --T 1 --price 9.516258202222488 --flag c
```

The script requires `mpmath==1.3.0`, which the development lock provides. Use the constants in `protocol.json` for all scores.
