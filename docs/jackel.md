# Jäckel IV — "Let's Be Rational" Implied Volatility

The `fast_vollib.jackel` module implements Peter Jäckel's *"Let's Be Rational"*
(2016) algorithm. It uses a rational initial guess with four branches and three
Householder(3) iterations. The CPU and GPU implementations use normalised Black prices.

Use `fast_implied_volatility(..., solver="jackel")` or `fast_implied_volatility_black`.
These functions apply the row policy and use `on_error` for rows that give `NaN`.

Jäckel meets the tolerance on the strict rows inside its domain in the canonical grid
and varied-parameter sample, with two exceptions. Both exceptions occur near the money
on the PyTorch CPU backend. The tests used both sets during development.
The base relative tolerance is 10⁻¹². The protocol also includes rounding terms.

Jäckel refuses quotes below the smallest normal double after it normalises the
out-of-the-money price. It also refuses some quotes near the upper price bound.
For these refusals, the signal suggests `solver="halley"`.

---

## Why a separate solver?

The `backends/` module implements Halley, the default solver for functional calls.
It applies Halley's method within a bracket to the out-of-the-money price.
Both solvers use the same row policy and exact-quote reference.

| Property | `backends/` Halley | `jackel/` Householder(3)×3 |
|---|---|---|
| Algorithm | Halley on the log of the out-of-the-money price, sign-checked bracket | Jäckel LBR rational guess + HH(3) |
| Validation tolerance (relative, plus rounding terms) | 10⁻⁸ | 10⁻¹² |
| Declared domain | every strict row | normalised price ≥ smallest normal double |
| Backends | NumPy, Numba, PyTorch (fused Triton on CUDA), JAX | NumPy + Numba, PyTorch, JAX |
| Gradient support | no | yes (`implied_volatility_autograd`) |

See [Benchmarks](benchmarks.md) for timing methods and results.
The v0.1.4 timings below are for 100,000 options on H100 NVL.
That benchmark used K = F·exp(U(−0.3, 0.3)) and σ = U(0.10, 0.60).

Halley meets its tolerance on all strict rows in the canonical grid, varied-parameter
sample, fixtures and stress set. The stress set was generated after the solver changes.

On a separate development set, Halley returns inaccurate finite values on 6–36 strict
rows, depending on the backend. These rows have spot-form quotes within a few multiples
of the smallest subnormal double. Their reference volatility is below 10⁻¹⁶.
Halley also returns finite values for two inconsistent quotes.

The native PyTorch and JAX Jäckel routes support gradients.

---

## Validated scope and known limitations

The v0.2.5 tests use an exact-quote reference and a fixed tolerance protocol.
The ranges below include NumPy, PyTorch and JAX results on CPU and GPU.

**Refusals near the upper price bound.** The stress set has 1,446 strict rows.
Halley solves all these rows on every tested route.
Jäckel returns `NaN` with a signal on 306–328 rows per route.
Of these, 119 are outside its domain and 187–209 are near the upper price bound.
Most of the near-bound quotes are within a few ULPs of the bound.
The subtraction in the normalised Jäckel calculation loses precision near this bound.
The verification check then refuses these results. Use `solver="halley"` for these stress-set quotes.

**Coverage against v0.2.4.** The x86-64 NumPy development set has 1,962 strict
rows in the Jäckel domain. Jäckel returns finite values on 1,669 rows in v0.2.5 and
1,810 rows in v0.2.4. The result is below the coverage floor in the protocol.

Of the 220 rows which v0.2.5 refuses after v0.2.4 returned a finite value, 177 met
the tolerance in v0.2.4. Halley solves 173 of those 177 rows.
For the other 43 rows, the v0.2.4 values exceed the tolerance.
In total, 50 finite v0.2.4 Jäckel values exceed the tolerance on this set.
All finite v0.2.5 Jäckel values meet it.
Halley refuses four rows for which v0.2.4 returned values within their wide consistency sets.

**Finite values outside the tolerance.** On the stress set, 1–5 finite Jäckel
values per route exceed the full tolerance without a signal. Among these misses, the
maximum relative error is 1.6 × 10⁻¹¹. The largest error-to-tolerance ratio is 3.8.
Across all finite strict outputs, the maximum relative error is 0.019.
That row passes because its quote gives a wide rounding interval.

**PyTorch backend.** The native PyTorch port uses `torch.special.ndtr` for its initial
guess. This function loses relative accuracy in the deep lower tail.
The public route checks the result by repricing the quote.
The tests did not isolate the effect of the initial guess.

**Throughput.** Jäckel uses NumPy on the host to classify rows and verify solutions.
It uses double-double precision near the price bounds.
The PyTorch and JAX routes transfer data between the device and host.
In a shared-host test of 10⁶ options, NumPy time increased from 0.32 s to 0.55 s.
JAX time increased from 0.34 s to 0.46 s. The PyTorch time stayed the same.
Halley runs without this host stage. See [Benchmarks](benchmarks.md).

**Direct kernels.** The direct kernels have no domain check, repricing check or error signal.
The experimental `jackel_iv_triton` kernel meets the tolerance on 4,230 of 5,474 strict grid rows.

---

## Algorithm overview

The algorithm operates in the *normalised* Black domain to decouple the IV
problem from forward price and time-to-expiry:

1. **Normalisation** — reduce any call/put/ITM/OTM input to an OTM call
   `β = price / √(FK)` with reduced log-moneyness `x ≤ 0`.

2. **Boundary values** — compute the inflection point `s_c = √(2|x|)` and one
   Newton step left/right to get `(s_l, β_l)` and `(s_h, β_h)`.

3. **Initial guess** — cubic Hermite interpolation in (β, σ) space for the
   interior (zones 2/3); boundary values for the extremes (zones 1/4).

4. **Householder(3) × 3** — three iterations of the third-order rational
   Householder step with a three-branch objective function (log-space for
   near-zero β, complementary log-space for near-`β_max`, linear otherwise).

5. **Denormalisation** — `σ = σ̂ / √T`.

---

## Quick start

```python
import numpy as np
from fast_vollib.jackel.jackel_iv import jackel_iv_black

F     = 100.0
K     = np.array([95.0, 100.0, 105.0])
T     = 1.0
price = np.array([8.02,  7.97,  5.57])   # undiscounted Black-76 call prices

sigma = jackel_iv_black(price, F, K, T, is_call=True)
# array([0.2000..., 0.2000..., 0.2000...])
```

All inputs are **undiscounted** (Black-76 convention).  The `jackel_iv_black`
function handles put-call reduction internally — pass `is_call=False` for puts.

---

## API reference

### `jackel_iv_black` (NumPy / Numba)

```python
from fast_vollib.jackel.jackel_iv import jackel_iv_black

jackel_iv_black(
    price,       # undiscounted option price — ndarray or scalar
    F,           # forward price — float or ndarray
    K,           # strike — ndarray or scalar
    T,           # time to expiry in years — float or ndarray
    is_call=True # True = call, False = put — bool or bool ndarray
) -> np.ndarray  # annualised IV; NaN for degenerate inputs
```

!!! warning "Direct kernels"
    These kernels have no row policy or `on_error` signal.
    At or near intrinsic value they can return `NaN` without a signal.
    They can return a value for quotes outside the declared domain.
    For the row policy, use `fast_implied_volatility(..., solver="jackel")`.

Uses the NumPy + Numba JIT chain (CPU).  Six parallel Numba kernels cover
preproc, boundary, Hermite guess, Householder loop, and postproc.

---

### `jackel_iv_black_torch` (PyTorch)

```python
import torch
from fast_vollib.jackel.torch_backend import jackel_iv_black_torch

price_t = torch.tensor([8.02, 7.97, 5.57], dtype=torch.float64, device="cuda")
K_t     = torch.tensor([95.0, 100.0, 105.0], dtype=torch.float64, device="cuda")

sigma_t = jackel_iv_black_torch(price_t, F=100.0, K=K_t, T=1.0, is_call=True)
```

The Householder loop is wrapped with `torch.compile(dynamic=True)` so the
erfcx evaluations and branch dispatches fuse into a single CUDA kernel.

**Performance (v0.1.4 development measurement):** ~2.7 ms GPU compute / 100k options (CUDA events, H100 NVL).

---

### `jackel_iv_black_jax` (JAX)

```python
import jax.numpy as jnp
from fast_vollib.jackel.jax_backend import jackel_iv_black_jax

price_j = jnp.array([8.02, 7.97, 5.57])
K_j     = jnp.array([95.0, 100.0, 105.0])

sigma_j = jackel_iv_black_jax(price_j, F=100.0, K=K_j, T=1.0, is_call=True)
```

Uses `jax.lax.fori_loop` inside a `@jax.jit`-compiled function so the full
Householder loop is XLA-fused into one GPU kernel.  Requires `jax.config.x64 = True`
(enabled automatically at import time).

**Performance (v0.1.4 development measurement):** ~2.4 ms / 100k options (H100 NVL).

---

### `jackel_iv_triton` (Triton — experimental)

```python
import torch
from fast_vollib.jackel.triton_kernels import jackel_iv_triton

price_t = torch.tensor([8.02, 7.97, 5.57], dtype=torch.float64, device="cuda")
K_t     = torch.tensor([95.0, 100.0, 105.0], dtype=torch.float64, device="cuda")

sigma_t = jackel_iv_triton(price_t, F=100.0, K=K_t, T=1.0, is_call=True)
```

!!! warning "Experimental kernel"
    This kernel meets the tolerance on 4,230 of 5,474 strict rows in the canonical grid.
    It has no domain check, repricing check or error signal.
    For the public Jäckel route, use `fast_implied_volatility(..., solver="jackel")`.
    For the fused CUDA Halley kernel, select `solver="halley"` on the PyTorch backend.

One `@triton.jit` kernel performs preprocessing, the boundary calculation, the
Hermite initial guess and three Householder(3) iterations in one pass over HBM.
Triton 3.x does not provide `erfcx`. The kernel uses
`exp(x/2)·N(h+t) − exp(−x/2)·N(h−t)` for the normalised Black function.

**Performance (v0.1.4 development measurement):** ~0.056 ms GPU compute / 100k
options (CUDA events, H100 NVL); not re-measured for v0.2.5.

---

## Backend selection

The Jäckel module does **not** go through the `backends/` routing layer — each
function is called directly.  Choose based on your environment:

| Situation | Recommended call |
|---|---|
| Row policy and signals (recommended) | `fast_implied_volatility(..., solver="jackel")` |
| CPU only, raw kernel | `jackel_iv_black` (NumPy + Numba) |
| GPU, existing torch pipeline | `jackel_iv_black_torch` |
| GPU, existing JAX pipeline | `jackel_iv_black_jax` |
| GPU, experimental (fails validation) | `jackel_iv_triton` |

---

## Benchmarking

Run the dedicated benchmark script:

```bash
uv run python scripts/jackel_triton_bench.py
```

This script:

1. Generates the canonical 100k near-ATM grid
2. Computes reference prices via `py_lets_be_rational` (Jäckel C oracle)
3. Runs correctness check (`max_rel_err < 1e-8`)
4. Times the Triton kernel with CUDA events (20 timed rounds after 5 warm-up)

---

## Correctness

Parity tests live in `tests/test_jackel/test_parity.py` and run automatically
with the test suite (requires the `dev` dependency group):

```bash
uv sync                       # installs py-lets-be-rational via dev group
uv run pytest tests/test_jackel/ -v
```

The parity tests compare results with the independent Python package `py_lets_be_rational`.
`tests/test_iv_adjudicated.py` and `tests/test_iv_domain.py` use the exact-quote reference.
The protocol is in `tests/fixtures/iv_adjudication/`.
Use `scripts/validation_matrix.py` to write a record for each row.
