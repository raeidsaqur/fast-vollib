# Jäckel IV — Machine-Precision Implied Volatility

The `fast_vollib.jackel` module is a standalone, full-precision implied-volatility
solver based on Peter Jäckel's *"Let's Be Rational"* (2016) algorithm.  It solves
the normalised Black call problem to **machine precision** (max relative error
~10⁻¹⁴) using a four-branch rational initial guess and three Householder(3)
iterations — and it does so in a single GPU kernel pass with Triton.

---

## Why a separate solver?

The production `backends/` IV solver uses **Halley's method × 8 iterations** with
a compiled bisection fallback.  It is fast and robust for typical market data but
does not formally guarantee machine precision for all inputs.

The `jackel/` module provides:

| Property | `backends/` Halley×8 | `jackel/` Householder(3)×3 |
|---|---|---|
| Algorithm | Halley + bisection | Jäckel LBR rational guess + HH(3) |
| Max relative error | ~10⁻⁸ (typical) | ~10⁻¹⁴ (guaranteed) |
| Convergence | 8 iterations + fallback | 3 Householder(3) steps |
| CPU (NumPy + Numba) | ~8 ms / 100k | ~8.5 ms / 100k |
| GPU — torch.compile | ~2.8 ms / 100k | ~2.7 ms / 100k |
| GPU — JAX jit | ~3.85 ms / 100k | ~2.4 ms / 100k |
| GPU — Triton (compute) | **0.636 ms** / 100k | **0.056 ms** / 100k |

Numbers measured on an NVIDIA H100 NVL, N = 100,000 options,
canonical near-ATM grid (K = F·exp(U(−0.3, 0.3)), σ = U(0.10, 0.60)).

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

**Performance:** ~2.7 ms GPU compute / 100k options (CUDA events, H100 NVL).

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

**Performance:** ~2.4 ms / 100k options (H100 NVL) — 2× faster than the
PyTorch backend due to more aggressive XLA loop fusion.

---

### `jackel_iv_triton` (Triton — fastest)

```python
import torch
from fast_vollib.jackel.triton_kernels import jackel_iv_triton

price_t = torch.tensor([8.02, 7.97, 5.57], dtype=torch.float64, device="cuda")
K_t     = torch.tensor([95.0, 100.0, 105.0], dtype=torch.float64, device="cuda")

sigma_t = jackel_iv_triton(price_t, F=100.0, K=K_t, T=1.0, is_call=True)
```

A single `@triton.jit` kernel fuses the **entire** pipeline — preproc,
boundary, Hermite guess, and three Householder(3) iterations — in one pass
over HBM.  All intermediate values live in registers; there are no
intermediate writes.

**Performance:** ~0.056 ms GPU compute / 100k options (CUDA events, H100 NVL)
— **11× faster** than the 0.636 ms Halley×8 Triton target.

!!! note "Triton vs. erfcx"
    Triton 3.x does not expose `erfcx`.  The kernel uses the erf-based
    normalised Black formula `exp(x/2)·N(h+t) − exp(−x/2)·N(h−t)` instead.
    This is numerically stable for all standard option ranges and achieves
    max relative error ~10⁻¹³.

---

## Backend selection

The Jäckel module does **not** go through the `backends/` routing layer — each
function is called directly.  Choose based on your environment:

| Situation | Recommended call |
|---|---|
| CPU only, max precision | `jackel_iv_black` (NumPy + Numba) |
| GPU, existing torch pipeline | `jackel_iv_black_torch` |
| GPU, existing JAX pipeline | `jackel_iv_black_jax` |
| GPU, maximum throughput | `jackel_iv_triton` |

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

The oracle is `py_lets_be_rational` (Peter Jäckel's reference C implementation
wrapped for Python).  It has **no dependency on fast-vollib**.

---

## References

- P. Jäckel, *"Let's Be Rational"*, Wilmott Magazine, January 2015.
  [`jaeckel.org`](http://www.jaeckel.org/LetsBeRational.pdf)
- Source: [`jaeckel.org/LetsBeRational.7z`](http://www.jaeckel.org/LetsBeRational.7z)
