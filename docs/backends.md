# Backend Selection

fast-vollib supports four numeric backends. All backends expose the same API and produce numerically equivalent results.

| Backend | When to use |
|---|---|
| `numpy` | Default; works everywhere; no GPU required |
| `numba` | JIT-compiled CPU loops; faster than NumPy for large batches |
| `torch` | GPU acceleration on CUDA hardware |
| `jax` | JIT-compiled CPU/GPU/TPU; functional programming style |

---

## Automatic resolution

When `backend="auto"` (the default), fast-vollib resolves the backend at call time
using this priority order:

1. Explicit `backend=` kwarg on the function call
2. `fast_vollib.set_backend(name)` process-level override
3. `FAST_VOLLIB_BACKEND` environment variable
4. `torch` — if `torch.cuda.is_available()` returns `True`
5. `jax` — if JAX is importable
6. `numpy` — always available as the final fallback

!!! note
    The `numba` backend is **not** inserted into the `auto` resolution chain because
    it is a CPU-only backend like NumPy.  Use it explicitly via `backend="numba"` or
    `FAST_VOLLIB_BACKEND=numba` when you want JIT-compiled CPU acceleration without
    a GPU.

---

## Setting a backend

### Per-session (process-level)

```python
import fast_vollib

fast_vollib.set_backend("torch")   # all subsequent calls use PyTorch

# Reset to auto-resolution
fast_vollib.set_backend("auto")
```

### Per-call

Every pricing, IV, and Greek function accepts a `backend` keyword:

```python
price = fast_vollib.fast_black_scholes(
    flag="c", S=100, K=100, t=0.25, r=0.05, sigma=0.20,
    backend="numpy",
)
```

### Via environment variable

```bash
export FAST_VOLLIB_BACKEND=torch
python my_script.py
```

### Inspecting the active backend

```python
print(fast_vollib.get_backend())         # resolved backend for "auto"
print(fast_vollib.get_backend("torch"))  # pass an explicit value to validate it
```

---

## Native tensor output

Most public functions default to `return_as="dataframe"`, which materializes a
`pandas.DataFrame`. Pass `return_as="numpy"` if you want a `numpy.ndarray`, or
pass `return_native=True` on the PyTorch and JAX backends to receive the
backend's native type instead:

```python
# Returns a torch.Tensor (float64)
price = fast_vollib.fast_black_scholes(
    flag="c", S=100, K=100, t=0.25, r=0.05, sigma=0.20,
    backend="torch",
    return_native=True,
)
```

!!! note
    `return_native=True` has no effect on the NumPy backend — NumPy arrays
    are already the native type.

For `get_all_greeks`, `return_native=True` returns a `dict` mapping each Greek
name to a native backend array or tensor.

---

## Backend availability

| Backend | `pip install` | Notes |
|---|---|---|
| `numpy` | bundled | Always available |
| `numba` | `pip install "fast-vollib[numba]"` | CPU-only; JIT-compiled parallel loops |
| `torch` | `pip install "fast-vollib[torch]"` | CPU wheels are cross-platform; GPU requires CUDA |
| `jax` | `pip install "fast-vollib[jax]"` | CPU-only by default; add `jax[cuda13]` for GPU |

---

## JAX compilation on the first call

The first implied-volatility call in a process compiles the JAX solver.
For rows near the price bounds, Halley compiles a second version with higher precision.
For a batch of 10⁶ rows on GH200, the first call took 5.3 s in v0.2.5 and 1.8 s in v0.2.4.
Later calls in the same process reuse the compiled code.

To reuse this code in other processes, enable the JAX compilation cache before the first call.

```python
import jax

jax.config.update("jax_compilation_cache_dir", "/path/to/jax-cache")
# fast-vollib's kernels compile in under JAX's default 1 s threshold; cache them anyway
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
```

In this test, the batch size was 10⁵. The shared H100 host ran JAX 0.9.2.

| Device | Solver | First call without cached code | First call with cached code |
|---|---|---:|---:|
| GPU | Halley | 2.4 s | 0.6 s |
| GPU | Jäckel | 1.8 s | 0.2 s |
| CPU | Halley | 0.4 s | 0.2 s |
| CPU | Jäckel | 1.0 s | 0.3 s |

JAX reported CPU feature differences when it loaded compiled code from the CPU cache.
Use a separate CPU cache directory for each machine.


---

## Numba backend

The Numba backend compiles Black-Scholes pricing, Greeks, and the
Halley+bisection IV solver to native machine code via `@numba.njit(parallel=True)`.
Each batch incurs a single Python→native dispatch; the Halley loop and
bisection fallback execute entirely within the compiled kernel.

Kernels are compiled on **first call** and cached to `__pycache__` for
subsequent process starts.  Expect a ~1–2 s warm-up on the first call; all
subsequent calls hit the cache and are near-instant.

```python
import fast_vollib

# Use the numba backend for a single call
price = fast_vollib.fast_black_scholes(
    flag="c", S=100, K=100, t=0.25, r=0.05, sigma=0.20,
    backend="numba",
)

# Or set it process-wide
fast_vollib.set_backend("numba")
```

```bash
# Or via environment variable
export FAST_VOLLIB_BACKEND=numba
python my_script.py
```

### When to choose numba over numpy

- Large batches (≥ 10 k options) on a CPU-only machine
- Environments where PyTorch / JAX cannot be installed (e.g. minimal Docker images)
- When you need deterministic, portable CPU performance without GPU drivers

## Jäckel IV — direct solver functions

The `fast_vollib.jackel` module is **not** routed through the backend system
described above.  It is a self-contained implementation of Peter Jäckel's
*"Let's Be Rational"* algorithm and exposes one function per backend:

| Backend | Import | Notes |
|---|---|---|
| NumPy + Numba (CPU) | `fast_vollib.jackel.jackel_iv.jackel_iv_black` | Parallel Numba kernels; ~8.5 ms / 100k |
| PyTorch (GPU) | `fast_vollib.jackel.torch_backend.jackel_iv_black_torch` | `torch.compile` fused; ~2.7 ms / 100k |
| JAX (GPU) | `fast_vollib.jackel.jax_backend.jackel_iv_black_jax` | XLA fused; ~2.4 ms / 100k |
| Triton (GPU) | `fast_vollib.jackel.triton_kernels.jackel_iv_triton` | **Experimental**: fails validation (4,230 of 5,474 strict grid rows) |

The v0.1.4 timings are for 100,000 options on H100 NVL. The table lists direct kernels.
For the row policy and error signal, use `fast_implied_volatility(..., solver="jackel")`.
See [Jäckel IV](jackel.md) for the validation results.
