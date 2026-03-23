# Backend Selection

fast-vollib supports three numeric backends. All backends expose the same API and produce numerically equivalent results.

| Backend | When to use |
|---|---|
| `numpy` | Default; works everywhere; no GPU required |
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
price = fast_vollib.vectorized_black_scholes(
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

By default all backends return `numpy.ndarray`. Pass `return_native=True` to
receive the backend's native type instead:

```python
# Returns a torch.Tensor (float64)
price = fast_vollib.vectorized_black_scholes(
    flag="c", S=100, K=100, t=0.25, r=0.05, sigma=0.20,
    backend="torch",
    return_native=True,
)
```

!!! note
    `return_native=True` has no effect on the NumPy backend — NumPy arrays
    are already the native type.

---

## Backend availability

| Backend | `pip install` | Notes |
|---|---|---|
| `numpy` | bundled | Always available |
| `torch` | `pip install "fast-vollib[torch]"` | CPU wheels are cross-platform; GPU requires CUDA |
| `jax` | `pip install "fast-vollib[jax]"` | CPU-only by default; add `jax[cuda13]` for GPU |
