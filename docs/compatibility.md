# Compatibility

fastiv is designed as a drop-in replacement for
[`py_vollib`](https://github.com/vollib/py_vollib) and
[`py_vollib_vectorized`](https://github.com/marcdemers/py_vollib_vectorized).

---

## What is preserved

| Aspect | Notes |
|---|---|
| Argument order | Preserved exactly for all pricing, IV, and Greek entry points |
| `flag` values | `"c"` / `"p"` strings work as before |
| `return_as="dataframe"` | Returns `pandas.DataFrame`, same as upstream |
| `return_as="series"` | Returns `pandas.Series`, same as upstream |
| Default NumPy output | `return_as="numpy"` returns `numpy.ndarray` |

---

## New kwargs (additive only)

fastiv adds two optional keyword-only arguments to every function. They are
silently consumed when not provided, so existing call sites continue to work
without changes.

| Kwarg | Default | Description |
|---|---|---|
| `backend` | `"auto"` | Override the numeric backend for this call |
| `return_native` | `False` | Return the backend's native tensor type |

---

## Runtime monkey-patch

`patch_py_vollib()` replaces the implementations inside the `py_vollib`
namespace at runtime. This is useful for codebases that import from `py_vollib`
directly and cannot be changed.

```python
# At program startup, before any py_vollib imports are used:
import fastiv
fastiv.patch_py_vollib()

# Now these transparently use fastiv under the hood:
from py_vollib.black_scholes import black_scholes
from py_vollib.black_scholes.implied_volatility import implied_volatility
from py_vollib.black_scholes.greeks.numerical import delta
```

**Namespaces patched:**

- `py_vollib.black` — `black`, IV, greeks
- `py_vollib.black_scholes` — `black_scholes`, IV, greeks
- `py_vollib.black_scholes_merton` — `black_scholes_merton`, IV, greeks

!!! warning "Requires `py_vollib`"
    `patch_py_vollib()` raises `ImportError` if `py_vollib` is not installed.

---

## Known differences

| Behaviour | `py_vollib_vectorized` | fastiv |
|---|---|---|
| Below-intrinsic IV | Returns `NaN` silently | Controlled by `on_error=` (`"warn"` / `"raise"` / `"ignore"`) |
| Backend | NumPy only | NumPy, PyTorch, JAX |
| GPU support | No | Yes (PyTorch + JAX) |
| `q` in Black-76 | Treated as zero | Correctly set to `r` (Black-76 forward pricing) |
