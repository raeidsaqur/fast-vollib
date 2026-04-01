# Compatibility

fast-vollib is designed as a drop-in replacement for
[`py_vollib`](https://github.com/vollib/py_vollib) and
[`py_vollib_vectorized`](https://github.com/marcdemers/py_vollib_vectorized).

---

## What is preserved

| Aspect | Notes |
|---|---|
| Argument order | Preserved exactly for all pricing, IV, and Greek entry points |
| `flag` values | `"c"` / `"p"` strings work as before |
| `return_as="dataframe"` | Default return path remains `pandas.DataFrame` |
| `return_as="series"` | Returns `pandas.Series`, same as upstream |
| NumPy array output | `return_as="numpy"` returns `numpy.ndarray` |

---

## New kwargs (additive only)

fast-vollib adds two optional keyword-only arguments to every function. They are
silently consumed when not provided, so existing call sites continue to work
without changes.

| Kwarg | Default | Description |
|---|---|---|
| `backend` | `"auto"` | Override the numeric backend for this call |
| `return_native` | `False` | Return the backend's native tensor type |

---

## Runtime monkey-patch

fast-vollib provides two separate patch helpers:

- `patch_py_vollib()` for the scalar `py_vollib` namespace
- `patch_py_vollib_vectorized()` for the `py_vollib_vectorized` namespace

Use the helper that matches the package already imported by your application.

### `patch_py_vollib()`

`patch_py_vollib()` replaces implementations inside the scalar `py_vollib`
namespace at runtime. This is useful for codebases that import from
`py_vollib` directly and cannot be changed.

```python
# At program startup, before any py_vollib imports are used:
import fast_vollib
fast_vollib.patch_py_vollib()

# Now these transparently use fast-vollib under the hood:
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

### `patch_py_vollib_vectorized()`

`patch_py_vollib_vectorized()` rewires the `py_vollib_vectorized` top-level,
`models`, `implied_volatility`, `greeks`, and `api` modules to the
fast-vollib implementations.

```python
import fast_vollib
fast_vollib.patch_py_vollib_vectorized()

from py_vollib_vectorized import vectorized_black_scholes
```

!!! warning "Requires `py_vollib_vectorized`"
    `patch_py_vollib_vectorized()` raises `ImportError` if
    `py_vollib_vectorized` is not installed.

---

## Known differences

| Behaviour | `py_vollib_vectorized` | fast-vollib |
|---|---|---|
| Below-intrinsic IV | Returns `NaN` silently | Controlled by `on_error=` (`"warn"` / `"raise"` / `"ignore"`) |
| Backend | NumPy only | NumPy, PyTorch, JAX |
| GPU support | No | Yes (PyTorch + JAX) |
| `q` in Black-76 | Treated as zero | Correctly set to `r` (Black-76 forward pricing) |
