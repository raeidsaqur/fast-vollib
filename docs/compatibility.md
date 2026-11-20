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

# These now call fast-vollib, with py_vollib's own signatures:
from py_vollib.black_scholes import black_scholes
from py_vollib.black_scholes.implied_volatility import implied_volatility
from py_vollib.black_scholes.greeks.numerical import delta
```

**Functions patched** (in `py_vollib.black`, `py_vollib.black_scholes` and
`py_vollib.black_scholes_merton`):

| Surface | Patched | Notes |
|---|---|---|
| `black`, `black_scholes`, `black_scholes_merton` | yes | |
| `<model>.implied_volatility.implied_volatility` | yes | `black_scholes_merton` keeps py_vollib's order `(price, S, K, t, r, q, flag)` |
| `<model>.greeks.numerical.{delta, gamma, rho, theta, vega}` | yes | Black-76 rho follows py_vollib's convention, `-t * price / 100` per percentage point |
| `<model>.greeks.analytical.*` | no | left as upstream |

- `inspect.signature` shows the original py_vollib signature.
  Positional arguments and keyword arguments follow that signature.
- A scalar input gives a Python `float`. An array input gives a `numpy.ndarray`.
- The Black-76 functions use the Black-76 model. The Black-Scholes-Merton
  functions use `q` on all backends.
- The patch changes only py_vollib functions. A second call makes no further changes.
- You can also use `backend=`, `on_error=` and `dtype=` as keyword arguments.

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

| Behaviour | `py_vollib` | `patch_py_vollib()` |
|---|---|---|
| IV of a price no volatility can produce (e.g. below intrinsic) | Raises (e.g. `BelowIntrinsicException`) | `NaN` with one warning per call; `on_error="raise"` raises `ValueError`, `on_error="ignore"` is silent |
