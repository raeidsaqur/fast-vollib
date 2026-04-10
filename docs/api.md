# API Reference

All public symbols are importable directly from `fast_vollib`:

```python
from fast_vollib import fast_black_scholes, fast_implied_volatility, ...
```

---

## Common parameters

The following parameters appear across pricing, IV, and Greek functions:

| Parameter | Type | Description |
|---|---|---|
| `flag` | `str \| array-like` | `"c"` for call, `"p"` for put |
| `S` | `float \| array-like` | Underlying spot price |
| `F` | `float \| array-like` | Forward price (Black-76 only) |
| `K` | `float \| array-like` | Strike price |
| `t` | `float \| array-like` | Time to expiry in years |
| `r` | `float \| array-like` | Risk-free rate (continuous, annualized) |
| `sigma` | `float \| array-like` | Volatility (annualized) |
| `q` | `float \| array-like` | Continuous dividend yield (BSM only) |
| `model` | `str` | `"black"`, `"black_scholes"` (default), or `"black_scholes_merton"` |
| `return_as` | `str` | Output container; see each function below |
| `dtype` | `numpy.dtype` | Input coercion dtype; default `numpy.float64` |
| `backend` | `str` | `"auto"` (default), `"numpy"`, `"torch"`, or `"jax"` |
| `return_native` | `bool` | Return backend-native arrays for torch/jax instead of formatted pandas/NumPy output |

All array-like parameters are broadcast against each other using NumPy
broadcasting rules.

---

## Pricing

Pricing functions default to `return_as="dataframe"`. Supported container
formats are:

- `return_as="dataframe"`: `pandas.DataFrame`
- `return_as="series"`: `pandas.Series`
- `return_as="numpy"`: `numpy.ndarray`
- `return_native=True` with `backend="torch"` or `backend="jax"`: native tensor/array

### `fast_black`

Black-76 model for options on futures.

```python
fast_vollib.fast_black(
    flag,
    F,
    K,
    t,
    r,
    sigma,
    *,
    return_as="dataframe",
    dtype=numpy.float64,
    backend="auto",
    return_native=False,
)
```

**Returns:** Option price(s) shaped by broadcasting rules.

### `fast_black_scholes`

Black-Scholes model for European equity options without dividends.

```python
fast_vollib.fast_black_scholes(
    flag,
    S,
    K,
    t,
    r,
    sigma,
    *,
    return_as="dataframe",
    dtype=numpy.float64,
    backend="auto",
    return_native=False,
)
```

**Returns:** Option price(s) shaped by broadcasting rules.

### `fast_black_scholes_merton`

Black-Scholes-Merton model with a continuous dividend yield.

```python
fast_vollib.fast_black_scholes_merton(
    flag,
    S,
    K,
    t,
    r,
    sigma,
    q,
    *,
    return_as="dataframe",
    dtype=numpy.float64,
    backend="auto",
    return_native=False,
)
```

**Returns:** Option price(s) shaped by broadcasting rules.

---

## Implied Volatility

IV functions share the same output conventions as pricing functions:
`"dataframe"`, `"series"`, `"numpy"`, or native torch/jax arrays via
`return_native=True`.

### `fast_implied_volatility`

Solve for implied volatility given a market price.

```python
fast_vollib.fast_implied_volatility(
    price,
    S,
    K,
    t,
    r,
    flag,
    q=None,
    *,
    on_error="warn",
    model="black_scholes",
    return_as="dataframe",
    dtype=numpy.float64,
    backend="auto",
    return_native=False,
)
```

**Returns:** Implied volatility shaped by broadcasting rules. Below-intrinsic
inputs return `NaN` when `on_error` is `"warn"` or `"ignore"`.

!!! note "`on_error`"
    - `"raise"`: raise `ValueError` for below-intrinsic inputs
    - `"warn"`: emit a warning and return `NaN`
    - `"ignore"`: silently return `NaN`

### `fast_implied_volatility_black`

Convenience wrapper for Black-76 IV. The positional argument order matches
`py_vollib`: `price, F, K, r, t, flag`.

```python
fast_vollib.fast_implied_volatility_black(
    price,
    F,
    K,
    r,
    t,
    flag,
    *,
    on_error="warn",
    return_as="dataframe",
    dtype=numpy.float64,
    backend="auto",
    return_native=False,
)
```

**Returns:** Implied volatility under Black-76.

---

## Greeks

The individual Greek functions are exported under the compatibility aliases
`vectorized_delta`, `vectorized_gamma`, `vectorized_theta`, `vectorized_rho`,
and `vectorized_vega`.

Their signature is:

```python
fast_vollib.vectorized_<greek>(
    flag,
    S,
    K,
    t,
    r,
    sigma,
    q=None,
    *,
    model="black_scholes",
    return_as="dataframe",
    dtype=numpy.float64,
    backend="auto",
    return_native=False,
)
```

Supported outputs:

- `return_as="dataframe"`: `pandas.DataFrame`
- `return_as="series"`: `pandas.Series`
- `return_as="numpy"`: `numpy.ndarray`
- `return_native=True` with `backend="torch"` or `backend="jax"`: native tensor/array

### `vectorized_delta`

First derivative of price with respect to the underlying (∂V/∂S).

### `vectorized_gamma`

Second derivative of price with respect to the underlying (∂²V/∂S²).

### `vectorized_theta`

Rate of change of price with respect to time (∂V/∂t), expressed as daily decay.

### `vectorized_rho`

Sensitivity to the risk-free rate (∂V/∂r).

### `vectorized_vega`

Sensitivity to implied volatility (∂V/∂σ), expressed per 1% move in vol.

---

## `get_all_greeks`

Compute all five Greeks in a single vectorized call.

```python
fast_vollib.get_all_greeks(
    flag,
    S,
    K,
    t,
    r,
    sigma,
    q=None,
    *,
    model="black_scholes",
    return_as="dataframe",
    dtype=numpy.float64,
    backend="auto",
    return_native=False,
)
```

**Returns:**

- `return_as="dataframe"` (default): `pandas.DataFrame` with columns `delta`, `gamma`, `theta`, `rho`, `vega`
- `return_as="json"`: JSON string mapping Greek name to values
- `return_as="dict"`: Python `dict[str, numpy.ndarray]`
- `return_native=True` with `backend="torch"` or `backend="jax"`: `dict[str, native array]`

---

## DataFrame helper

### `price_dataframe`

Price, solve IV, and compute Greeks for every row of a DataFrame in one call.

```python
fast_vollib.price_dataframe(
    df,
    *,
    flag_col,
    underlying_price_col,
    strike_col,
    annualized_tte_col,
    riskfree_rate_col,
    sigma_col=None,
    price_col=None,
    dividend_col=None,
    model="black_scholes",
    inplace=False,
    dtype=numpy.float64,
    backend="auto",
    return_native=False,
)
```

You must supply at least one of `sigma_col` or `price_col`:

- `sigma_col`: compute `Price` plus Greeks
- `price_col`: compute `IV` plus Greeks
- both: use the supplied price and volatility data as inputs for the Greek calculation; the helper does not duplicate those input columns into the output

**Returns:** A new `pandas.DataFrame`, or `None` when `inplace=True`. Added
columns are:

- `Price` when `sigma_col` is provided
- `IV` when `price_col` is provided
- `delta`, `gamma`, `theta`, `rho`, `vega`

`price_dataframe` always materializes pandas output; `return_native` only
affects intermediate backend execution.

---

## Backend management

### `get_backend`

```python
fast_vollib.get_backend(explicit=None) -> str
```

Return the backend that would be used for a given explicit choice, or the
auto-resolved backend when called without arguments.

### `set_backend`

```python
fast_vollib.set_backend(name: str) -> None
```

Set a process-level backend override. Valid values are `"auto"`, `"numpy"`,
`"torch"`, and `"jax"`.

---

## Jäckel IV — machine-precision solver

The `fast_vollib.jackel` module provides a standalone implementation of Peter
Jäckel's *"Let's Be Rational"* algorithm.  These functions are called directly
(not routed through `backend=`).  See [Jäckel IV](jackel.md) for full
documentation.

### `jackel_iv_black` (CPU — NumPy + Numba)

```python
from fast_vollib.jackel.jackel_iv import jackel_iv_black

jackel_iv_black(price, F, K, T, is_call=True) -> np.ndarray
```

| Parameter | Type | Description |
|---|---|---|
| `price` | `float \| ndarray` | Undiscounted option price |
| `F` | `float \| ndarray` | Forward price |
| `K` | `float \| ndarray` | Strike |
| `T` | `float \| ndarray` | Time to expiry (years) |
| `is_call` | `bool \| ndarray` | `True` = call, `False` = put |

**Returns:** Annualised implied volatility. `NaN` for degenerate inputs
(zero price, below intrinsic, zero expiry).

### `jackel_iv_black_torch` (GPU — PyTorch)

```python
from fast_vollib.jackel.torch_backend import jackel_iv_black_torch

jackel_iv_black_torch(price, F, K, T, is_call=True) -> torch.Tensor
```

Inputs must be `torch.float64` tensors on the same device.  The Householder
loop is compiled with `torch.compile(dynamic=True)`.

### `jackel_iv_black_jax` (GPU — JAX)

```python
from fast_vollib.jackel.jax_backend import jackel_iv_black_jax

jackel_iv_black_jax(price, F, K, T, is_call=True) -> jax.Array
```

Uses `jax.lax.fori_loop` inside a `@jax.jit`-compiled function.
Float64 mode is enabled automatically at import.

### `jackel_iv_triton` (GPU — Triton, fastest)

```python
from fast_vollib.jackel.triton_kernels import jackel_iv_triton

jackel_iv_triton(price, F, K, T, is_call=True) -> torch.Tensor
```

Single-pass `@triton.jit` kernel.  All pipeline stages (preproc, boundary,
Hermite guess, Householder×3) execute in registers with one HBM read and one
HBM write per element.

---

## Compatibility

### `patch_py_vollib`

```python
fast_vollib.patch_py_vollib() -> None
```

Monkey-patch the scalar `py_vollib` namespace with fast-vollib implementations.
Requires `py_vollib` to be installed.

### `patch_py_vollib_vectorized`

```python
fast_vollib.patch_py_vollib_vectorized() -> None
```

Monkey-patch the `py_vollib_vectorized` namespace with fast-vollib
implementations. Requires `py_vollib_vectorized` to be installed.

See [Compatibility](compatibility.md) for examples and caveats.
