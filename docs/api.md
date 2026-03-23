# API Reference

All public symbols are importable directly from `fast_vollib`:

```python
from fast_vollib import vectorized_black_scholes, vectorized_implied_volatility, ...
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
| `return_as` | `str` | `"dataframe"` (default), `"series"`, or `"numpy"` |
| `dtype` | `numpy.dtype` | Output dtype; default `numpy.float64` |
| `backend` | `str` | `"auto"` (default), `"numpy"`, `"torch"`, or `"jax"` |
| `return_native` | `bool` | Return backend-native tensor instead of `ndarray`; default `False` |

All array-like parameters are broadcast against each other following NumPy
broadcasting rules.

---

## Pricing

### `vectorized_black`

Black-76 model for options on futures.

```python
fast_vollib.vectorized_black(
    flag,
    F,        # forward price
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

---

### `vectorized_black_scholes`

Black-Scholes model for European equity options (no dividends).

```python
fast_vollib.vectorized_black_scholes(
    flag,
    S,        # spot price
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

---

### `vectorized_black_scholes_merton`

Black-Scholes-Merton model with a continuous dividend yield.

```python
fast_vollib.vectorized_black_scholes_merton(
    flag,
    S,
    K,
    t,
    r,
    sigma,
    q,        # continuous dividend yield (required)
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

### `vectorized_implied_volatility`

Solve for implied volatility given a market price.

```python
fast_vollib.vectorized_implied_volatility(
    price,    # observed market price
    S,
    K,
    t,
    r,
    flag,
    q=None,   # required when model="black_scholes_merton"
    *,
    on_error="warn",   # "raise", "warn", or "ignore"
    model="black_scholes",
    return_as="dataframe",
    dtype=numpy.float64,
    backend="auto",
    return_native=False,
)
```

**Returns:** Implied volatility (σ) shaped by broadcasting rules. Returns
`NaN` for options below intrinsic value when `on_error` is `"warn"` or
`"ignore"`.

!!! note "`on_error`"
    - `"raise"` — raise `ValueError` for below-intrinsic inputs
    - `"warn"` (default) — emit a `RuntimeWarning` and return `NaN`
    - `"ignore"` — silently return `NaN`

---

### `vectorized_implied_volatility_black`

Convenience wrapper: solves IV under the Black-76 model.
Argument order matches `py_vollib` (`price, F, K, r, t, flag`).

```python
fast_vollib.vectorized_implied_volatility_black(
    price,
    F,        # forward price
    K,
    r,
    t,        # note: r before t (py_vollib convention)
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

All Greek functions share the same signature:

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

### `vectorized_delta`

First derivative of price with respect to the underlying (∂V/∂S).

### `vectorized_gamma`

Second derivative of price with respect to the underlying (∂²V/∂S²).

### `vectorized_theta`

Rate of change of price with respect to time (∂V/∂t). Expressed as daily
decay (divided by 365).

### `vectorized_rho`

Sensitivity to the risk-free rate (∂V/∂r).

### `vectorized_vega`

Sensitivity to implied volatility (∂V/∂σ). Expressed per 1% move in vol
(divided by 100).

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

**Returns:** When `return_as="dataframe"` (default), a `pandas.DataFrame`
with columns `["delta", "gamma", "theta", "rho", "vega"]`. When
`return_native=True`, a `dict` mapping Greek name to native array.

---

## DataFrame helper

### `price_dataframe`

Price, compute IV, and compute Greeks for every row of a DataFrame in one
call. Supply `sigma_col` to price options; supply `price_col` to solve IV;
supply both to use the provided values as-is for Greeks.

```python
fast_vollib.price_dataframe(
    df,                           # pandas.DataFrame
    *,
    flag_col,                     # required
    underlying_price_col,         # required
    strike_col,                   # required
    annualized_tte_col,           # required
    riskfree_rate_col,            # required
    sigma_col=None,               # supply to price; result written to "Price"
    price_col=None,               # supply to solve IV; result written to "IV"
    dividend_col=None,            # optional; required for "black_scholes_merton"
    model="black_scholes",
    inplace=False,
    dtype=numpy.float64,
    backend="auto",
    return_native=False,
)
```

**Returns:** A new `pandas.DataFrame` (or `None` when `inplace=True`) with
computed columns appended. Columns added: `"Price"` (if `sigma_col` was
given), `"IV"` (if `price_col` was given), plus `delta`, `gamma`, `theta`,
`rho`, `vega`.

---

## Backend management

### `get_backend`

```python
fast_vollib.get_backend(explicit=None) -> str
```

Return the backend that would be used for a given `explicit` value (or the
auto-resolved backend when called with no arguments).

### `set_backend`

```python
fast_vollib.set_backend(name: str) -> None
```

Set a process-level backend override. Valid values: `"auto"`, `"numpy"`,
`"torch"`, `"jax"`.

---

## Compatibility

### `patch_py_vollib`

```python
fast_vollib.patch_py_vollib() -> None
```

Monkey-patch the `py_vollib` and `py_vollib_vectorized` namespaces with
fast-vollib's implementations. After this call, any code that imports from
`py_vollib` will transparently use fast_vollib. Requires `py_vollib` to be
installed.

See [Compatibility](compatibility.md) for details.
