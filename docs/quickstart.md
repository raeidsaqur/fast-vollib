# Quick Start

## Pricing

### Black-Scholes (equity options)

```python
import fast_vollib

# Single option
price = fast_vollib.vectorized_black_scholes(
    flag="c",      # "c" = call, "p" = put
    S=100.0,       # underlying price
    K=100.0,       # strike price
    t=0.25,        # time to expiry in years
    r=0.05,        # risk-free rate (continuous)
    sigma=0.20,    # volatility
    return_as="numpy",
)

# Vectorized batch — mix scalars and arrays freely
import numpy as np

prices = fast_vollib.vectorized_black_scholes(
    flag=np.array(["c", "c", "p", "p"]),
    S=100.0,                              # scalar broadcasts
    K=np.array([95, 100, 100, 105]),
    t=0.25,
    r=0.05,
    sigma=np.array([0.18, 0.20, 0.20, 0.22]),
    return_as="numpy",
)
```

### Black-76 (futures options)

```python
price = fast_vollib.vectorized_black(
    flag="c",
    F=100.0,   # forward price
    K=100.0,
    t=0.25,
    r=0.05,
    sigma=0.20,
    return_as="numpy",
)
```

### Black-Scholes-Merton (continuous dividends)

```python
price = fast_vollib.vectorized_black_scholes_merton(
    flag="c",
    S=100.0,
    K=100.0,
    t=0.25,
    r=0.05,
    sigma=0.20,
    q=0.02,    # continuous dividend yield
    return_as="numpy",
)
```

---

## Implied Volatility

```python
# Recover IV from market prices
iv = fast_vollib.vectorized_implied_volatility(
    price=np.array([3.63, 6.04, 4.27]),
    S=100.0,
    K=np.array([100, 95, 105]),
    t=0.25,
    r=0.05,
    flag=np.array(["c", "c", "p"]),
    return_as="numpy",
)

# Black-76 IV
iv_black = fast_vollib.vectorized_implied_volatility_black(
    price=3.5,
    F=100.0,
    K=100.0,
    r=0.05,
    t=0.25,
    flag="c",
    return_as="numpy",
)
```

---

## Greeks

### Individual Greeks

```python
delta = fast_vollib.vectorized_delta(
    flag="c", S=100, K=100, t=0.25, r=0.05, sigma=0.20,
    return_as="numpy",
)

gamma = fast_vollib.vectorized_gamma(
    flag="c", S=100, K=100, t=0.25, r=0.05, sigma=0.20,
    return_as="numpy",
)
```

### All Greeks at once

```python
greeks_df = fast_vollib.get_all_greeks(
    flag=["c", "p"],
    S=100.0,
    K=100.0,
    t=0.25,
    r=0.05,
    sigma=0.20,
)
# Returns a pandas DataFrame:
#    delta  gamma     theta      rho     vega
# 0   0.54   0.019  -0.0148   0.121   0.197
# 1  -0.46   0.019  -0.0098  -0.129   0.197
```

---

## DataFrame helper

`price_dataframe` processes an entire DataFrame in one call. You must supply
either `sigma_col` (to compute prices + greeks) or `price_col` (to compute IV
+ greeks), or both.

```python
import pandas as pd

df = pd.DataFrame({
    "flag":   ["c", "p", "c"],
    "S":      [100, 100, 105],
    "K":      [100, 100, 100],
    "t":      [0.25, 0.25, 0.50],
    "r":      [0.05, 0.05, 0.05],
    "sigma":  [0.20, 0.20, 0.18],
})

result = fast_vollib.price_dataframe(
    df,
    flag_col="flag",
    underlying_price_col="S",
    strike_col="K",
    annualized_tte_col="t",
    riskfree_rate_col="r",
    sigma_col="sigma",
)
# result has columns: Price, delta, gamma, theta, rho, vega
```

---

## Return formats

All pricing, IV, and Greek functions accept a `return_as` keyword:

| `return_as` | Output type |
|---|---|
| `"dataframe"` (default) | `pandas.DataFrame` |
| `"series"` | `pandas.Series` |
| `"numpy"` | `numpy.ndarray` |

Pass `return_native=True` to get the backend's native tensor/array type (e.g.
`torch.Tensor` when using the PyTorch backend) instead of a NumPy array.

---

## Backend selection

```python
# Use a specific backend for all calls in this session
fast_vollib.set_backend("torch")

# Override per-call
price = fast_vollib.vectorized_black_scholes(
    ...,
    backend="jax",
)
```

See [Backend Selection](backends.md) for more.
