from __future__ import annotations

import numpy as np

from . import backends
from .config import get_backend
from .types import BackendLiteral
from .utils.broadcast import maybe_format_data_and_broadcast, preprocess_flags
from .utils.formatting import format_named_output
from .utils.validation import ensure_on_error, validate_data


def _backend_module(name: str):
    if name == "torch":
        return backends.torch_backend
    if name == "jax":
        return backends.jax_backend
    return backends.numpy_backend


def vectorized_implied_volatility(price, S, K, t, r, flag, q=None, *, on_error="warn", model="black_scholes", return_as="dataframe", dtype=np.float64, backend: BackendLiteral = "auto", return_native: bool = False, **kwargs):
    del kwargs
    ensure_on_error(on_error)
    flag = preprocess_flags(flag)
    if model == "black_scholes_merton":
        if q is None:
            raise ValueError("Must pass a `q` to black scholes merton model (annualized continuous dividend yield).")
        price, S, K, t, r, q, flag = maybe_format_data_and_broadcast(price, S, K, t, r, q, flag, dtype=dtype)
        validate_data(price, S, K, t, r, q)
    else:
        price, S, K, t, r, flag = maybe_format_data_and_broadcast(price, S, K, t, r, flag, dtype=dtype)
        validate_data(price, S, K, t, r)
    backend_name = get_backend(backend)
    values = _backend_module(backend_name).implied_volatility(model, price, S, K, t, r, flag, q=q, on_error=on_error)
    if return_native and backend_name in {"torch", "jax"}:
        return _backend_module(backend_name).to_native(values)
    if return_as == "numpy":
        return values
    return format_named_output(values, "IV", return_as)


def vectorized_implied_volatility_black(price, F, K, r, t, flag, *, on_error="warn", return_as="dataframe", dtype=np.float64, backend: BackendLiteral = "auto", return_native: bool = False, **kwargs):
    del kwargs
    return vectorized_implied_volatility(
        price,
        F,
        K,
        t,
        r,
        flag,
        on_error=on_error,
        model="black",
        return_as=return_as,
        dtype=dtype,
        backend=backend,
        return_native=return_native,
    )
