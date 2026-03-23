from __future__ import annotations

import numpy as np

from . import backends
from .config import get_backend
from .types import BackendLiteral
from .utils.broadcast import maybe_format_data_and_broadcast, preprocess_flags
from .utils.formatting import format_named_output
from .utils.validation import validate_data


def _backend_module(name: str):
    if name == "torch":
        return backends.torch_backend
    if name == "jax":
        return backends.jax_backend
    return backends.numpy_backend


def _finalize(
    values: np.ndarray, return_as: str, name: str, backend_name: str, return_native: bool
):
    if return_native and backend_name in {"torch", "jax"}:
        return _backend_module(backend_name).to_native(values)
    if return_as == "numpy":
        return values
    return format_named_output(values, name, return_as)


def vectorized_black(
    flag,
    F,
    K,
    t,
    r,
    sigma,
    *,
    return_as="dataframe",
    dtype=np.float64,
    backend: BackendLiteral = "auto",
    return_native: bool = False,
):
    flag = preprocess_flags(flag)
    F, K, t, r, sigma, flag = maybe_format_data_and_broadcast(F, K, t, r, sigma, flag, dtype=dtype)
    validate_data(F, K, t, r, sigma)
    backend_name = get_backend(backend)
    values = _backend_module(backend_name).price_black(flag, F, K, t, r, sigma)
    return _finalize(values, return_as, "Price", backend_name, return_native)


def vectorized_black_scholes(
    flag,
    S,
    K,
    t,
    r,
    sigma,
    *,
    return_as="dataframe",
    dtype=np.float64,
    backend: BackendLiteral = "auto",
    return_native: bool = False,
):
    flag = preprocess_flags(flag)
    S, K, t, r, sigma, flag = maybe_format_data_and_broadcast(S, K, t, r, sigma, flag, dtype=dtype)
    validate_data(S, K, t, r, sigma)
    backend_name = get_backend(backend)
    values = _backend_module(backend_name).price_black_scholes(flag, S, K, t, r, sigma)
    return _finalize(values, return_as, "Price", backend_name, return_native)


def vectorized_black_scholes_merton(
    flag,
    S,
    K,
    t,
    r,
    sigma,
    q,
    *,
    return_as="dataframe",
    dtype=np.float64,
    backend: BackendLiteral = "auto",
    return_native: bool = False,
):
    flag = preprocess_flags(flag)
    S, K, t, r, sigma, q, flag = maybe_format_data_and_broadcast(
        S, K, t, r, sigma, q, flag, dtype=dtype
    )
    validate_data(S, K, t, r, sigma, q)
    backend_name = get_backend(backend)
    values = _backend_module(backend_name).price_black_scholes_merton(flag, S, K, t, r, sigma, q)
    return _finalize(values, return_as, "Price", backend_name, return_native)
