from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from . import backends
from .config import get_backend
from .types import BackendLiteral, ReturnAsLiteral
from .utils.broadcast import maybe_format_data_and_broadcast, preprocess_flags
from .utils.formatting import format_named_output
from .utils.validation import validate_data

if TYPE_CHECKING:
    from ._typing import ArrayLike, FlagLike, OptionalArrayLike  # noqa: F401


def _finalize(
    values: np.ndarray, return_as: str, name: str, backend_name: str, return_native: bool
):
    if return_native and backend_name in {"torch", "jax"}:
        return backends.get_module(backend_name).to_native(values)
    if return_as == "numpy":
        return values
    return format_named_output(values, name, return_as)


def fast_black(
    flag: FlagLike,
    F: ArrayLike,
    K: ArrayLike,
    t: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    *,
    return_as: ReturnAsLiteral = "dataframe",
    dtype=np.float64,
    backend: BackendLiteral = "auto",
    return_native: bool = False,
):
    flag = preprocess_flags(flag)
    F, K, t, r, sigma, flag = maybe_format_data_and_broadcast(F, K, t, r, sigma, flag, dtype=dtype)
    validate_data(F, K, t, r, sigma)
    backend_name = get_backend(backend)
    values = backends.get_module(backend_name).price_black(flag, F, K, t, r, sigma)
    return _finalize(values, return_as, "Price", backend_name, return_native)


def fast_black_scholes(
    flag: FlagLike,
    S: ArrayLike,
    K: ArrayLike,
    t: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    *,
    return_as: ReturnAsLiteral = "dataframe",
    dtype=np.float64,
    backend: BackendLiteral = "auto",
    return_native: bool = False,
):
    flag = preprocess_flags(flag)
    S, K, t, r, sigma, flag = maybe_format_data_and_broadcast(S, K, t, r, sigma, flag, dtype=dtype)
    validate_data(S, K, t, r, sigma)
    backend_name = get_backend(backend)
    values = backends.get_module(backend_name).price_black_scholes(flag, S, K, t, r, sigma)
    return _finalize(values, return_as, "Price", backend_name, return_native)


def fast_black_scholes_merton(
    flag: FlagLike,
    S: ArrayLike,
    K: ArrayLike,
    t: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    q: ArrayLike,
    *,
    return_as: ReturnAsLiteral = "dataframe",
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
    values = backends.get_module(backend_name).price_black_scholes_merton(
        flag, S, K, t, r, sigma, q
    )
    return _finalize(values, return_as, "Price", backend_name, return_native)
