from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from . import backends
from .config import get_backend
from .types import BackendLiteral, ModelLiteral, OnErrorLiteral, ReturnAsLiteral
from .utils.broadcast import maybe_format_data_and_broadcast, preprocess_flags
from .utils.formatting import format_named_output
from .utils.validation import ensure_on_error, validate_data

if TYPE_CHECKING:
    from ._typing import ArrayLike, FlagLike, OptionalArrayLike  # noqa: F401


def fast_implied_volatility(
    price: ArrayLike,
    S: ArrayLike,
    K: ArrayLike,
    t: ArrayLike,
    r: ArrayLike,
    flag: FlagLike,
    q: OptionalArrayLike = None,
    *,
    on_error: OnErrorLiteral = "warn",
    model: ModelLiteral = "black_scholes",
    return_as: ReturnAsLiteral = "dataframe",
    dtype=np.float64,
    backend: BackendLiteral = "auto",
    return_native: bool = False,
    **kwargs,
):
    del kwargs
    ensure_on_error(on_error)
    flag = preprocess_flags(flag)
    if model == "black_scholes_merton":
        if q is None:
            raise ValueError(
                "Must pass a `q` to black scholes merton model (annualized continuous dividend yield)."
            )
        price, S, K, t, r, q, flag = maybe_format_data_and_broadcast(
            price, S, K, t, r, q, flag, dtype=dtype
        )
        validate_data(price, S, K, t, r, q)
    else:
        price, S, K, t, r, flag = maybe_format_data_and_broadcast(
            price, S, K, t, r, flag, dtype=dtype
        )
        validate_data(price, S, K, t, r)
    backend_name = get_backend(backend)
    values = backends.get_module(backend_name).implied_volatility(
        model, price, S, K, t, r, flag, q=q, on_error=on_error
    )
    if return_native and backend_name in {"torch", "jax"}:
        return backends.get_module(backend_name).to_native(values)
    if return_as == "numpy":
        return values
    return format_named_output(values, "IV", return_as)


def fast_implied_volatility_black(
    price: ArrayLike,
    F: ArrayLike,
    K: ArrayLike,
    r: ArrayLike,
    t: ArrayLike,
    flag: FlagLike,
    *,
    on_error: OnErrorLiteral = "warn",
    return_as: ReturnAsLiteral = "dataframe",
    dtype=np.float64,
    backend: BackendLiteral = "auto",
    return_native: bool = False,
    **kwargs,
):
    del kwargs
    return fast_implied_volatility(
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
