from __future__ import annotations

import numpy as np

from . import backends
from .config import get_backend
from .types import BackendLiteral
from .utils.broadcast import maybe_format_data_and_broadcast, preprocess_flags
from .utils.formatting import format_named_output
from .utils.validation import validate_data


def _greek(
    name: str,
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
    dtype=np.float64,
    backend: BackendLiteral = "auto",
    return_native: bool = False,
):
    flag = preprocess_flags(flag)
    if model == "black_scholes_merton":
        if q is None:
            raise ValueError(
                "Must pass a `q` to black scholes merton model (annualized continuous dividend yield)."
            )
        S, K, t, r, sigma, q, flag = maybe_format_data_and_broadcast(
            S, K, t, r, sigma, q, flag, dtype=dtype
        )
        validate_data(S, K, t, r, sigma, q)
    else:
        S, K, t, r, sigma, flag = maybe_format_data_and_broadcast(
            S, K, t, r, sigma, flag, dtype=dtype
        )
        validate_data(S, K, t, r, sigma)
    backend_name = get_backend(backend)
    values = backends.get_module(backend_name).greeks(model, flag, S, K, t, r, sigma, q=q)[name]
    if return_native and backend_name in {"torch", "jax"}:
        return backends.get_module(backend_name).to_native(values)
    if return_as == "numpy":
        return values
    return format_named_output(values, name, return_as)


def delta(
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
    dtype=np.float64,
    backend: BackendLiteral = "auto",
    return_native: bool = False,
):
    return _greek(
        "delta",
        flag,
        S,
        K,
        t,
        r,
        sigma,
        q,
        model=model,
        return_as=return_as,
        dtype=dtype,
        backend=backend,
        return_native=return_native,
    )


def gamma(
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
    dtype=np.float64,
    backend: BackendLiteral = "auto",
    return_native: bool = False,
):
    return _greek(
        "gamma",
        flag,
        S,
        K,
        t,
        r,
        sigma,
        q,
        model=model,
        return_as=return_as,
        dtype=dtype,
        backend=backend,
        return_native=return_native,
    )


def rho(
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
    dtype=np.float64,
    backend: BackendLiteral = "auto",
    return_native: bool = False,
):
    return _greek(
        "rho",
        flag,
        S,
        K,
        t,
        r,
        sigma,
        q,
        model=model,
        return_as=return_as,
        dtype=dtype,
        backend=backend,
        return_native=return_native,
    )


def theta(
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
    dtype=np.float64,
    backend: BackendLiteral = "auto",
    return_native: bool = False,
):
    return _greek(
        "theta",
        flag,
        S,
        K,
        t,
        r,
        sigma,
        q,
        model=model,
        return_as=return_as,
        dtype=dtype,
        backend=backend,
        return_native=return_native,
    )


def vega(
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
    dtype=np.float64,
    backend: BackendLiteral = "auto",
    return_native: bool = False,
):
    return _greek(
        "vega",
        flag,
        S,
        K,
        t,
        r,
        sigma,
        q,
        model=model,
        return_as=return_as,
        dtype=dtype,
        backend=backend,
        return_native=return_native,
    )
