from __future__ import annotations

import numpy as np
import pandas as pd

from . import backends
from .config import get_backend
from .greeks import delta, gamma, rho, theta, vega
from .implied_volatility import fast_implied_volatility, fast_implied_volatility_black
from .models import fast_black, fast_black_scholes, fast_black_scholes_merton
from .utils.broadcast import maybe_format_data_and_broadcast, preprocess_flags
from .utils.formatting import format_greeks_output
from .utils.validation import validate_data


def get_all_greeks(
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
    dtype=None,
    backend="auto",
    return_native=False,
):
    # Single-pass: preprocess once, call backend greeks() once (avoids 5x redundant d1/d2/exp/ndtr)
    _dtype = dtype if dtype is not None else np.float64
    flag = preprocess_flags(flag)
    if model == "black_scholes_merton":
        if q is None:
            raise ValueError(
                "Must pass a `q` to black scholes merton model (annualized continuous dividend yield)."
            )
        S, K, t, r, sigma, q, flag = maybe_format_data_and_broadcast(
            S, K, t, r, sigma, q, flag, dtype=_dtype
        )
        validate_data(S, K, t, r, sigma, q)
    else:
        S, K, t, r, sigma, flag = maybe_format_data_and_broadcast(
            S, K, t, r, sigma, flag, dtype=_dtype
        )
        validate_data(S, K, t, r, sigma)
    backend_name = get_backend(backend)
    bmod = backends.get_module(backend_name)
    data = bmod.greeks(model, flag, S, K, t, r, sigma, q=q)
    if return_native and backend_name in {"torch", "jax"}:
        data = {k: bmod.to_native(v) for k, v in data.items()}
    return format_greeks_output(data, return_as)


def price_dataframe(
    df: pd.DataFrame,
    *,
    flag_col=None,
    underlying_price_col=None,
    strike_col=None,
    annualized_tte_col=None,
    riskfree_rate_col=None,
    sigma_col=None,
    price_col=None,
    dividend_col=None,
    model="black_scholes",
    inplace=False,
    dtype=None,
    backend="auto",
    return_native=False,
):
    if flag_col is None:
        raise ValueError("You must specify a `flag_col` argument!")
    if underlying_price_col is None:
        raise ValueError("You must specify a `underlying_price_col` argument!")
    if strike_col is None:
        raise ValueError("You must specify a `strike_col` argument!")
    if annualized_tte_col is None:
        raise ValueError("You must specify a `annualized_tte_col` argument!")
    if riskfree_rate_col is None:
        raise ValueError("You must specify a `riskfree_rate_col` argument!")

    flag = df[flag_col]
    S = df[underlying_price_col]
    K = df[strike_col]
    t = df[annualized_tte_col]
    r = df[riskfree_rate_col]
    q = df[dividend_col] if dividend_col is not None and dividend_col in df.columns else None

    output = df if inplace else pd.DataFrame(index=df.index)

    sigma = df[sigma_col] if sigma_col is not None else None
    price = df[price_col] if price_col is not None else None

    if sigma is not None and price is None:
        if model == "black":
            priced = fast_black(
                flag,
                S,
                K,
                t,
                r,
                sigma,
                return_as="numpy",
                dtype=dtype,
                backend=backend,
                return_native=return_native,
            )
        elif model == "black_scholes":
            priced = fast_black_scholes(
                flag,
                S,
                K,
                t,
                r,
                sigma,
                return_as="numpy",
                dtype=dtype,
                backend=backend,
                return_native=return_native,
            )
        else:
            priced = fast_black_scholes_merton(
                flag,
                S,
                K,
                t,
                r,
                sigma,
                q,
                return_as="numpy",
                dtype=dtype,
                backend=backend,
                return_native=return_native,
            )
        output["Price"] = priced
        price = priced

    if price is not None and sigma is None:
        if model == "black":
            implied = fast_implied_volatility_black(
                price,
                S,
                K,
                r,
                t,
                flag,
                return_as="numpy",
                dtype=dtype,
                backend=backend,
                return_native=return_native,
            )
        else:
            implied = fast_implied_volatility(
                price,
                S,
                K,
                t,
                r,
                flag,
                q=q,
                model=model,
                return_as="numpy",
                dtype=dtype,
                backend=backend,
                return_native=return_native,
            )
        output["IV"] = implied
        sigma = implied

    if sigma is None and price is None:
        raise ValueError("You must specify either `sigma_col`, `price_col`, or both!")

    greeks = get_all_greeks(
        flag,
        S,
        K,
        t,
        r,
        sigma,
        q=q,
        model=model,
        return_as="dataframe",
        dtype=dtype,
        backend=backend,
        return_native=False,
    )
    for column in greeks.columns:
        output[column] = greeks[column].to_numpy()

    if inplace:
        return None
    return output
