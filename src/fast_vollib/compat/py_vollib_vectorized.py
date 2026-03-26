from __future__ import annotations

from ..api import get_all_greeks, price_dataframe
from ..greeks import delta, gamma, rho, theta, vega
from ..implied_volatility import fast_implied_volatility, fast_implied_volatility_black
from ..models import fast_black, fast_black_scholes, fast_black_scholes_merton


def patch_py_vollib_vectorized() -> None:
    """Replace py_vollib_vectorized functions with fast_vollib implementations.

    https://github.com/marcdemers/py_vollib_vectorized

    After calling this, any code that calls ``py_vollib_vectorized.vectorized_black_scholes(...)``
    (or any other exported function) will transparently use the fast_vollib backend.
    Signatures are identical, so no call-site changes are needed.

    Raises
    ------
    ImportError
        If ``py_vollib_vectorized`` is not installed.
    """
    try:
        import py_vollib_vectorized
        import py_vollib_vectorized.api
        import py_vollib_vectorized.greeks
        import py_vollib_vectorized.implied_volatility
        import py_vollib_vectorized.models
    except ImportError as exc:
        raise ImportError(
            "You must have py_vollib_vectorized installed to use patch_py_vollib_vectorized()."
        ) from exc

    # --- Pricing models ---
    py_vollib_vectorized.models.vectorized_black = fast_black
    py_vollib_vectorized.models.vectorized_black_scholes = fast_black_scholes
    py_vollib_vectorized.models.vectorized_black_scholes_merton = fast_black_scholes_merton
    py_vollib_vectorized.vectorized_black = fast_black
    py_vollib_vectorized.vectorized_black_scholes = fast_black_scholes
    py_vollib_vectorized.vectorized_black_scholes_merton = fast_black_scholes_merton

    # --- Implied volatility ---
    py_vollib_vectorized.implied_volatility.vectorized_implied_volatility = fast_implied_volatility
    py_vollib_vectorized.implied_volatility.vectorized_implied_volatility_black = (
        fast_implied_volatility_black
    )
    py_vollib_vectorized.vectorized_implied_volatility = fast_implied_volatility
    py_vollib_vectorized.vectorized_implied_volatility_black = fast_implied_volatility_black

    # --- Greeks ---
    py_vollib_vectorized.greeks.delta = delta
    py_vollib_vectorized.greeks.gamma = gamma
    py_vollib_vectorized.greeks.rho = rho
    py_vollib_vectorized.greeks.theta = theta
    py_vollib_vectorized.greeks.vega = vega
    py_vollib_vectorized.vectorized_delta = delta
    py_vollib_vectorized.vectorized_gamma = gamma
    py_vollib_vectorized.vectorized_rho = rho
    py_vollib_vectorized.vectorized_theta = theta
    py_vollib_vectorized.vectorized_vega = vega

    # --- High-level API ---
    py_vollib_vectorized.api.get_all_greeks = get_all_greeks
    py_vollib_vectorized.api.price_dataframe = price_dataframe
    py_vollib_vectorized.get_all_greeks = get_all_greeks
    py_vollib_vectorized.price_dataframe = price_dataframe


def patch_py_vollib() -> None:
    """Replace py_vollib (scalar) functions with fast_vollib implementations.

    Raises
    ------
    ImportError
        If ``py_vollib`` is not installed.
    """
    from functools import update_wrapper

    try:
        import py_vollib.black
        import py_vollib.black.greeks.numerical
        import py_vollib.black.implied_volatility
        import py_vollib.black_scholes
        import py_vollib.black_scholes.greeks.numerical
        import py_vollib.black_scholes.implied_volatility
        import py_vollib.black_scholes_merton
        import py_vollib.black_scholes_merton.greeks.numerical
        import py_vollib.black_scholes_merton.implied_volatility
    except ImportError as exc:
        raise ImportError("You must have py_vollib installed to use patch_py_vollib().") from exc

    py_vollib.black.black = update_wrapper(fast_black, fast_black)
    py_vollib.black_scholes.black_scholes = update_wrapper(fast_black_scholes, fast_black_scholes)
    py_vollib.black_scholes_merton.black_scholes_merton = update_wrapper(
        fast_black_scholes_merton, fast_black_scholes_merton
    )

    py_vollib.black.implied_volatility.implied_volatility = update_wrapper(
        fast_implied_volatility_black, fast_implied_volatility_black
    )
    py_vollib.black_scholes.implied_volatility.implied_volatility = update_wrapper(
        fast_implied_volatility, fast_implied_volatility
    )
    py_vollib.black_scholes_merton.implied_volatility.implied_volatility = update_wrapper(
        fast_implied_volatility, fast_implied_volatility
    )

    py_vollib.black.greeks.numerical.delta = update_wrapper(delta, delta)
    py_vollib.black.greeks.numerical.gamma = update_wrapper(gamma, gamma)
    py_vollib.black.greeks.numerical.rho = update_wrapper(rho, rho)
    py_vollib.black.greeks.numerical.theta = update_wrapper(theta, theta)
    py_vollib.black.greeks.numerical.vega = update_wrapper(vega, vega)

    py_vollib.black_scholes.greeks.numerical.delta = update_wrapper(delta, delta)
    py_vollib.black_scholes.greeks.numerical.gamma = update_wrapper(gamma, gamma)
    py_vollib.black_scholes.greeks.numerical.rho = update_wrapper(rho, rho)
    py_vollib.black_scholes.greeks.numerical.theta = update_wrapper(theta, theta)
    py_vollib.black_scholes.greeks.numerical.vega = update_wrapper(vega, vega)

    py_vollib.black_scholes_merton.greeks.numerical.delta = update_wrapper(delta, delta)
    py_vollib.black_scholes_merton.greeks.numerical.gamma = update_wrapper(gamma, gamma)
    py_vollib.black_scholes_merton.greeks.numerical.rho = update_wrapper(rho, rho)
    py_vollib.black_scholes_merton.greeks.numerical.theta = update_wrapper(theta, theta)
    py_vollib.black_scholes_merton.greeks.numerical.vega = update_wrapper(vega, vega)
