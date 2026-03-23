from __future__ import annotations

from functools import update_wrapper

from ..greeks import delta, gamma, rho, theta, vega
from ..implied_volatility import vectorized_implied_volatility, vectorized_implied_volatility_black
from ..models import vectorized_black, vectorized_black_scholes, vectorized_black_scholes_merton


def patch_py_vollib() -> None:
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

    py_vollib.black.black = update_wrapper(vectorized_black, vectorized_black)
    py_vollib.black_scholes.black_scholes = update_wrapper(vectorized_black_scholes, vectorized_black_scholes)
    py_vollib.black_scholes_merton.black_scholes_merton = update_wrapper(vectorized_black_scholes_merton, vectorized_black_scholes_merton)

    py_vollib.black.implied_volatility.implied_volatility = update_wrapper(vectorized_implied_volatility_black, vectorized_implied_volatility_black)
    py_vollib.black_scholes.implied_volatility.implied_volatility = update_wrapper(vectorized_implied_volatility, vectorized_implied_volatility)
    py_vollib.black_scholes_merton.implied_volatility.implied_volatility = update_wrapper(vectorized_implied_volatility, vectorized_implied_volatility)

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
