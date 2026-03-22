from __future__ import annotations

import numpy as np

from . import numpy_backend


def is_available() -> bool:
    try:
        import jax  # noqa: F401
    except ImportError:
        return False
    return True


def to_native(values: np.ndarray):
    import jax.numpy as jnp

    return jnp.asarray(values)


def from_native(values) -> np.ndarray:
    return np.asarray(values)


price_black = numpy_backend.price_black
price_black_scholes = numpy_backend.price_black_scholes
price_black_scholes_merton = numpy_backend.price_black_scholes_merton
implied_volatility = numpy_backend.implied_volatility
greeks = numpy_backend.greeks
