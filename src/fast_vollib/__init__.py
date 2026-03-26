from .api import get_all_greeks, price_dataframe
from .compat.py_vollib_vectorized import patch_py_vollib, patch_py_vollib_vectorized
from .config import get_backend, set_backend
from .greeks import (
    delta as vectorized_delta,
    gamma as vectorized_gamma,
    rho as vectorized_rho,
    theta as vectorized_theta,
    vega as vectorized_vega,
)
from .implied_volatility import (
    fast_implied_volatility,
    fast_implied_volatility_black,
)
from .models import (
    fast_black,
    fast_black_scholes,
    fast_black_scholes_merton,
)

__version__ = "0.1.1"

__all__ = [
    "get_all_greeks",
    "get_backend",
    "patch_py_vollib",
    "patch_py_vollib_vectorized",
    "price_dataframe",
    "set_backend",
    "fast_black",
    "fast_black_scholes",
    "fast_black_scholes_merton",
    "vectorized_delta",
    "vectorized_gamma",
    "fast_implied_volatility",
    "fast_implied_volatility_black",
    "vectorized_rho",
    "vectorized_theta",
    "vectorized_vega",
]
