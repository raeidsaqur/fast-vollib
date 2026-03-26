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

try:
    from ._version import __version__
except ImportError:  # pragma: no cover - fallback for source trees without build hooks
    try:
        from importlib.metadata import version as _pkg_version
    except ImportError:  # pragma: no cover - Python < 3.8
        from importlib_metadata import version as _pkg_version  # type: ignore[no-redef]

    try:
        __version__ = _pkg_version("fast-vollib")
    except Exception:  # pragma: no cover - package metadata unavailable
        __version__ = "0.0.0"

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
