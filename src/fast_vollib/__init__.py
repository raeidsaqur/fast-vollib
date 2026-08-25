from typing import TYPE_CHECKING

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

try:
    from .jackel.differentiable import implied_volatility_autograd
except ImportError:  # pragma: no cover - torch is an optional dependency
    implied_volatility_autograd = None  # type: ignore[assignment]
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

if TYPE_CHECKING:  # pragma: no cover - import-time cost is the point of the guard
    from . import instruments


def __getattr__(name: str):
    """Expose ``fast_vollib.instruments`` without importing it at startup.

    The instruments package is an optional API surface; charging every bare
    ``import fast_vollib`` for it would be a regression for the functional API,
    which is what most callers want. Resolving it here keeps
    ``fast_vollib.instruments`` working as an attribute while deferring the
    import to first use. ``import fast_vollib.instruments`` is unaffected.
    """
    if name == "instruments":
        import importlib

        module = importlib.import_module(".instruments", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted([*globals(), "instruments"])


__all__ = [
    "get_all_greeks",
    "instruments",
    "get_backend",
    "patch_py_vollib",
    "patch_py_vollib_vectorized",
    "price_dataframe",
    "set_backend",
    "fast_black",
    "fast_black_scholes",
    "fast_black_scholes_merton",
    "implied_volatility_autograd",
    "vectorized_delta",
    "vectorized_gamma",
    "fast_implied_volatility",
    "fast_implied_volatility_black",
    "vectorized_rho",
    "vectorized_theta",
    "vectorized_vega",
]
