"""Opt-in shape-aware typing for fast-vollib.

Design constraints
------------------
* Zero runtime cost on the default install path. ``jaxtyping`` is only
  imported when ``TYPE_CHECKING`` is true (static checkers) or when the
  user explicitly calls :func:`enable_runtime_checks`.
* No decorators are applied anywhere in the library. Annotations are
  pure PEP-563 strings (every module uses ``from __future__ import
  annotations``), so they are never evaluated on a hot call.
* Runtime checks, when enabled, are scoped to the *public dispatch
  layer* (``api``, ``models``, ``greeks``, ``implied_volatility``) and
  the top-level backend entry points. Inner closures captured by
  ``torch.compile`` / ``@numba.njit`` / ``@jax.jit`` and Triton kernels
  are never touched.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Union

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    # Shape-aware aliases used throughout the library signatures.
    # These are only evaluated by static checkers or when the opt-in
    # runtime hook materializes annotations via ``typing.get_type_hints``.
    from jaxtyping import Bool, Float, Shaped

    Float1D = Float[np.ndarray, "n"]
    Bool1D = Bool[np.ndarray, "n"]
    FlagArray = Shaped[np.ndarray, "n"]  # '<U1' char array of 'c' / 'p'
    OptionalFloat1D = Union[Float1D, None]
else:
    # Runtime fallbacks — never exercised unless someone imports these
    # names at runtime (we don't). Kept as ``Any`` so accidental runtime
    # references remain cheap and correct.
    Float1D = Any
    Bool1D = Any
    FlagArray = Any
    OptionalFloat1D = Any


# Permissive public-API input type. The top-level functions accept
# anything that ``maybe_format_data_and_broadcast`` can coerce: scalars,
# lists, numpy arrays, pandas Series, single-column DataFrames.
ArrayLike = Union[float, int, "list[float]", np.ndarray, pd.Series, pd.DataFrame]
OptionalArrayLike = Union[ArrayLike, None]
FlagLike = Union[str, "list[str]", np.ndarray, pd.Series]


_PUBLIC_MODULES: tuple[str, ...] = (
    "fast_vollib.api",
    "fast_vollib.models",
    "fast_vollib.greeks",
    "fast_vollib.implied_volatility",
    "fast_vollib.backends.numpy_backend",
    "fast_vollib.backends.torch_backend",
    "fast_vollib.backends.jax_backend",
    "fast_vollib.backends.numba_backend",
)


def enable_runtime_checks(modules: tuple[str, ...] | None = None) -> object:
    """Install ``jaxtyping``'s import hook scoped to the public API.

    This is **opt-in**. It is never called by the library itself. Intended
    for tests and local debugging.

    The hook rewrites annotations on *already-loaded* modules only if they
    are re-imported after the hook is installed. To use it reliably, call
    before importing ``fast_vollib``::

        from fast_vollib._typing import enable_runtime_checks
        enable_runtime_checks()
        import fast_vollib  # annotations now checked

    Parameters
    ----------
    modules:
        Optional override. Defaults to the public dispatch layer. Backend
        *inner closures*, triton kernels, numba ``@njit`` factories, and
        anything under ``fast_vollib.jackel`` are deliberately excluded.
    """
    try:
        from jaxtyping import install_import_hook
    except ImportError as exc:  # pragma: no cover - optional dep
        raise RuntimeError(
            "fast_vollib._typing.enable_runtime_checks() requires the "
            "'typecheck' extra: `pip install fast-vollib[typecheck]`."
        ) from exc

    target = modules if modules is not None else _PUBLIC_MODULES
    # ``beartype`` is the shape-checker backend jaxtyping drives.
    return install_import_hook(target, "beartype.beartype")


__all__ = [
    "ArrayLike",
    "Bool1D",
    "FlagArray",
    "FlagLike",
    "Float1D",
    "OptionalArrayLike",
    "OptionalFloat1D",
    "enable_runtime_checks",
]
