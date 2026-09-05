"""Domain checks for curve parameters, written to survive a JAX trace.

The same two-halves rule the process validators follow: static properties --
bool, complex, rank, dtype -- are knowable while tracing and are always checked;
a *value* is not, so a value check runs only when
:func:`fast_vollib._array_api.concrete_float` can read the number eagerly.  Code
that called ``float(x)`` unconditionally would break ``jax.grad`` on a curve that
was otherwise fine.

The object handed in is returned unchanged, never replaced by its float, so a
``requires_grad`` tensor an optimizer is stepping stays that tensor.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .._array_api import concrete_float
from .errors import RateValidationError

__all__ = ["ensure_scalar_parameter"]

_REAL_SCALARS = (int, float, np.integer, np.floating)


def _is_native(value: Any) -> bool:
    """Whether ``value`` is a torch tensor or a JAX array.

    Spelled here rather than imported from :mod:`fast_vollib._random_api`: this
    package depends on the array API and nothing else, so a curve can be built
    in a process that has no simulation machinery loaded at all.
    """
    root = type(value).__module__.partition(".")[0]
    return root in {"torch", "jax", "jaxlib"}


def ensure_scalar_parameter(
    value: Any,
    *,
    field: str,
    positive: bool = False,
    non_negative: bool = False,
) -> Any:
    """Check one curve parameter and hand back the caller's own object.

    Parameters
    ----------
    value : object
        A real Python number, a NumPy scalar, or a zero-dimensional native
        array. Real integer values are accepted as well as floating-point
        values; the original object is returned without conversion.
    field : str
        Used in the message, so a failure names the parameter.
    positive, non_negative : bool
        At most one is meaningful; ``positive`` implies ``non_negative``.

    Raises
    ------
    RateValidationError
        If the value is a bool, complex, non-scalar, non-finite,
        or outside the requested sign domain.
    """
    if isinstance(value, (bool, np.bool_)):
        raise RateValidationError(f"{field} must be a real number, not a bool.")
    if isinstance(value, (complex, np.complexfloating)):
        raise RateValidationError(f"{field} must be a real number, not complex.")

    if _is_native(value) or isinstance(value, np.ndarray):
        ndim = getattr(value, "ndim", None)
        if ndim is None or ndim != 0:
            raise RateValidationError(
                f"{field} must be a scalar; got an array with shape "
                f"{getattr(value, 'shape', '?')}. A curve holds one value per name, "
                f"not a term structure."
            )
    elif not isinstance(value, _REAL_SCALARS):
        raise RateValidationError(
            f"{field} must be a real number or a scalar array; got {type(value).__name__}."
        )

    as_float = concrete_float(value)
    if as_float is not None:
        if as_float != as_float or as_float in (float("inf"), float("-inf")):
            raise RateValidationError(f"{field} must be finite; got {as_float!r}.")
        if positive and as_float <= 0.0:
            raise RateValidationError(f"{field} must be strictly positive; got {as_float!r}.")
        if non_negative and as_float < 0.0:
            raise RateValidationError(f"{field} must be non-negative; got {as_float!r}.")
    return value
