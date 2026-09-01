"""Shared construction-time validators for the surface value objects.

Every public container in this package -- points, market state, observations,
predictions, samples, grid specifications -- is a frozen dataclass whose
``__post_init__`` replaces each field with an *owned, read-only* array.  The
helpers here are what performs that replacement, so the ownership rule is
stated once rather than re-implemented per container.

Two properties are load-bearing and are tested per container:

*The container owns its arrays.*  Every input is copied and the copy is marked
non-writable, so a caller who keeps and later mutates the array they passed in
cannot change a number that has already been reported.

*A label survives a JSON round trip unchanged.*  Surface and point identifiers
are strings or integers, never floats and never booleans.  A float label would
come back from JSON as a different value than it went in as, and alignment
keyed on it would silently match the wrong rows.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .errors import SurfaceTypeError, SurfaceValidationError

__all__ = [
    "owned_bool_1d",
    "owned_float_1d",
    "owned_float_2d",
    "owned_labels",
    "read_only",
]


def read_only(array: np.ndarray) -> np.ndarray:
    """Mark ``array`` non-writable and return it."""
    array.setflags(write=False)
    return array


def owned_float_1d(value: Any, name: str, n: int | None = None) -> np.ndarray:
    """An owned, read-only float64 1-D copy of ``value``.

    A scalar broadcasts to length ``n`` when one is given, which is what lets a
    single maturity or a single forward stand in for a whole column without the
    caller building the repeat themselves.
    """
    array = np.array(value, dtype=np.float64, copy=True)
    if array.ndim == 0:
        if n is None:
            array = array.reshape(1)
        else:
            array = np.full(n, array.item(), dtype=np.float64)
    if array.ndim != 1:
        raise SurfaceValidationError(f"{name} must be one-dimensional; got shape {array.shape}.")
    return read_only(array)


def owned_float_2d(value: Any, name: str, shape: tuple[int, int] | None = None) -> np.ndarray:
    """An owned, read-only float64 2-D copy of ``value``.

    Used for the sample axis of :class:`~fast_vollib.surface.prediction.SurfaceSamples`
    and for the quantile block of a prediction, where the first axis indexes
    draws or levels and the second indexes points.
    """
    array = np.array(value, dtype=np.float64, copy=True)
    if array.ndim != 2:
        raise SurfaceValidationError(f"{name} must be two-dimensional; got shape {array.shape}.")
    if shape is not None and array.shape != shape:
        raise SurfaceValidationError(f"{name} must have shape {shape}; got {array.shape}.")
    return read_only(array)


def owned_bool_1d(value: Any, name: str, n: int) -> np.ndarray:
    """An owned, read-only boolean 1-D copy of ``value`` with length ``n``."""
    array = np.asarray(value)
    if array.ndim == 0:
        array = np.repeat(array, n)
    if array.ndim != 1:
        raise SurfaceValidationError(f"{name} must be one-dimensional; got shape {array.shape}.")
    if array.shape != (n,):
        raise SurfaceValidationError(f"{name} must have shape ({n},); got {array.shape}.")
    if array.dtype != np.bool_:
        raise SurfaceTypeError(f"{name} must be a boolean array; got dtype {array.dtype}.")
    return read_only(np.array(array, dtype=bool, copy=True))


def _is_json_label(value: Any) -> bool:
    """Whether ``value`` is a non-null JSON scalar usable as a label."""
    if isinstance(value, (bool, np.bool_)):
        return False
    if isinstance(value, (int, np.integer)):
        return True
    return isinstance(value, str)


def owned_labels(value: Any, n: int, name: str) -> np.ndarray:
    """An owned, read-only label array of strings and/or integers.

    Integer inputs keep an ``int64`` dtype (fast grouping); anything else is
    normalized to an object array of plain ``str`` / ``int``.  Booleans,
    floats, ``None``, and NaN are rejected -- a label must survive a JSON round
    trip unchanged.
    """
    array = np.asarray(value)
    if array.ndim == 0:
        array = np.repeat(array, n)
    if array.ndim != 1:
        raise SurfaceValidationError(f"{name} must be one-dimensional; got shape {array.shape}.")
    if array.shape != (n,):
        raise SurfaceValidationError(f"{name} must have shape ({n},); got {array.shape}.")
    if array.dtype == np.bool_:
        raise SurfaceTypeError(f"{name} must be string or integer labels; got a boolean array.")
    if np.issubdtype(array.dtype, np.integer):
        return read_only(np.array(array, dtype=np.int64, copy=True))
    if np.issubdtype(array.dtype, np.floating) or np.issubdtype(array.dtype, np.complexfloating):
        raise SurfaceTypeError(
            f"{name} must be string or integer labels; got dtype {array.dtype}. "
            "Floating-point labels do not round-trip through JSON."
        )
    values = array.tolist()
    normalized: list[Any] = []
    for item in values:
        if isinstance(item, np.generic):
            item = item.item()
        if not _is_json_label(item):
            raise SurfaceTypeError(
                f"{name} entries must be non-null strings or integers "
                f"(booleans excluded); got {item!r}."
            )
        normalized.append(int(item) if isinstance(item, (int, np.integer)) else str(item))
    if normalized and all(isinstance(item, int) for item in normalized):
        return read_only(np.array(normalized, dtype=np.int64))
    out = np.empty(n, dtype=object)
    out[:] = normalized
    return read_only(out)
