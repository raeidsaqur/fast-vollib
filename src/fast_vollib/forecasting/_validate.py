"""Host ownership and validation shared by forecast records."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np

from .._array_api import get_namespace
from .errors import ForecastError


def instant(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ForecastError("forecast instants must be timezone-aware")
    return value.astimezone(timezone.utc)


def text(value: str, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ForecastError(f"{field} must be nonempty text")
    return value


def names(values: Any, field: str) -> tuple[str, ...]:
    if isinstance(values, str):
        raise ForecastError(f"{field} must be a sequence of names")
    result = tuple(text(x, field) for x in values)
    if not result or len(set(result)) != len(result):
        raise ForecastError(f"{field} must be nonempty and unique")
    return result


def owned(value: Any, *, dtype: Any = None) -> Any:
    """Detach native buffers at the reporting boundary and seal their host copy."""
    host = get_namespace(value).to_numpy(value) if hasattr(value, "dtype") else value
    arr = np.array(host, dtype=dtype, copy=True)
    if arr.dtype.kind not in "biufUS":
        raise ForecastError("report buffers require real numeric, bool or text values")
    return np.frombuffer(arr.tobytes(), dtype=arr.dtype).reshape(arr.shape)


def values_and_valid(
    values: Any, valid: Any, ndim: int, *, positive: bool = False
) -> tuple[Any, Any]:
    array = owned(values)
    if array.ndim != ndim or array.dtype.kind != "f" or array.shape[0] < 1:
        raise ForecastError(f"values require a nonempty sample axis and {ndim} floating dimensions")
    mask_shape = array.shape[:-1]
    finite = np.isfinite(array).all(axis=-1)
    if positive:
        finite &= (array > 0).all(axis=-1)
    if valid is None:
        mask = owned(finite, dtype=bool)
    else:
        mask = owned(valid)
        if mask.dtype != np.dtype(bool) or mask.shape != mask_shape:
            raise ForecastError("validity must be boolean and match sample/time axes")
        if np.any(mask & ~finite):
            raise ForecastError("valid cells must contain finite supported values")
    return array, mask


def sample_count(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ForecastError("n_samples must be a positive integer")
    return value
