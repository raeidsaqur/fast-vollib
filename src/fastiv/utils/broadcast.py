from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def _is_scalar_like(value: object) -> bool:
    return np.isscalar(value) or isinstance(value, str)


def to_numpy(value: object, dtype: np.dtype | type | None = None) -> np.ndarray:
    if isinstance(value, pd.Series):
        array = value.to_numpy()
    elif isinstance(value, pd.DataFrame):
        if value.shape[1] != 1:
            raise ValueError("Expected a single-column DataFrame.")
        array = value.iloc[:, 0].to_numpy()
    elif isinstance(value, np.ndarray):
        array = value
    elif _is_scalar_like(value):
        array = np.asarray([value])
    elif isinstance(value, Sequence):
        array = np.asarray(value)
    else:
        array = np.asarray(value)
    if dtype is not None:
        return array.astype(dtype, copy=False)
    return array


def preprocess_flags(flag: object) -> np.ndarray:
    arr = to_numpy(flag)
    arr = np.asarray(arr, dtype=str)
    arr = np.char.lower(arr.astype(str))
    valid = np.isin(arr, ["c", "p"])
    if not np.all(valid):
        raise ValueError("Flags must be 'c' or 'p'.")
    return arr


def maybe_format_data_and_broadcast(*values: object, dtype: np.dtype | type = np.float64) -> tuple[np.ndarray, ...]:
    prepared: list[np.ndarray] = []
    for value in values:
        if isinstance(value, np.ndarray) and value.dtype.kind in {"U", "S", "O"}:
            prepared.append(value)
        elif isinstance(value, str):
            prepared.append(np.asarray([value]))
        else:
            prepared.append(to_numpy(value, dtype=dtype))
    return tuple(np.broadcast_arrays(*prepared))
