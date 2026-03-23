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
    # Ensure single-character '<U1' array (4 bytes per element in UCS-4)
    arr = np.asarray(arr, dtype="<U1")
    # Fast vectorized lowercase via ASCII bit trick: OR 0x20 sets lowercase bit
    # for A-Z ('C'=0x43→'c'=0x63, 'P'=0x50→'p'=0x70); idempotent for a-z.
    # View as uint32 to operate on character code points directly.
    arr_lower = (arr.view(np.uint32) | np.uint32(0x20)).view("<U1")
    valid = (arr_lower == "c") | (arr_lower == "p")
    if not np.all(valid):
        raise ValueError("Flags must be 'c' or 'p'.")
    return arr_lower


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
