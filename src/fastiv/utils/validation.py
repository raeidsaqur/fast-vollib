from __future__ import annotations

import warnings

import numpy as np


def validate_numeric_inputs(*values: np.ndarray) -> None:
    for value in values:
        # not .all() avoids creating a temporary ~isfinite array (2x faster at large N)
        if not np.isfinite(value).all():
            raise ValueError("Inputs must be finite.")


def validate_data(*values: np.ndarray) -> None:
    numeric = [value for value in values if value.dtype.kind not in {"U", "S", "O"}]
    validate_numeric_inputs(*numeric)


def handle_error(message: str, on_error: str) -> None:
    if on_error == "raise":
        raise ValueError(message)
    if on_error == "warn":
        warnings.warn(message, stacklevel=3)


def ensure_on_error(value: str) -> str:
    if value not in {"raise", "warn", "ignore"}:
        raise ValueError("`on_error` must be 'raise', 'warn', or 'ignore'.")
    return value
