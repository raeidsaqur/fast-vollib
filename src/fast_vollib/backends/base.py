from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BackendResult:
    values: np.ndarray
    native: object | None = None
