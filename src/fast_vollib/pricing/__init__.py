"""Model-specific pricers that do not fit the vectorized backend dispatch.

:mod:`fast_vollib.models` holds the Black family: closed-form, elementwise, and
implemented once per backend so a torch tensor is priced on the device it lives
on.  This package holds the pricers that cannot be written that way -- ones whose
price is a numerical integral of a characteristic function, evaluated on the
host in float64 with a deterministic quadrature.

The split is deliberate rather than historical.  Putting a quadrature-based
pricer behind the same dispatcher would advertise a torch backend that stages
through host memory and drops the autograd tape, which is precisely the silent
substitution the instruments layer's capability metadata exists to prevent.  A
pricer here declares host-only support and means it.

Importing this package pulls in numpy only.
"""

from __future__ import annotations

from .heston import (
    DEFAULT_QUADRATURE_NODES,
    FORMULATIONS,
    heston_call_price,
    heston_characteristic_function,
    heston_price,
)

__all__ = [
    "DEFAULT_QUADRATURE_NODES",
    "FORMULATIONS",
    "heston_call_price",
    "heston_characteristic_function",
    "heston_price",
]
