"""Pricers that do not fit the vectorized backend dispatch.

:mod:`fast_vollib.models` holds the Black family: closed-form, elementwise, and
implemented once per backend so a torch tensor is priced on the device it lives
on.  This package holds what cannot be written that way, for two different
reasons.

**The Fourier modules are host-side float64, and mean it.**  ``heston`` -- and
the jump and stochastic-rate pricers that will join it -- compute a price as a
numerical integral of a characteristic function on a deterministic
Gauss-Legendre node set.  Putting that behind the same dispatcher would
advertise a torch backend that stages through host memory and drops the autograd
tape, which is exactly the silent substitution the instruments layer's
capability metadata exists to prevent.  A pricer in those modules declares
host-only support and means it.

**``fixed_income`` is array-API native, and that is not an exception to the
rule so much as a different case.**  A present value is a weighted sum of
discount factors; there is no quadrature and no complex arithmetic, so it can be
written once against :mod:`fast_vollib._array_api` and evaluate in whatever
namespace the curve hands back.  A curve holding a ``requires_grad`` tensor
therefore produces a present value that differentiates back to the curve's
parameters.  The host-only rule above is scoped to the Fourier modules; it was
never a statement about the package as a whole, and this module is why the
distinction is now written down.

Importing this package pulls in NumPy only.  SciPy is reached from inside
``_gauss_legendre`` rather than at module level, and
:mod:`fast_vollib.rates` is imported lazily by ``fixed_income`` so that a caller
with a duck-typed curve of their own never loads it.
"""

from __future__ import annotations

from .bates import (
    BATES_QUADRATURE_NODES,
    bates_characteristic_function,
    bates_price,
)
from .bcc97 import (
    BCC97_QUADRATURE_NODES,
    bcc97_characteristic_function,
    bcc97_discounted_transform,
    bcc97_forward_measure,
    bcc97_price,
)
from .fixed_income import present_value
from .heston import (
    DEFAULT_QUADRATURE_NODES,
    FORMULATIONS,
    heston_call_price,
    heston_characteristic_function,
    heston_price,
)

__all__ = [
    "BATES_QUADRATURE_NODES",
    "BCC97_QUADRATURE_NODES",
    "DEFAULT_QUADRATURE_NODES",
    "FORMULATIONS",
    "bates_characteristic_function",
    "bates_price",
    "bcc97_characteristic_function",
    "bcc97_discounted_transform",
    "bcc97_forward_measure",
    "bcc97_price",
    "heston_call_price",
    "heston_characteristic_function",
    "heston_price",
    "present_value",
]
