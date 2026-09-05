"""Discount curves: what a cashflow at a future date is worth now.

The layer this package occupies is narrow on purpose.  A curve holds
*valuation inputs* -- a rate, a set of model parameters, a term structure --
and answers one question about them.  It is not a contract (those are in
:mod:`fast_vollib.instruments`, and hold terms only), not a process (those are
in :mod:`fast_vollib.processes`, and hold dynamics only), and not a pricer
(:func:`fast_vollib.pricing.present_value` combines a contract and a curve
without either knowing about the other).

:class:`DiscountCurve` is structural, so a caller's own curve works with no
registration.  :class:`FlatDiscountCurve` is one continuously compounded rate.
:class:`CIRDiscountCurve` binds four risk-neutral Cox-Ingersoll-Ross parameters
to the analytic term structure in :mod:`fast_vollib.rates.cir`; it holds no
process and cannot be simulated, which is what keeps the same object usable as a
discounting input and as a model description.

*The kernel is array-API native.*  It is written once against
:mod:`fast_vollib._array_api` rather than once per backend, so a bond price is a
NumPy float from Python inputs, a torch tensor carrying a gradient back to
``kappa`` from torch inputs, and a traceable value under ``jax.jit`` from JAX
inputs. That is a property of how the kernel is written, and it is tested
directly rather than advertised.

Importing this package pulls in neither torch, jax, numba, nor triton, and it
imports no other fast-vollib package except the array API. The dependency runs
one way: :mod:`fast_vollib.pricing` and :mod:`fast_vollib.simulation` reach in
here, and nothing here reaches back.

Examples
--------
>>> from fast_vollib.rates import CIRDiscountCurve, FlatDiscountCurve
>>> float(round(FlatDiscountCurve(rate=0.03).discount_factor(2.0), 12))
0.941764533584
>>> curve = CIRDiscountCurve(kappa=0.3, theta=0.04, volatility=0.1, initial_rate=0.04)
>>> float(round(curve.discount_factor(1.0), 12))
0.960840811993
"""

from __future__ import annotations

from .base import DiscountCurve
from .cir import (
    cir_affine_coefficients,
    cir_discount_factor,
    cir_instantaneous_forward_rate,
    cir_integrated_rate_coefficients,
    cir_integrated_rate_transform,
    cir_zero_rate,
)
from .curves import (
    EXTRAPOLATIONS,
    CIRDiscountCurve,
    FlatDiscountCurve,
    InterpolatedDiscountCurve,
)
from .errors import RateError, RateValidationError

__all__ = [
    "CIRDiscountCurve",
    "DiscountCurve",
    "EXTRAPOLATIONS",
    "FlatDiscountCurve",
    "InterpolatedDiscountCurve",
    "RateError",
    "RateValidationError",
    "cir_affine_coefficients",
    "cir_discount_factor",
    "cir_instantaneous_forward_rate",
    "cir_integrated_rate_coefficients",
    "cir_integrated_rate_transform",
    "cir_zero_rate",
]
