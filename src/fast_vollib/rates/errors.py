"""Typed failures for the rates package.

The same shape every other package in the library uses: one root that a caller
can catch to mean "this library refused", and one subclass that also derives
from the built-in a caller would otherwise have written ``except`` for.

A curve refuses rather than returning a plausible number.  A negative maturity,
a mean-reversion speed of zero, a pillar out of order: each of those has an
answer that the arithmetic will happily produce and that nobody should act on.
"""

from __future__ import annotations

__all__ = ["RateError", "RateValidationError"]


class RateError(Exception):
    """Base class for every error raised by :mod:`fast_vollib.rates`."""


class RateValidationError(RateError, ValueError):
    """A curve or kernel argument is outside the domain the formula holds on.

    Also a :class:`ValueError`, so ``except ValueError`` written before this
    package existed keeps working.
    """
