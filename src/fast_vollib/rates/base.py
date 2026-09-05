"""What a discount curve has to be, and deliberately nothing more.

A curve answers one question -- what is one unit at ``maturity`` worth now --
and the protocol says so with one method.  Everything else a real curve might
carry (pillars, a parameterization, a bootstrap, a currency, a fixing source) is
the concrete type's business, not a requirement placed on every caller who wants
to discount a cashflow.

Structural, not a base class to inherit from.  An object with a compliant
:meth:`~DiscountCurve.discount_factor` works in
:func:`fast_vollib.pricing.present_value` with no registration and no import of
this module, which is what lets a caller discount against their own firm's curve
without vendoring it into the library.  The numerical claims fast-vollib makes
are about the curves it ships.

Two properties every shipped curve holds, and a user curve is expected to:

*``P(0, 0) == 1`` exactly.*  Not to within a tolerance.  A cashflow due today is
worth its face value, and a present value that quietly multiplied it by
0.9999999999999999 would be wrong in a way no test tolerance would catch.

*The namespace is the curve's, not the caller's.*  ``discount_factor`` returns a
value in whatever array namespace its own parameters live in, so a curve holding
a torch tensor returns a torch tensor with the autograd tape attached, and the
present value built from it is differentiable in the curve's parameters.

Examples
--------
>>> from fast_vollib.rates import DiscountCurve, FlatDiscountCurve
>>> isinstance(FlatDiscountCurve(rate=0.03), DiscountCurve)
True

``isinstance`` against a runtime-checkable protocol checks that the method
exists, not that it behaves; it is a convenience for error messages rather than
a verification.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

__all__ = ["DiscountCurve"]


@runtime_checkable
class DiscountCurve(Protocol):
    """The present value of one unit paid at ``maturity``."""

    def discount_factor(self, maturity: Any) -> Any:
        """``P(0, maturity)``, in the curve's own array namespace.

        Parameters
        ----------
        maturity : float or array
            A year fraction from the valuation date, non-negative. Never a
            date: calendars and day counts belong in a layer above this one.

        Returns
        -------
        The discount factor. ``P(0, 0)`` is exactly ``1``.
        """
        ...  # pragma: no cover - protocol declaration
