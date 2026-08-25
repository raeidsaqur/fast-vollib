"""Valuation-time observations, kept separate from contract terms.

A contract is what was agreed; market inputs are what is true now.  Storing the
second on the first is the mistake this split exists to prevent: an option
object that carries a spot price is stale the moment it is constructed,
compares unequal to an identical option observed a second later, and cannot be
serialized without smuggling a market snapshot into the record.

So :class:`VanillaMarketInputs` is passed alongside a contract, never attached
to one.  It is frozen, and it holds arrays: unlike contract terms, market
observations are genuinely vectors, and they may be torch or jax arrays when
the caller wants a native route.

``underlying`` is deliberately neutrally named.  Under Black-76 it is a
forward; under Black-Scholes and Black-Scholes-Merton it is a spot.  The
explicit ``model`` argument at the call site fixes which -- the market object
does not guess, and neither does the asset class.

Negative rates and negative dividend yields are valid.  Nothing here rejects
them.

Examples
--------
>>> from fast_vollib.instruments import VanillaMarketInputs
>>> market = VanillaMarketInputs(underlying=100.0, rate=-0.005, volatility=0.2)
>>> market.rate
-0.005
>>> market.dividend_yield is None
True
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .errors import MissingMarketInputError

if TYPE_CHECKING:
    from .._typing import ArrayLike, OptionalArrayLike

__all__ = ["VanillaMarketInputs"]


@dataclass(frozen=True, slots=True, kw_only=True)
class VanillaMarketInputs:
    """Market observations for pricing, Greeks, or implied volatility.

    Parameters
    ----------
    underlying : array-like
        The forward under Black-76; the spot under Black-Scholes and
        Black-Scholes-Merton.  The requested model fixes the meaning.
    rate : array-like
        Continuously compounded risk-free rate.  May be negative.
    volatility : array-like, optional
        Required to price or to compute Greeks; not used for inversion.
    dividend_yield : array-like, optional
        Continuous dividend yield.  Required by Black-Scholes-Merton and
        ignored by the other two models.  May be negative.
    price : array-like, optional
        Observed option prices, required to invert implied volatility.
        Interpreted as the price of the whole position: it is divided by the
        contract's notional before inversion.

    Notes
    -----
    Fields accept scalars or array-like values and follow the broadcasting
    rules of the selected pricing backend.
    """

    underlying: ArrayLike
    rate: ArrayLike
    volatility: OptionalArrayLike = None
    dividend_yield: OptionalArrayLike = None
    price: OptionalArrayLike = None

    def require(self, field: str, *, operation: str) -> Any:
        """Return a market field, or say precisely which one is missing.

        Parameters
        ----------
        field : str
            One of the field names of this class.
        operation : str
            What the caller was trying to do; quoted back in the message so the
            error explains why the field was needed.

        Raises
        ------
        MissingMarketInputError
            If the field is ``None``.

        Examples
        --------
        >>> from fast_vollib.instruments import MissingMarketInputError, VanillaMarketInputs
        >>> market = VanillaMarketInputs(underlying=100.0, rate=0.02)
        >>> try:
        ...     market.require("volatility", operation="price")
        ... except MissingMarketInputError as error:
        ...     print(error)
        VanillaMarketInputs.volatility is required to price, but was not supplied.
        """
        value = getattr(self, field)
        if value is None:
            raise MissingMarketInputError(
                f"VanillaMarketInputs.{field} is required to {operation}, but was not supplied."
            )
        return value
