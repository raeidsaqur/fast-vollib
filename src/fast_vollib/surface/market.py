"""The market state a surface is quoted against: forwards, discounting, carry.

An implied volatility is a number in a coordinate system, and the coordinate
system is a market.  ``k = log(K / F(T))`` cannot be computed without a forward
curve; a price-space error cannot be computed without a discount factor; a
vega weight cannot be computed without both.  :class:`SurfaceMarket` is that
state as a value: term-structured, immutable, and carrying its own provenance.

*The market is never inferred.*  Nothing in this package substitutes ``F = 1``
or ``r = 0`` when a market is missing.  A computation that needs one and is not
given one raises
:class:`~fast_vollib.surface.errors.MissingMarketStateError` instead, because a
price computed against an invented forward looks exactly like a price computed
against the real one and is not comparable to anything.

Interpolation between pillars is a modelling choice, so it is a field rather
than a hidden convention: ``interpolation='log_linear'`` is linear in
:math:`\\log F` against :math:`T` (equivalently, a piecewise-constant
instantaneous growth rate), and ``'exact'`` refuses any maturity that is not a
pillar.  Extrapolation is the same: ``'flat'`` holds the end pillar, ``'error'``
declines.

Examples
--------
>>> from fast_vollib.surface import SurfaceMarket
>>> market = SurfaceMarket.from_spot(spot=100.0, T=[0.25, 1.0], rate=0.03, carry=0.01)
>>> float(round(market.forward_at(1.0), 8))
102.020134
>>> float(round(market.discount_at(1.0), 8))
0.97044553
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ._validate import owned_float_1d
from .errors import SurfaceValidationError

__all__ = ["EXTRAPOLATIONS", "INTERPOLATIONS", "SurfaceMarket"]

#: Accepted term-structure interpolation policies.
INTERPOLATIONS = ("log_linear", "exact")

#: Accepted term-structure extrapolation policies.
EXTRAPOLATIONS = ("flat", "error")


@dataclass(frozen=True, slots=True)
class SurfaceMarket:
    """A forward, discount, and carry term structure on maturity pillars.

    Parameters
    ----------
    T:
        Maturity pillars in years, shape ``(M,)``, strictly increasing and
        strictly positive.
    forward:
        Forward level at each pillar, shape ``(M,)`` or a scalar, strictly
        positive.
    rate:
        Continuously compounded discount rate at each pillar, shape ``(M,)`` or
        a scalar.  Used only for discounting; it does not re-derive the
        forward, which is supplied directly.
    carry:
        Continuously compounded carry (dividend / convenience) yield at each
        pillar.  Recorded for adapters that need it; the forward is authoritative.
    interpolation:
        ``'log_linear'`` (default) or ``'exact'``.
    extrapolation:
        ``'flat'`` (default) or ``'error'``.
    source:
        Free-text provenance -- which curve, as of when, from where.

    Raises
    ------
    SurfaceValidationError
        If pillars are not strictly increasing, a forward is not positive, or a
        policy name is unknown.

    Notes
    -----
    The forward is stored rather than derived from ``spot``, ``rate``, and
    ``carry``, because a real forward curve is quoted, not computed: implying
    it from a rate and a dividend yield silently asserts a model of the
    dividend stream.  :meth:`from_spot` is available for the synthetic case
    where that model *is* the intent.
    """

    T: Any
    forward: Any
    rate: Any = 0.0
    carry: Any = 0.0
    interpolation: str = "log_linear"
    extrapolation: str = "flat"
    source: str | None = None

    def __post_init__(self) -> None:
        T = owned_float_1d(self.T, "T")
        m = T.size
        if m == 0:
            raise SurfaceValidationError("A market needs at least one maturity pillar.")
        if not bool(np.all(np.isfinite(T))) or not bool(np.all(T > 0.0)):
            raise SurfaceValidationError("Market pillars must be finite and strictly positive.")
        if m > 1 and not bool(np.all(np.diff(T) > 0.0)):
            raise SurfaceValidationError(
                f"Market pillars must be strictly increasing; got {T.tolist()}."
            )
        forward = owned_float_1d(self.forward, "forward", m)
        rate = owned_float_1d(self.rate, "rate", m)
        carry = owned_float_1d(self.carry, "carry", m)
        for name, array in (("forward", forward), ("rate", rate), ("carry", carry)):
            if array.size != m:
                raise SurfaceValidationError(
                    f"{name} must have one value per pillar ({m}); got {array.size}."
                )
            if not bool(np.all(np.isfinite(array))):
                raise SurfaceValidationError(f"{name} must be finite everywhere.")
        if not bool(np.all(forward > 0.0)):
            raise SurfaceValidationError("forward must be strictly positive at every pillar.")
        if self.interpolation not in INTERPOLATIONS:
            raise SurfaceValidationError(
                f"interpolation must be one of {INTERPOLATIONS}; got {self.interpolation!r}."
            )
        if self.extrapolation not in EXTRAPOLATIONS:
            raise SurfaceValidationError(
                f"extrapolation must be one of {EXTRAPOLATIONS}; got {self.extrapolation!r}."
            )

        object.__setattr__(self, "T", T)
        object.__setattr__(self, "forward", forward)
        object.__setattr__(self, "rate", rate)
        object.__setattr__(self, "carry", carry)

    # -- constructors --------------------------------------------------------
    @classmethod
    def flat(
        cls,
        *,
        forward: float,
        rate: float = 0.0,
        carry: float = 0.0,
        source: str | None = None,
    ) -> SurfaceMarket:
        """A market whose forward, rate, and carry do not vary with maturity.

        The single pillar is nominal; with ``extrapolation='flat'`` the curve is
        constant everywhere, which is the intended meaning.
        """
        return cls(T=[1.0], forward=forward, rate=rate, carry=carry, source=source)

    @classmethod
    def from_spot(
        cls,
        *,
        spot: float,
        T: Any,
        rate: Any = 0.0,
        carry: Any = 0.0,
        source: str | None = None,
    ) -> SurfaceMarket:
        """A market whose forward is ``F(T) = S e^{(r - q) T}``.

        This is a model of the forward, not an observation of it, and is what a
        synthetic benchmark wants.  ``source`` should say so.
        """
        pillars = np.atleast_1d(np.asarray(T, dtype=np.float64))
        rates = np.broadcast_to(np.atleast_1d(np.asarray(rate, dtype=np.float64)), pillars.shape)
        carries = np.broadcast_to(np.atleast_1d(np.asarray(carry, dtype=np.float64)), pillars.shape)
        spot_value = float(spot)
        if not np.isfinite(spot_value) or spot_value <= 0.0:
            raise SurfaceValidationError(
                f"spot must be finite and strictly positive; got {spot!r}."
            )
        return cls(
            T=pillars,
            forward=spot_value * np.exp((rates - carries) * pillars),
            rate=rates,
            carry=carries,
            source=source,
        )

    # -- term-structure lookup ----------------------------------------------
    @property
    def n_pillars(self) -> int:
        """Number of maturity pillars."""
        return int(self.T.size)

    def forward_at(self, T: Any) -> Any:
        """The forward at ``T``, interpolated by the declared policy."""
        return np.exp(self._interpolate(np.log(self.forward), T, "forward"))

    def rate_at(self, T: Any) -> Any:
        """The discount rate at ``T``, interpolated by the declared policy."""
        return self._interpolate(self.rate, T, "rate")

    def carry_at(self, T: Any) -> Any:
        """The carry yield at ``T``, interpolated by the declared policy."""
        return self._interpolate(self.carry, T, "carry")

    def discount_at(self, T: Any) -> Any:
        """The discount factor ``e^{-r(T) T}`` at ``T``."""
        maturity = np.asarray(T, dtype=np.float64)
        return np.exp(-self.rate_at(maturity) * maturity)

    def strikes_at(self, k: Any, T: Any) -> Any:
        """``K = F(T) e^k``, the strike a canonical coordinate names."""
        return self.forward_at(T) * np.exp(np.asarray(k, dtype=np.float64))

    def to_dict(self) -> dict[str, Any]:
        """The market as a JSON-safe mapping."""
        return {
            "T": [float(value) for value in self.T],
            "forward": [float(value) for value in self.forward],
            "rate": [float(value) for value in self.rate],
            "carry": [float(value) for value in self.carry],
            "interpolation": self.interpolation,
            "extrapolation": self.extrapolation,
            "source": self.source,
        }

    # -- internals -----------------------------------------------------------
    def _interpolate(self, values: np.ndarray, T: Any, name: str) -> Any:
        query = np.asarray(T, dtype=np.float64)
        if not bool(np.all(np.isfinite(query))):
            raise SurfaceValidationError(f"Cannot read {name} at a non-finite maturity.")
        pillars = self.T
        if self.extrapolation == "error":
            outside = (query < pillars[0]) | (query > pillars[-1])
            if bool(np.any(outside)):
                raise SurfaceValidationError(
                    f"Maturity outside the market's pillars [{pillars[0]}, {pillars[-1]}] "
                    f"and extrapolation='error'."
                )
        if self.interpolation == "exact":
            missing = ~np.isin(query, pillars)
            if bool(np.any(missing)):
                raise SurfaceValidationError(
                    f"interpolation='exact' requires every maturity to be a pillar; "
                    f"{int(np.sum(missing))} of {query.size} are not."
                )
        if pillars.size == 1:
            return np.broadcast_to(values[0], query.shape) + np.zeros_like(query)
        return np.interp(query, pillars, values)
