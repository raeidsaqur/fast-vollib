"""Spread-consistency diagnostics: does a predicted IV price inside the quotes?

A fit error in implied-volatility units says nothing about whether the model's
price would have been *tradeable*.  These diagnostics convert the predicted
implied volatility to a forward-normalised option price and compare it to the
observed bid/ask on the same scale:

* ``price_rmse`` -- root-mean-square distance to the quote midpoint;
* ``outside_percentage`` -- share of priced quotes whose model price falls
  strictly outside ``[bid, ask]``;
* ``mean_miss_width`` -- when outside, how far outside, measured in
  spread-widths, averaged over the outside quotes alone.

The denominators are deliberately distinct.  ``price_rmse`` and
``outside_percentage`` divide by the number of quotes that were both quoted and
priced; ``mean_miss_width`` divides by the number of misses.  A dataset without
bid/ask carries **no** spread block at all -- ``None``, never a zero that reads
as perfect agreement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from .._array_api import get_namespace
from ..surface.transforms import normalized_black_call

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .._array_api import ArrayNS
    from .fit import PredictionClassification
    from .quotes import SurfaceQuotes

__all__ = ["SpreadSums", "normalized_option_price", "spread_consistency"]

#: Floor on the spread width used when expressing a miss in spread-widths.
_WIDTH_FLOOR = 1e-12


def normalized_option_price(
    iv: Any, k: Any, T: Any, is_call: Any = None, *, xp: "ArrayNS | None" = None
) -> Any:
    """Forward-normalised Black option price, ``price / (F * e^{-rT})``.

    Calls delegate to :func:`fast_vollib.surface.transforms.normalized_black_call`
    -- one kernel, shared with the surface harness -- and puts follow from
    put-call parity ``p = c - (1 - e^k)``.  Namespace-generic: numpy, torch, and
    jax inputs all stay on their own backend.

    Parameters
    ----------
    iv, k, T:
        Implied volatility, forward log-moneyness, and maturity in years,
        broadcastable to a common shape.
    is_call:
        Optional per-point call/put selector.  ``None`` prices every point as a
        call.
    xp:
        Optional array namespace; inferred from the inputs when omitted.

    Examples
    --------
    >>> import numpy as np
    >>> from fast_vollib.diagnostics import normalized_option_price
    >>> call = normalized_option_price(np.array([0.2]), np.array([0.0]), np.array([1.0]))
    >>> put = normalized_option_price(
    ...     np.array([0.2]), np.array([0.0]), np.array([1.0]), np.array([False])
    ... )
    >>> bool(np.allclose(call, put))  # at k = 0 parity is the zero adjustment
    True
    """
    xp = xp or get_namespace(iv, k, T)
    iv_a = xp.asarray(iv, like=iv)
    k_a = xp.asarray(k, like=iv)
    T_a = xp.asarray(T, like=iv)
    w = iv_a * iv_a * T_a
    call = normalized_black_call(k_a, w, xp)
    if is_call is None:
        return call
    flags = xp.asarray(is_call, like=is_call)
    parity = 1.0 - xp.exp(k_a)
    return xp.where(flags, call, call - parity)


@dataclass(frozen=True, slots=True)
class SpreadSums:
    """Additive spread-consistency sums and counts for one group of quotes."""

    #: Rows carrying a finite ``(bid, ask)`` pair, whether or not they were priced.
    eligible_quote_count: int = 0
    #: Eligible rows whose prediction was valid, so a model price exists.
    priced_quote_count: int = 0
    #: Eligible rows the model could not price (invalid prediction).
    unpriced_quote_count: int = 0
    #: Sum of squared distances between the model price and the quote midpoint.
    midpoint_squared_error_sum: float = 0.0
    #: Priced rows whose model price falls strictly outside ``[bid, ask]``.
    outside_count: int = 0
    #: Sum of miss distances in spread-widths over the outside rows.
    outside_width_sum: float = 0.0

    def merge(self, other: SpreadSums) -> SpreadSums:
        """Pooled sums of two disjoint groups.  Associative and commutative."""
        return SpreadSums(
            eligible_quote_count=self.eligible_quote_count + other.eligible_quote_count,
            priced_quote_count=self.priced_quote_count + other.priced_quote_count,
            unpriced_quote_count=self.unpriced_quote_count + other.unpriced_quote_count,
            midpoint_squared_error_sum=self.midpoint_squared_error_sum
            + other.midpoint_squared_error_sum,
            outside_count=self.outside_count + other.outside_count,
            outside_width_sum=self.outside_width_sum + other.outside_width_sum,
        )

    @property
    def price_rmse(self) -> float | None:
        """RMS distance to the quote midpoint, or ``None`` with nothing priced."""
        if self.priced_quote_count == 0:
            return None
        return float(np.sqrt(self.midpoint_squared_error_sum / self.priced_quote_count))

    @property
    def outside_percentage(self) -> float | None:
        """Percentage of priced quotes missed, or ``None`` with nothing priced."""
        if self.priced_quote_count == 0:
            return None
        return float(100.0 * self.outside_count / self.priced_quote_count)

    @property
    def mean_miss_width(self) -> float | None:
        """Mean miss in spread-widths over the misses, or ``None`` without any."""
        if self.outside_count == 0:
            return None
        return float(self.outside_width_sum / self.outside_count)


def spread_consistency(
    pred_iv: Any,
    quotes: "SurfaceQuotes",
    *,
    classification: "PredictionClassification | None" = None,
) -> SpreadSums | None:
    """Spread sums for one sample, or ``None`` when the quotes carry no bid/ask.

    Only rows that are *targets with a valid prediction* can be priced; an
    eligible quote whose prediction was invalid is counted as unpriced rather
    than silently dropped, so ``eligible = priced + unpriced`` always holds.
    """
    from .fit import classify_predictions

    if not quotes.has_spread:
        return None
    pred = np.asarray(pred_iv, dtype=np.float64)
    if pred.shape != (quotes.n,):
        raise ValueError(f"pred_iv must have shape ({quotes.n},); got {pred.shape}.")
    if classification is None:
        classification = classify_predictions(pred, quotes.iv)

    bid = np.asarray(quotes.bid, dtype=np.float64)
    ask = np.asarray(quotes.ask, dtype=np.float64)
    eligible = np.isfinite(bid) & np.isfinite(ask)
    priced = eligible & classification.valid
    if not bool(priced.any()):
        return SpreadSums(
            eligible_quote_count=int(eligible.sum()),
            priced_quote_count=0,
            unpriced_quote_count=int(eligible.sum()),
        )

    is_call = None if quotes.is_call is None else np.asarray(quotes.is_call, dtype=bool)[priced]
    price = np.asarray(
        normalized_option_price(
            pred[priced],
            np.asarray(quotes.k, dtype=np.float64)[priced],
            np.asarray(quotes.T, dtype=np.float64)[priced],
            is_call,
        ),
        dtype=np.float64,
    )
    bid_p, ask_p = bid[priced], ask[priced]
    midpoint = 0.5 * (bid_p + ask_p)
    outside = (price < bid_p) | (price > ask_p)
    miss = np.maximum(0.0, np.maximum(bid_p - price, price - ask_p))
    width = miss / np.maximum(ask_p - bid_p, _WIDTH_FLOOR)
    residual = price - midpoint
    return SpreadSums(
        eligible_quote_count=int(eligible.sum()),
        priced_quote_count=int(priced.sum()),
        unpriced_quote_count=int(eligible.sum() - priced.sum()),
        midpoint_squared_error_sum=float(np.sum(residual * residual)),
        outside_count=int(outside.sum()),
        outside_width_sum=float(np.sum(width[outside])),
    )
