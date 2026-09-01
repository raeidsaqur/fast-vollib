"""Where a surface is asked a question: coordinates, identity, and provenance.

:class:`SurfacePoints` is the query domain shared by every model family in this
package.  A calibrator is handed observations at points, a definite surface is
evaluated at points, a distribution is sampled at points, and an evaluation
compares a prediction to an observation *point by point*.  Keeping that one
object means a forecaster and a generative model can be scored on exactly the
same rows without either of them agreeing on a file format first.

The canonical coordinates are forward log-moneyness :math:`k = \\log(K / F(T))`
and maturity :math:`T` in year fractions.  They are canonical because the
no-arbitrage conditions are stated in them: total variance :math:`w = \\sigma^2 T`
is the natural dependent variable, and :math:`k` is the variable Durrleman's
condition differentiates against.  Strike, spot-moneyness, delta, and
day-count-derived maturities are all *inputs*, converted by an explicit adapter
that records what it used.

*A conversion that is not recorded is not reproducible.*  Every adapter attaches
a :class:`CoordinateConvention` naming the source coordinate, the market state
it consumed, and the maturity convention, so a surface loaded a year later
still says which forward curve turned its strikes into moneyness.  Points built
directly in canonical coordinates carry the identity convention rather than
nothing at all.

Examples
--------
>>> import numpy as np
>>> from fast_vollib.surface import SurfacePoints
>>> points = SurfacePoints(k=[-0.1, 0.0, 0.1], T=[0.5, 0.5, 0.5])
>>> points.n, points.surface_ids()
(3, [0])
>>> points.k.flags.writeable
False
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from ._validate import owned_float_1d, owned_labels, read_only
from .errors import SurfaceValidationError

if TYPE_CHECKING:  # pragma: no cover - typing only, keeps the market import acyclic
    from .market import SurfaceMarket

__all__ = [
    "CoordinateConvention",
    "SurfacePoints",
    "points_from_forward_delta",
    "points_from_spot_moneyness",
    "points_from_strikes",
]

#: The convention attached to points built directly in canonical coordinates.
#: Naming it, rather than leaving the field empty, keeps "no conversion was
#: performed" distinguishable from "nobody recorded what the conversion was".
CANONICAL_CONVENTION_SOURCE = "forward_log_moneyness"


@dataclass(frozen=True, slots=True)
class CoordinateConvention:
    """How a set of points came to be in canonical coordinates.

    Attributes
    ----------
    source : str
        The coordinate the caller supplied: ``'forward_log_moneyness'``,
        ``'strike'``, ``'spot_moneyness'``, ``'forward_delta'``.
    maturity : str
        How ``T`` was expressed: ``'year_fraction'`` when it was already a year
        fraction, otherwise the day count that produced one (``'act/365f'``).
    market_source : str | None
        Free-text provenance of the market state consumed by the conversion --
        the forward curve, spot, and carry that turned strikes into moneyness.
        ``None`` when the conversion needed no market state.
    notes : str | None
        Anything else a reader needs in order to reproduce the conversion.
    """

    source: str = CANONICAL_CONVENTION_SOURCE
    maturity: str = "year_fraction"
    market_source: str | None = None
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """The convention as a JSON-safe mapping."""
        return {
            "source": self.source,
            "maturity": self.maturity,
            "market_source": self.market_source,
            "notes": self.notes,
        }


@dataclass(frozen=True, slots=True)
class SurfacePoints:
    """Coordinates a surface may be asked about, with stable identity.

    Parameters
    ----------
    k, T:
        Forward log-moneyness and maturity in years, each shape ``(N,)``.
        Every ``k`` is finite and every ``T`` is finite and strictly positive.
    surface_id:
        Shape ``(N,)`` labels -- strings or integers -- grouping points into
        surfaces (a day, an asset-day, a sample id).  A scalar labels every
        row; ``None`` means a single surface labelled ``0``.
    point_id:
        Optional shape ``(N,)`` stable per-point labels, unique within each
        surface.  They are what alignment keys on by default.
    convention:
        How the coordinates were produced.  Defaults to the canonical identity
        convention.

    Notes
    -----
    Every stored array is an owned, read-only copy.  Duplicate coordinates are
    legitimate -- two exchanges quoting the same strike and expiry are two
    observations, not one -- and are preserved as separate rows, which is why
    ``point_id`` rather than ``(k, T)`` is the identity of a row.

    Examples
    --------
    >>> import numpy as np
    >>> from fast_vollib.surface import SurfacePoints
    >>> points = SurfacePoints(
    ...     k=[-0.05, 0.05], T=[0.25, 0.25], surface_id="2026-08-31", point_id=[1, 2]
    ... )
    >>> points.n, points.has_point_ids
    (2, True)
    """

    k: Any
    T: Any
    surface_id: Any = None
    point_id: Any = None
    convention: CoordinateConvention = CoordinateConvention()

    def __post_init__(self) -> None:
        k = owned_float_1d(self.k, "k")
        n = k.size
        T = owned_float_1d(self.T, "T", n)
        if T.size != n:
            raise SurfaceValidationError(
                f"k and T must have the same length; got {n} and {T.size}."
            )
        if not bool(np.all(np.isfinite(k))):
            raise SurfaceValidationError("k must be finite everywhere.")
        if not bool(np.all(np.isfinite(T))):
            raise SurfaceValidationError("T must be finite everywhere.")
        if n and not bool(np.all(T > 0.0)):
            raise SurfaceValidationError("T must be strictly positive everywhere.")

        if self.surface_id is None:
            surface_id = read_only(np.zeros(n, dtype=np.int64))
        else:
            surface_id = owned_labels(self.surface_id, n, "surface_id")

        point_id = None
        if self.point_id is not None:
            point_id = owned_labels(self.point_id, n, "point_id")
            if _has_duplicate_pairs(surface_id, point_id):
                raise SurfaceValidationError("point_id must be unique within each surface_id.")

        if not isinstance(self.convention, CoordinateConvention):
            raise SurfaceValidationError(
                f"convention must be a CoordinateConvention; got {type(self.convention).__name__}."
            )

        object.__setattr__(self, "k", k)
        object.__setattr__(self, "T", T)
        object.__setattr__(self, "surface_id", surface_id)
        object.__setattr__(self, "point_id", point_id)

    # -- size and structure --------------------------------------------------
    @property
    def n(self) -> int:
        """Number of points."""
        return int(self.k.size)

    def __len__(self) -> int:
        return self.n

    @property
    def has_point_ids(self) -> bool:
        """Whether stable per-point labels are carried."""
        return self.point_id is not None

    def surface_ids(self) -> list[Any]:
        """Distinct surface labels in order of first appearance, as plain scalars."""
        return list(dict.fromkeys(self.surface_id.tolist()))

    def subset(self, index: Any) -> SurfacePoints:
        """The points selected by a boolean mask or an index array."""
        index = np.asarray(index)
        return SurfacePoints(
            k=self.k[index],
            T=self.T[index],
            surface_id=self.surface_id[index],
            point_id=None if self.point_id is None else self.point_id[index],
            convention=self.convention,
        )

    def surfaces(self) -> Iterator[tuple[Any, SurfacePoints]]:
        """Yield ``(label, points)`` per surface, in order of first appearance."""
        for label in self.surface_ids():
            yield label, self.subset(self.surface_id == label)

    def maturities(self, *, decimals: int = 6) -> np.ndarray:
        """Distinct maturities, ascending, bucketed on ``round(T, decimals)``."""
        return np.unique(np.round(self.T, decimals))

    def strikes(self, market: "SurfaceMarket") -> np.ndarray:
        """``K = F(T) e^k`` under ``market``, shape ``(N,)``."""
        return market.forward_at(self.T) * np.exp(self.k)

    def matches_domain(self, other: SurfacePoints) -> bool:
        """Whether ``other`` asks about the same coordinates and point labels.

        Ignores ``surface_id`` alone.  A draw from a generative model is a
        different surface than the grid that defined its domain, and requiring
        the labels to agree would make it impossible to say that the two
        describe the same nodes -- which is the only thing this check is for.
        """
        if self.n != other.n or self.has_point_ids != other.has_point_ids:
            return False
        same = bool(np.array_equal(self.k, other.k)) and bool(np.array_equal(self.T, other.T))
        if self.point_id is not None:
            same = same and bool(np.array_equal(self.point_id, other.point_id))
        return same

    def equals(self, other: SurfacePoints) -> bool:
        """Whether ``other`` carries the same coordinates and identity, row for row.

        Compared exactly.  Two point sets that agree only to floating-point
        tolerance describe different queries, and treating them as one is the
        implicit alignment this package refuses to perform.
        """
        if self.n != other.n:
            return False
        if self.has_point_ids != other.has_point_ids:
            return False
        same = bool(np.array_equal(self.k, other.k)) and bool(np.array_equal(self.T, other.T))
        same = same and bool(np.array_equal(self.surface_id, other.surface_id))
        if self.point_id is not None:
            same = same and bool(np.array_equal(self.point_id, other.point_id))
        return same


def _has_duplicate_pairs(left: np.ndarray, right: np.ndarray) -> bool:
    """Whether any ``(left, right)`` pair repeats."""
    if left.size == 0:
        return False
    pairs = list(zip(left.tolist(), right.tolist()))
    return len(set(pairs)) != len(pairs)


# --- coordinate adapters -----------------------------------------------------


def points_from_strikes(
    K: Any,
    T: Any,
    *,
    forward: Any,
    surface_id: Any = None,
    point_id: Any = None,
    maturity: str = "year_fraction",
    market_source: str | None = None,
) -> SurfacePoints:
    """Points from strikes and a forward: ``k = log(K / F)``.

    Parameters
    ----------
    K, T:
        Strikes and maturities in years, each shape ``(N,)`` or broadcastable
        to it.  Strikes are finite and strictly positive.
    forward:
        The forward at each point, scalar or shape ``(N,)``.  A term structure
        is supplied per point rather than per pillar so the caller decides how
        it was interpolated instead of this function guessing.
    market_source:
        Provenance of the forward curve, recorded on the returned convention.

    Examples
    --------
    >>> from fast_vollib.surface import points_from_strikes
    >>> points = points_from_strikes([90.0, 100.0], [1.0, 1.0], forward=100.0)
    >>> float(round(points.k[1], 12))
    0.0
    """
    strikes = np.array(K, dtype=np.float64, copy=True)
    forwards = np.array(forward, dtype=np.float64, copy=True)
    strikes, forwards = np.broadcast_arrays(np.atleast_1d(strikes), np.atleast_1d(forwards))
    if strikes.ndim != 1:
        raise SurfaceValidationError(
            f"K and forward must broadcast to 1-D; got shape {strikes.shape}."
        )
    if not bool(np.all(np.isfinite(strikes))) or not bool(np.all(strikes > 0.0)):
        raise SurfaceValidationError("K must be finite and strictly positive.")
    if not bool(np.all(np.isfinite(forwards))) or not bool(np.all(forwards > 0.0)):
        raise SurfaceValidationError("forward must be finite and strictly positive.")
    return SurfacePoints(
        k=np.log(strikes / forwards),
        T=T,
        surface_id=surface_id,
        point_id=point_id,
        convention=CoordinateConvention(
            source="strike", maturity=maturity, market_source=market_source
        ),
    )


def points_from_spot_moneyness(
    m: Any,
    T: Any,
    *,
    forward: Any,
    spot: Any,
    surface_id: Any = None,
    point_id: Any = None,
    maturity: str = "year_fraction",
    market_source: str | None = None,
) -> SurfacePoints:
    """Points from spot log-moneyness ``m = log(K / S)``: ``k = m + log(S / F)``.

    The two moneyness measures differ by the carry over the life of the option,
    which is small at a week and is not small at two years.  Converting
    requires both spot and the forward, and refusing to accept one without the
    other is the point of this adapter.
    """
    spot_arr = np.array(spot, dtype=np.float64, copy=True)
    forwards = np.array(forward, dtype=np.float64, copy=True)
    moneyness = np.atleast_1d(np.array(m, dtype=np.float64, copy=True))
    moneyness, forwards, spot_arr = np.broadcast_arrays(
        moneyness, np.atleast_1d(forwards), np.atleast_1d(spot_arr)
    )
    if not bool(np.all(np.isfinite(spot_arr))) or not bool(np.all(spot_arr > 0.0)):
        raise SurfaceValidationError("spot must be finite and strictly positive.")
    if not bool(np.all(np.isfinite(forwards))) or not bool(np.all(forwards > 0.0)):
        raise SurfaceValidationError("forward must be finite and strictly positive.")
    return SurfacePoints(
        k=moneyness + np.log(spot_arr / forwards),
        T=T,
        surface_id=surface_id,
        point_id=point_id,
        convention=CoordinateConvention(
            source="spot_moneyness", maturity=maturity, market_source=market_source
        ),
    )


def points_from_forward_delta(
    delta: Any,
    T: Any,
    iv: Any,
    *,
    is_call: Any = True,
    surface_id: Any = None,
    point_id: Any = None,
    maturity: str = "year_fraction",
    market_source: str | None = None,
) -> SurfacePoints:
    """Points from an undiscounted forward delta and the implied volatility at it.

    The Black forward delta of a call is :math:`N(d_1)` with
    :math:`d_1 = (-k + w/2)/\\sqrt{w}` and :math:`w = \\sigma^2 T`, so inverting
    it is exact:

    .. math::

        k = \\tfrac{1}{2} w - \\sqrt{w}\\, N^{-1}(\\Delta_{\\text{call}}).

    Delta-quoted surfaces are the reason this adapter needs an implied
    volatility as an *input*: the coordinate is defined in terms of the very
    quantity being observed.  Supplying it explicitly keeps that circularity in
    the caller's hands, where it can be resolved by the convention the venue
    actually quotes, rather than resolved silently here.

    Parameters
    ----------
    delta:
        Undiscounted forward delta, shape ``(N,)``.  Call deltas lie in
        ``(0, 1)``; put deltas in ``(-1, 0)``.
    is_call:
        Per-point call/put flags, or a single flag for every point.  A put
        delta is converted with :math:`\\Delta_{\\text{call}} = 1 + \\Delta_{\\text{put}}`.

    Examples
    --------
    >>> from fast_vollib.surface import points_from_forward_delta
    >>> points = points_from_forward_delta([0.5], [1.0], [0.2])
    >>> float(round(points.k[0], 12))
    0.02
    """
    from scipy.special import ndtri

    deltas = np.atleast_1d(np.array(delta, dtype=np.float64, copy=True))
    maturity_years = np.atleast_1d(np.array(T, dtype=np.float64, copy=True))
    sigma = np.atleast_1d(np.array(iv, dtype=np.float64, copy=True))
    calls = np.atleast_1d(np.asarray(is_call, dtype=bool))
    deltas, maturity_years, sigma, calls = np.broadcast_arrays(deltas, maturity_years, sigma, calls)
    if not bool(np.all(np.isfinite(sigma))) or not bool(np.all(sigma > 0.0)):
        raise SurfaceValidationError("iv must be finite and strictly positive to invert a delta.")
    call_delta = np.where(calls, deltas, 1.0 + deltas)
    if not bool(np.all((call_delta > 0.0) & (call_delta < 1.0))):
        raise SurfaceValidationError(
            "delta must invert to a call delta strictly inside (0, 1); "
            "0 and 1 are the degenerate wings, where no finite strike exists."
        )
    w = sigma * sigma * maturity_years
    sqrt_w = np.sqrt(w)
    return SurfacePoints(
        k=0.5 * w - sqrt_w * ndtri(call_delta),
        T=maturity_years,
        surface_id=surface_id,
        point_id=point_id,
        convention=CoordinateConvention(
            source="forward_delta",
            maturity=maturity,
            market_source=market_source,
            notes="k inverted from the Black forward delta at the supplied implied volatility.",
        ),
    )
