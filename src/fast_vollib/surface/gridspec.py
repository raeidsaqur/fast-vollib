"""The mesh an evaluation is run on, declared before any model is asked for a number.

Arbitrage conditions are *discrete* statements about neighbouring nodes: a
butterfly needs three strikes, a calendar needs two maturities.  Which nodes
those are is therefore part of the result, not an implementation detail, and
two models compared on different meshes have not been compared.
:class:`SurfaceGridSpec` is that mesh as a value -- coordinates, topology,
market state, and which nodes a generator actually produced -- so a report can
say what it checked.

*A grid is a question, not an answer.*  The spec holds no implied volatilities.
Filling it is :func:`~fast_vollib.surface.materialize.materialize_surface`,
which evaluates a definite surface at the grid's own points and returns an
:class:`~fast_vollib.surface.grid.IVSurface`.  Keeping the two apart is what
lets the same declared grid be applied to a calibrated smile, a forecast, and
every draw of a generative model without any of them agreeing on anything else.

Two topologies are supported, matching :class:`~fast_vollib.surface.grid.IVSurface`:

* ``'shared_moneyness'`` -- one forward-log-moneyness axis common to every
  maturity.  Calendar arbitrage is total-variance monotonicity.
* ``'fixed_strike'`` -- a shared strike vector under a term-varying forward, so
  ``k`` is two-dimensional.  Calendar arbitrage is undiscounted-call
  monotonicity at fixed strike, which is the coordinate-correct form there.

Examples
--------
>>> import numpy as np
>>> from fast_vollib.surface import SurfaceGridSpec, SurfaceMarket
>>> grid = SurfaceGridSpec(
...     k=np.linspace(-0.3, 0.3, 7),
...     T=[0.25, 1.0],
...     market=SurfaceMarket.flat(forward=100.0),
... )
>>> grid.shape, grid.n_nodes
((7, 2), 14)
>>> grid.to_points().n
14
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ._validate import read_only
from .errors import SurfaceValidationError
from .market import SurfaceMarket
from .points import CoordinateConvention, SurfacePoints

__all__ = ["TOPOLOGIES", "SurfaceGridSpec"]

#: Accepted grid topologies, matching :class:`~fast_vollib.surface.grid.IVSurface`.
TOPOLOGIES = ("shared_moneyness", "fixed_strike")


@dataclass(frozen=True, slots=True)
class SurfaceGridSpec:
    """A declared ``(Nk, Nt)`` evaluation mesh in canonical coordinates.

    Parameters
    ----------
    k:
        Forward log-moneyness, shape ``(Nk,)`` for a shared axis or
        ``(Nk, Nt)`` for a fixed-strike grid.  Finite everywhere, and strictly
        increasing down the moneyness axis -- an unordered axis makes the
        neighbour stencils that detect butterflies meaningless.
    T:
        Maturities in years, shape ``(Nt,)``, finite, strictly positive, and
        strictly increasing.
    market:
        Market state used to convert to strikes and prices.  Optional: a grid
        that is only ever used in implied-volatility space needs none, and a
        computation that needs one and does not have it raises rather than
        inventing it.
    topology:
        ``'shared_moneyness'`` (default) or ``'fixed_strike'``.  Inferred from
        the rank of ``k`` when it disagrees is *not* allowed; a 2-D ``k`` with
        ``'shared_moneyness'`` is an error, because the two forms check
        different calendar conditions.
    native_mask:
        Optional ``(Nk, Nt)`` boolean marking nodes a generator produced
        directly.  Nodes outside it are interpolation artifacts, and the
        arbitrage report separates violations by origin on exactly this basis.
    name:
        Optional label for the grid, carried into reports.

    Raises
    ------
    SurfaceValidationError
        On a non-monotone axis, a non-positive maturity, a topology that
        disagrees with the rank of ``k``, or a mask of the wrong shape.
    """

    k: Any
    T: Any
    market: SurfaceMarket | None = None
    topology: str = "shared_moneyness"
    native_mask: Any = None
    name: str | None = None

    def __post_init__(self) -> None:
        if self.topology not in TOPOLOGIES:
            raise SurfaceValidationError(
                f"topology must be one of {TOPOLOGIES}; got {self.topology!r}."
            )
        T = read_only(np.array(self.T, dtype=np.float64, copy=True).reshape(-1))
        if T.size == 0:
            raise SurfaceValidationError("A grid needs at least one maturity.")
        if not bool(np.all(np.isfinite(T))) or not bool(np.all(T > 0.0)):
            raise SurfaceValidationError("T must be finite and strictly positive everywhere.")
        if T.size > 1 and not bool(np.all(np.diff(T) > 0.0)):
            raise SurfaceValidationError(f"T must be strictly increasing; got {T.tolist()}.")

        k = np.array(self.k, dtype=np.float64, copy=True)
        if k.ndim not in (1, 2):
            raise SurfaceValidationError(f"k must be one- or two-dimensional; got shape {k.shape}.")
        if self.topology == "shared_moneyness" and k.ndim != 1:
            raise SurfaceValidationError(
                "topology='shared_moneyness' requires a 1-D k; a 2-D k describes a "
                "fixed-strike grid, whose calendar condition is a different one."
            )
        if self.topology == "fixed_strike" and k.ndim != 2:
            raise SurfaceValidationError(
                "topology='fixed_strike' requires a 2-D k of shape (Nk, Nt)."
            )
        if k.ndim == 2 and k.shape[1] != T.size:
            raise SurfaceValidationError(
                f"A 2-D k must have one column per maturity ({T.size}); got {k.shape[1]}."
            )
        if k.size == 0:
            raise SurfaceValidationError("A grid needs at least one moneyness node.")
        if not bool(np.all(np.isfinite(k))):
            raise SurfaceValidationError("k must be finite everywhere.")
        if k.shape[0] > 1 and not bool(np.all(np.diff(k, axis=0) > 0.0)):
            raise SurfaceValidationError(
                "k must be strictly increasing down the moneyness axis; the butterfly "
                "and slope stencils read neighbours in that order."
            )
        k = read_only(k)

        if self.market is not None and not isinstance(self.market, SurfaceMarket):
            raise SurfaceValidationError(
                f"market must be a SurfaceMarket or None; got {type(self.market).__name__}."
            )

        native_mask = None
        if self.native_mask is not None:
            native_mask = np.asarray(self.native_mask)
            if native_mask.dtype != np.bool_:
                raise SurfaceValidationError(
                    f"native_mask must be a boolean array; got dtype {native_mask.dtype}."
                )
            expected = (k.shape[0], T.size)
            if native_mask.shape != expected:
                raise SurfaceValidationError(
                    f"native_mask must have shape {expected}; got {native_mask.shape}."
                )
            native_mask = read_only(np.array(native_mask, dtype=bool, copy=True))

        object.__setattr__(self, "k", k)
        object.__setattr__(self, "T", T)
        object.__setattr__(self, "native_mask", native_mask)

    # -- geometry ------------------------------------------------------------
    @property
    def Nk(self) -> int:
        """Number of moneyness nodes."""
        return int(self.k.shape[0])

    @property
    def Nt(self) -> int:
        """Number of maturities."""
        return int(self.T.size)

    @property
    def shape(self) -> tuple[int, int]:
        """The ``(Nk, Nt)`` mesh shape."""
        return (self.Nk, self.Nt)

    @property
    def n_nodes(self) -> int:
        """Total number of mesh nodes."""
        return self.Nk * self.Nt

    @property
    def shared_k(self) -> bool:
        """Whether the moneyness axis is shared across maturities."""
        return self.topology == "shared_moneyness"

    def k2d(self) -> np.ndarray:
        """Log-moneyness broadcast to ``(Nk, Nt)``."""
        if self.k.ndim == 2:
            return self.k
        return np.broadcast_to(self.k[:, None], self.shape)

    def T2d(self) -> np.ndarray:
        """Maturity broadcast to ``(Nk, Nt)``."""
        return np.broadcast_to(self.T[None, :], self.shape)

    # -- constructors --------------------------------------------------------
    @classmethod
    def uniform(
        cls,
        *,
        k_min: float,
        k_max: float,
        n_k: int,
        T: Any,
        market: SurfaceMarket | None = None,
        name: str | None = None,
    ) -> SurfaceGridSpec:
        """A shared-moneyness grid with ``n_k`` equally spaced nodes in ``[k_min, k_max]``."""
        if n_k < 1:
            raise SurfaceValidationError(f"n_k must be at least 1; got {n_k}.")
        if not (k_max > k_min):
            raise SurfaceValidationError(
                f"k_max must exceed k_min; got k_min={k_min!r}, k_max={k_max!r}."
            )
        return cls(k=np.linspace(k_min, k_max, n_k), T=T, market=market, name=name)

    @classmethod
    def from_strikes(
        cls,
        K: Any,
        T: Any,
        *,
        market: SurfaceMarket,
        name: str | None = None,
    ) -> SurfaceGridSpec:
        """A fixed-strike grid: ``k_ij = log(K_i / F(T_j))`` under ``market``.

        The topology is ``'fixed_strike'`` even when the forward happens to be
        flat, because the caller declared strikes.  A flat-forward grid that
        should be treated as shared-moneyness is built with the shared-moneyness
        constructor from the resulting single column.
        """
        strikes = np.array(K, dtype=np.float64, copy=True).reshape(-1)
        if not bool(np.all(np.isfinite(strikes))) or not bool(np.all(strikes > 0.0)):
            raise SurfaceValidationError("K must be finite and strictly positive.")
        maturities = np.array(T, dtype=np.float64, copy=True).reshape(-1)
        forwards = np.atleast_1d(market.forward_at(maturities))
        return cls(
            k=np.log(strikes[:, None] / forwards[None, :]),
            T=maturities,
            market=market,
            topology="fixed_strike",
            name=name,
        )

    # -- conversions ---------------------------------------------------------
    def to_points(self, *, surface_id: Any = None, point_ids: bool = True) -> SurfacePoints:
        """The mesh flattened to scattered points, in C order over ``(Nk, Nt)``.

        Parameters
        ----------
        surface_id:
            Label for every node; ``None`` means a single surface labelled ``0``.
        point_ids:
            Whether to attach ``0 .. Nk*Nt - 1`` flat indices as stable point
            labels.  They are what lets a prediction be put back on the mesh
            without relying on the order it came back in.
        """
        n = self.n_nodes
        return SurfacePoints(
            k=self.k2d().reshape(-1),
            T=self.T2d().reshape(-1),
            surface_id=surface_id,
            point_id=np.arange(n, dtype=np.int64) if point_ids else None,
            convention=CoordinateConvention(
                source="forward_log_moneyness",
                market_source=None if self.market is None else self.market.source,
                notes=f"grid {self.name!r} ({self.topology}, {self.Nk}x{self.Nt})",
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """The specification as a JSON-safe mapping."""
        return {
            "name": self.name,
            "topology": self.topology,
            "Nk": self.Nk,
            "Nt": self.Nt,
            "k": self.k.reshape(-1).tolist(),
            "T": self.T.tolist(),
            "market": None if self.market is None else self.market.to_dict(),
            "has_native_mask": self.native_mask is not None,
        }

    def require_market(self, what: str) -> SurfaceMarket:
        """The market state, or a :class:`~fast_vollib.surface.errors.MissingMarketStateError`."""
        from .errors import MissingMarketStateError

        if self.market is None:
            raise MissingMarketStateError(
                f"{what} needs the market state this grid was quoted against, and the "
                f"grid carries none. Attach a SurfaceMarket rather than accepting a "
                f"default forward, which would produce a number that is not a price."
            )
        return self.market
