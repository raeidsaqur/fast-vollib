"""Turning a model into a grid, and a grid back into a model -- both explicitly.

Two conversions live here, and neither is allowed to happen by accident.

**Model to grid.**  :func:`materialize_surface` evaluates a
:class:`~fast_vollib.surface.protocols.DefiniteIVSurface` at a declared
:class:`~fast_vollib.surface.gridspec.SurfaceGridSpec` and returns an
:class:`~fast_vollib.surface.grid.IVSurface`.  Every node is the model's own
output, so the arbitrage report that follows is a statement about the model.
This is the path that lets an SVI fit, a Heston surface, a forecast, and one
draw of a generative model be checked by identical code.

**Grid to model.**  :class:`GridIVSurface` wraps an existing grid so it can be
asked about points that are not nodes.  It requires an explicit interpolation
``policy`` -- there is no default -- because the choice is a modelling decision
with consequences the report will attribute to the model: interpolating implied
volatility linearly between two maturities can manufacture a calendar violation
that interpolating total variance does not.  :class:`IVSurface` itself gains no
``evaluate`` method, so a grid can never be silently treated as a continuous
surface.

*Interpolated nodes are labelled.*  A grid built by interpolation carries a
``native_mask``, and the arbitrage report separates native violations from
interpolation-induced ones on exactly that basis, which is how a genuine model
defect stays distinguishable from an artifact of the mesh someone chose.

Examples
--------
>>> import numpy as np
>>> from fast_vollib.surface import (
...     SurfaceGridSpec, SurfaceMarket, SurfacePrediction, materialize_surface,
... )
>>> class FlatSurface:
...     def evaluate(self, points, *, market=None):
...         return SurfacePrediction(points=points, iv=np.full(points.n, 0.2))
>>> grid = SurfaceGridSpec(
...     k=np.linspace(-0.2, 0.2, 5), T=[0.5, 1.0], market=SurfaceMarket.flat(forward=100.0)
... )
>>> surface = materialize_surface(FlatSurface(), grid)
>>> surface.iv.shape, bool(surface.validate().passed)
((5, 2), True)
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from .errors import SurfaceDomainError, SurfaceValidationError
from .grid import IVSurface
from .gridspec import SurfaceGridSpec
from .points import SurfacePoints
from .prediction import SurfacePrediction, SurfaceSamples

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .market import SurfaceMarket
    from .protocols import DefiniteIVSurface

__all__ = [
    "EXTRAPOLATIONS",
    "POLICIES",
    "GridIVSurface",
    "materialize_samples",
    "materialize_surface",
]

#: Interpolation policies :class:`GridIVSurface` accepts.  ``'total_variance'``
#: interpolates :math:`w = \\sigma^2 T` bilinearly and recovers
#: :math:`\\sigma = \\sqrt{w / T}`; ``'implied_volatility'`` interpolates
#: :math:`\\sigma` directly; ``'nearest'`` snaps to the closest node.
POLICIES = ("total_variance", "implied_volatility", "nearest")

#: What happens outside the grid's own range.  ``'invalid'`` marks the point
#: unanswered, ``'error'`` raises, ``'clamp'`` evaluates at the boundary node.
EXTRAPOLATIONS = ("invalid", "error", "clamp")


def materialize_surface(
    surface: "DefiniteIVSurface",
    grid: SurfaceGridSpec,
    *,
    market: "SurfaceMarket | None" = None,
    surface_id: Any = None,
    t_index: int | None = None,
) -> IVSurface:
    """Evaluate ``surface`` on ``grid`` and return the resulting materialized mesh.

    Parameters
    ----------
    surface:
        Any definite surface.  It is asked for exactly the grid's nodes, in
        C order over ``(Nk, Nt)``, and its answer is reshaped back.
    grid:
        The declared mesh.  Its market state is passed to the surface unless
        ``market`` overrides it.
    market:
        Market state to evaluate against, overriding the grid's own.
    surface_id:
        Label attached to the query points, for a surface that distinguishes
        them.
    t_index:
        Optional calendar-time index recorded on the resulting
        :class:`~fast_vollib.surface.grid.IVSurface`.

    Returns
    -------
    An :class:`~fast_vollib.surface.grid.IVSurface` whose invalid nodes are
    ``NaN``.  Nodes the model declined are missing quotes, not zeros: every
    downstream statistic excludes them and counts them.

    Raises
    ------
    SurfaceValidationError
        If the surface returns a prediction on different points than it was
        asked about.  Row order is the contract; a surface that reorders is
        detected here rather than producing a silently transposed mesh.
    """
    points = grid.to_points(surface_id=surface_id)
    state = market if market is not None else grid.market
    prediction = surface.evaluate(points, market=state)
    if not isinstance(prediction, SurfacePrediction):
        raise SurfaceValidationError(
            f"A definite surface must return a SurfacePrediction; got {type(prediction).__name__}."
        )
    if prediction.n != points.n:
        raise SurfaceValidationError(
            f"The surface answered {prediction.n} of {points.n} requested points. "
            f"Evaluation returns one row per query row, in query order."
        )
    if not prediction.points.equals(points):
        raise SurfaceValidationError(
            "The surface returned a prediction on different points than it was asked "
            "about. Row order is the alignment contract; reordering here would put "
            "values on the wrong nodes."
        )
    iv = np.where(prediction.valid, prediction.iv, np.nan).reshape(grid.shape)
    forward: Any = 1.0
    rate: Any = 0.0
    carry: Any = 0.0
    if state is not None:
        forward = np.asarray(state.forward_at(grid.T), dtype=np.float64)
        rate = np.asarray(state.rate_at(grid.T), dtype=np.float64)
        carry = np.asarray(state.carry_at(grid.T), dtype=np.float64)
    return IVSurface(
        k=np.array(grid.k, copy=True),
        T=np.array(grid.T, copy=True),
        iv=iv,
        forward=forward,
        r=rate,
        q=carry,
        native_mask=grid.native_mask,
        t_index=t_index,
    )


def materialize_samples(
    samples: SurfaceSamples,
    grid: SurfaceGridSpec,
    *,
    market: "SurfaceMarket | None" = None,
) -> Iterator[IVSurface]:
    """Yield each draw of ``samples`` as its own materialized grid.

    The draws must have been taken at ``grid.to_points()``; the check is exact,
    for the same reason alignment elsewhere is exact.  Yielding rather than
    returning a list keeps a thousand-sample evaluation from holding a thousand
    meshes at once.

    Raises
    ------
    SurfaceValidationError
        If the sampled points are not the grid's own nodes in the grid's order.
    """
    expected = grid.to_points(surface_id=None)
    if not samples.points.matches_domain(expected):
        raise SurfaceValidationError(
            "Samples were drawn at points other than this grid's nodes, so they "
            "cannot be reshaped onto it. Sample at grid.to_points() to evaluate a "
            "distribution on a declared mesh."
        )
    state = market if market is not None else grid.market
    forward: Any = 1.0
    rate: Any = 0.0
    carry: Any = 0.0
    if state is not None:
        forward = np.asarray(state.forward_at(grid.T), dtype=np.float64)
        rate = np.asarray(state.rate_at(grid.T), dtype=np.float64)
        carry = np.asarray(state.carry_at(grid.T), dtype=np.float64)
    for index in range(samples.n_samples):
        iv = np.where(samples.valid[index], samples.iv[index], np.nan).reshape(grid.shape)
        yield IVSurface(
            k=np.array(grid.k, copy=True),
            T=np.array(grid.T, copy=True),
            iv=iv,
            forward=forward,
            r=rate,
            q=carry,
            native_mask=grid.native_mask,
            t_index=index,
        )


@dataclass(frozen=True, slots=True)
class GridIVSurface:
    """A materialized grid presented as a definite surface, under a stated policy.

    Parameters
    ----------
    surface:
        The grid to read.  Its own ``iv`` is never modified.
    policy:
        Required.  One of :data:`POLICIES`.  There is no default because the
        choice changes the answer and, through it, the arbitrage report: linear
        interpolation of implied volatility across maturities does not preserve
        total-variance monotonicity, and linear interpolation of total variance
        does.
    extrapolation:
        One of :data:`EXTRAPOLATIONS`; ``'invalid'`` by default, which reports a
        point outside the mesh as unanswered rather than inventing a value for
        it.

    Notes
    -----
    Every point this returns is an interpolated node unless it coincides with a
    grid node.  A caller materializing this back onto a mesh should pass the
    resulting ``native_mask`` so the report can attribute violations correctly;
    :meth:`native_mask_for` computes it.

    Examples
    --------
    >>> import numpy as np
    >>> from fast_vollib.surface import GridIVSurface, IVSurface, SurfacePoints
    >>> grid = IVSurface.from_logmoneyness(
    ...     np.array([-0.1, 0.0, 0.1]), np.array([0.5, 1.0]), np.full((3, 2), 0.2)
    ... )
    >>> reader = GridIVSurface(grid, policy="total_variance")
    >>> prediction = reader.evaluate(SurfacePoints(k=[0.05], T=[0.75]))
    >>> float(round(prediction.iv[0], 12))
    0.2
    """

    surface: IVSurface
    policy: str = "total_variance"
    extrapolation: str = "invalid"

    def __post_init__(self) -> None:
        if not isinstance(self.surface, IVSurface):
            raise SurfaceValidationError(
                f"surface must be an IVSurface; got {type(self.surface).__name__}."
            )
        if self.policy not in POLICIES:
            raise SurfaceValidationError(f"policy must be one of {POLICIES}; got {self.policy!r}.")
        if self.extrapolation not in EXTRAPOLATIONS:
            raise SurfaceValidationError(
                f"extrapolation must be one of {EXTRAPOLATIONS}; got {self.extrapolation!r}."
            )
        axis = np.asarray(self.surface.k, dtype=np.float64)
        if axis.shape[0] > 1 and not bool(np.all(np.diff(axis, axis=0) > 0.0)):
            raise SurfaceValidationError(
                "GridIVSurface reads along the moneyness axis, so the grid's k must be "
                "strictly increasing; sort the grid before wrapping it."
            )
        namespace = self.surface.namespace()
        if namespace.name != "numpy":
            raise SurfaceValidationError(
                f"GridIVSurface reads a host grid; this surface is on the "
                f"{namespace.name!r} backend. Materialize it to numpy first -- "
                f"interpolation here is a host operation and would break the tape."
            )

    def evaluate(
        self,
        points: SurfacePoints,
        *,
        market: "SurfaceMarket | None" = None,
    ) -> SurfacePrediction:
        """Interpolated implied volatility at ``points`` under the declared policy."""
        del market  # the grid carries its own market state; interpolation needs none
        if not isinstance(points, SurfacePoints):
            raise SurfaceValidationError(
                f"points must be a SurfacePoints; got {type(points).__name__}."
            )
        iv = _interpolate_grid(self.surface, points.k, points.T, self.policy, self.extrapolation)
        return SurfacePrediction(points=points, iv=iv)

    def native_mask_for(self, grid: SurfaceGridSpec, *, decimals: int = 12) -> np.ndarray:
        """Which nodes of ``grid`` coincide with this surface's own nodes.

        Coincidence is decided on coordinates rounded to ``decimals``, because a
        mesh rebuilt by ``linspace`` reproduces a node to within rounding rather
        than bit-for-bit, and calling such a node interpolated would over-report
        interpolation artifacts.
        """
        source_k = np.round(np.unique(np.asarray(self.surface.k).reshape(-1)), decimals)
        source_T = np.round(np.asarray(self.surface.T).reshape(-1), decimals)
        target_k = np.round(grid.k2d(), decimals)
        target_T = np.round(grid.T2d(), decimals)
        return np.isin(target_k, source_k) & np.isin(target_T, source_T)


def _interpolate_grid(
    surface: IVSurface,
    kq: np.ndarray,
    Tq: np.ndarray,
    policy: str,
    extrapolation: str,
) -> np.ndarray:
    """Bilinear (or nearest) read of ``surface`` at scattered ``(kq, Tq)``."""
    T = np.asarray(surface.T, dtype=np.float64).reshape(-1)
    iv = np.asarray(surface.iv, dtype=np.float64)
    k = np.asarray(surface.k, dtype=np.float64)
    values = iv * iv * T[None, :] if policy == "total_variance" else iv

    out = np.full(kq.shape, np.nan, dtype=np.float64)
    if T.size == 0 or iv.size == 0:
        return out

    below = Tq < T[0]
    above = Tq > T[-1]
    outside_T = below | above
    if extrapolation == "error" and bool(np.any(outside_T)):
        raise SurfaceDomainError(
            f"{int(np.sum(outside_T))} point(s) lie outside the grid's maturity range "
            f"[{T[0]}, {T[-1]}] and extrapolation='error'."
        )
    clamped_T = np.clip(Tq, T[0], T[-1])

    if T.size == 1:
        upper = np.zeros(kq.shape, dtype=np.intp)
        lower = upper
        weight = np.zeros(kq.shape, dtype=np.float64)
    else:
        upper = np.clip(np.searchsorted(T, clamped_T, side="left"), 1, T.size - 1)
        lower = upper - 1
        span = T[upper] - T[lower]
        weight = (clamped_T - T[lower]) / span

    if policy == "nearest":
        weight = np.where(weight >= 0.5, 1.0, 0.0)

    left = _column_values(k, values, lower, kq, policy, extrapolation)
    right = _column_values(k, values, upper, kq, policy, extrapolation)
    blended = (1.0 - weight) * left + weight * right

    if policy == "total_variance":
        with np.errstate(invalid="ignore"):
            out = np.sqrt(np.where(blended >= 0.0, blended, np.nan) / clamped_T)
    else:
        out = blended
    if extrapolation == "invalid":
        out = np.where(outside_T, np.nan, out)
    return out


def _column_values(
    k: np.ndarray,
    values: np.ndarray,
    columns: np.ndarray,
    kq: np.ndarray,
    policy: str,
    extrapolation: str,
) -> np.ndarray:
    """Interpolate each query point along the moneyness axis of its column."""
    out = np.full(kq.shape, np.nan, dtype=np.float64)
    for column in np.unique(columns):
        selected = columns == column
        axis = k if k.ndim == 1 else k[:, column]
        column_values = values[:, column]
        query = kq[selected]
        if extrapolation == "error":
            outside = (query < axis[0]) | (query > axis[-1])
            if bool(np.any(outside)):
                raise SurfaceDomainError(
                    f"{int(np.sum(outside))} point(s) lie outside the grid's moneyness "
                    f"range [{axis[0]}, {axis[-1]}] and extrapolation='error'."
                )
        if policy == "nearest":
            index = np.clip(np.searchsorted(axis, query, side="left"), 0, axis.size - 1)
            lower = np.clip(index - 1, 0, axis.size - 1)
            pick = np.where(
                np.abs(axis[index] - query) <= np.abs(axis[lower] - query), index, lower
            )
            interpolated = column_values[pick]
        else:
            interpolated = np.interp(query, axis, column_values)
        if extrapolation == "invalid":
            interpolated = np.where((query < axis[0]) | (query > axis[-1]), np.nan, interpolated)
        out[selected] = interpolated
    return out
