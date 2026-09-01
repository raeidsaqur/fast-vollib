"""A surface as a mean plus a handful of numbers, with signs that survive a re-run.

An ensemble of implied-volatility surfaces -- a year of one underlying, a batch
of generated samples, a family of stressed scenarios -- is not a cloud of
independent nodes.  A few smooth modes (a level, a skew, a term slope) explain
almost all of it, so a whole surface can be carried as a short score vector
against a fixed basis.  :func:`fit_factor_basis` computes that basis with one
:func:`numpy.linalg.svd` of the mean-centred node matrix, and
:class:`FactorIVSurface` turns a score vector back into an evaluable
:class:`~fast_vollib.surface.protocols.DefiniteIVSurface`.

*The sign convention is the point of this module.*  The singular value
decomposition determines each component only up to sign: :math:`u_j s_j v_j^T`
is unchanged when both :math:`u_j` and :math:`v_j` are negated, so LAPACK is
free to return either, and which one it returns depends on the driver, the BLAS
build, and the platform.  A basis that inherited that freedom would give scores
whose meaning -- "one unit of skew" or its negative -- changed when the
environment did, and a stored score vector would silently decode into a
different surface on a different machine.  Every component here is therefore
fixed so that **the first of its largest-magnitude entries is positive** -- first
by flat node index, and largest within a relative band rather than exactly, for
the reason below.  It is an arbitrary rule, deterministic, and stated once; that
is all a convention has to be.  Stating it without the band would be worse than
saying nothing: on a symmetric grid the two extreme entries of a skew mode tie,
and a caller who checked ``loadings[j][argmax(abs(loadings[j]))] > 0`` would find
it false on this package's own fixture.

*A tie is a near-tie.*  Two entries that are equal in exact arithmetic come back
from a decomposition differing by a unit in the last place, so a rule that only
recognised exact equality would leave the pivot -- and with it the sign of the
whole component -- to rounding.  That is not hypothetical: the skew mode of a
symmetric moneyness grid is antisymmetric, its two extreme entries are equal in
magnitude and opposite in sign, and permuting the ensemble is enough to swap
which of them the last bit calls larger.  Magnitudes within
:data:`_PIVOT_TIE_RTOL` of the largest are therefore treated as tied, and the
lowest flat node index among them wins.

*Missing nodes are refused, not filled.*  A singular value decomposition has no
principled treatment of a matrix with holes, and the two silent repairs -- zeros
or the column mean -- both write a fabricated number into the mean vector and
into every loading, where nothing downstream can see it any more.  An ensemble
member with a ``NaN`` node is rejected by name, and the caller materializes a
complete grid first.

Two design decisions are worth stating.

**The value space is configurable and defaults to total variance.**  Factoring
:math:`w = \\sigma^2 T` rather than :math:`\\sigma` puts the model's arithmetic
in the coordinates the no-arbitrage conditions are written in and in the space
the safe interpolation policy of
:class:`~fast_vollib.surface.materialize.GridIVSurface` works in, so the linear
algebra and the grid read happen in the same variable rather than in two.  It
does *not* make a reconstruction arbitrage-free: a large enough score can still
break calendar monotonicity, and the arbitrage harness is what says so.

**Evaluation reuses the one interpolator this package has.**  A factor surface
reconstructs its reference grid and reads it through
:class:`~fast_vollib.surface.materialize.GridIVSurface` under an explicitly
declared policy, defaulting to the space the basis was fitted in.  There is no
second interpolator here, so an off-node value from a factor model and an
off-node value from any other materialized grid are produced by identical code
and are attributed identically by the report.

Examples
--------
>>> import numpy as np
>>> from fast_vollib.surface import IVSurface, SurfaceObservations, SurfacePoints
>>> from fast_vollib.surface.fitting.factors import (
...     FactorSurfaceCalibrator, fit_factor_basis,
... )
>>> k = np.linspace(-0.2, 0.2, 5)
>>> T = np.array([0.5, 1.0])
>>> mean_w = 0.04 + 0.02 * np.ones((5, 1)) * T[None, :]
>>> level = 0.01 * np.ones((5, 1)) * T[None, :]
>>> skew = 0.005 * k[:, None] * np.ones((1, 2))
>>> ensemble = [
...     IVSurface.from_total_variance(k, T, mean_w + a * level + b * skew)
...     for a, b in [(-2.0, 1.0), (-1.0, -1.0), (1.0, -1.0), (2.0, 1.0)]
... ]
>>> basis = fit_factor_basis(ensemble, n_factors=2)
>>> basis.n_factors, basis.n_nodes, basis.value_space
(2, 10, 'total_variance')

The convention holds on every component, whatever LAPACK returned.  The pivot
is the lowest node index whose magnitude ties the largest, which for the skew
component is one of a pair equal in magnitude and opposite in sign:

>>> magnitude = np.abs(basis.loadings)
>>> tied = magnitude >= (1.0 - 1e-9) * magnitude.max(axis=1, keepdims=True)
>>> pivots = np.argmax(tied, axis=1)
>>> bool(np.all(basis.loadings[np.arange(2), pivots] > 0.0))
True

A calibrator projects one observed surface onto the basis, and the resulting
surface declines the points the basis does not cover:

>>> observations = SurfaceObservations.from_points(
...     basis.grid.to_points(), iv=ensemble[0].iv.reshape(-1)
... )
>>> fitted = FactorSurfaceCalibrator(basis=basis).fit(observations)
>>> bool(np.allclose(fitted.scores, basis.project(ensemble[0].total_variance())))
True
>>> fitted.evaluate(SurfacePoints(k=[0.0, 3.0], T=[0.75, 0.75])).valid.tolist()
[True, False]
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from .._validate import owned_float_1d, owned_float_2d
from ..errors import SurfaceCalibrationError, SurfaceValidationError
from ..grid import IVSurface
from ..gridspec import SurfaceGridSpec
from ..materialize import EXTRAPOLATIONS, POLICIES, GridIVSurface
from ..observations import SurfaceObservations
from ..points import SurfacePoints
from ..prediction import SurfacePrediction

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..market import SurfaceMarket
    from ..protocols import RNGInput

__all__ = [
    "VALUE_SPACES",
    "FactorIVSurface",
    "FactorPCARecipe",
    "FactorSurfaceCalibrator",
    "SurfaceFactorBasis",
    "fit_factor_basis",
]

#: The spaces a basis may be fitted in.  ``'total_variance'`` factors
#: :math:`w = \\sigma^2 T`; ``'implied_volatility'`` factors :math:`\\sigma`
#: itself.  The two give different bases: :math:`\\sigma \\mapsto \\sigma^2 T`
#: is not linear, so a rank-two ensemble in one space is not rank-two in the
#: other.
VALUE_SPACES = ("total_variance", "implied_volatility")

# How far the rows of a stored basis may stray from orthonormality.  A LAPACK
# SVD delivers orthonormality to a few units in the last place; 1e-8 is loose
# enough never to reject a genuine decomposition and tight enough to catch
# loadings that were rescaled, stacked in the wrong order, or never
# orthogonalized at all.
_ORTHONORMAL_ATOL = 1e-8

# How far the explained-variance ratios may sum past one before the vector is
# rejected as something other than a set of ratios.  Summing at most a few
# thousand float64 terms cannot drift this far.
_RATIO_SUM_ATOL = 1e-9

# How close to the largest absolute entry of a component another entry must be
# to count as tied with it.  Without a band the tie rule would be dead code: an
# antisymmetric mode -- a skew, the second component of almost every real
# ensemble -- has its extreme values equal in magnitude in exact arithmetic and
# unequal by one unit in the last place after a decomposition, so the pivot, and
# with it the sign of the whole component, would be chosen by rounding.  The
# band is six orders above the ~1e-15 componentwise noise of a float64 SVD and
# far below the difference a smooth mode makes between genuinely distinct nodes.
_PIVOT_TIE_RTOL = 1e-9


def _component_signs(loadings: np.ndarray) -> np.ndarray:
    """The sign that puts each component into the module's convention.

    Parameters
    ----------
    loadings:
        Right singular vectors, shape ``(n_factors, n_nodes)``.

    Returns
    -------
    Shape ``(n_factors,)`` of ``+1.0`` and ``-1.0``.  Multiplying each row by
    its sign makes its pivot entry positive.  The pivot is the *lowest* flat
    node index whose magnitude is within :data:`_PIVOT_TIE_RTOL` of the row's
    largest, so entries that are equal in exact arithmetic are treated as equal
    here; an all-zero row is left alone.

    Notes
    -----
    The band is what makes the tie rule operative.  A plain
    :func:`numpy.argmax` breaks a tie at the lowest index only when the two
    magnitudes are bit-for-bit identical, which a computed singular vector's
    almost never are -- and for an antisymmetric mode the two candidates carry
    opposite signs, so the last bit of the decomposition would decide the sign
    of the whole component.

    Examples
    --------
    >>> import numpy as np
    >>> from fast_vollib.surface.fitting.factors import _component_signs
    >>> _component_signs(np.array([[1.0, -1.0, 0.0], [-2.0, 2.0, 0.5]])).tolist()
    [1.0, -1.0]

    A tie that rounding has broken by one unit in the last place is still a tie:

    >>> _component_signs(np.array([[-1.0, 1.0 + 2.0e-16]])).tolist()
    [-1.0]
    """
    magnitude = np.abs(loadings)
    tied = magnitude >= (1.0 - _PIVOT_TIE_RTOL) * magnitude.max(axis=1, keepdims=True)
    pivot = np.argmax(tied, axis=1)
    entry = loadings[np.arange(loadings.shape[0]), pivot]
    return np.where(entry < 0.0, -1.0, 1.0)


def _to_value_space(iv: np.ndarray, T: np.ndarray, value_space: str) -> np.ndarray:
    """Implied volatilities expressed in ``value_space``."""
    if value_space == "total_variance":
        return iv * iv * T
    return iv


def _to_implied_volatility(values: np.ndarray, T: np.ndarray, value_space: str) -> np.ndarray:
    """``value_space`` values expressed as implied volatilities.

    A non-positive value is not an implied volatility -- a negative total
    variance is not a variance, and a zero volatility prices every option at
    intrinsic -- so it becomes ``NaN`` here and the nodes that read it come back
    invalid rather than as a plausible number the model does not stand behind.
    """
    with np.errstate(invalid="ignore"):
        positive = np.where(values > 0.0, values, np.nan)
        if value_space == "total_variance":
            return np.sqrt(positive / T)
        return positive


def _require_complete(values: np.ndarray, subject: str) -> np.ndarray:
    """``values`` with every node present, or a :class:`SurfaceValidationError`."""
    missing = int(np.count_nonzero(~np.isfinite(values)))
    if missing:
        raise SurfaceValidationError(
            f"Every node must be present for a singular value decomposition; {subject} "
            f"has {missing} missing node value(s). There is no principled decomposition "
            f"of a matrix with holes, and imputing zeros or a column mean would write a "
            f"fabricated number into the mean vector and into every loading, where "
            f"nothing downstream can see it. Materialize a complete grid first."
        )
    return values


def _read_nodes(
    grid: SurfaceGridSpec,
    node_values: np.ndarray,
    points: SurfacePoints,
    extrapolation: str,
) -> np.ndarray:
    """``node_values`` interpolated at ``points`` as a linear operator.

    Notes
    -----
    The read goes through :class:`~fast_vollib.surface.materialize.GridIVSurface`
    with ``policy='implied_volatility'``, which is the policy that interpolates
    the node array *as given*, without squaring or rooting it.  That is what is
    wanted here: a mean vector and a loading are coordinates, not volatilities --
    a loading is negative on half the mesh -- and the projection is only the
    least-squares problem it claims to be if the design matrix is the same
    linear map that :meth:`FactorIVSurface.evaluate` applies to the same nodes.
    Points outside the mesh come back ``NaN`` under ``extrapolation='invalid'``
    and are dropped by the caller.
    """
    surface = IVSurface(k=grid.k, T=grid.T, iv=node_values.reshape(grid.shape))
    reader = GridIVSurface(surface, policy="implied_volatility", extrapolation=extrapolation)
    return reader.evaluate(points).iv


@dataclass(frozen=True, slots=True)
class SurfaceFactorBasis:
    """A mean surface and an orthonormal set of modes on one reference grid.

    Parameters
    ----------
    grid:
        The reference mesh every stored vector lives on.  Node vectors are
        flattened in C order over ``(Nk, Nt)``, matching
        :meth:`~fast_vollib.surface.gridspec.SurfaceGridSpec.to_points`, so a
        flat index means the same node here and in a prediction.
    mean:
        The ensemble mean, shape ``(n_nodes,)``, in ``value_space``.
    loadings:
        The modes, shape ``(n_factors, n_nodes)``, one component per row, with
        orthonormal rows.  In a basis built by :func:`fit_factor_basis` they are
        the right singular vectors of the centred ensemble, in non-increasing
        order of singular value, each fixed to the sign convention of this
        module.  A basis assembled by hand is taken as given: the convention is
        applied where the arbitrary sign arises, which is the decomposition, and
        checking it here would reject a legitimate basis rotated or written down
        elsewhere.
    singular_values:
        Shape ``(n_factors,)``, non-negative and non-increasing.
    explained_variance_ratio:
        Shape ``(n_factors,)``, each :math:`s_j^2` over the sum of *all*
        squared singular values of the fit.  Non-increasing, and summing to one
        only when nothing was truncated -- which is what makes a truncated basis
        visibly a truncation.
    value_space:
        One of :data:`VALUE_SPACES`; the space the mean and loadings are
        expressed in, and the space a score vector is linear in.

    Raises
    ------
    SurfaceValidationError
        On a shape that does not match the grid, a non-finite entry, a
        non-monotone spectrum, ratios that sum past one, or loadings whose rows
        are not orthonormal.  The orthonormality check is what lets
        :meth:`project` be a dot product rather than a least-squares solve.

    Notes
    -----
    Every stored array is an owned, read-only copy, so a caller who keeps and
    later mutates the array they passed in cannot change a score that has
    already been reported.

    Examples
    --------
    >>> import numpy as np
    >>> from fast_vollib.surface import SurfaceGridSpec
    >>> from fast_vollib.surface.fitting.factors import SurfaceFactorBasis
    >>> grid = SurfaceGridSpec(k=[-0.1, 0.0, 0.1], T=[1.0])
    >>> basis = SurfaceFactorBasis(
    ...     grid=grid,
    ...     mean=[0.04, 0.04, 0.04],
    ...     loadings=[[-1.0, 0.0, 1.0]] / np.sqrt(2.0),
    ...     singular_values=[0.01],
    ...     explained_variance_ratio=[1.0],
    ... )
    >>> basis.n_factors, basis.n_nodes
    (1, 3)
    >>> basis.reconstruct([np.sqrt(2.0) * 0.001]).round(3).tolist()
    [[0.039], [0.04], [0.041]]
    """

    grid: SurfaceGridSpec
    mean: Any
    loadings: Any
    singular_values: Any
    explained_variance_ratio: Any
    value_space: str = "total_variance"

    def __post_init__(self) -> None:
        if not isinstance(self.grid, SurfaceGridSpec):
            raise SurfaceValidationError(
                f"grid must be a SurfaceGridSpec; got {type(self.grid).__name__}."
            )
        if self.value_space not in VALUE_SPACES:
            raise SurfaceValidationError(
                f"value_space must be one of {VALUE_SPACES}; got {self.value_space!r}."
            )

        n_nodes = self.grid.n_nodes
        mean = owned_float_1d(self.mean, "mean")
        if mean.size != n_nodes:
            raise SurfaceValidationError(
                f"mean must have one entry per grid node ({n_nodes}); got {mean.size}."
            )
        if not bool(np.all(np.isfinite(mean))):
            raise SurfaceValidationError(
                "mean must be finite at every node; a missing entry there would "
                "propagate into every reconstruction the basis produces."
            )

        loadings = np.array(self.loadings, dtype=np.float64, copy=True)
        if loadings.ndim != 2:
            raise SurfaceValidationError(
                f"loadings must be two-dimensional (n_factors, n_nodes); "
                f"got shape {loadings.shape}."
            )
        n_factors = loadings.shape[0]
        if n_factors < 1:
            raise SurfaceValidationError(
                "loadings must carry at least one component; a basis with no direction "
                "describes nothing the mean does not already describe."
            )
        loadings = owned_float_2d(loadings, "loadings", (n_factors, n_nodes))
        if not bool(np.all(np.isfinite(loadings))):
            raise SurfaceValidationError("loadings must be finite everywhere.")
        gram = loadings @ loadings.T
        deviation = float(np.max(np.abs(gram - np.eye(n_factors))))
        if deviation > _ORTHONORMAL_ATOL:
            raise SurfaceValidationError(
                f"loadings must have orthonormal rows to within {_ORTHONORMAL_ATOL}; the "
                f"Gram matrix deviates from the identity by {deviation!r}. Scores are read "
                f"off the basis by projection, which is only the least-squares answer when "
                f"the rows are orthonormal."
            )

        singular_values = owned_float_1d(self.singular_values, "singular_values")
        if singular_values.size != n_factors:
            raise SurfaceValidationError(
                f"singular_values must have one entry per component ({n_factors}); "
                f"got {singular_values.size}."
            )
        if not bool(np.all(np.isfinite(singular_values))) or not bool(
            np.all(singular_values >= 0.0)
        ):
            raise SurfaceValidationError("singular_values must be finite and non-negative.")
        if n_factors > 1 and not bool(np.all(np.diff(singular_values) <= 0.0)):
            raise SurfaceValidationError(
                f"singular_values must be non-increasing; got {singular_values.tolist()}. "
                f"Component order is what 'the first two factors' means."
            )

        ratio = owned_float_1d(self.explained_variance_ratio, "explained_variance_ratio")
        if ratio.size != n_factors:
            raise SurfaceValidationError(
                f"explained_variance_ratio must have one entry per component "
                f"({n_factors}); got {ratio.size}."
            )
        if not bool(np.all(np.isfinite(ratio))) or not bool(np.all(ratio >= 0.0)):
            raise SurfaceValidationError(
                "explained_variance_ratio must be finite and non-negative."
            )
        if n_factors > 1 and not bool(np.all(np.diff(ratio) <= 0.0)):
            raise SurfaceValidationError(
                f"explained_variance_ratio must be non-increasing; got {ratio.tolist()}."
            )
        total = float(ratio.sum())
        if total > 1.0 + _RATIO_SUM_ATOL:
            raise SurfaceValidationError(
                f"explained_variance_ratio must sum to at most 1; got {total!r}. A "
                f"fraction of the ensemble's variance cannot exceed the whole of it."
            )

        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "loadings", loadings)
        object.__setattr__(self, "singular_values", singular_values)
        object.__setattr__(self, "explained_variance_ratio", ratio)

    # -- geometry ------------------------------------------------------------
    @property
    def n_factors(self) -> int:
        """Number of components carried."""
        return int(self.loadings.shape[0])

    @property
    def n_nodes(self) -> int:
        """Number of nodes on the reference grid."""
        return int(self.grid.n_nodes)

    # -- the linear model ----------------------------------------------------
    def reconstruct(self, scores: Any) -> np.ndarray:
        """The ``(Nk, Nt)`` node values implied by ``scores``, in ``value_space``.

        Parameters
        ----------
        scores:
            Shape ``(n_factors,)`` coefficients against :attr:`loadings`.

        Returns
        -------
        ``mean + scores @ loadings``, reshaped to the reference grid.
        """
        coefficients = self._scores(scores)
        return (self.mean + coefficients @ self.loadings).reshape(self.grid.shape)

    def project(self, values: Any) -> np.ndarray:
        """The scores of a complete set of node ``values``.

        Parameters
        ----------
        values:
            Node values in ``value_space``, shape ``(Nk, Nt)`` or
            ``(n_nodes,)`` flattened in C order.  Every node must be present.

        Returns
        -------
        ``loadings @ (values - mean)``, shape ``(n_factors,)``.  The rows are
        orthonormal, so this dot product *is* the least-squares projection onto
        the subspace they span.

        Raises
        ------
        SurfaceValidationError
            If the shape does not match the grid, or any node is missing.
        """
        array = np.asarray(values, dtype=np.float64).reshape(-1)
        if array.size != self.n_nodes:
            raise SurfaceValidationError(
                f"values must cover every grid node ({self.n_nodes}); got {array.size}."
            )
        _require_complete(array, "the projected surface")
        return self.loadings @ (array - self.mean)

    def truncate(self, n_factors: int) -> SurfaceFactorBasis:
        """The same basis restricted to its first ``n_factors`` components.

        The explained-variance ratios are carried over unchanged rather than
        renormalized, so a truncated basis still reports what fraction of the
        original ensemble it accounts for; renormalizing would make every
        truncation claim to explain everything.
        """
        if isinstance(n_factors, bool) or not isinstance(n_factors, int):
            raise SurfaceValidationError(
                f"n_factors must be an integer; got {type(n_factors).__name__}."
            )
        if not (1 <= n_factors <= self.n_factors):
            raise SurfaceValidationError(
                f"n_factors must lie in [1, {self.n_factors}]; got {n_factors}. A basis "
                f"cannot be truncated to components it does not carry."
            )
        return SurfaceFactorBasis(
            grid=self.grid,
            mean=self.mean,
            loadings=self.loadings[:n_factors],
            singular_values=self.singular_values[:n_factors],
            explained_variance_ratio=self.explained_variance_ratio[:n_factors],
            value_space=self.value_space,
        )

    def _scores(self, scores: Any) -> np.ndarray:
        """A validated, owned copy of a score vector."""
        array = owned_float_1d(scores, "scores")
        if array.size != self.n_factors:
            raise SurfaceValidationError(
                f"scores must have one entry per component ({self.n_factors}); got {array.size}."
            )
        if not bool(np.all(np.isfinite(array))):
            raise SurfaceValidationError(
                "scores must be finite everywhere; a non-finite score reconstructs a "
                "surface that is missing wherever the component is non-zero."
            )
        return array


@dataclass(frozen=True, slots=True)
class FactorIVSurface:
    """A definite surface carried as a score vector against a basis.

    Parameters
    ----------
    basis:
        The :class:`SurfaceFactorBasis` the scores are expressed in.
    scores:
        Shape ``(basis.n_factors,)`` coefficients, finite.
    policy:
        How the reconstructed grid is read off the mesh; one of
        :data:`~fast_vollib.surface.materialize.POLICIES`.  ``None`` (the
        default) means the space the basis was fitted in, which makes the
        evaluated surface the factor model itself: the same linear map away
        from the nodes as on them.  Naming a different policy is a deliberate
        decoupling of the read space from the fit space -- reading an
        implied-volatility basis in total variance, say -- and makes off-node
        values a different function from the model that produced the scores.
    extrapolation:
        One of :data:`~fast_vollib.surface.materialize.EXTRAPOLATIONS`;
        ``'invalid'`` by default.  A basis is a set of directions on one mesh
        and says nothing beyond it.

    Notes
    -----
    Satisfies :class:`~fast_vollib.surface.protocols.DefiniteIVSurface`.  It
    returns one row per query row, in query order, and marks a point outside
    the reference grid invalid rather than extending the last mode outwards.

    Examples
    --------
    >>> import numpy as np
    >>> from fast_vollib.surface import SurfaceGridSpec, SurfacePoints
    >>> from fast_vollib.surface.fitting.factors import FactorIVSurface, SurfaceFactorBasis
    >>> basis = SurfaceFactorBasis(
    ...     grid=SurfaceGridSpec(k=[-0.1, 0.0, 0.1], T=[1.0]),
    ...     mean=[0.04, 0.04, 0.04],
    ...     loadings=[[-1.0, 0.0, 1.0]] / np.sqrt(2.0),
    ...     singular_values=[0.01],
    ...     explained_variance_ratio=[1.0],
    ... )
    >>> surface = FactorIVSurface(basis=basis, scores=[0.0])
    >>> surface.policy
    'total_variance'
    >>> float(round(surface.evaluate(SurfacePoints(k=[0.0], T=[1.0])).iv[0], 6))
    0.2
    """

    basis: SurfaceFactorBasis
    scores: Any
    policy: str | None = None
    extrapolation: str = "invalid"
    _reader: Any = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.basis, SurfaceFactorBasis):
            raise SurfaceValidationError(
                f"basis must be a SurfaceFactorBasis; got {type(self.basis).__name__}."
            )
        policy = self.basis.value_space if self.policy is None else self.policy
        if policy not in POLICIES:
            raise SurfaceValidationError(f"policy must be one of {POLICIES}; got {policy!r}.")
        if self.extrapolation not in EXTRAPOLATIONS:
            raise SurfaceValidationError(
                f"extrapolation must be one of {EXTRAPOLATIONS}; got {self.extrapolation!r}."
            )
        if self.extrapolation == "clamp":
            raise SurfaceValidationError(
                "extrapolation='clamp' is not available for fitting. Clamping reads an "
                "off-mesh quote at the nearest node's maturity, so the design column is "
                "built at the clamped T while the target is the quote's own -- a two-year "
                "quote is then fitted as if it were a one-year one, and the resulting "
                "scores reproduce neither. Use 'invalid' to exclude such quotes or "
                "'error' to refuse them; clamping remains available on FactorIVSurface, "
                "where the maturity being read is the one asked for."
            )
        object.__setattr__(self, "scores", self.basis._scores(self.scores))
        object.__setattr__(self, "policy", policy)

    def evaluate(
        self,
        points: SurfacePoints,
        *,
        market: "SurfaceMarket | None" = None,
    ) -> SurfacePrediction:
        """Implied volatility at ``points``, one row per query row, in query order.

        Parameters
        ----------
        points:
            Where to evaluate.  Points outside the reference grid come back
            invalid under the default extrapolation.
        market:
            Accepted and unused.  A factor model in implied-volatility
            coordinates is defined without prices, so it needs no market state
            and does not pretend to consume one.
        """
        del market  # the model lives in IV coordinates; no price is formed here
        if not isinstance(points, SurfacePoints):
            raise SurfaceValidationError(
                f"points must be a SurfacePoints; got {type(points).__name__}."
            )
        return self._grid_reader().evaluate(points)

    def to_grid(self) -> IVSurface:
        """The reconstruction materialized on the basis's own reference grid.

        Implied volatilities, not ``value_space`` values: a node whose
        reconstructed value is not a volatility -- a negative total variance --
        is ``NaN`` here, exactly as a missing quote is.
        """
        grid = self.basis.grid
        values = self.basis.reconstruct(self.scores)
        iv = _to_implied_volatility(values, grid.T2d(), self.basis.value_space)
        return IVSurface(k=grid.k, T=grid.T, iv=iv, native_mask=grid.native_mask)

    def parameters(self) -> dict[str, Any]:
        """The fitted parameters as a JSON-safe mapping."""
        return {
            "scores": [float(value) for value in self.scores],
            "value_space": str(self.basis.value_space),
            "policy": str(self.policy),
            "extrapolation": str(self.extrapolation),
        }

    def _grid_reader(self) -> GridIVSurface:
        """The grid adapter this surface reads through, built once."""
        cached = self._reader
        if cached is None:
            cached = GridIVSurface(
                self.to_grid(),
                policy=str(self.policy),
                extrapolation=self.extrapolation,
            )
            object.__setattr__(self, "_reader", cached)
        return cached


@dataclass(frozen=True, slots=True)
class FactorSurfaceCalibrator:
    """Projects one set of observations onto a fixed basis by least squares.

    Parameters
    ----------
    basis:
        The :class:`SurfaceFactorBasis` to project onto.  It is configuration,
        not state: the calibrator never learns it and never updates it, so two
        fits on two days share nothing and the second cannot depend on the
        first.
    policy, extrapolation:
        Passed to every :class:`FactorIVSurface` this returns, and
        ``extrapolation`` also governs the reads that build the design matrix:
        under the default ``'invalid'`` a quote outside the reference mesh is
        an excluded row, and under ``'error'`` it is a hard failure.
    use_weights:
        Whether to honour the observations' ``weight`` column as least-squares
        weights.  Rows of zero weight are excluded rather than merely
        down-weighted, so they cannot contribute to the rank of the design.

    Notes
    -----
    Satisfies :class:`~fast_vollib.surface.protocols.SurfaceCalibrator`.

    *The design matrix is the model's own linear map.*  Each column is one
    loading interpolated from the reference grid to the observation points, and
    the target is the observed value in the basis's space minus the mean
    interpolated the same way.  Because
    :meth:`FactorIVSurface.evaluate` reads the reconstruction back through the
    same interpolator, a fully observed surface projects to the scores that
    reproduce it, rather than to the scores of some nearby smoothing.

    *A basis speaks only for its own grid.*  Observations with no implied
    volatility, with zero weight, or lying outside the reference mesh are
    excluded from the least squares; if too few remain, or if those that remain
    cannot tell two factors apart, the fit raises rather than returning scores
    that split arbitrarily between indistinguishable directions.

    Examples
    --------
    >>> import numpy as np
    >>> from fast_vollib.surface import SurfaceGridSpec, SurfaceObservations
    >>> from fast_vollib.surface.fitting.factors import (
    ...     FactorSurfaceCalibrator, SurfaceFactorBasis,
    ... )
    >>> basis = SurfaceFactorBasis(
    ...     grid=SurfaceGridSpec(k=[-0.1, 0.0, 0.1], T=[1.0]),
    ...     mean=[0.04, 0.04, 0.04],
    ...     loadings=[[-1.0, 0.0, 1.0]] / np.sqrt(2.0),
    ...     singular_values=[0.01],
    ...     explained_variance_ratio=[1.0],
    ... )
    >>> observations = SurfaceObservations(
    ...     k=[-0.1, 0.1], T=[1.0, 1.0], iv=[np.sqrt(0.039), np.sqrt(0.041)]
    ... )
    >>> surface = FactorSurfaceCalibrator(basis=basis).fit(observations)
    >>> float(round(surface.scores[0], 6))
    0.001414
    """

    basis: SurfaceFactorBasis
    policy: str | None = None
    extrapolation: str = "invalid"
    use_weights: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.basis, SurfaceFactorBasis):
            raise SurfaceValidationError(
                f"basis must be a SurfaceFactorBasis; got {type(self.basis).__name__}."
            )
        if self.policy is not None and self.policy not in POLICIES:
            raise SurfaceValidationError(f"policy must be one of {POLICIES}; got {self.policy!r}.")
        if self.extrapolation not in EXTRAPOLATIONS:
            raise SurfaceValidationError(
                f"extrapolation must be one of {EXTRAPOLATIONS}; got {self.extrapolation!r}."
            )
        if self.extrapolation == "clamp":
            raise SurfaceValidationError(
                "extrapolation='clamp' is not available for fitting. Clamping reads an "
                "off-mesh quote at the nearest node's maturity, so the design column is "
                "built at the clamped T while the target is the quote's own -- a two-year "
                "quote is then fitted as if it were a one-year one, and the resulting "
                "scores reproduce neither. Use 'invalid' to exclude such quotes or "
                "'error' to refuse them; clamping remains available on FactorIVSurface, "
                "where the maturity being read is the one asked for."
            )

    def fit(
        self,
        observations: SurfaceObservations,
        *,
        rng: "RNGInput" = None,
    ) -> FactorIVSurface:
        """Project ``observations`` onto :attr:`basis` and return the surface.

        Parameters
        ----------
        observations:
            Scattered quotes in canonical coordinates.  They need not lie on
            the reference grid, but they must lie inside it.
        rng:
            Accepted and unused: a least-squares projection is deterministic,
            and a calibrator that took randomness it did not need would make a
            reproducible fit look like a lucky one.

        Raises
        ------
        SurfaceCalibrationError
            If fewer observations are usable than the basis has components, if
            the usable observations leave two components indistinguishable, or
            if the solve returns a non-finite score.
        """
        del rng  # deterministic
        if not isinstance(observations, SurfaceObservations):
            raise SurfaceValidationError(
                f"observations must be a SurfaceObservations; got {type(observations).__name__}."
            )
        basis = self.basis
        n_factors = basis.n_factors
        points = observations.points
        mean_at = _read_nodes(basis.grid, basis.mean, points, self.extrapolation)
        design = np.column_stack(
            [
                _read_nodes(basis.grid, loading, points, self.extrapolation)
                for loading in basis.loadings
            ]
        )

        usable = ~np.isnan(observations.iv) & np.isfinite(mean_at)
        usable &= np.all(np.isfinite(design), axis=1)
        weight = observations.weight if self.use_weights else None
        if weight is not None:
            usable &= weight > 0.0
        n_used = int(np.count_nonzero(usable))
        if n_used < n_factors:
            raise SurfaceCalibrationError(
                f"A projection onto {n_factors} factor(s) must use at least {n_factors} "
                f"observation(s); got {n_used}. Fewer observations than directions leaves "
                f"the scores undetermined, and the least-squares answer would be one "
                f"arbitrary member of a family that all fit the quotes equally well."
            )

        target = (
            _to_value_space(observations.iv[usable], observations.T[usable], basis.value_space)
            - mean_at[usable]
        )
        matrix = design[usable]
        if weight is not None:
            root = np.sqrt(weight[usable])
            matrix = matrix * root[:, None]
            target = target * root

        scores, _, rank, _ = np.linalg.lstsq(matrix, target, rcond=None)
        if int(rank) < n_factors:
            raise SurfaceCalibrationError(
                f"The observations must resolve all {n_factors} factor(s); the design "
                f"matrix has rank {int(rank)}. Repeated coordinates, or points that see "
                f"only part of the mesh, make two components indistinguishable there, and "
                f"any split of the score between them fits the quotes equally well."
            )
        if not bool(np.all(np.isfinite(scores))):
            raise SurfaceCalibrationError(
                f"The projection returned non-finite scores {scores.tolist()}; the "
                f"observations do not determine a surface this basis stands behind."
            )
        return FactorIVSurface(
            basis=basis,
            scores=scores,
            policy=self.policy,
            extrapolation=self.extrapolation,
        )


def fit_factor_basis(
    surfaces: Sequence[Any],
    *,
    grid: SurfaceGridSpec | None = None,
    n_factors: int | None = None,
    value_space: str = "total_variance",
) -> SurfaceFactorBasis:
    """Build a factor basis from an ensemble of surfaces on one common grid.

    Parameters
    ----------
    surfaces:
        At least two :class:`~fast_vollib.surface.grid.IVSurface` objects, or
        at least two :class:`~fast_vollib.surface.observations.SurfaceObservations`
        already materialized onto ``grid`` -- one row per node, in the grid's
        own C order.  Every member must be complete: a ``NaN`` node is refused
        by name rather than imputed.
    grid:
        The reference mesh.  ``None`` takes it from the first
        :class:`~fast_vollib.surface.grid.IVSurface`; it is required when the
        ensemble is given as observations, which carry rows rather than a mesh.
    n_factors:
        How many components to keep.  ``None`` (the default) keeps the
        ensemble's numerical rank -- the components whose singular value exceeds
        ``max(n_surfaces, n_nodes) * eps * max|X|``, the criterion
        :func:`numpy.linalg.matrix_rank` uses, measured at the scale of the
        *uncentred* ensemble because that is where the subtraction that loses
        the digits happens.  Components below it are directions in the rounding
        error rather than in the data: they differ between LAPACK builds, so a
        score expressed in one is not reproducible and is not offered.
    value_space:
        One of :data:`VALUE_SPACES`.  Defaults to ``'total_variance'``.

    Returns
    -------
    A :class:`SurfaceFactorBasis` whose loadings are the right singular vectors
    of the mean-centred ensemble, each fixed to this module's sign convention.

    Raises
    ------
    SurfaceValidationError
        On an unknown ``value_space``, an ensemble of fewer than two surfaces,
        a member on a different mesh, a member with a missing node, or
        observations given without a grid.
    SurfaceCalibrationError
        When the ensemble does not vary at all, or when more components are
        requested than it resolves.

    Notes
    -----
    The decomposition is a single :func:`numpy.linalg.svd` with
    ``full_matrices=False`` on the ``(n_surfaces, n_nodes)`` centred matrix.  It
    is deterministic and exact to float64: there is no randomized range finder
    and therefore no seed, which is why this function takes no ``rng``.

    Examples
    --------
    >>> import numpy as np
    >>> from fast_vollib.surface import IVSurface
    >>> from fast_vollib.surface.fitting.factors import fit_factor_basis
    >>> k, T = np.linspace(-0.2, 0.2, 5), np.array([1.0])
    >>> ensemble = [
    ...     IVSurface.from_total_variance(k, T, np.full((5, 1), 0.04 + shift))
    ...     for shift in (-0.01, 0.0, 0.01)
    ... ]
    >>> basis = fit_factor_basis(ensemble, n_factors=1)
    >>> float(round(basis.explained_variance_ratio[0], 9))
    1.0
    >>> basis.loadings.round(6).tolist()
    [[0.447214, 0.447214, 0.447214, 0.447214, 0.447214]]
    """
    if value_space not in VALUE_SPACES:
        raise SurfaceValidationError(
            f"value_space must be one of {VALUE_SPACES}; got {value_space!r}."
        )
    if isinstance(surfaces, (str, bytes)) or not isinstance(surfaces, Sequence):
        raise SurfaceValidationError(
            f"surfaces must be a sequence of IVSurface or SurfaceObservations; "
            f"got {type(surfaces).__name__}."
        )
    if len(surfaces) < 2:
        raise SurfaceValidationError(
            f"An ensemble must hold at least 2 surfaces; got {len(surfaces)}. One surface "
            f"is its own mean, so the centred matrix is zero and there is no direction to "
            f"find."
        )
    reference = _reference_grid(surfaces[0], grid)
    matrix = np.stack(
        [
            _node_values(member, index, reference, value_space)
            for index, member in enumerate(surfaces)
        ]
    )

    mean = matrix.mean(axis=0)
    centred = matrix - mean
    _, singular_values, right = np.linalg.svd(centred, full_matrices=False)

    # The numerical-rank criterion of numpy.linalg.matrix_rank, measured at the
    # scale of the *uncentred* ensemble. Centring subtracts numbers of that
    # size, so every entry of the centred matrix carries an absolute error of
    # order eps * max|X| however small the entry itself is -- three bit-for-bit
    # identical surfaces do not centre to zero, because the mean of three equal
    # floats is not exactly either of them. Judging the spectrum against the
    # centred matrix's own norm would hand that rounding error back as a
    # component, sign convention and all.
    resolution = (
        max(centred.shape) * float(np.finfo(np.float64).eps) * float(np.max(np.abs(matrix)))
    )
    rank = int(np.count_nonzero(singular_values > resolution))
    if rank == 0:
        raise SurfaceCalibrationError(
            f"An ensemble must vary by more than the rounding error of its own mean "
            f"({resolution!r}); the largest singular value here is "
            f"{float(singular_values[0])!r}. Every surface is the ensemble mean, so "
            f"there is no direction to decompose, and returning arbitrary orthonormal "
            f"rows would invent structure the data does not contain."
        )
    total = float(np.sum(singular_values**2))
    kept = rank if n_factors is None else n_factors
    if isinstance(kept, bool) or not isinstance(kept, int):
        raise SurfaceValidationError(
            f"n_factors must be an integer or None; got {type(n_factors).__name__}."
        )
    if kept < 1:
        raise SurfaceValidationError(f"n_factors must be at least 1; got {kept}.")
    if kept > rank:
        raise SurfaceCalibrationError(
            f"n_factors must be at most the {rank} component(s) this ensemble resolves; "
            f"got {kept}. Beyond the numerical rank the singular vectors describe the "
            f"decomposition's own rounding error, and they differ between LAPACK builds, "
            f"so a score expressed in one of them would not survive a change of machine."
        )

    loadings = right[:kept]
    loadings = _component_signs(loadings)[:, None] * loadings
    return SurfaceFactorBasis(
        grid=reference,
        mean=mean,
        loadings=loadings,
        singular_values=singular_values[:kept],
        explained_variance_ratio=(singular_values**2 / total)[:kept],
        value_space=value_space,
    )


def _reference_grid(first: Any, grid: SurfaceGridSpec | None) -> SurfaceGridSpec:
    """The mesh the ensemble is decomposed on."""
    if grid is not None:
        if not isinstance(grid, SurfaceGridSpec):
            raise SurfaceValidationError(
                f"grid must be a SurfaceGridSpec or None; got {type(grid).__name__}."
            )
        return grid
    if not isinstance(first, IVSurface):
        raise SurfaceValidationError(
            "grid must be given when the ensemble is supplied as observations; a long "
            "table of rows does not declare a mesh, and guessing one from the distinct "
            "coordinates would silently decide the node order every loading is indexed by."
        )
    axis = np.asarray(first.k, dtype=np.float64)
    return SurfaceGridSpec(
        k=axis,
        T=np.asarray(first.T, dtype=np.float64),
        topology="shared_moneyness" if axis.ndim == 1 else "fixed_strike",
    )


def _node_values(
    member: Any,
    index: int,
    grid: SurfaceGridSpec,
    value_space: str,
) -> np.ndarray:
    """One ensemble member as a complete flat node vector in ``value_space``."""
    if isinstance(member, IVSurface):
        namespace = member.namespace()
        if namespace.name != "numpy":
            raise SurfaceValidationError(
                f"Surface {index} must be a host (numpy) surface; got the "
                f"{namespace.name!r} backend. The decomposition is a host operation, and "
                f"moving a device tensor here silently would break the tape it belongs to."
            )
        iv = np.asarray(member.iv, dtype=np.float64)
        if iv.shape != grid.shape:
            raise SurfaceValidationError(
                f"Surface {index} must be on the same reference grid, of shape "
                f"{grid.shape}; got {iv.shape}. A basis is a set of directions on one "
                f"mesh, and surfaces on different meshes share no node vector to "
                f"decompose."
            )
        if not (
            np.array_equal(np.asarray(member.k, dtype=np.float64), grid.k)
            and np.array_equal(np.asarray(member.T, dtype=np.float64), grid.T)
        ):
            raise SurfaceValidationError(
                f"Surface {index} must be on the same reference grid; its coordinates "
                f"differ from the reference mesh. Node j of one loading would otherwise "
                f"mean a different strike and maturity for each member of the ensemble."
            )
        values = _to_value_space(iv, grid.T2d(), value_space).reshape(-1)
        return _require_complete(values, f"surface {index}")
    if isinstance(member, SurfaceObservations):
        if member.n != grid.n_nodes:
            raise SurfaceValidationError(
                f"Observation set {index} must carry one row per grid node "
                f"({grid.n_nodes}); got {member.n}. Materialize it onto the reference "
                f"grid first -- an incomplete set has no complete node vector."
            )
        if not (
            np.array_equal(member.k, grid.k2d().reshape(-1))
            and np.array_equal(member.T, grid.T2d().reshape(-1))
        ):
            raise SurfaceValidationError(
                f"Observation set {index} must be ordered as the reference grid's own "
                f"nodes, C order over (Nk, Nt); its coordinates differ. Reordering it "
                f"here would put values on the wrong nodes."
            )
        values = _to_value_space(member.iv, member.T, value_space)
        return _require_complete(values, f"observation set {index}")
    raise SurfaceValidationError(
        f"Ensemble member {index} must be an IVSurface or a SurfaceObservations; "
        f"got {type(member).__name__}."
    )


@dataclass(frozen=True, slots=True)
class FactorPCARecipe:
    """The two-phase lifecycle of a factor model, as one configurable object.

    A factor model is not calibrated from one day's quotes: its basis is
    estimated from an ensemble, and only then is a single surface projected onto
    it.  That is two steps with two different inputs, and a calibrator alone
    cannot express it -- which is why the capability registry advertises *this*
    for ``factor-pca`` and marks it as requiring training.

    :meth:`train` estimates the basis from a set of surfaces; :meth:`calibrator`
    turns a trained basis into an ordinary
    :class:`~fast_vollib.surface.protocols.SurfaceCalibrator`.  Neither step
    mutates the recipe, so the same configuration can train two bases from two
    ensembles without either affecting the other.

    Examples
    --------
    >>> import numpy as np
    >>> from fast_vollib.surface import IVSurface, SurfaceObservations
    >>> from fast_vollib.surface.fitting.factors import FactorPCARecipe
    >>> k = np.array([-0.1, 0.0, 0.1])
    >>> T = np.array([1.0])
    >>> ensemble = [
    ...     IVSurface.from_logmoneyness(k, T, np.full((3, 1), level))
    ...     for level in (0.18, 0.20, 0.22)
    ... ]
    >>> recipe = FactorPCARecipe(n_factors=1)
    >>> basis = recipe.train(ensemble)
    >>> basis.n_factors
    1
    """

    n_factors: int = 3
    value_space: str = "total_variance"
    policy: str | None = None
    extrapolation: str = "invalid"
    use_weights: bool = True

    def __post_init__(self) -> None:
        if int(self.n_factors) < 1:
            raise SurfaceValidationError(f"n_factors must be at least 1; got {self.n_factors!r}.")
        if self.value_space not in VALUE_SPACES:
            raise SurfaceValidationError(
                f"value_space must be one of {VALUE_SPACES}; got {self.value_space!r}."
            )
        if self.policy is not None and self.policy not in POLICIES:
            raise SurfaceValidationError(f"policy must be one of {POLICIES}; got {self.policy!r}.")
        if self.extrapolation not in EXTRAPOLATIONS:
            raise SurfaceValidationError(
                f"extrapolation must be one of {EXTRAPOLATIONS}; got {self.extrapolation!r}."
            )
        if self.extrapolation == "clamp":
            raise SurfaceValidationError(
                "extrapolation='clamp' is not available for fitting; see "
                "FactorSurfaceCalibrator for why a clamped maturity fits a quote as a "
                "different quote."
            )

    def train(self, surfaces: Any, *, grid: SurfaceGridSpec | None = None) -> SurfaceFactorBasis:
        """Estimate the basis from an ensemble of materialized surfaces."""
        return fit_factor_basis(
            surfaces,
            n_factors=int(self.n_factors),
            value_space=self.value_space,
            grid=grid,
        )

    def calibrator(self, basis: SurfaceFactorBasis) -> FactorSurfaceCalibrator:
        """The single-surface calibrator that projects onto ``basis``."""
        return FactorSurfaceCalibrator(
            basis=basis,
            policy=self.policy,
            extrapolation=self.extrapolation,
            use_weights=self.use_weights,
        )
