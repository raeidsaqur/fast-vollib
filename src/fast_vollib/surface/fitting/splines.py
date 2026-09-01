"""Penalized tensor-product B-splines on total variance: flexible, and still bounded.

A spline is the natural non-parametric answer to "what surface fits these
quotes".  It is also the natural way to produce a number nobody should believe:
a cubic evaluated one knot-width outside the data it was fitted to diverges
like the cube of the distance, and it does so smoothly, so the number it
returns looks exactly like the numbers it returned inside the data.  This
module is the honest form of that fit -- :class:`SplineIVSurface` carries the
knot span it was built on and declines every point outside it, and
:class:`SplineSurfaceCalibrator` produces one only when the linear algebra
behind it was actually solvable.

*The dependent variable is total variance, not implied volatility.*  The fit is
of :math:`w(k, T) = \\sigma^2 T`, because that is the quantity the conditions
are stated in: the calendar condition is monotonicity of :math:`w` in :math:`T`,
Durrleman's function differentiates :math:`w` in :math:`k`, and :math:`w` is
close to linear in :math:`T` where :math:`\\sigma` is not.  Fitting
:math:`\\sigma` would put the curvature of the square root into the T-direction
of the basis and spend knots on an artifact of the parameterization.  Implied
volatility is recovered on the way out as :math:`\\sigma = \\sqrt{\\max(w, 0)/T}`,
and a coefficient combination that lands on a negative total variance produces
an *invalid* point rather than a NaN that nobody counts.

*The fit does not depend on the order the quotes arrived in.*  Least squares
is an accumulation over rows, and floating-point summation is not associative,
so shuffling a quote file perturbs the last bits of the factorization and,
through a stiff solve, rather more than the last bits of the coefficients.
Rows are therefore sorted into a canonical :math:`(T, k, \\text{weight}, w)`
order before the design matrix is built, which makes the fit *bitwise* a
function of the multiset of observations.  Reproducibility that holds to a
tolerance is a claim about the tolerance; this one is a claim about the data.

*A basis that outnumbers the data is a choice, not an accident.*  Knot counts
chosen automatically are capped so the basis in each direction cannot outnumber
the distinct coordinates that support it, and the degree is reduced when there
are too few distinct maturities to carry it -- two expiries cannot support a
cubic in :math:`T`, and pretending otherwise puts three free coefficients where
there are two numbers.  A caller who asks for a specific interior-knot count
gets it verbatim, because deliberate over-parameterization regularised by the
penalty is a legitimate thing to want; what they do not get is a silent
unpenalised solve of a deficient system, which raises
:class:`~fast_vollib.surface.errors.SurfaceCalibrationError` instead.

The estimator is a P-spline: with :math:`A` the tensor-product design matrix,
:math:`W` the diagonal of observation weights and :math:`D` the second-difference
operator in each direction, the coefficients minimise

.. math::

    \\left\\| W^{1/2} (A c - w) \\right\\|^2
    + \\lambda_k \\left\\| (D_k \\otimes I) c \\right\\|^2
    + \\lambda_T \\left\\| (I \\otimes D_T) c \\right\\|^2.

*The normal equations are never formed.*  That objective is a single linear
least-squares problem in the design matrix stacked on top of
:math:`\\sqrt{\\lambda_k}(D_k \\otimes I)` and
:math:`\\sqrt{\\lambda_T}(I \\otimes D_T)`, and it is solved that way rather
than by a Cholesky factorization of :math:`A^\\top W A + \\lambda P`.  The
accuracy argument for that is real but small here -- a B-spline design matrix is
well-conditioned by construction, the basis being locally supported and nearly
orthogonal, so :math:`\\operatorname{cond}(A)` runs in the tens even at a
hundred coefficients and squaring it costs about one digit rather than half of
them.  The argument that decides it is the other one: :func:`numpy.linalg.lstsq`
*reports the rank it found*, which turns "this basis is not identified by this
data" into a number the calibrator checks, instead of an exception a
factorization may or may not raise.

The basis rows come from :func:`scipy.interpolate.BSpline.design_matrix`;
nothing here re-derives Cox-de Boor, Black prices, densities, or arbitrage
conditions -- those live in :mod:`fast_vollib.surface.transforms`,
:mod:`fast_vollib.surface.arbitrage`, and :mod:`fast_vollib.surface.metrics`,
and a second copy of them would be a second thing to keep correct.

Examples
--------
>>> import numpy as np
>>> from fast_vollib.surface import SurfaceObservations, SurfacePoints
>>> from fast_vollib.surface.fitting.splines import SplineSurfaceCalibrator
>>> k, T = np.meshgrid(np.linspace(-0.3, 0.3, 13), [0.25, 0.5, 1.0, 2.0], indexing="ij")
>>> w = T * (0.04 + 0.02 * k + 0.30 * k**2)
>>> observations = SurfaceObservations(k=k.ravel(), T=T.ravel(), iv=np.sqrt(w / T).ravel())
>>> surface = SplineSurfaceCalibrator(smoothing_k=0.0, smoothing_t=0.0).fit(observations)
>>> surface.degree_k, surface.degree_t
(3, 3)
>>> prediction = surface.evaluate(SurfacePoints(k=[0.0, 5.0], T=[1.0, 1.0]))
>>> prediction.valid.tolist()
[True, False]
>>> float(round(prediction.iv[0], 6))
0.2
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.interpolate import BSpline

from .._validate import owned_float_1d, owned_float_2d
from ..errors import SurfaceCalibrationError, SurfaceValidationError
from ..points import SurfacePoints
from ..prediction import SurfacePrediction

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..market import SurfaceMarket
    from ..observations import SurfaceObservations
    from ..protocols import RNGInput

__all__ = [
    "DEFAULT_MAX_INTERIOR_KNOTS_K",
    "DEFAULT_SMOOTHING",
    "MATURITY_DECIMALS",
    "PENALTY_ORDER",
    "SplineIVSurface",
    "SplineSmileCalibrator",
    "SplineSurfaceCalibrator",
]

#: Upper bound on the number of interior knots placed automatically in the
#: moneyness direction.  Six interior knots is roughly one degree of freedom per
#: two quotes on a liquid single-expiry chain, which is flexible enough for a
#: skew and a wing and short of the point where the fit starts interpolating
#: bid-ask noise.
DEFAULT_MAX_INTERIOR_KNOTS_K = 6

#: Default penalty weight :math:`\\lambda` in each direction.  Small enough to
#: be a *stabiliser* rather than a smoother -- it keeps the stacked system full
#: rank where a basis function is thinly supported -- and large
#: enough to be visible in a recovery test, so exact-reproduction claims set it
#: to zero rather than relying on it being negligible.
DEFAULT_SMOOTHING = 1e-6

#: Decimal places at which two maturities are treated as one expiry, matching
#: :meth:`~fast_vollib.surface.observations.SurfaceObservations.smiles`.  Year
#: fractions that differ in the twelfth digit are one expiry quoted twice, and
#: counting them as two would let a cubic in :math:`T` claim support it does
#: not have.
MATURITY_DECIMALS = 6

#: Difference order of the roughness penalty.  Second differences penalize
#: curvature and leave the bilinear sheets :math:`c_{ab} = \\alpha + \\beta a +
#: \\gamma b + \\delta ab` unpenalized, so an infinite penalty shrinks towards a
#: plane in the coefficients rather than towards zero -- which is the limit a
#: volatility surface should have.
PENALTY_ORDER = 2


@dataclass(frozen=True, slots=True)
class SplineIVSurface:
    """A tensor-product B-spline in total variance, defined on its knot span.

    Parameters
    ----------
    knots_k, knots_t:
        Full (clamped) knot vectors in forward log-moneyness and in maturity,
        each non-decreasing with at least ``2 * degree + 2`` entries.  Maturity
        knots are strictly positive.
    degree_k, degree_t:
        B-spline degrees, non-negative integers.  Three is a cubic.
    coefficients:
        Shape ``(len(knots_k) - degree_k - 1, len(knots_t) - degree_t - 1)``
        coefficients of the tensor-product basis, k-major.

    Returns
    -------
    Nothing; this is a value object.  :meth:`evaluate` returns a
    :class:`~fast_vollib.surface.prediction.SurfacePrediction`.

    Raises
    ------
    SurfaceValidationError
        On a knot vector that is too short for its degree, not non-decreasing,
        or non-positive in maturity, and on a coefficient block whose shape
        disagrees with the knots.

    Notes
    -----
    Satisfies :class:`~fast_vollib.surface.protocols.DefiniteIVSurface`.  The
    domain is *derived* from the knot vectors rather than stored beside them,
    so it cannot drift out of agreement with the basis it describes; outside it
    :meth:`evaluate` marks points invalid instead of extrapolating, because a
    degree-:math:`p` polynomial continued past its last knot grows like the
    :math:`p`-th power of the distance and returns a plausible-looking implied
    volatility that is an artifact of the basis.

    A single observed maturity gives the degenerate span ``[T0, T0]`` with
    degree 0, whose basis is the constant one.  That is the honest domain of a
    smile: it speaks for the expiry it was fitted to and for no other.

    Examples
    --------
    >>> from fast_vollib.surface import SurfacePoints
    >>> from fast_vollib.surface.fitting.splines import SplineIVSurface
    >>> surface = SplineIVSurface(
    ...     knots_k=[-0.2, -0.2, 0.2, 0.2],
    ...     knots_t=[0.5, 0.5, 2.0, 2.0],
    ...     degree_k=1,
    ...     degree_t=1,
    ...     coefficients=[[0.02, 0.08], [0.02, 0.08]],
    ... )
    >>> surface.domain_k, surface.domain_t
    ((-0.2, 0.2), (0.5, 2.0))
    >>> float(round(surface.total_variance(0.0, 1.25)[0], 12))
    0.05
    >>> surface.evaluate(SurfacePoints(k=[0.0], T=[3.0])).valid.tolist()
    [False]
    """

    knots_k: Any
    knots_t: Any
    degree_k: int
    degree_t: int
    coefficients: Any

    def __post_init__(self) -> None:
        degree_k = _validated_degree(self.degree_k, "degree_k")
        degree_t = _validated_degree(self.degree_t, "degree_t")
        knots_k = _validated_knots(self.knots_k, "knots_k", degree_k, positive=False)
        knots_t = _validated_knots(self.knots_t, "knots_t", degree_t, positive=True)
        shape = (knots_k.size - degree_k - 1, knots_t.size - degree_t - 1)
        coefficients = owned_float_2d(self.coefficients, "coefficients", shape)
        if not bool(np.all(np.isfinite(coefficients))):
            raise SurfaceValidationError(
                "coefficients must be finite everywhere; got a non-finite entry. "
                "A non-finite coefficient poisons every evaluation whose basis "
                "functions overlap it, including points far from the one that "
                "produced it."
            )
        object.__setattr__(self, "knots_k", knots_k)
        object.__setattr__(self, "knots_t", knots_t)
        object.__setattr__(self, "degree_k", degree_k)
        object.__setattr__(self, "degree_t", degree_t)
        object.__setattr__(self, "coefficients", coefficients)

    # -- geometry ------------------------------------------------------------
    @property
    def n_basis_k(self) -> int:
        """Number of basis functions in the moneyness direction."""
        return int(self.knots_k.size - self.degree_k - 1)

    @property
    def n_basis_t(self) -> int:
        """Number of basis functions in the maturity direction."""
        return int(self.knots_t.size - self.degree_t - 1)

    @property
    def domain_k(self) -> tuple[float, float]:
        """The closed moneyness interval the surface speaks for."""
        return float(self.knots_k[self.degree_k]), float(self.knots_k[-self.degree_k - 1])

    @property
    def domain_t(self) -> tuple[float, float]:
        """The closed maturity interval the surface speaks for."""
        return float(self.knots_t[self.degree_t]), float(self.knots_t[-self.degree_t - 1])

    # -- evaluation ----------------------------------------------------------
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
            The coordinates to answer.
        market:
            Accepted and unused: a spline in :math:`(k, T)` is defined without
            reference to a forward curve or a discount factor.

        Returns
        -------
        A :class:`~fast_vollib.surface.prediction.SurfacePrediction` whose
        ``valid`` entry is ``False`` at every point outside the knot span (with
        ``iv`` reported as ``NaN``) and at every point whose fitted total
        variance is not strictly positive (with ``iv`` reported as ``0``, the
        value :math:`\\sqrt{\\max(w, 0)/T}` takes there).

        Raises
        ------
        SurfaceValidationError
            If ``points`` is not a
            :class:`~fast_vollib.surface.points.SurfacePoints`.
        """
        del market  # a spline in (k, T) is defined without a market state
        if not isinstance(points, SurfacePoints):
            raise SurfaceValidationError(
                f"points must be a SurfacePoints; got {type(points).__name__}. "
                f"The container carries the identity a prediction is aligned on, "
                f"which a bare pair of arrays cannot."
            )
        inside = self._inside(points.k, points.T)
        iv = np.full(points.n, np.nan, dtype=np.float64)
        valid = np.zeros(points.n, dtype=bool)
        if bool(np.any(inside)):
            variance = self._total_variance(points.k[inside], points.T[inside])
            iv[inside] = np.sqrt(np.maximum(variance, 0.0) / points.T[inside])
            valid[inside] = variance > 0.0
        return SurfacePrediction(points=points, iv=iv, valid=valid)

    def total_variance(self, k: Any, T: Any) -> np.ndarray:
        """Fitted total variance :math:`w = \\sigma^2 T`, ``NaN`` outside the domain.

        Parameters
        ----------
        k, T:
            Forward log-moneyness and maturity, broadcastable against each
            other.

        Returns
        -------
        An array of the broadcast shape (at least one-dimensional), holding
        :math:`w` where the point lies in the knot span and ``NaN`` elsewhere.

        Examples
        --------
        >>> from fast_vollib.surface.fitting.splines import SplineIVSurface
        >>> surface = SplineIVSurface(
        ...     knots_k=[0.0, 0.0, 1.0, 1.0],
        ...     knots_t=[1.0, 1.0, 2.0, 2.0],
        ...     degree_k=1,
        ...     degree_t=1,
        ...     coefficients=[[0.04, 0.08], [0.04, 0.08]],
        ... )
        >>> surface.total_variance([0.5, 9.0], 1.0).tolist()
        [0.04, nan]
        """
        moneyness = np.atleast_1d(np.asarray(k, dtype=np.float64))
        maturity = np.atleast_1d(np.asarray(T, dtype=np.float64))
        moneyness, maturity = np.broadcast_arrays(moneyness, maturity)
        flat_k = np.ascontiguousarray(moneyness).reshape(-1)
        flat_t = np.ascontiguousarray(maturity).reshape(-1)
        out = np.full(flat_k.size, np.nan, dtype=np.float64)
        inside = self._inside(flat_k, flat_t)
        if bool(np.any(inside)):
            out[inside] = self._total_variance(flat_k[inside], flat_t[inside])
        return out.reshape(moneyness.shape)

    def roughness(self) -> float:
        """The penalized roughness: summed squared second differences of the coefficients.

        Returns
        -------
        :math:`\\|(D_k \\otimes I) c\\|^2 + \\|(I \\otimes D_T) c\\|^2`, the exact
        quantity :attr:`SplineSurfaceCalibrator.smoothing_k` and
        :attr:`SplineSurfaceCalibrator.smoothing_t` multiply, so a caller can
        confirm what raising them bought.
        """
        across_k = np.diff(self.coefficients, n=PENALTY_ORDER, axis=0)
        across_t = np.diff(self.coefficients, n=PENALTY_ORDER, axis=1)
        return float((across_k * across_k).sum() + (across_t * across_t).sum())

    def parameters(self) -> dict[str, Any]:
        """The fitted parameters as a JSON-safe mapping.

        Returns
        -------
        Exactly the constructor's keyword arguments, as plain lists and ints, so
        ``SplineIVSurface(**json.loads(json.dumps(surface.parameters())))``
        rebuilds the same surface.  Nothing derived is included: a stored
        domain or basis count could be edited into disagreement with the knots
        it was supposed to describe.

        Examples
        --------
        >>> import json
        >>> from fast_vollib.surface.fitting.splines import SplineIVSurface
        >>> surface = SplineIVSurface(
        ...     knots_k=[0.0, 0.0, 1.0, 1.0],
        ...     knots_t=[1.0, 1.0, 2.0, 2.0],
        ...     degree_k=1,
        ...     degree_t=1,
        ...     coefficients=[[0.04, 0.08], [0.04, 0.08]],
        ... )
        >>> sorted(surface.parameters())
        ['coefficients', 'degree_k', 'degree_t', 'knots_k', 'knots_t']
        >>> rebuilt = SplineIVSurface(**json.loads(json.dumps(surface.parameters())))
        >>> bool((rebuilt.coefficients == surface.coefficients).all())
        True
        """
        return {
            "knots_k": [float(value) for value in self.knots_k],
            "knots_t": [float(value) for value in self.knots_t],
            "degree_k": int(self.degree_k),
            "degree_t": int(self.degree_t),
            "coefficients": [[float(value) for value in row] for row in self.coefficients],
        }

    # -- internals -----------------------------------------------------------
    def _inside(self, k: np.ndarray, T: np.ndarray) -> np.ndarray:
        """Boolean mask of the points lying in the closed knot span."""
        k_lo, k_hi = self.domain_k
        t_lo, t_hi = self.domain_t
        return (k >= k_lo) & (k <= k_hi) & (T >= t_lo) & (T <= t_hi)

    def _total_variance(self, k: np.ndarray, T: np.ndarray) -> np.ndarray:
        """Total variance at points already known to lie in the knot span."""
        k_lo, k_hi = self.domain_k
        t_lo, t_hi = self.domain_t
        basis_k = _basis_rows(np.clip(k, k_lo, k_hi), self.knots_k, self.degree_k)
        basis_t = _basis_rows(np.clip(T, t_lo, t_hi), self.knots_t, self.degree_t)
        return np.einsum("ia,ab,ib->i", basis_k, self.coefficients, basis_t)


@dataclass(frozen=True, slots=True)
class SplineSurfaceCalibrator:
    """Fits one penalized tensor-product spline in total variance to observations.

    Parameters
    ----------
    degree_k, degree_t:
        Requested B-spline degrees, defaulting to cubic in both directions.
        Each is *reduced* to ``n_distinct - 1`` when the observations carry too
        few distinct coordinates to support it, so two expiries give a linear
        term structure and one expiry gives a constant.  The reduction is
        reported on the fitted surface's :attr:`SplineIVSurface.degree_t`
        rather than logged, because the degree that ran is part of the fit.
    n_interior_knots_k, n_interior_knots_t:
        Interior knot counts, or ``None`` (default) to choose them from the
        data.  Automatic counts are capped at ``n_distinct - degree - 1`` --
        and, in :math:`k`, at :data:`DEFAULT_MAX_INTERIOR_KNOTS_K` -- so the
        basis never outnumbers the coordinates supporting it.  An explicit
        count is honoured verbatim: deliberate over-parameterization leaning on
        the penalty is a legitimate request, and refusing it here would make
        the penalty untestable.
    smoothing_k, smoothing_t:
        The penalty weights :math:`\\lambda_k, \\lambda_T` on the squared
        second differences of the coefficients, finite and non-negative.
        Default :data:`DEFAULT_SMOOTHING`; set both to zero for an unpenalised
        least-squares fit.
    use_weights:
        Whether to honour the observations' ``weight`` column.  A calibrator
        that ignored supplied weights without saying so would make a weighted
        experiment silently unweighted.

    Returns
    -------
    Nothing; :meth:`fit` returns a :class:`SplineIVSurface`.

    Raises
    ------
    SurfaceValidationError
        On a negative degree, a negative knot count, or a negative or
        non-finite smoothing weight.

    Notes
    -----
    Satisfies :class:`~fast_vollib.surface.protocols.SurfaceCalibrator`.  Holds
    configuration only -- knots, degrees, and coefficients belong to the
    surface that comes out, never to the object that made it -- so two fits on
    two days share nothing and the second cannot depend on the first.

    Knot placement is quantile-spaced on the observed :math:`k`, which puts an
    equal number of quotes between consecutive knots and therefore puts the
    spline's freedom where the data is.  In :math:`T` the interior knots are
    quantiles of the *distinct* maturities, because an expiry with three
    hundred strikes and an expiry with twenty are one pillar each.  When the
    quantiles are not strictly increasing -- a handful of repeated strikes --
    the placement falls back to uniform spacing across the observed range: a
    repeated knot raises the multiplicity and drops a derivative there, which
    is a modelling statement nobody made.

    Examples
    --------
    >>> import numpy as np
    >>> from fast_vollib.surface import SurfaceObservations, SurfacePoints
    >>> from fast_vollib.surface.fitting.splines import SplineSurfaceCalibrator
    >>> k, T = np.meshgrid(np.linspace(-0.2, 0.2, 9), [0.5, 1.0], indexing="ij")
    >>> observations = SurfaceObservations(k=k.ravel(), T=T.ravel(), iv=np.full(18, 0.2))
    >>> surface = SplineSurfaceCalibrator(smoothing_k=0.0, smoothing_t=0.0).fit(observations)
    >>> surface.degree_t  # two expiries cannot carry a cubic in T
    1
    >>> float(round(surface.evaluate(SurfacePoints(k=[0.1], T=[0.75])).iv[0], 9))
    0.2
    """

    degree_k: int = 3
    degree_t: int = 3
    n_interior_knots_k: int | None = None
    n_interior_knots_t: int | None = None
    smoothing_k: float = DEFAULT_SMOOTHING
    smoothing_t: float = DEFAULT_SMOOTHING
    use_weights: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "degree_k", _validated_degree(self.degree_k, "degree_k"))
        object.__setattr__(self, "degree_t", _validated_degree(self.degree_t, "degree_t"))
        object.__setattr__(
            self,
            "n_interior_knots_k",
            _validated_count(self.n_interior_knots_k, "n_interior_knots_k"),
        )
        object.__setattr__(
            self,
            "n_interior_knots_t",
            _validated_count(self.n_interior_knots_t, "n_interior_knots_t"),
        )
        object.__setattr__(
            self, "smoothing_k", _validated_smoothing(self.smoothing_k, "smoothing_k")
        )
        object.__setattr__(
            self, "smoothing_t", _validated_smoothing(self.smoothing_t, "smoothing_t")
        )

    def fit(
        self,
        observations: "SurfaceObservations",
        *,
        rng: "RNGInput" = None,
    ) -> SplineIVSurface:
        """Calibrate a spline to ``observations`` and return it.

        Parameters
        ----------
        observations:
            Scattered quotes for **one** surface.  Rows whose implied
            volatility is missing, and rows whose weight is zero when weights
            are honoured, are dropped before anything is measured.
        rng:
            Accepted and unused.  A penalized least-squares solve is
            deterministic, and the signature says so rather than leaving a
            caller to discover it.

        Returns
        -------
        A :class:`SplineIVSurface` whose knot span is the observed
        :math:`(k, T)` range.

        Raises
        ------
        SurfaceCalibrationError
            If no observation is usable; if the problem is unpenalised and has
            fewer usable observations than basis functions; or if the stacked
            design and penalty do not have full column rank, which is what a
            basis function with no observation under it and no penalty over it
            looks like to the solver -- the case in which the normal equations
            would have been singular.
        """
        del rng  # a penalized least-squares solve draws no randomness
        k, maturity, variance, weight = _usable_rows(observations, self.use_weights)
        if k.size == 0:
            raise SurfaceCalibrationError(
                "A spline fit must have at least one usable observation; got none. "
                "Every implied volatility is missing or every weight is zero, so "
                "there is no surface to stand behind."
            )
        distinct_k = np.unique(k)
        distinct_t = np.unique(np.round(maturity, MATURITY_DECIMALS))
        degree_k = _reduced_degree(self.degree_k, distinct_k.size)
        degree_t = _reduced_degree(self.degree_t, distinct_t.size)
        count_k = self.n_interior_knots_k
        if count_k is None:
            count_k = _automatic_count(distinct_k.size, degree_k, DEFAULT_MAX_INTERIOR_KNOTS_K)
        count_t = self.n_interior_knots_t
        if count_t is None:
            count_t = _automatic_count(distinct_t.size, degree_t, None)
        knots_k = _knot_vector(k, degree_k, count_k, float(distinct_k[0]), float(distinct_k[-1]))
        knots_t = _knot_vector(
            distinct_t, degree_t, count_t, float(maturity.min()), float(maturity.max())
        )

        n_k = knots_k.size - degree_k - 1
        n_t = knots_t.size - degree_t - 1
        n_basis = int(n_k * n_t)
        unpenalised = self.smoothing_k <= 0.0 and self.smoothing_t <= 0.0
        if unpenalised and k.size < n_basis:
            raise SurfaceCalibrationError(
                f"An unpenalised spline fit must have at least as many usable "
                f"observations as basis functions ({n_basis}); got {k.size}. With no "
                f"penalty there is nothing to resolve the deficient directions, so "
                f"the normal equations have a null space and the coefficients that "
                f"come out of them are arbitrary rather than fitted."
            )

        basis_k = _basis_rows(k, knots_k, degree_k)
        basis_t = _basis_rows(maturity, knots_t, degree_t)
        design = (basis_k[:, :, None] * basis_t[:, None, :]).reshape(k.size, n_basis)
        root_weight = np.sqrt(weight)
        blocks = [design * root_weight[:, None]]
        targets = [root_weight * variance]
        for penalty, across_moneyness in ((self.smoothing_k, True), (self.smoothing_t, False)):
            if penalty <= 0.0:
                continue
            operator = _penalty_operator(n_k, n_t, across_moneyness=across_moneyness)
            if operator.shape[0] == 0:
                continue
            blocks.append(np.sqrt(penalty) * operator)
            targets.append(np.zeros(operator.shape[0], dtype=np.float64))

        coefficients, _, rank, _ = np.linalg.lstsq(
            np.vstack(blocks), np.concatenate(targets), rcond=None
        )
        if int(rank) < n_basis:
            raise SurfaceCalibrationError(
                f"The stacked design and penalty must have full column rank "
                f"{n_basis}; got rank {int(rank)}. Some basis function has no "
                f"observation under it and no penalty over it, so its coefficient is "
                f"unidentified and the normal equations for it would be singular; add "
                f"knots where the quotes are, or raise smoothing_k / smoothing_t."
            )
        if not bool(np.all(np.isfinite(coefficients))):
            raise SurfaceCalibrationError(
                "The solved coefficients must be finite everywhere; got a non-finite "
                "entry. The system was reported full rank but is numerically "
                "degenerate, and a surface built from it would report rounding error "
                "as a volatility."
            )
        return SplineIVSurface(
            knots_k=knots_k,
            knots_t=knots_t,
            degree_k=degree_k,
            degree_t=degree_t,
            coefficients=coefficients.reshape(n_k, n_t),
        )


@dataclass(frozen=True, slots=True)
class SplineSmileCalibrator:
    """Fits one penalized spline across strike, at a single maturity.

    Parameters
    ----------
    degree:
        Requested B-spline degree in :math:`k`, reduced when there are too few
        distinct strikes to support it.
    n_interior_knots:
        Interior knot count in :math:`k`, or ``None`` to choose it from the
        data; see :class:`SplineSurfaceCalibrator`.
    smoothing:
        Penalty weight :math:`\\lambda_k` on the squared second differences.
    use_weights:
        Whether to honour the observations' ``weight`` column.

    Returns
    -------
    Nothing; :meth:`fit` returns a :class:`SplineIVSurface` whose maturity
    domain is the single point it was fitted at.

    Raises
    ------
    SurfaceValidationError
        On a negative degree, knot count, or smoothing weight.

    Notes
    -----
    Satisfies :class:`~fast_vollib.surface.protocols.SurfaceCalibrator`, and is
    the surface calibrator with ``degree_t = 0`` -- not a separate estimator.
    What it adds is a *refusal*: given quotes at more than one expiry it raises
    rather than pooling them, because a curve fitted through two term
    structures at once belongs to neither, and the pooling would be invisible
    in the returned object.  Split the observations with
    :meth:`~fast_vollib.surface.observations.SurfaceObservations.smiles` first,
    or use :class:`SplineSurfaceCalibrator`.

    The resulting surface declines every maturity but the one it saw.  That is
    not a limitation being worked around; it is the whole content of a smile.

    Examples
    --------
    >>> import numpy as np
    >>> from fast_vollib.surface import SurfaceObservations, SurfacePoints
    >>> from fast_vollib.surface.fitting.splines import SplineSmileCalibrator
    >>> k = np.linspace(-0.25, 0.25, 11)
    >>> observations = SurfaceObservations(k=k, T=np.full(11, 0.5), iv=0.2 + 0.4 * k**2)
    >>> smile = SplineSmileCalibrator(smoothing=0.0).fit(observations)
    >>> smile.degree_t, smile.n_basis_t
    (0, 1)
    >>> smile.evaluate(SurfacePoints(k=[0.0, 0.0], T=[0.5, 1.0])).valid.tolist()
    [True, False]
    """

    degree: int = 3
    n_interior_knots: int | None = None
    smoothing: float = DEFAULT_SMOOTHING
    use_weights: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "degree", _validated_degree(self.degree, "degree"))
        object.__setattr__(
            self, "n_interior_knots", _validated_count(self.n_interior_knots, "n_interior_knots")
        )
        object.__setattr__(self, "smoothing", _validated_smoothing(self.smoothing, "smoothing"))

    def fit(
        self,
        observations: "SurfaceObservations",
        *,
        rng: "RNGInput" = None,
    ) -> SplineIVSurface:
        """Calibrate a single-maturity smile to ``observations``.

        Raises
        ------
        SurfaceCalibrationError
            If the usable observations do not all share one maturity, or if the
            underlying surface fit fails for any of its own reasons.
        """
        del rng  # a penalized least-squares solve draws no randomness
        _, maturity, _, _ = _usable_rows(observations, self.use_weights)
        expiries = np.unique(np.round(maturity, MATURITY_DECIMALS))
        if expiries.size != 1:
            raise SurfaceCalibrationError(
                f"A smile fit must be given exactly one maturity; got "
                f"{expiries.size} ({expiries.tolist()[:5]}). Pooling several "
                f"expiries into one curve would produce a smile belonging to none "
                f"of them, and the returned surface would carry no record that it "
                f"had happened."
            )
        return SplineSurfaceCalibrator(
            degree_k=self.degree,
            degree_t=0,
            n_interior_knots_k=self.n_interior_knots,
            n_interior_knots_t=0,
            smoothing_k=self.smoothing,
            smoothing_t=0.0,
            use_weights=self.use_weights,
        ).fit(observations)


# --- validation helpers ------------------------------------------------------


def _validated_degree(value: Any, name: str) -> int:
    """A non-negative, non-boolean integer degree."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise SurfaceValidationError(
            f"{name} must be an integer; got {type(value).__name__}. A fractional "
            f"degree names no B-spline basis."
        )
    if int(value) < 0:
        raise SurfaceValidationError(
            f"{name} must be non-negative; got {int(value)}. Degree zero is already "
            f"the piecewise constant, below which there is nothing."
        )
    return int(value)


def _validated_count(value: Any, name: str) -> int | None:
    """A non-negative, non-boolean integer knot count, or ``None`` for automatic."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise SurfaceValidationError(
            f"{name} must be an integer or None; got {type(value).__name__}. None "
            f"means the count is chosen from the data, which is not the same "
            f"request as any particular number."
        )
    if int(value) < 0:
        raise SurfaceValidationError(
            f"{name} must be non-negative; got {int(value)}. Zero interior knots is "
            f"a single polynomial piece, which is the sparsest basis there is."
        )
    return int(value)


def _validated_smoothing(value: Any, name: str) -> float:
    """A finite, non-negative penalty weight."""
    number = float(value)
    if not np.isfinite(number):
        raise SurfaceValidationError(
            f"{name} must be finite; got {value!r}. An infinite penalty is a request "
            f"for the unpenalized limit's null space, which is a different estimator "
            f"and should be asked for as one."
        )
    if number < 0.0:
        raise SurfaceValidationError(
            f"{name} must be non-negative; got {number!r}. A negative weight rewards "
            f"roughness and makes the objective unbounded below."
        )
    return number


def _validated_knots(value: Any, name: str, degree: int, *, positive: bool) -> np.ndarray:
    """An owned, read-only, non-decreasing knot vector long enough for ``degree``."""
    knots = owned_float_1d(value, name)
    required = 2 * degree + 2
    if knots.size < required:
        raise SurfaceValidationError(
            f"{name} must hold at least 2 * degree + 2 = {required} entries; got "
            f"{knots.size}. A clamped spline of degree {degree} repeats each end "
            f"knot {degree + 1} times before it spans a single basis function."
        )
    if not bool(np.all(np.isfinite(knots))):
        raise SurfaceValidationError(
            f"{name} must be finite everywhere; got a non-finite knot. An infinite "
            f"knot describes no interval, so no basis function is supported on it."
        )
    if not bool(np.all(np.diff(knots) >= 0.0)):
        raise SurfaceValidationError(
            f"{name} must be non-decreasing; got {knots.tolist()[:6]}. Cox-de Boor "
            f"reads the knots as interval boundaries, and an out-of-order pair names "
            f"an interval of negative width."
        )
    if positive and not bool(np.all(knots > 0.0)):
        raise SurfaceValidationError(
            f"{name} must be strictly positive; got {knots.tolist()[:6]}. A maturity "
            f"of zero has no total variance to fit and no volatility to recover from "
            f"one."
        )
    return knots


# --- fitting internals -------------------------------------------------------


def _usable_rows(
    observations: "SurfaceObservations", use_weights: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Usable ``(k, T, w, weight)`` rows in canonical order.

    Rows with a missing implied volatility -- and, when weights are honoured,
    rows with zero weight -- are dropped, the total variance :math:`w =
    \\sigma^2 T` is formed, and the survivors are sorted on
    :math:`(T, k, \\text{weight}, w)`.  The sort is what makes the fit bitwise
    independent of the order the observations were supplied in: identical rows
    are interchangeable, so the sorted arrays are a function of the multiset
    alone, and every floating-point accumulation downstream sees the same
    summation order.
    """
    observed = ~np.isnan(observations.iv)
    if use_weights and observations.weight is not None:
        usable = observed & (observations.weight > 0.0)
        weight = np.array(observations.weight[usable], dtype=np.float64)
    else:
        usable = observed
        weight = np.ones(int(np.count_nonzero(usable)), dtype=np.float64)
    k = np.array(observations.k[usable], dtype=np.float64)
    maturity = np.array(observations.T[usable], dtype=np.float64)
    iv = np.array(observations.iv[usable], dtype=np.float64)
    variance = iv * iv * maturity
    order = np.lexsort((variance, weight, k, maturity))
    return k[order], maturity[order], variance[order], weight[order]


def _reduced_degree(requested: int, n_distinct: int) -> int:
    """``requested``, lowered to what ``n_distinct`` distinct coordinates can carry.

    A clamped spline of degree :math:`p` with no interior knots already spans
    :math:`p + 1` basis functions, so :math:`p \\le n - 1` is the condition for
    the data to outnumber the coefficients.  Two maturities give a linear term
    structure; one gives a constant.
    """
    return max(0, min(int(requested), int(n_distinct) - 1))


def _automatic_count(n_distinct: int, degree: int, cap: int | None) -> int:
    """Interior knots to place when the caller did not choose a number."""
    count = max(0, int(n_distinct) - int(degree) - 1)
    return count if cap is None else min(count, int(cap))


def _interior_knots(sample: np.ndarray, count: int, lo: float, hi: float) -> np.ndarray:
    """``count`` strictly increasing interior knots inside ``(lo, hi)``."""
    if count <= 0 or not (hi > lo):
        return np.zeros(0, dtype=np.float64)
    probabilities = np.linspace(0.0, 1.0, count + 2)[1:-1]
    knots = np.quantile(np.asarray(sample, dtype=np.float64), probabilities)
    knots = knots[(knots > lo) & (knots < hi)]
    if knots.size != count or (count > 1 and not bool(np.all(np.diff(knots) > 0.0))):
        knots = np.linspace(lo, hi, count + 2)[1:-1]
    return np.ascontiguousarray(knots, dtype=np.float64)


def _knot_vector(sample: np.ndarray, degree: int, count: int, lo: float, hi: float) -> np.ndarray:
    """A clamped knot vector spanning ``[lo, hi]`` with ``count`` interior knots."""
    interior = _interior_knots(sample, count, lo, hi)
    end = degree + 1
    return np.concatenate(
        [np.full(end, lo, dtype=np.float64), interior, np.full(end, hi, dtype=np.float64)]
    )


def _basis_rows(x: np.ndarray, knots: np.ndarray, degree: int) -> np.ndarray:
    """Dense B-spline basis rows: entry ``(i, a)`` is :math:`B_a(x_i)`.

    Delegates to :meth:`scipy.interpolate.BSpline.design_matrix`, which
    evaluates Cox-de Boor's recursion once and correctly; the only thing added
    here is the empty-input case, which the sparse constructor does not accept,
    and densification, because the tensor product below is dense anyway at the
    tens-of-basis-functions scale an option surface has.
    """
    n_basis = knots.size - degree - 1
    if x.size == 0:
        return np.zeros((0, n_basis), dtype=np.float64)
    matrix = BSpline.design_matrix(x, knots, degree, extrapolate=False)
    return np.asarray(matrix.toarray(), dtype=np.float64)


def _penalty_operator(n_k: int, n_t: int, *, across_moneyness: bool) -> np.ndarray:
    """The operator whose squared norm one smoothing weight multiplies.

    :math:`D_k \\otimes I` in the moneyness direction and :math:`I \\otimes D_T`
    in the maturity direction, laid out for the k-major flattening of the
    coefficient block.  Stacking these under the design matrix is what turns
    the penalized objective into an ordinary least-squares problem.
    """
    if across_moneyness:
        return np.kron(_difference_matrix(n_k), np.eye(n_t, dtype=np.float64))
    return np.kron(np.eye(n_k, dtype=np.float64), _difference_matrix(n_t))


def _difference_matrix(size: int) -> np.ndarray:
    """The order-:data:`PENALTY_ORDER` difference operator on ``size`` coefficients.

    Degenerate by design when the basis is shorter than the difference order:
    a two-coefficient direction has no second difference, so the penalty in it
    is zero rather than an error.  There is nothing there to be rough.
    """
    if size <= PENALTY_ORDER:
        return np.zeros((0, size), dtype=np.float64)
    return np.diff(np.eye(size, dtype=np.float64), n=PENALTY_ORDER, axis=0)
