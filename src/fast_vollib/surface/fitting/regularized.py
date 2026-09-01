"""Quadratic penalties and parameter-path coupling, with no model inside them.

Every surface fitter richer than a constant has more freedom than a day of
quotes can pin down: a spline with more knots than usable strikes, a polynomial
or factor basis evaluated over a wide log-moneyness range, a parameterization
whose Jacobian is nearly degenerate at the point it converged to.  The cure
always has the same shape -- add a quadratic penalty that says what "not wiggly"
means, and solve the enlarged least-squares problem -- and that shape knows
nothing about SVI, Heston, splines, tickers, or dates.  So it is written once,
here, and the model-specific fitters import it instead of each growing a
private copy that drifts.

The module is deliberately asset-agnostic.  There is no Black price, no total
variance and no arbitrage condition in it; those live in
:mod:`fast_vollib.surface.transforms` and :mod:`fast_vollib.surface.arbitrage`
and are not re-derived.  What is here is linear algebra with a contract.

*The penalized problem is solved by stacking, never by normal equations.*
:func:`solve_penalized_least_squares` minimizes
:math:`\\|W^{1/2}(Ac - y)\\|^2 + \\sum_i \\lambda_i\\|L_ic\\|^2` by handing
:func:`numpy.linalg.lstsq` the stacked design
:math:`[W^{1/2}A;\\ \\sqrt{\\lambda_1}L_1;\\ \\dots]` against the stacked target
:math:`[W^{1/2}y;\\ 0;\\ \\dots]`.  Assembling
:math:`A^{\\top}WA + \\sum_i \\lambda_iL_i^{\\top}L_i` instead would square the
condition number: a design conditioned at :math:`10^8` -- ordinary for a
polynomial basis over a wide strike range -- becomes :math:`10^{16}`, at which
point float64 has no digits left and the returned coefficients are noise shaped
like a fit.  The stacked factorization costs a constant factor more and keeps
every digit the quotes actually support.  The measured gap on the fixture in
the tests is seven orders of magnitude in the recovered coefficients.

That is the failure mode this module exists to prevent: a fit that returns a
plausible coefficient vector from a numerically dead system, with nothing in
the return value to say so.  Every solve therefore comes back with a
:class:`LeastSquaresDiagnostics` record -- residual norm, penalty norm,
effective rank, condition number -- so a caller learns that the system was rank
deficient here, rather than inferring it from a strange-looking surface three
steps downstream.

:func:`couple_parameter_sequence` is the same idea applied *across* surfaces
rather than within one.  Parameters fitted independently to consecutive days
jump when the quote set changes rather than when the market does; coupling
penalizes differences along the sequence index and returns a path that moves
only as much as the data insists on.  It is generic in the same way: it never
learns what the parameters mean, so an SVI path, a factor-score path, and a
single scalar level are all served by the same operator.

Examples
--------
>>> import numpy as np
>>> from fast_vollib.surface.fitting.regularized import (
...     TikhonovPenalty,
...     couple_parameter_sequence,
...     solve_penalized_least_squares,
... )
>>> x = np.linspace(-1.0, 1.0, 5)
>>> design = np.column_stack([np.ones_like(x), x])
>>> coefficients, diagnostics = solve_penalized_least_squares(design, 2.0 + 3.0 * x)
>>> [float(round(value, 9)) for value in coefficients]
[2.0, 3.0]
>>> float(round(diagnostics.residual_norm, 9))
0.0

Shrinking that fit toward zero, and smoothing a jagged parameter path:

>>> shrunk, _ = solve_penalized_least_squares(
...     design, 2.0 + 3.0 * x, penalties=[TikhonovPenalty.ridge(2, 1.0)]
... )
>>> bool(np.linalg.norm(shrunk) < np.linalg.norm(coefficients))
True
>>> path = [0.20, 0.30, 0.20, 0.30, 0.20]
>>> [float(round(value, 4)) for value in couple_parameter_sequence(path, weight=1e12)]
[0.24, 0.24, 0.24, 0.24, 0.24]
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np

from .._validate import owned_float_2d
from ..errors import SurfaceCalibrationError, SurfaceValidationError

__all__ = [
    "MAX_DIFFERENCE_ORDER",
    "LeastSquaresDiagnostics",
    "TikhonovPenalty",
    "couple_parameter_sequence",
    "difference_matrix",
    "solve_penalized_least_squares",
]

#: Highest difference order whose binomial coefficients are still exact in
#: float64: :math:`\\binom{56}{28} < 2^{53} < \\binom{57}{28}`.  Past this the
#: operator would silently stop being the integer stencil it advertises, and
#: the annihilation property below would hold only to rounding.
MAX_DIFFERENCE_ORDER = 56


def difference_matrix(n: int, order: int = 1) -> np.ndarray:
    """The finite-difference operator of a given order on ``n`` points.

    Parameters
    ----------
    n:
        Number of points the operator acts on, at least one.
    order:
        Difference order.  ``1`` gives first differences
        :math:`x_{i+1} - x_i`; ``2`` gives second differences
        :math:`x_i - 2x_{i+1} + x_{i+2}`; ``0`` gives the identity, which is
        the ridge operator.

    Returns
    -------
    numpy.ndarray
        Shape ``(n - order, n)``, float64.  Row :math:`i` carries
        :math:`(-1)^{p-j}\\binom{p}{j}` in column :math:`i + j` for
        :math:`j = 0, \\dots, p`, and zero elsewhere.  A fresh, writable array
        on every call -- nothing here is cached, so a caller may scale or
        stack it in place without reaching another caller's matrix.

    Raises
    ------
    SurfaceValidationError
        If ``n`` or ``order`` is not an integer; if either is negative; if
        ``order >= n``, which would leave the operator with no rows; or if
        ``order`` exceeds :data:`MAX_DIFFERENCE_ORDER`.

    Notes
    -----
    The stencil is built by convolving ``[-1, 1]`` with itself ``order`` times
    in ``int64`` and cast to float64 once, so the entries are the exact
    binomial coefficients rather than a floating-point approximation of them.
    That exactness is load-bearing: an operator of order :math:`p` maps every
    polynomial of degree below :math:`p` in the index to *exactly* zero, in
    floating point and not merely to rounding, which is what makes a penalty
    built from it a statement about curvature rather than about round-off.

    Examples
    --------
    >>> import numpy as np
    >>> from fast_vollib.surface.fitting.regularized import difference_matrix
    >>> difference_matrix(4, 1).tolist()
    [[-1.0, 1.0, 0.0, 0.0], [0.0, -1.0, 1.0, 0.0], [0.0, 0.0, -1.0, 1.0]]
    >>> difference_matrix(4, 2).tolist()
    [[1.0, -2.0, 1.0, 0.0], [0.0, 1.0, -2.0, 1.0]]

    A second difference annihilates any affine sequence, exactly:

    >>> (difference_matrix(5, 2) @ (3.0 + 2.0 * np.arange(5.0))).tolist()
    [0.0, 0.0, 0.0]
    """
    size = _as_count(n, "n")
    degree = _as_count(order, "order")
    if size < 1:
        raise SurfaceValidationError(
            f"n must be at least 1; got {size}. An operator on no points has no rows "
            f"and no columns, and would make every downstream shape check vacuous."
        )
    if degree > MAX_DIFFERENCE_ORDER:
        raise SurfaceValidationError(
            f"order must be at most {MAX_DIFFERENCE_ORDER}; got {degree}. Beyond that "
            f"the binomial coefficients exceed 2**53 and stop being exact in float64, "
            f"so the operator would no longer be the integer stencil it advertises."
        )
    if degree >= size:
        raise SurfaceValidationError(
            f"order must be smaller than n; got order {degree} on {size} points. A "
            f"difference of order p needs p + 1 points to have a single row, and an "
            f"operator with no rows is a penalty that silently does nothing."
        )
    stencil = np.array([1], dtype=np.int64)
    for _ in range(degree):
        stencil = np.convolve(stencil, np.array([-1, 1], dtype=np.int64))
    matrix = np.zeros((size - degree, size), dtype=np.float64)
    rows = np.arange(size - degree)
    for offset, coefficient in enumerate(stencil.tolist()):
        matrix[rows, rows + offset] = float(coefficient)
    return matrix


@dataclass(frozen=True, slots=True)
class TikhonovPenalty:
    """One quadratic penalty :math:`\\lambda\\|Lc\\|^2` on a coefficient vector.

    Parameters
    ----------
    operator:
        The matrix :math:`L`, shape ``(rows, n_parameters)``.  Copied on
        construction and marked read-only, so a caller who keeps and later
        mutates the array they passed in cannot change a penalty that has
        already been applied to a reported fit.
    weight:
        The multiplier :math:`\\lambda`, finite and non-negative.  Zero is
        admissible and means the penalty is inert: it is kept in the stack so
        that sweeping a weight down to zero does not change which system is
        being solved, only its contents.

    Raises
    ------
    SurfaceValidationError
        If the operator is not a finite two-dimensional array with at least one
        column, or if the weight is negative or not finite.

    Notes
    -----
    The class carries no notion of what the coefficients mean.  ``L`` may be a
    difference operator on spline coefficients, an identity (ridge), a
    contrast against a prior parameter vector, or anything else quadratic; the
    solver only needs its shape.

    Examples
    --------
    >>> from fast_vollib.surface.fitting.regularized import TikhonovPenalty
    >>> penalty = TikhonovPenalty.difference(4, 0.5, order=2)
    >>> penalty.operator.shape, penalty.n_parameters, penalty.n_rows
    ((2, 4), 4, 2)
    >>> float(penalty.value([1.0, 2.0, 3.0, 4.0]))   # affine: no curvature
    0.0
    >>> float(TikhonovPenalty.ridge(3, 2.0).value([1.0, 0.0, 1.0]))
    4.0
    """

    operator: np.ndarray
    weight: float

    def __post_init__(self) -> None:
        operator = owned_float_2d(self.operator, "operator")
        if operator.shape[1] == 0:
            raise SurfaceValidationError(
                f"operator must have at least one column; got shape {operator.shape}. "
                f"The column count is the number of coefficients the penalty speaks "
                f"about, and a penalty about nothing cannot be checked against a design."
            )
        if not bool(np.isfinite(operator).all()):
            raise SurfaceValidationError(
                "operator must be finite; got a non-finite entry. A NaN row would "
                "spread through the stacked factorization into every coefficient, so "
                "the fit would come back all-NaN with no indication of which penalty "
                "caused it."
            )
        weight = float(self.weight)
        if not np.isfinite(weight):
            raise SurfaceValidationError(
                f"weight must be finite; got {self.weight!r}. An infinite penalty is a "
                f"hard constraint, which belongs in the design matrix rather than in a "
                f"term added to a finite residual."
            )
        if weight < 0.0:
            raise SurfaceValidationError(
                f"weight must be non-negative; got {weight!r}. A negative weight pays "
                f"the objective for roughness, so the minimum runs off to infinity and "
                f"the returned coefficients would record where the solver stopped "
                f"rather than where the data is."
            )
        object.__setattr__(self, "operator", operator)
        object.__setattr__(self, "weight", weight)

    @property
    def n_parameters(self) -> int:
        """Number of coefficients the penalty acts on -- the operator's columns."""
        return int(self.operator.shape[1])

    @property
    def n_rows(self) -> int:
        """Number of rows the penalty contributes to the stacked design."""
        return int(self.operator.shape[0])

    @classmethod
    def ridge(cls, n_parameters: int, weight: float) -> TikhonovPenalty:
        """A ridge penalty :math:`\\lambda\\|c\\|^2` on ``n_parameters`` coefficients.

        Parameters
        ----------
        n_parameters:
            Size of the coefficient vector.
        weight:
            The multiplier :math:`\\lambda`.

        Returns
        -------
        TikhonovPenalty
            The identity operator at that weight.  Shrinks toward zero, which
            is a statement about the coefficients' scale and not about their
            smoothness -- use :meth:`difference` when the claim is smoothness.
        """
        return cls(operator=difference_matrix(n_parameters, 0), weight=weight)

    @classmethod
    def difference(cls, n_parameters: int, weight: float, *, order: int = 2) -> TikhonovPenalty:
        """A difference penalty :math:`\\lambda\\|Dc\\|^2` on ``n_parameters`` coefficients.

        Parameters
        ----------
        n_parameters:
            Size of the coefficient vector.
        weight:
            The multiplier :math:`\\lambda`.
        order:
            Difference order, default ``2``.  Order 1 penalizes slope and pulls
            the coefficient vector toward a constant; order 2 penalizes
            curvature and pulls it toward an affine sequence.  Which one is the
            modelling claim, so it is named rather than defaulted silently.

        Returns
        -------
        TikhonovPenalty
            :func:`difference_matrix` at that order and weight.
        """
        return cls(operator=difference_matrix(n_parameters, order), weight=weight)

    def weighted_operator(self) -> np.ndarray:
        """:math:`\\sqrt{\\lambda}L`, the block this penalty adds to a stacked design.

        Returns
        -------
        numpy.ndarray
            A fresh writable array of shape ``(n_rows, n_parameters)``.  The
            square root is taken once, here, so no caller re-derives the
            convention that the weight multiplies the *squared* operator norm.
        """
        return np.sqrt(self.weight) * np.asarray(self.operator, dtype=np.float64)

    def value(self, coefficients: Any) -> float:
        """The penalty :math:`\\lambda\\|Lc\\|^2` at ``coefficients``.

        Parameters
        ----------
        coefficients:
            Shape ``(n_parameters,)``.

        Returns
        -------
        float
            The contribution this penalty makes to the objective.

        Raises
        ------
        SurfaceValidationError
            If ``coefficients`` is not a finite one-dimensional array of the
            operator's width.
        """
        vector = _finite_1d(coefficients, "coefficients")
        if vector.size != self.n_parameters:
            raise SurfaceValidationError(
                f"coefficients must have one entry per operator column; got "
                f"{vector.size} for an operator of width {self.n_parameters}. A penalty "
                f"applied to the wrong coefficients would report a finite, wrong number."
            )
        transformed = np.asarray(self.operator, dtype=np.float64) @ vector
        return float(self.weight * float(transformed @ transformed))


@dataclass(frozen=True, slots=True)
class LeastSquaresDiagnostics:
    """What the solver saw while solving, reported rather than discarded.

    Parameters
    ----------
    residual_norm:
        :math:`\\|W^{1/2}(Ac - y)\\|_2`, the weighted data misfit alone.  The
        penalty rows are excluded: a number that mixed the two could be driven
        down by tightening the penalty, which is the opposite of what a misfit
        is for.
    penalty_norm:
        :math:`(\\sum_i \\lambda_i\\|L_ic\\|^2)^{1/2}`, the roughness paid for.
        Reported next to the residual so the trade between them is visible
        rather than hidden inside one scalar objective.
    effective_rank:
        Rank of the *stacked* system at the solver's cutoff.  Below
        ``n_parameters`` the fit is one member of an infinite family, and
        :func:`numpy.linalg.lstsq` returned the minimum-norm member.
    condition_number:
        Ratio of largest to smallest singular value of the stacked system, and
        **infinity whenever the system is rank deficient** -- which is what the
        condition number of a singular operator is.  The distinction matters
        because :func:`numpy.linalg.lstsq` returns only ``min(rows, columns)``
        singular values: on a wide system, the null directions are simply not in
        the list, and a ratio taken over what is there would report a
        comfortable number for an operator with an infinite solution family.  It is the conditioning of the
        problem *as solved*, so adding a penalty is expected to lower it.
    n_parameters:
        Width of the design, carried so ``effective_rank`` can be read without
        the design in hand.

    Notes
    -----
    A rank-deficient answer is reported, not refused.  Fitting a deliberately
    over-complete basis is a legitimate thing to ask for, and the minimum-norm
    solution is a defensible answer to it; what is not defensible is returning
    that answer with nothing attached to say the system was degenerate.

    Examples
    --------
    >>> import numpy as np
    >>> from fast_vollib.surface.fitting.regularized import solve_penalized_least_squares
    >>> design = np.ones((3, 2))                      # two identical columns
    >>> _, diagnostics = solve_penalized_least_squares(design, [1.0, 1.0, 1.0])
    >>> diagnostics.effective_rank, diagnostics.rank_deficient
    (1, True)
    >>> diagnostics.to_dict()["n_parameters"]
    2
    """

    residual_norm: float
    penalty_norm: float
    effective_rank: int
    condition_number: float
    n_parameters: int

    def __post_init__(self) -> None:
        for name in ("residual_norm", "penalty_norm"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise SurfaceValidationError(
                    f"{name} must be finite and non-negative; got {value!r}. A norm is a "
                    f"length, and a negative or NaN length means the solve was not "
                    f"described by the record that reports it."
                )
            object.__setattr__(self, name, value)
        n_parameters = _as_count(self.n_parameters, "n_parameters")
        effective_rank = _as_count(self.effective_rank, "effective_rank")
        if not 0 <= effective_rank <= n_parameters:
            raise SurfaceValidationError(
                f"effective_rank must lie in [0, n_parameters]; got {effective_rank} of "
                f"{n_parameters}. A rank above the number of coefficients describes a "
                f"different system than the one whose width is reported."
            )
        condition_number = float(self.condition_number)
        if np.isnan(condition_number) or condition_number < 1.0:
            raise SurfaceValidationError(
                f"condition_number must be at least 1 (infinity allowed); got "
                f"{condition_number!r}. The ratio of the largest singular value to the "
                f"smallest is never below one, so a smaller value means the record was "
                f"filled in from something other than the solved system."
            )
        object.__setattr__(self, "n_parameters", n_parameters)
        object.__setattr__(self, "effective_rank", effective_rank)
        object.__setattr__(self, "condition_number", condition_number)

    @property
    def rank_deficient(self) -> bool:
        """Whether the stacked system left at least one direction undetermined."""
        return self.effective_rank < self.n_parameters

    def to_dict(self) -> dict[str, Any]:
        """The record as a JSON-safe mapping.

        Returns
        -------
        dict
            An infinite ``condition_number`` becomes ``None``: ``Infinity`` is
            not valid JSON, and a record that has to be repaired before it can
            be written down is a record that will be written down wrong.
        """
        condition = self.condition_number
        return {
            "residual_norm": float(self.residual_norm),
            "penalty_norm": float(self.penalty_norm),
            "effective_rank": int(self.effective_rank),
            "condition_number": None if np.isinf(condition) else float(condition),
            "n_parameters": int(self.n_parameters),
            "rank_deficient": bool(self.rank_deficient),
        }


def solve_penalized_least_squares(
    design: Any,
    target: Any,
    *,
    weights: Any = None,
    penalties: Iterable[TikhonovPenalty] = (),
    rcond: float | None = None,
) -> tuple[np.ndarray, LeastSquaresDiagnostics]:
    """Minimize :math:`\\|W^{1/2}(Ac - y)\\|^2 + \\sum_i \\lambda_i\\|L_ic\\|^2`.

    Parameters
    ----------
    design:
        The matrix :math:`A`, shape ``(n_rows, n_parameters)``, finite.
    target:
        The vector :math:`y`, shape ``(n_rows,)``, finite.
    weights:
        Optional non-negative row weights :math:`w_i` multiplying the *squared*
        residual of row :math:`i`, so a weight of 4 is exactly the row entered
        four times.  ``None`` means unweighted, and is not the same as passing
        ones: it skips the scaling entirely rather than multiplying by one.
    penalties:
        Any iterable of :class:`TikhonovPenalty`, each acting on
        ``n_parameters`` coefficients.  Weights of zero are kept, so sweeping a
        weight to zero changes the contents of the system and not its shape.
    rcond:
        Singular-value cutoff passed to :func:`numpy.linalg.lstsq`, relative to
        the largest singular value.  ``None`` uses that function's
        machine-precision default.

    Returns
    -------
    coefficients : numpy.ndarray
        Shape ``(n_parameters,)``, the minimizer.  When the stacked system is
        rank deficient this is the minimum-norm minimizer, and
        ``diagnostics.rank_deficient`` says so.
    diagnostics : LeastSquaresDiagnostics
        Residual norm, penalty norm, effective rank, and condition number of
        the system that was actually solved.

    Raises
    ------
    SurfaceValidationError
        If the shapes disagree, if any input is non-finite, if a weight is
        negative, or if a penalty acts on a different number of coefficients
        than the design has columns.
    SurfaceCalibrationError
        If the underlying SVD does not converge.  No fallback solution is
        substituted: a caller who receives coefficients from a solve that
        failed has no way to know they are meaningless.

    Notes
    -----
    The objective is minimized by stacking, never by normal equations.  With
    :math:`\\tilde{A} = [W^{1/2}A;\\ \\sqrt{\\lambda_1}L_1;\\ \\dots]` and
    :math:`\\tilde{y} = [W^{1/2}y;\\ 0;\\ \\dots]`, the objective is exactly
    :math:`\\|\\tilde{A}c - \\tilde{y}\\|^2`, so one call to
    :func:`numpy.linalg.lstsq` -- deterministic, and minimum-norm on a rank
    deficient system -- answers it.  Forming
    :math:`A^{\\top}WA + \\sum_i \\lambda_iL_i^{\\top}L_i` would be cheaper and
    would square the condition number: on the 40-by-12 Vandermonde fixture in
    the tests the stacked path recovers the generating coefficients to
    :math:`10^{-8}` and the normal equations to :math:`0.26`, on the same data
    and in the same precision.

    *Weights are variances-in-reverse, not standard deviations.*  ``weights``
    multiplies the squared residual, so a caller holding quote standard errors
    passes :math:`1/\\sigma_i^2`, not :math:`1/\\sigma_i`.  Getting this
    backwards halves nothing loudly -- it silently re-weights the fit -- which
    is why the convention is stated on both the parameter and here.

    Examples
    --------
    >>> import numpy as np
    >>> from fast_vollib.surface.fitting.regularized import (
    ...     TikhonovPenalty,
    ...     solve_penalized_least_squares,
    ... )
    >>> x = np.linspace(-1.0, 1.0, 7)
    >>> design = np.column_stack([np.ones_like(x), x, x * x])
    >>> target = 1.0 - 0.5 * x + 2.0 * x * x
    >>> coefficients, diagnostics = solve_penalized_least_squares(design, target)
    >>> [float(round(value, 9)) for value in coefficients]
    [1.0, -0.5, 2.0]
    >>> diagnostics.rank_deficient
    False

    A ridge penalty shrinks the same fit and pays for it in residual:

    >>> shrunk, ridged = solve_penalized_least_squares(
    ...     design, target, penalties=[TikhonovPenalty.ridge(3, 1.0)]
    ... )
    >>> bool(np.linalg.norm(shrunk) < np.linalg.norm(coefficients))
    True
    >>> bool(ridged.residual_norm > diagnostics.residual_norm)
    True
    """
    matrix = _finite_2d(design, "design")
    observed = _finite_1d(target, "target")
    n_rows, n_parameters = matrix.shape
    if n_rows == 0 or n_parameters == 0:
        raise SurfaceValidationError(
            f"design must have at least one row and one column; got shape "
            f"{matrix.shape}. An empty design has no fit, and returning a zero "
            f"coefficient vector for one would invent a model nobody asked for."
        )
    if observed.size != n_rows:
        raise SurfaceValidationError(
            f"target must have one entry per design row; got {observed.size} entries "
            f"for {n_rows} rows. Silently truncating to the shorter of the two would "
            f"fit a different data set than the caller assembled."
        )
    if weights is None:
        blocks = [matrix]
        stacked_target = [observed]
        root: np.ndarray | None = None
    else:
        row_weights = _finite_1d(weights, "weights")
        if row_weights.size != n_rows:
            raise SurfaceValidationError(
                f"weights must have one entry per design row; got {row_weights.size} "
                f"entries for {n_rows} rows. A partially weighted fit is not a fit "
                f"under any stated objective."
            )
        if bool((row_weights < 0.0).any()):
            raise SurfaceValidationError(
                f"weights must be non-negative; got a minimum of "
                f"{float(row_weights.min())!r}. A negative weight pays the objective "
                f"for missing a row, so the minimum runs off to infinity."
            )
        root = np.sqrt(row_weights)
        blocks = [root[:, None] * matrix]
        stacked_target = [root * observed]
    listed = tuple(penalties)
    for index, penalty in enumerate(listed):
        if not isinstance(penalty, TikhonovPenalty):
            raise SurfaceValidationError(
                f"penalties[{index}] must be a TikhonovPenalty; got "
                f"{type(penalty).__name__}. The weight convention -- lambda multiplies "
                f"the squared operator norm -- lives in that class, and a bare matrix "
                f"would arrive without it."
            )
        if penalty.n_parameters != n_parameters:
            raise SurfaceValidationError(
                f"penalties[{index}] must act on {n_parameters} coefficients to match "
                f"the design; got {penalty.n_parameters}. A penalty of the wrong width "
                f"regularizes a different vector than the one being fitted."
            )
        blocks.append(penalty.weighted_operator())
        stacked_target.append(np.zeros(penalty.n_rows, dtype=np.float64))
    stacked_design = blocks[0] if len(blocks) == 1 else np.vstack(blocks)
    stacked = stacked_target[0] if len(stacked_target) == 1 else np.concatenate(stacked_target)
    try:
        solution, _residuals, rank, singular = np.linalg.lstsq(stacked_design, stacked, rcond=rcond)
    except np.linalg.LinAlgError as error:  # pragma: no cover - LAPACK non-convergence
        raise SurfaceCalibrationError(
            f"The penalized least-squares solve did not converge: {error}. No fallback "
            f"coefficients are returned, because a caller cannot tell a failed solve "
            f"from a successful one by looking at the numbers it produced."
        ) from error
    coefficients = np.asarray(solution, dtype=np.float64)
    misfit = matrix @ coefficients - observed
    if root is None:
        residual_norm = float(np.sqrt(float(misfit @ misfit)))
    else:
        weighted = root * misfit
        residual_norm = float(np.sqrt(float(weighted @ weighted)))
    penalty_norm = float(np.sqrt(sum(penalty.value(coefficients) for penalty in listed)))
    # Rank deficiency first: lstsq reports only min(rows, columns) singular
    # values, so on a wide system the zero ones never appear and singular[-1] is
    # a healthy positive number belonging to a range direction. Reading the ratio
    # there would report a well-conditioned operator with a null space.
    if int(rank) < int(n_parameters) or singular.size == 0 or singular[-1] <= 0.0:
        condition_number = float("inf")
    else:
        condition_number = float(singular[0] / singular[-1])
    diagnostics = LeastSquaresDiagnostics(
        residual_norm=residual_norm,
        penalty_norm=penalty_norm,
        effective_rank=int(rank),
        condition_number=condition_number,
        n_parameters=int(n_parameters),
    )
    return coefficients, diagnostics


def couple_parameter_sequence(
    parameters: Any,
    *,
    weight: float,
    order: int = 1,
    data_weights: Any = None,
) -> np.ndarray:
    """Smooth a sequence of independently fitted parameter vectors along its index.

    Minimizes :math:`\\sum_t v_t\\|p_t - \\hat{p}_t\\|^2 + \\lambda\\|DP\\|_F^2`
    over the whole path, where :math:`D` is the order-``order`` difference
    operator on the sequence index and :math:`\\hat{P}` are the independent
    fits.  Nothing here knows what a parameter means, so the same call serves
    an SVI path, a factor-score path, and a single scalar level.

    Parameters
    ----------
    parameters:
        The independent fits :math:`\\hat{P}`, shape ``(n_steps,)`` or
        ``(n_steps, n_parameters)``, oldest step first, finite.  A
        one-dimensional input comes back one-dimensional.
    weight:
        The coupling weight :math:`\\lambda`, finite and non-negative.  Zero
        returns the input path; larger values pull it toward the null space of
        :math:`D` -- a constant path at order 1, an affine path at order 2.
    order:
        Difference order along the sequence, at least 1.  Order 0 is refused:
        the identity operator shrinks the path toward zero rather than coupling
        its steps, which is a different claim wearing this function's name.
    data_weights:
        Optional non-negative per-step weights :math:`v_t`, defaulting to ones.
        A step of weight zero is one whose own fit is not trusted; its
        parameters are then reconstructed from its neighbours through the
        coupling instead of being taken at face value.

    Returns
    -------
    numpy.ndarray
        The coupled path, same shape and step order as ``parameters``, a fresh
        writable array.

    Raises
    ------
    SurfaceValidationError
        If the sequence is ragged, non-finite, shorter than ``order + 1``, or
        if ``weight``, ``order``, or ``data_weights`` are out of range --
        including too few strictly positive step weights to determine the path.
    SurfaceCalibrationError
        If the underlying SVD does not converge.

    Notes
    -----
    The columns do not interact.  The objective separates over parameter
    columns, so coupling a block is identical to coupling each column on its
    own, and parameters on wildly different scales need no standardization
    first -- scaling a column scales its data and penalty terms equally.

    The minimizer solves the dense linear system
    :math:`(V + \\lambda D^{\\top}D)P = V\\hat{P}` -- the Whittaker-Henderson
    graduation of the path -- but that system is not what is assembled here.
    Its condition number is :math:`\\lambda\\|D\\|^2/\\min_t v_t`, about
    :math:`4\\times10^{12}` at :math:`\\lambda = 10^{12}`; solving it directly
    recovers the constant limit of a five-step unit-scale path with an error of
    :math:`1.2\\times10^{-5}`, while the equivalent stacked system
    :math:`[V^{1/2};\\ \\sqrt{\\lambda}D]`, conditioned at the square root of
    that, recovers it to :math:`2.2\\times10^{-11}`.  Both are exact
    factorizations and neither iterates; the stacked one keeps six more digits,
    so it is the one used.

    *This function refuses where the solver reports.*
    :func:`solve_penalized_least_squares` answers a rank-deficient system with
    the minimum-norm solution and records the deficiency, because fitting an
    over-complete basis is a legitimate request.  A coupled path has no such
    reading: if the step weights leave a direction of the path undetermined,
    every member of an infinite family fits equally well and the one the solver
    happens to return is an artifact of the algorithm.  So the precondition --
    at least ``order`` strictly positive step weights when the path is coupled,
    and all of them positive when it is not -- is checked up front, and its
    violation raises rather than returning a plausible path.

    Examples
    --------
    >>> import numpy as np
    >>> from fast_vollib.surface.fitting.regularized import couple_parameter_sequence
    >>> path = [0.20, 0.30, 0.20, 0.30, 0.20]
    >>> [float(round(value, 12)) for value in couple_parameter_sequence(path, weight=0.0)]
    [0.2, 0.3, 0.2, 0.3, 0.2]
    >>> [float(round(value, 4)) for value in couple_parameter_sequence(path, weight=1e12)]
    [0.24, 0.24, 0.24, 0.24, 0.24]

    A step whose own fit is not trusted is reconstructed from its neighbours:

    >>> filled = couple_parameter_sequence(
    ...     [1.0, 2.0, 999.0, 4.0, 5.0], weight=1.0, data_weights=[1.0, 1.0, 0.0, 1.0, 1.0]
    ... )
    >>> float(round(filled[2], 9))
    3.0

    Several parameters are coupled together, each column on its own scale:

    >>> block = np.column_stack([path, np.multiply(path, 100.0)])
    >>> couple_parameter_sequence(block, weight=5.0, order=2).shape
    (5, 2)
    """
    fits = _finite_2d_or_1d(parameters, "parameters")
    flat = fits.ndim == 1
    block = fits[:, None] if flat else fits
    n_steps, n_parameters = block.shape
    if n_parameters == 0:
        raise SurfaceValidationError(
            f"parameters must have at least one column; got shape {fits.shape}. "
            f"A path of empty vectors has nothing to couple."
        )
    degree = _as_count(order, "order")
    if degree < 1:
        raise SurfaceValidationError(
            f"order must be at least 1; got {degree}. Order 0 is the identity operator, "
            f"which shrinks every step toward zero instead of coupling it to its "
            f"neighbours -- a different claim under this function's name."
        )
    if n_steps < degree + 1:
        raise SurfaceValidationError(
            f"parameters must have at least order + 1 = {degree + 1} steps; got "
            f"{n_steps}. A difference of order {degree} has no rows on a shorter "
            f"sequence, so the request describes no coupling at all."
        )
    coupling = float(weight)
    if not np.isfinite(coupling):
        raise SurfaceValidationError(
            f"weight must be finite; got {weight!r}. An infinite coupling is a hard "
            f"constraint that the path lie in the null space of the operator, which is "
            f"a projection rather than a smoothing."
        )
    if coupling < 0.0:
        raise SurfaceValidationError(
            f"weight must be non-negative; got {coupling!r}. A negative coupling pays "
            f"the objective for jumps between steps, so the smoothed path would run "
            f"away from the fits it is supposed to reconcile."
        )
    if data_weights is None:
        step_weights = np.ones(n_steps, dtype=np.float64)
    else:
        step_weights = _finite_1d(data_weights, "data_weights")
        if step_weights.size != n_steps:
            raise SurfaceValidationError(
                f"data_weights must have one weight per step; got {step_weights.size} "
                f"for {n_steps} steps. A partially weighted path is not a path under "
                f"any stated objective."
            )
        if bool((step_weights < 0.0).any()):
            raise SurfaceValidationError(
                f"data_weights must be non-negative; got a minimum of "
                f"{float(step_weights.min())!r}. A negative step weight pays the "
                f"objective for missing that step's own fit."
            )
    positive = int(np.count_nonzero(step_weights > 0.0))
    required = degree if coupling > 0.0 else n_steps
    if positive < required:
        raise SurfaceValidationError(
            f"data_weights must have at least {required} strictly positive entries; got "
            f"{positive} of {n_steps}. The null space of an order-{degree} difference "
            f"is the polynomials of degree below {degree}, which vanish at fewer than "
            f"{degree} indices, so with fewer anchored steps than that the path is "
            f"undetermined along a whole direction and any answer would be one "
            f"arbitrary member of an infinite family."
        )
    root = np.sqrt(step_weights)
    operator = difference_matrix(n_steps, degree)
    stacked_design = np.vstack([np.diag(root), np.sqrt(coupling) * operator])
    stacked_target = np.vstack(
        [root[:, None] * block, np.zeros((operator.shape[0], n_parameters), dtype=np.float64)]
    )
    try:
        solution, _residuals, _rank, _singular = np.linalg.lstsq(
            stacked_design, stacked_target, rcond=None
        )
    except np.linalg.LinAlgError as error:  # pragma: no cover - LAPACK non-convergence
        raise SurfaceCalibrationError(
            f"The coupled least-squares solve did not converge: {error}. No fallback "
            f"path is returned, because a caller cannot tell a failed solve from a "
            f"successful one by looking at the path it produced."
        ) from error
    coupled = np.asarray(solution, dtype=np.float64)
    return coupled[:, 0].copy() if flat else coupled


def _as_count(value: Any, name: str) -> int:
    """``value`` as a plain non-negative integer, or a typed error."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise SurfaceValidationError(
            f"{name} must be an integer; got {type(value).__name__}. A fractional "
            f"dimension has no matrix, and truncating one silently would build an "
            f"operator of the wrong size."
        )
    count = int(value)
    if count < 0:
        raise SurfaceValidationError(
            f"{name} must be non-negative; got {count}. Negative sizes and orders have "
            f"no operator, and numpy would read them as counting from the end."
        )
    return count


def _as_float_array(value: Any, name: str) -> np.ndarray:
    """An owned float64 copy of ``value``, or a typed error for ragged input."""
    try:
        return np.array(value, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as error:
        raise SurfaceValidationError(
            f"{name} must be a rectangular array of real numbers; numpy read it as "
            f"ragged or non-numeric ({error})."
        ) from error


def _require_finite(array: np.ndarray, name: str) -> None:
    """Raise unless every entry of ``array`` is finite."""
    if bool(np.isfinite(array).all()):
        return
    offending = int(np.count_nonzero(~np.isfinite(array)))
    raise SurfaceValidationError(
        f"{name} must be finite; got {offending} non-finite entries. A single NaN "
        f"spreads through the factorization into every coefficient, so the solve would "
        f"return an all-NaN answer with nothing in it to say which row caused that."
    )


def _finite_1d(value: Any, name: str) -> np.ndarray:
    """A finite, owned, one-dimensional float64 copy of ``value``."""
    array = _as_float_array(value, name)
    if array.ndim != 1:
        raise SurfaceValidationError(f"{name} must be one-dimensional; got shape {array.shape}.")
    _require_finite(array, name)
    return array


def _finite_2d(value: Any, name: str) -> np.ndarray:
    """A finite, owned, two-dimensional float64 copy of ``value``."""
    array = _as_float_array(value, name)
    if array.ndim != 2:
        raise SurfaceValidationError(f"{name} must be two-dimensional; got shape {array.shape}.")
    _require_finite(array, name)
    return array


def _finite_2d_or_1d(value: Any, name: str) -> np.ndarray:
    """A finite, owned float64 copy of ``value``, one- or two-dimensional."""
    array = _as_float_array(value, name)
    if array.ndim not in (1, 2):
        raise SurfaceValidationError(
            f"{name} must be one- or two-dimensional -- steps, or steps by parameters; "
            f"got shape {array.shape}."
        )
    _require_finite(array, name)
    return array
