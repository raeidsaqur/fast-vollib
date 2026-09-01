"""A linear Gaussian state space whose state is a value, never an attribute.

The dynamics of an implied-volatility surface are normally written on its
*parameters*: fit a representation to each day, and the fitted parameter path
is a short, low-dimensional time series that a linear Gaussian model can
filter, smooth, and project.  This module is that machinery --
:class:`GaussianState`, :class:`LinearGaussianModel`, the four recursions
(:func:`kalman_predict`, :func:`kalman_update`, :func:`kalman_filter`,
:func:`kalman_smooth`), and :class:`StateSpaceForecaster`, which composes them
with a calibrator into a
:class:`~fast_vollib.surface.protocols.SurfaceForecaster`.

**The design decision is that nothing here hides filter state.**  Every step
takes a state and returns a new one: :func:`kalman_predict` and
:func:`kalman_update` are free functions on values, and every container is a
frozen dataclass over owned read-only arrays.  The usual alternative -- a
``KalmanFilter`` object carrying ``self.x`` and ``self.P`` and mutating them in
a ``.step()`` method -- makes the answer depend on how many times the object
was stepped before, which is invisible at the call site.

*The failure mode this prevents is an experiment nobody can replay.*  A filter
that owns its state returns something different for the second of two identical
runs whenever the runs share the object, and it does so silently: the numbers
stay plausible, the covariance stays positive, and no part of the output
records that the state was warm.  Handing the state over explicitly makes a
warm start something a reader can see in the code that asked for it -- and it
is why :class:`StateSpaceForecaster` can promise that two forecasts from the
same history are the same forecast.

*Symmetry and positivity are maintained, not assumed.*  The covariance update
uses the Joseph form, and every recursion replaces its result :math:`P` with
:math:`(P + P^{\\top}) / 2` before building the next state.  Both are cheap
insurance against the arithmetic drift that turns a covariance into a matrix
with a negative eigenvalue a few hundred steps later, at which point the
Cholesky factorization fails and the filter reports an error whose cause is
several thousand operations upstream.

*A missing observation is skipped, never imputed.*  ``observation_mask`` (or a
``NaN`` entry, which means "no quote" everywhere in this package) removes a row
from the update, so the step contributes no innovation and no likelihood and
leaves the state with the uncertainty the prediction gave it.  Filling the gap
with a fitted value instead would feed the model's own forecast back to it as
evidence.

Examples
--------
>>> import numpy as np
>>> from fast_vollib.surface.fitting.state_space import (
...     GaussianState, LinearGaussianModel, kalman_filter,
... )
>>> model = LinearGaussianModel.random_walk(1, transition_variance=1e-4,
...                                         observation_variance=1e-6)
>>> initial = GaussianState.isotropic([0.20], 1.0)
>>> path = kalman_filter(np.array([[0.20], [0.22], [0.21]]), model, initial)
>>> float(round(path.terminal.mean[0], 6))
0.210096
>>> path.n_steps
3
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from .._validate import owned_bool_1d, owned_float_1d, owned_float_2d, read_only
from ..errors import SurfaceCalibrationError, SurfaceValidationError
from ..protocols import ForecastHorizon
from .prior import FlatIVSurface, FlatVolatilityCalibrator

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..observations import SurfaceObservations
    from ..protocols import DefiniteIVSurface, RNGInput, SurfaceCalibrator

__all__ = [
    "COVARIANCE_TOLERANCE_FACTOR",
    "DEFAULT_INITIAL_VARIANCE",
    "DEFAULT_OBSERVATION_VARIANCE",
    "DEFAULT_TRANSITION_VARIANCE",
    "FilteredPath",
    "GaussianState",
    "KalmanUpdate",
    "LinearGaussianModel",
    "StateSpaceForecaster",
    "flat_level_parameters",
    "flat_surface_from_parameters",
    "kalman_filter",
    "kalman_predict",
    "kalman_smooth",
    "kalman_update",
]

#: How much asymmetry and how negative an eigenvalue a covariance may carry
#: before it is rejected, as a multiple of :math:`n \\, \\varepsilon \\,
#: \\max_{ij} |P_{ij}|` with :math:`\\varepsilon` the float64 machine epsilon.
#: The scale is the matrix's own, so the gate means the same thing for a
#: covariance in basis points as for one in units of variance.  16 is one
#: doubling above the round-off a Cholesky solve followed by two symmetric
#: matrix products accumulates, which is exactly what the Joseph form and the
#: Rauch-Tung-Striebel step do to their inputs.
COVARIANCE_TOLERANCE_FACTOR = 16.0

#: Default per-step variance of the parameter random walk, on the scale of
#: implied volatility itself: a standard deviation of 0.01 per step, which is
#: an ordinary daily move of an at-the-money level.  A parameter vector on
#: another scale needs another number, which is why this is configuration and
#: not a constant buried in the recursion.
DEFAULT_TRANSITION_VARIANCE = 1e-4

#: Default variance of the measurement noise on a fitted parameter: a standard
#: deviation of 0.001, the order of the calibration error of a level fitted to
#: a liquid smile.  It is not zero, because a fitted parameter is an estimate.
DEFAULT_OBSERVATION_VARIANCE = 1e-6

#: Default variance of the initial state, strictly positive and deliberately
#: diffuse relative to an implied volatility: the first observation should move
#: the state essentially all the way, rather than being averaged against a
#: prior nobody chose.
DEFAULT_INITIAL_VARIANCE = 1.0

_LOG_TWO_PI = float(np.log(2.0 * np.pi))


def _symmetrise(matrix: np.ndarray) -> np.ndarray:
    """:math:`(P + P^{\\top}) / 2`, the symmetric part of ``matrix``.

    Applied to every covariance a recursion produces.  In exact arithmetic
    :math:`F P F^{\\top} + Q` and the Joseph form are symmetric already; in
    float64 the two triangles differ in their last bits because the products
    are summed in different orders, and that difference compounds.  Taking the
    symmetric part costs one addition per entry and makes the result *bitwise*
    symmetric -- floating-point addition is commutative -- so a downstream
    ``eigvalsh`` or Cholesky sees the matrix the algebra intended.
    """
    return (matrix + matrix.T) / 2.0


def _covariance_tolerance(matrix: np.ndarray) -> float:
    """The predeclared symmetry and positivity slack for ``matrix``."""
    if matrix.size == 0:
        return 0.0
    scale = float(np.max(np.abs(matrix)))
    return COVARIANCE_TOLERANCE_FACTOR * matrix.shape[0] * float(np.finfo(np.float64).eps) * scale


def _validated_covariance(value: Any, name: str, n: int) -> np.ndarray:
    """An owned, read-only ``(n, n)`` covariance, checked symmetric and PSD."""
    matrix = owned_float_2d(value, name, (n, n))
    if not bool(np.all(np.isfinite(matrix))):
        raise SurfaceValidationError(
            f"{name} must be finite everywhere; got a non-finite entry. A covariance "
            f"with a NaN or an infinity describes no distribution, and every recursion "
            f"downstream would propagate it silently."
        )
    tolerance = _covariance_tolerance(matrix)
    asymmetry = float(np.max(np.abs(matrix - matrix.T))) if matrix.size else 0.0
    if asymmetry > tolerance:
        raise SurfaceValidationError(
            f"{name} must be symmetric to within {tolerance:.3e}; got a maximum "
            f"|P - P.T| of {asymmetry:.3e}. An asymmetric matrix is not a covariance, "
            f"and the eigenvalue check below would silently read only one triangle of it."
        )
    if matrix.size:
        smallest = float(np.linalg.eigvalsh(matrix)[0])
        if smallest < -tolerance:
            raise SurfaceValidationError(
                f"{name} must be positive semi-definite to within {tolerance:.3e}; got a "
                f"smallest eigenvalue of {smallest:.3e}. A negative variance direction "
                f"makes the log-likelihood complex and the Cholesky factorization fail "
                f"several steps later, where the cause is no longer visible."
            )
    return matrix


def _square(value: Any, name: str) -> np.ndarray:
    """An owned, read-only square float matrix, finite everywhere."""
    matrix = np.array(value, dtype=np.float64, copy=True)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise SurfaceValidationError(f"{name} must be a square matrix; got shape {matrix.shape}.")
    if not bool(np.all(np.isfinite(matrix))):
        raise SurfaceValidationError(
            f"{name} must be finite everywhere; got a non-finite entry. A non-finite "
            f"transition turns the whole filtered path into NaN one step later."
        )
    return read_only(matrix)


def _cholesky(matrix: np.ndarray, subject: str, remedy: str) -> tuple[np.ndarray, bool]:
    """The Cholesky factorization of ``matrix``, or a typed calibration error."""
    try:
        return cho_factor(matrix, lower=True)
    except np.linalg.LinAlgError as error:
        smallest = float(np.linalg.eigvalsh(matrix)[0]) if matrix.size else 0.0
        raise SurfaceCalibrationError(
            f"{subject} must be positive definite; got a smallest eigenvalue of "
            f"{smallest:.3e}. {remedy}"
        ) from error


@dataclass(frozen=True, slots=True)
class GaussianState:
    """A Gaussian belief about the state: a mean and a covariance.

    Parameters
    ----------
    mean:
        The state mean, shape ``(n,)``, finite everywhere.
    covariance:
        The state covariance, shape ``(n, n)``, finite, symmetric, and positive
        semi-definite -- each to within
        :data:`COVARIANCE_TOLERANCE_FACTOR` times :math:`n \\, \\varepsilon`
        times the largest entry of the matrix, so the gate scales with the
        numbers it is applied to.

    Notes
    -----
    The covariance is stored exactly as given and is *not* symmetrised on the
    way in.  A container that quietly repaired its input would make the Joseph
    form's own symmetry untestable and would hide an asymmetric matrix arriving
    from a caller's code; the recursions in this module symmetrise their own
    results before building the next state, which is where that belongs.

    Both arrays are owned read-only copies, so a caller who keeps and later
    mutates what they passed in cannot change a state that has already been
    reported.

    Examples
    --------
    >>> from fast_vollib.surface.fitting.state_space import GaussianState
    >>> state = GaussianState(mean=[0.2, 0.0], covariance=[[1e-4, 0.0], [0.0, 1e-6]])
    >>> state.dimension
    2
    >>> state.mean.flags.writeable
    False
    """

    mean: Any
    covariance: Any

    def __post_init__(self) -> None:
        mean = owned_float_1d(self.mean, "mean")
        if not bool(np.all(np.isfinite(mean))):
            raise SurfaceValidationError(
                "mean must be finite everywhere; got a non-finite entry. A state mean "
                "that is NaN is not an uncertain estimate, it is no estimate at all."
            )
        covariance = _validated_covariance(self.covariance, "covariance", mean.size)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "covariance", covariance)

    @property
    def dimension(self) -> int:
        """Number of state components."""
        return int(self.mean.size)

    @property
    def trace(self) -> float:
        """Total variance :math:`\\operatorname{tr} P`: the scalar summary of how unsure it is."""
        return float(np.trace(self.covariance))

    @classmethod
    def isotropic(cls, mean: Any, variance: float) -> GaussianState:
        """A state with covariance ``variance * I``.

        Parameters
        ----------
        mean:
            The state mean, shape ``(n,)``.
        variance:
            The common variance of every component, finite and non-negative.

        Examples
        --------
        >>> from fast_vollib.surface.fitting.state_space import GaussianState
        >>> GaussianState.isotropic([0.2], 0.25).covariance.tolist()
        [[0.25]]
        """
        value = float(variance)
        if not np.isfinite(value):
            raise SurfaceValidationError(f"variance must be finite; got {variance!r}.")
        if value < 0.0:
            raise SurfaceValidationError(
                f"variance must be non-negative; got {value!r}. A negative variance "
                f"describes no distribution."
            )
        location = owned_float_1d(mean, "mean")
        return cls(mean=location, covariance=value * np.eye(location.size))


@dataclass(frozen=True, slots=True)
class LinearGaussianModel:
    """The matrices of a linear Gaussian state space, and nothing else.

    The model is :math:`x_t = F x_{t-1} + b + w_t` with
    :math:`w_t \\sim N(0, Q)`, observed as :math:`y_t = H x_t + v_t` with
    :math:`v_t \\sim N(0, R)`.

    Parameters
    ----------
    transition_matrix:
        :math:`F`, shape ``(n, n)``, finite.
    transition_covariance:
        :math:`Q`, shape ``(n, n)``, symmetric positive semi-definite.
    observation_matrix:
        :math:`H`, shape ``(m, n)``, finite.
    observation_covariance:
        :math:`R`, shape ``(m, m)``, symmetric positive semi-definite.
    drift:
        Optional constant :math:`b`, shape ``(n,)``.  ``None`` is stored as an
        explicit zero vector, so every model has a drift and the recursion has
        no branch to get wrong.

    Notes
    -----
    Holds configuration only.  It is the same object across every step of every
    path it is used on, which is what makes ``kalman_predict(state, model)``
    a function of its arguments rather than of a history.

    A singular :math:`Q` or :math:`R` is admissible and is sometimes exactly
    what is meant -- a parameter that does not move, an observation with no
    measurement error -- but it is not free: the update needs
    :math:`H P H^{\\top} + R` to be positive definite and the smoother needs the
    predicted covariance to be, and both raise
    :class:`~fast_vollib.surface.errors.SurfaceCalibrationError` when it is not.

    Examples
    --------
    >>> from fast_vollib.surface.fitting.state_space import LinearGaussianModel
    >>> model = LinearGaussianModel.random_walk(2, transition_variance=1e-4,
    ...                                         observation_variance=1e-6)
    >>> model.state_dimension, model.observation_dimension
    (2, 2)
    >>> model.transition_matrix.tolist()
    [[1.0, 0.0], [0.0, 1.0]]
    """

    transition_matrix: Any
    transition_covariance: Any
    observation_matrix: Any
    observation_covariance: Any
    drift: Any = None

    def __post_init__(self) -> None:
        transition = _square(self.transition_matrix, "transition_matrix")
        n = transition.shape[0]
        transition_covariance = _validated_covariance(
            self.transition_covariance, "transition_covariance", n
        )
        observation = np.array(self.observation_matrix, dtype=np.float64, copy=True)
        if observation.ndim != 2:
            raise SurfaceValidationError(
                f"observation_matrix must be two-dimensional (m, n); got shape {observation.shape}."
            )
        if observation.shape[1] != n:
            raise SurfaceValidationError(
                f"observation_matrix must have {n} columns to match transition_matrix; "
                f"got shape {observation.shape}. The observation is a linear map out of "
                f"the state, so its column count is the state dimension."
            )
        if not bool(np.all(np.isfinite(observation))):
            raise SurfaceValidationError(
                "observation_matrix must be finite everywhere; got a non-finite entry."
            )
        observation = read_only(observation)
        observation_covariance = _validated_covariance(
            self.observation_covariance, "observation_covariance", observation.shape[0]
        )
        if self.drift is None:
            drift = read_only(np.zeros(n, dtype=np.float64))
        else:
            drift = owned_float_1d(self.drift, "drift", n)
            if drift.size != n:
                raise SurfaceValidationError(
                    f"drift must have one entry per state component ({n}); got {drift.size}."
                )
            if not bool(np.all(np.isfinite(drift))):
                raise SurfaceValidationError("drift must be finite everywhere.")
        object.__setattr__(self, "transition_matrix", transition)
        object.__setattr__(self, "transition_covariance", transition_covariance)
        object.__setattr__(self, "observation_matrix", observation)
        object.__setattr__(self, "observation_covariance", observation_covariance)
        object.__setattr__(self, "drift", drift)

    @property
    def state_dimension(self) -> int:
        """Number of state components :math:`n`."""
        return int(self.transition_matrix.shape[0])

    @property
    def observation_dimension(self) -> int:
        """Number of observation rows :math:`m`."""
        return int(self.observation_matrix.shape[0])

    @classmethod
    def random_walk(
        cls,
        dimension: int,
        *,
        transition_variance: float = DEFAULT_TRANSITION_VARIANCE,
        observation_variance: float = DEFAULT_OBSERVATION_VARIANCE,
    ) -> LinearGaussianModel:
        """The random walk plus noise: :math:`F = H = I`, isotropic :math:`Q` and :math:`R`.

        The default dynamics for a parameter path.  Its point forecast at every
        horizon is the filtered level, which is the state-space statement of
        the persistence baseline -- so a state-space forecaster starts from the
        thing it has to beat rather than from something more elaborate that has
        not been shown to help.

        Parameters
        ----------
        dimension:
            Number of parameters, at least 1.
        transition_variance, observation_variance:
            Non-negative variances of the walk's innovation and of the
            measurement noise.

        Examples
        --------
        >>> from fast_vollib.surface.fitting.state_space import LinearGaussianModel
        >>> LinearGaussianModel.random_walk(1, transition_variance=4e-4,
        ...                                 observation_variance=0.0).transition_covariance.tolist()
        [[0.0004]]
        """
        if isinstance(dimension, bool) or not isinstance(dimension, (int, np.integer)):
            raise SurfaceValidationError(
                f"dimension must be an integer; got {type(dimension).__name__}."
            )
        if int(dimension) < 1:
            raise SurfaceValidationError(f"dimension must be at least 1; got {int(dimension)}.")
        size = int(dimension)
        identity = np.eye(size)
        return cls(
            transition_matrix=identity,
            transition_covariance=float(transition_variance) * identity,
            observation_matrix=identity,
            observation_covariance=float(observation_variance) * identity,
        )


@dataclass(frozen=True, slots=True)
class KalmanUpdate:
    """What one measurement did: a new state, and the evidence it carried.

    Parameters
    ----------
    state:
        The posterior :class:`GaussianState` after the measurement.
    innovation:
        :math:`v = y - H x^-` over the rows present at this step, shape
        ``(m', )``.  Empty when the step was fully masked.
    innovation_covariance:
        :math:`S = H P^- H^{\\top} + R` over those rows, shape ``(m', m')``.
    gain:
        The Kalman gain :math:`K = P^- H^{\\top} S^{-1}`, shape ``(n, m')``.
        Reported because it is the quantity a steady-state analysis is about,
        and recomputing it from the states outside would duplicate the solve
        this update already did.
    log_likelihood:
        :math:`\\log p(y_t \\mid y_{1:t-1})` for the present rows -- the term
        this step contributes to the path's log-likelihood.  Exactly ``0.0``
        for a fully masked step, which is the log of an empty product and not a
        placeholder.

    Notes
    -----
    The innovation, its covariance, and the gain cover only the rows the mask
    kept, in their original order.  A step with a missing row therefore returns
    smaller arrays rather than padded ones: a padded innovation would have to
    invent a value for a measurement nobody made.
    """

    state: GaussianState
    innovation: Any
    innovation_covariance: Any
    gain: Any
    log_likelihood: float

    def __post_init__(self) -> None:
        if not isinstance(self.state, GaussianState):
            raise SurfaceValidationError(
                f"state must be a GaussianState; got {type(self.state).__name__}."
            )
        innovation = owned_float_1d(self.innovation, "innovation")
        m = innovation.size
        innovation_covariance = owned_float_2d(
            self.innovation_covariance, "innovation_covariance", (m, m)
        )
        gain = owned_float_2d(self.gain, "gain", (self.state.dimension, m))
        value = float(self.log_likelihood)
        if not np.isfinite(value):
            raise SurfaceValidationError(
                f"log_likelihood must be finite; got {value!r}. A non-finite "
                f"contribution means the innovation covariance was singular, which is "
                f"reported as an error rather than carried forward as a number."
            )
        object.__setattr__(self, "innovation", innovation)
        object.__setattr__(self, "innovation_covariance", innovation_covariance)
        object.__setattr__(self, "gain", gain)
        object.__setattr__(self, "log_likelihood", value)

    @property
    def n_observed(self) -> int:
        """How many observation rows were present at this step."""
        return int(self.innovation.size)


@dataclass(frozen=True, slots=True)
class FilteredPath:
    """One forward pass: the predicted and filtered states, and the evidence.

    Parameters
    ----------
    predicted:
        :math:`p(x_t \\mid y_{1:t-1})` for each step, in time order.
    filtered:
        :math:`p(x_t \\mid y_{1:t})` for each step, in time order.
    log_likelihood:
        :math:`\\sum_t \\log p(y_t \\mid y_{1:t-1})`, the path's total
        log-likelihood under the model that produced it.  It is the quantity to
        maximize over model parameters, and the reason the filter reports it
        rather than leaving a caller to re-derive it from the innovations.

    Notes
    -----
    Both sequences are kept because the Rauch-Tung-Striebel smoother needs
    both: its backward gain divides by the *predicted* covariance of the next
    step.  Storing only the filtered states would force the smoother to
    recompute a prediction the filter had already made, and any drift between
    the two computations would show up as a smoother that disagrees with its
    own filter.
    """

    predicted: Any
    filtered: Any
    log_likelihood: float

    def __post_init__(self) -> None:
        predicted = tuple(self.predicted)
        filtered = tuple(self.filtered)
        if len(predicted) != len(filtered):
            raise SurfaceValidationError(
                f"predicted and filtered must have one state per step; got "
                f"{len(predicted)} and {len(filtered)}."
            )
        for name, states in (("predicted", predicted), ("filtered", filtered)):
            for state in states:
                if not isinstance(state, GaussianState):
                    raise SurfaceValidationError(
                        f"{name} must contain GaussianState values; got {type(state).__name__}."
                    )
        value = float(self.log_likelihood)
        if not np.isfinite(value):
            raise SurfaceValidationError(f"log_likelihood must be finite; got {value!r}.")
        object.__setattr__(self, "predicted", predicted)
        object.__setattr__(self, "filtered", filtered)
        object.__setattr__(self, "log_likelihood", value)

    @property
    def n_steps(self) -> int:
        """Number of observation steps in the pass."""
        return len(self.filtered)

    @property
    def terminal(self) -> GaussianState:
        """The last filtered state -- where a forecast starts from."""
        return self.filtered[-1]


def kalman_predict(state: GaussianState, model: LinearGaussianModel) -> GaussianState:
    """One transition: :math:`x^- = F x + b`, :math:`P^- = F P F^{\\top} + Q`.

    Parameters
    ----------
    state:
        The current belief.
    model:
        The matrices to move it with.  Its state dimension must match.

    Returns
    -------
    GaussianState
        The prediction one step ahead.  Its covariance is symmetrised, so it is
        bitwise symmetric whatever order the products were summed in.

    Raises
    ------
    SurfaceValidationError
        If the arguments are of the wrong type or the dimensions disagree.

    Notes
    -----
    A pure function of two values: calling it twice on the same state returns
    the same prediction, and calling it ``h`` times is how a horizon is
    honoured.  There is no ``self`` to advance.

    Examples
    --------
    >>> from fast_vollib.surface.fitting.state_space import (
    ...     GaussianState, LinearGaussianModel, kalman_predict,
    ... )
    >>> model = LinearGaussianModel.random_walk(1, transition_variance=0.01,
    ...                                         observation_variance=1.0)
    >>> ahead = kalman_predict(GaussianState.isotropic([0.2], 0.04), model)
    >>> ahead.mean.tolist(), ahead.covariance.tolist()
    ([0.2], [[0.05]])
    """
    _check_state(state, "state")
    _check_model(model, "model")
    if state.dimension != model.state_dimension:
        raise SurfaceValidationError(
            f"state must have the model's state dimension ({model.state_dimension}); "
            f"got {state.dimension}. A transition matrix and a state of different "
            f"sizes describe two different systems."
        )
    transition = model.transition_matrix
    mean = transition @ state.mean + model.drift
    covariance = _symmetrise(
        transition @ state.covariance @ transition.T + model.transition_covariance
    )
    return GaussianState(mean=mean, covariance=covariance)


def kalman_update(
    state: GaussianState,
    model: LinearGaussianModel,
    observation: Any,
    *,
    observation_mask: Any = None,
) -> KalmanUpdate:
    """One measurement, in Joseph form, over the rows that are present.

    Parameters
    ----------
    state:
        The predicted belief :math:`p(x_t \\mid y_{1:t-1})`.
    model:
        The matrices :math:`H` and :math:`R`.
    observation:
        The measurement :math:`y_t`, shape ``(m,)``.  ``NaN`` marks a row as
        absent, the same convention
        :class:`~fast_vollib.surface.observations.SurfaceObservations` uses.
    observation_mask:
        Optional boolean shape ``(m,)`` selecting the rows present at this
        step.  ``None`` derives it as ``~isnan(observation)`` -- a ``NaN`` is a
        row nobody quoted, while an infinity is a corrupt one and is refused
        rather than quietly skipped.  A row the mask keeps must be finite; a row
        it drops may be anything.

    Returns
    -------
    KalmanUpdate
        The posterior state, the innovation and its covariance over the present
        rows, the gain, and this step's log-likelihood contribution.

    Raises
    ------
    SurfaceValidationError
        If the shapes disagree, or a row the mask keeps is not finite.
    SurfaceCalibrationError
        If :math:`S = H P^- H^{\\top} + R` is not positive definite.  That means
        the model claims the measurement carries no uncertainty in some
        direction *and* the state has none there either, so the update is not
        defined; it is reported rather than regularised, because a jitter added
        here would change the model without appearing in it.

    Notes
    -----
    The covariance update is the **Joseph form**

    .. math::

        P^+ = (I - K H) P^- (I - K H)^{\\top} + K R K^{\\top},

    not the shorter :math:`P^+ = (I - K H) P^-`.  The two are algebraically
    equal at the optimal gain, and they are not equal in floating point: the
    short form is a difference of two nearly equal matrices, so it loses
    symmetry immediately and, after enough steps, positive definiteness -- at
    which point the Cholesky factorization fails with a cause that is thousands
    of operations upstream.  The Joseph form is a sum of two positive
    semi-definite terms, so it stays symmetric positive semi-definite by
    construction, for any gain, at roughly twice the cost.

    Examples
    --------
    >>> import numpy as np
    >>> from fast_vollib.surface.fitting.state_space import (
    ...     GaussianState, LinearGaussianModel, kalman_update,
    ... )
    >>> model = LinearGaussianModel.random_walk(1, transition_variance=0.0,
    ...                                         observation_variance=1.0)
    >>> update = kalman_update(GaussianState.isotropic([0.0], 1.0), model, [1.0])
    >>> float(round(update.gain[0, 0], 12)), float(round(update.state.mean[0], 12))
    (0.5, 0.5)
    >>> masked = kalman_update(GaussianState.isotropic([0.0], 1.0), model, [np.nan])
    >>> masked.n_observed, masked.log_likelihood
    (0, 0.0)
    """
    _check_state(state, "state")
    _check_model(model, "model")
    if state.dimension != model.state_dimension:
        raise SurfaceValidationError(
            f"state must have the model's state dimension ({model.state_dimension}); "
            f"got {state.dimension}."
        )
    m = model.observation_dimension
    values = owned_float_1d(observation, "observation")
    if values.size != m:
        raise SurfaceValidationError(
            f"observation must have one entry per observation row ({m}); got {values.size}."
        )
    if observation_mask is None:
        mask = ~np.isnan(values)
    else:
        mask = owned_bool_1d(observation_mask, "observation_mask", m)
    present = np.flatnonzero(mask)
    n = state.dimension
    if present.size == 0:
        return KalmanUpdate(
            state=state,
            innovation=np.zeros(0),
            innovation_covariance=np.zeros((0, 0)),
            gain=np.zeros((n, 0)),
            log_likelihood=0.0,
        )
    y = values[present]
    if not bool(np.all(np.isfinite(y))):
        raise SurfaceValidationError(
            "observation must be finite on every row observation_mask keeps; got a "
            "non-finite entry. A row that is absent is declared absent by the mask (or "
            "by a NaN with no mask), never smuggled in as a NaN the update would use."
        )
    H = model.observation_matrix[present, :]
    R = model.observation_covariance[np.ix_(present, present)]
    P = state.covariance
    innovation = y - H @ state.mean
    innovation_covariance = _symmetrise(H @ P @ H.T + R)
    factor = _cholesky(
        innovation_covariance,
        "innovation covariance H P H' + R",
        "Give the state or the observation a strictly positive variance in that "
        "direction: with neither, the model asserts the measurement is already known "
        "exactly and there is nothing for the update to do.",
    )
    gain = cho_solve(factor, H @ P).T
    mean = state.mean + gain @ innovation
    closed_loop = np.eye(n) - gain @ H
    covariance = _symmetrise(closed_loop @ P @ closed_loop.T + gain @ R @ gain.T)
    log_determinant = 2.0 * float(np.sum(np.log(np.abs(np.diag(factor[0])))))
    quadratic = float(innovation @ cho_solve(factor, innovation))
    log_likelihood = -0.5 * (present.size * _LOG_TWO_PI + log_determinant + quadratic)
    return KalmanUpdate(
        state=GaussianState(mean=mean, covariance=covariance),
        innovation=innovation,
        innovation_covariance=innovation_covariance,
        gain=gain,
        log_likelihood=log_likelihood,
    )


def kalman_filter(
    observations: Any,
    model: LinearGaussianModel,
    initial: GaussianState,
    *,
    observation_mask: Any = None,
) -> FilteredPath:
    """Predict and update along a path, keeping every state on the way.

    Parameters
    ----------
    observations:
        Shape ``(S, m)``, one row per step in time order, oldest first.
        ``NaN`` marks an absent measurement.
    model:
        The matrices, the same at every step.
    initial:
        The belief *before* the first transition -- the state at step zero, one
        transition earlier than ``observations[0]``.  Each step is a
        :func:`kalman_predict` followed by a :func:`kalman_update`, with no
        special case for the first, so a warm start is expressed by the state
        passed in rather than by a flag.
    observation_mask:
        Optional boolean shape ``(S, m)`` presence mask, overriding the ``NaN``
        convention row by row.

    Returns
    -------
    FilteredPath
        The predicted and filtered states and the total log-likelihood.

    Raises
    ------
    SurfaceValidationError
        If ``observations`` is not a ``(S, m)`` array with ``S >= 1`` and ``m``
        the model's observation dimension, or the mask does not match it.
    SurfaceCalibrationError
        Propagated from :func:`kalman_update` when an innovation covariance is
        singular.

    Examples
    --------
    >>> import numpy as np
    >>> from fast_vollib.surface.fitting.state_space import (
    ...     GaussianState, LinearGaussianModel, kalman_filter,
    ... )
    >>> model = LinearGaussianModel.random_walk(1, transition_variance=1e-2,
    ...                                         observation_variance=1e-2)
    >>> path = kalman_filter(np.array([[1.0], [1.0], [1.0]]), model,
    ...                      GaussianState.isotropic([0.0], 1.0))
    >>> bool(path.terminal.mean[0] > 0.9), path.n_steps
    (True, 3)
    """
    _check_model(model, "model")
    _check_state(initial, "initial")
    values = np.array(observations, dtype=np.float64, copy=True)
    if values.ndim != 2:
        raise SurfaceValidationError(
            f"observations must be two-dimensional (steps, rows); got shape {values.shape}. "
            f"A one-dimensional path is ambiguous between one step of m rows and m steps "
            f"of one row, and guessing would silently transpose the problem."
        )
    if values.shape[0] < 1:
        raise SurfaceValidationError(
            f"observations must contain at least one step; got {values.shape[0]}. There is "
            f"nothing to filter, and returning the prior as a filtered path would report a "
            f"belief no data supports."
        )
    if values.shape[1] != model.observation_dimension:
        raise SurfaceValidationError(
            f"observations must have one column per observation row "
            f"({model.observation_dimension}); got {values.shape[1]}."
        )
    masks: Any = None
    if observation_mask is not None:
        masks = np.asarray(observation_mask)
        if masks.dtype != np.bool_ or masks.shape != values.shape:
            raise SurfaceValidationError(
                f"observation_mask must be a boolean array of shape {values.shape}; got "
                f"dtype {masks.dtype} and shape {masks.shape}."
            )
    predicted: list[GaussianState] = []
    filtered: list[GaussianState] = []
    state = initial
    total = 0.0
    for step in range(values.shape[0]):
        prior = kalman_predict(state, model)
        update = kalman_update(
            prior,
            model,
            values[step],
            observation_mask=None if masks is None else masks[step],
        )
        predicted.append(prior)
        filtered.append(update.state)
        total += update.log_likelihood
        state = update.state
    return FilteredPath(predicted=predicted, filtered=filtered, log_likelihood=total)


def kalman_smooth(path: FilteredPath, model: LinearGaussianModel) -> tuple[GaussianState, ...]:
    """The Rauch-Tung-Striebel backward pass over a filtered path.

    Parameters
    ----------
    path:
        A :class:`FilteredPath` from :func:`kalman_filter` under ``model``.
    model:
        The same matrices the forward pass used.  Passing different ones is not
        detectable here and would produce smoothed states that are the solution
        to no stated problem, which is why the model is an argument rather than
        something the path carries.

    Returns
    -------
    tuple of GaussianState
        :math:`p(x_t \\mid y_{1:S})` for each step, in time order.

    Raises
    ------
    SurfaceCalibrationError
        If a predicted covariance is singular, so the backward gain
        :math:`J_t = P^f_t F^{\\top} (P^p_{t+1})^{-1}` does not exist.  A state
        direction with no transition noise and no prior uncertainty is exactly
        determined by its own past, and there is nothing for the future to say
        about it.

    Notes
    -----
    The terminal smoothed state *is* the terminal filtered state -- the same
    object, not a recomputation -- because conditioning on :math:`y_{1:S}` at
    the last step is what the filter already did.  Recomputing it would put a
    round-off difference between the filter and the smoother at the one point
    where they must agree exactly.

    Examples
    --------
    >>> import numpy as np
    >>> from fast_vollib.surface.fitting.state_space import (
    ...     GaussianState, LinearGaussianModel, kalman_filter, kalman_smooth,
    ... )
    >>> model = LinearGaussianModel.random_walk(1, transition_variance=1e-2,
    ...                                         observation_variance=1e-2)
    >>> path = kalman_filter(np.array([[0.0], [1.0]]), model,
    ...                      GaussianState.isotropic([0.0], 1.0))
    >>> smoothed = kalman_smooth(path, model)
    >>> smoothed[-1] is path.terminal
    True
    >>> bool(smoothed[0].trace <= path.filtered[0].trace)
    True
    """
    if not isinstance(path, FilteredPath):
        raise SurfaceValidationError(f"path must be a FilteredPath; got {type(path).__name__}.")
    _check_model(model, "model")
    steps = path.n_steps
    smoothed: list[GaussianState] = [path.filtered[-1]]
    transition = model.transition_matrix
    for index in range(steps - 2, -1, -1):
        current = path.filtered[index]
        ahead_predicted = path.predicted[index + 1]
        ahead_smoothed = smoothed[0]
        factor = _cholesky(
            ahead_predicted.covariance,
            "predicted covariance F P F' + Q",
            "A direction with no transition noise and no prior uncertainty is already "
            "determined by the past, so the smoother has nothing to add there; give the "
            "transition a strictly positive variance in that direction to smooth it.",
        )
        gain = cho_solve(factor, transition @ current.covariance).T
        mean = current.mean + gain @ (ahead_smoothed.mean - ahead_predicted.mean)
        covariance = _symmetrise(
            current.covariance
            + gain @ (ahead_smoothed.covariance - ahead_predicted.covariance) @ gain.T
        )
        smoothed.insert(0, GaussianState(mean=mean, covariance=covariance))
    return tuple(smoothed)


def flat_level_parameters(surface: "DefiniteIVSurface") -> np.ndarray:
    """The one-parameter vector of a :class:`~fast_vollib.surface.fitting.FlatIVSurface`.

    The default extractor of :class:`StateSpaceForecaster`, paired with
    :func:`flat_surface_from_parameters`.

    Parameters
    ----------
    surface:
        A surface carrying a scalar ``level`` attribute.

    Returns
    -------
    numpy.ndarray
        Shape ``(1,)``, the level.

    Raises
    ------
    SurfaceValidationError
        If the surface has no ``level``.  Silently substituting an
        implied-volatility evaluated somewhere would make the state space
        describe a different quantity than the one the calibrator fitted.

    Examples
    --------
    >>> from fast_vollib.surface.fitting import FlatIVSurface
    >>> from fast_vollib.surface.fitting.state_space import flat_level_parameters
    >>> flat_level_parameters(FlatIVSurface(level=0.2)).tolist()
    [0.2]
    """
    level = getattr(surface, "level", None)
    if level is None:
        raise SurfaceValidationError(
            f"surface must expose a scalar 'level' to be extracted by "
            f"flat_level_parameters; got {type(surface).__name__}. Pass the extractor "
            f"that matches the calibrator you configured."
        )
    return np.array([float(level)], dtype=np.float64)


def flat_surface_from_parameters(parameters: Any) -> FlatIVSurface:
    """The inverse of :func:`flat_level_parameters`: a one-vector back to a flat surface.

    Parameters
    ----------
    parameters:
        Shape ``(1,)``, the forecast level.

    Returns
    -------
    FlatIVSurface
        The surface that level describes.

    Raises
    ------
    SurfaceValidationError
        If ``parameters`` does not have exactly one entry, or the level it
        carries is not a usable implied volatility.

    Examples
    --------
    >>> from fast_vollib.surface.fitting.state_space import flat_surface_from_parameters
    >>> flat_surface_from_parameters([0.25]).level
    0.25
    """
    values = owned_float_1d(parameters, "parameters")
    if values.size != 1:
        raise SurfaceValidationError(
            f"parameters must have exactly one entry to build a flat surface; got "
            f"{values.size}. The reconstructor and the extractor must describe the same "
            f"parameterization."
        )
    return FlatIVSurface(level=float(values[0]))


@dataclass(frozen=True, slots=True)
class StateSpaceForecaster:
    """Filters a fitted parameter path and projects it, horizon by horizon.

    The composition is deliberate and visible: a **calibrator** turns each
    element of the history into a definite surface, an **extractor** turns that
    surface into a parameter vector, a **linear Gaussian model** filters the
    resulting path, and a **reconstructor** turns the projected parameter
    vector back into a definite surface.  Nothing here knows what the
    parameters mean, so "state-space dynamics on an SVI fit" and "state-space
    dynamics on a flat level" are the same object with different parts.

    Parameters
    ----------
    calibrator:
        Fits one surface per element of the history.  Defaults to
        :class:`~fast_vollib.surface.fitting.FlatVolatilityCalibrator`, which
        makes the baseline as weak as it can honestly be.
    extract_parameters:
        Callable from a definite surface to a finite 1-D parameter vector, the
        same length for every element of the history.  Defaults to
        :func:`flat_level_parameters`, which matches the default calibrator.
    reconstruct_surface:
        Callable from a parameter vector back to a definite surface.  Defaults
        to :func:`flat_surface_from_parameters`.
    model:
        The dynamics.  ``None`` (default) builds
        :meth:`LinearGaussianModel.random_walk` at the dimension the extractor
        turns out to produce, using ``transition_variance`` and
        ``observation_variance``.  A supplied model overrides both and may have
        a larger state than the parameter vector -- a local linear trend, say --
        in which case ``initial_state`` is required, because there is no
        defensible way to guess a slope from one level.
    initial_state:
        Optional explicit :class:`GaussianState` to start the filter from.
        ``None`` uses the first extracted parameter vector as the mean and
        ``initial_variance`` on the diagonal, which requires the model's state
        and observation dimensions to agree.
    transition_variance, observation_variance, initial_variance:
        Variances for the default model and the default initial state, on the
        scale of the parameters themselves.  See
        :data:`DEFAULT_TRANSITION_VARIANCE`,
        :data:`DEFAULT_OBSERVATION_VARIANCE`, :data:`DEFAULT_INITIAL_VARIANCE`.
        ``initial_variance`` must be strictly positive: a state known exactly
        before any data has arrived cannot learn from the first observation.

    Notes
    -----
    Satisfies :class:`~fast_vollib.surface.protocols.SurfaceForecaster` and
    returns a **definite surface -- the point forecast**.  The filter does
    produce a predictive covariance over the *parameters*, and
    :meth:`forecast_state` reports it, but a covariance over parameters is not
    a distribution over surfaces: turning one into the other needs either a
    delta method through the reconstructor or a sampler, and both are model
    choices that have not been made here.  Reporting an interval this class did
    not derive would be inventing a calibrated uncertainty, so
    :meth:`forecast` does not, and a predictive distribution over surfaces
    remains a later capability.

    Holds configuration only.  Every array a forecast touches is local to the
    call, so two forecasts from the same history are the same forecast --
    bitwise -- whatever ran between them.

    Examples
    --------
    >>> from fast_vollib.surface import ForecastHorizon, SurfaceObservations, SurfacePoints
    >>> from fast_vollib.surface.fitting.state_space import StateSpaceForecaster
    >>> history = [
    ...     SurfaceObservations(k=[0.0], T=[1.0], iv=[0.20]),
    ...     SurfaceObservations(k=[0.0], T=[1.0], iv=[0.24]),
    ... ]
    >>> forecaster = StateSpaceForecaster(observation_variance=0.0)
    >>> surface = forecaster.forecast(history, ForecastHorizon(steps=3))
    >>> float(round(surface.evaluate(SurfacePoints(k=[0.1], T=[2.0])).iv[0], 6))
    0.24
    """

    calibrator: "SurfaceCalibrator" = field(default_factory=FlatVolatilityCalibrator)
    extract_parameters: Callable[[Any], Any] = flat_level_parameters
    reconstruct_surface: Callable[[Any], Any] = flat_surface_from_parameters
    model: LinearGaussianModel | None = None
    initial_state: GaussianState | None = None
    transition_variance: float = DEFAULT_TRANSITION_VARIANCE
    observation_variance: float = DEFAULT_OBSERVATION_VARIANCE
    initial_variance: float = DEFAULT_INITIAL_VARIANCE

    def __post_init__(self) -> None:
        if not callable(self.extract_parameters):
            raise SurfaceValidationError(
                f"extract_parameters must be callable; got "
                f"{type(self.extract_parameters).__name__}."
            )
        if not callable(self.reconstruct_surface):
            raise SurfaceValidationError(
                f"reconstruct_surface must be callable; got "
                f"{type(self.reconstruct_surface).__name__}."
            )
        if self.model is not None and not isinstance(self.model, LinearGaussianModel):
            raise SurfaceValidationError(
                f"model must be a LinearGaussianModel or None; got {type(self.model).__name__}."
            )
        if self.initial_state is not None and not isinstance(self.initial_state, GaussianState):
            raise SurfaceValidationError(
                f"initial_state must be a GaussianState or None; got "
                f"{type(self.initial_state).__name__}."
            )
        for name in ("transition_variance", "observation_variance"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise SurfaceValidationError(
                    f"{name} must be finite and non-negative; got {value!r}. A negative "
                    f"variance describes no distribution."
                )
            object.__setattr__(self, name, value)
        initial_variance = float(self.initial_variance)
        if not np.isfinite(initial_variance) or initial_variance <= 0.0:
            raise SurfaceValidationError(
                f"initial_variance must be finite and strictly positive; got "
                f"{initial_variance!r}. A state known exactly before any data has "
                f"arrived cannot be updated by the first observation."
            )
        object.__setattr__(self, "initial_variance", initial_variance)

    def parameter_path(
        self,
        history: Sequence["SurfaceObservations"],
        *,
        rng: "RNGInput" = None,
    ) -> np.ndarray:
        """Calibrate every element of ``history`` and stack the parameter vectors.

        Parameters
        ----------
        history:
            Observation sets, oldest first.
        rng:
            Randomness for the calibrator.  An integer builds **one** generator
            that every fit in the history shares, so the second day's fit does
            not replay the first day's stream; a generator is used as given and
            advances; ``None`` is passed through as ``None``, leaving a
            deterministic calibrator deterministic.

        Returns
        -------
        numpy.ndarray
            Shape ``(len(history), n_parameters)``, one row per element in time
            order.

        Raises
        ------
        SurfaceValidationError
            If ``history`` is not a non-empty sequence.
        SurfaceCalibrationError
            If the extractor returns something that is not a finite 1-D vector,
            or the vectors are not all the same length.  A path whose width
            changes describes a different model on different days.
        """
        if isinstance(history, str) or not isinstance(history, Sequence):
            raise SurfaceValidationError(
                f"history must be a sequence of SurfaceObservations, oldest first; got "
                f"{type(history).__name__}."
            )
        if len(history) == 0:
            raise SurfaceValidationError(
                "history is empty, so there is no parameter path to filter. A forecast "
                "from no observations would be a statement about the prior alone."
            )
        stream = rng if rng is None else np.random.default_rng(rng)
        rows: list[np.ndarray] = []
        for index, observations in enumerate(history):
            surface = self.calibrator.fit(observations, rng=stream)
            vector = np.asarray(self.extract_parameters(surface), dtype=np.float64)
            if vector.ndim != 1 or vector.size == 0:
                raise SurfaceCalibrationError(
                    f"extract_parameters must return a non-empty one-dimensional vector; "
                    f"got shape {vector.shape} at history index {index}. The state space "
                    f"is built on that vector, so its shape is the model's shape."
                )
            if not bool(np.all(np.isfinite(vector))):
                raise SurfaceCalibrationError(
                    f"extract_parameters must return finite values; got a non-finite "
                    f"entry at history index {index}. Filtering a NaN turns the whole "
                    f"path into NaN one step later."
                )
            if rows and vector.size != rows[0].size:
                raise SurfaceCalibrationError(
                    f"extract_parameters must return the same number of parameters at "
                    f"every step ({rows[0].size}); got {vector.size} at history index "
                    f"{index}."
                )
            rows.append(vector)
        return np.vstack(rows)

    def forecast_state(
        self,
        history: Sequence["SurfaceObservations"],
        horizon: "ForecastHorizon",
        *,
        rng: "RNGInput" = None,
    ) -> GaussianState:
        """The ``horizon``-step-ahead predictive state of the parameter vector.

        Parameters
        ----------
        history:
            Observation sets, oldest first.
        horizon:
            How far ahead.  ``horizon.steps`` transitions are applied, one
            :func:`kalman_predict` each -- so the horizon is honoured by
            construction rather than by a closed form that would have to be
            re-derived for every model.  ``horizon.step_years`` is accepted and
            unused: this model works in step units and does not age maturities,
            and guessing what a step is worth in calendar time is exactly what
            the protocol says never to do.
        rng:
            Passed to the calibrator; see :meth:`parameter_path`.

        Returns
        -------
        GaussianState
            The predictive distribution of the *state*.  Its covariance is a
            covariance over parameters, not over surfaces.

        Raises
        ------
        SurfaceValidationError
            If ``history`` is empty, ``horizon`` is not a
            :class:`~fast_vollib.surface.protocols.ForecastHorizon`, or a
            supplied model does not fit the extracted parameter vector.
        SurfaceCalibrationError
            Propagated from the calibrator, the extractor, or a singular
            innovation covariance in the filter.
        """
        return self._project(history, horizon, rng)[1]

    def forecast(
        self,
        history: Sequence["SurfaceObservations"],
        horizon: "ForecastHorizon",
        *,
        rng: "RNGInput" = None,
    ) -> "DefiniteIVSurface":
        """Forecast ``horizon`` ahead from ``history`` and rebuild the surface.

        Parameters
        ----------
        history:
            Observation sets, oldest first.
        horizon:
            How far ahead, in observation steps.
        rng:
            Passed to the calibrator; see :meth:`parameter_path`.

        Returns
        -------
        DefiniteIVSurface
            The **point forecast**: the surface the predicted mean parameter
            vector describes.  The predictive covariance is available from
            :meth:`forecast_state` and is deliberately not smuggled into this
            return value as an interval nobody derived.

        Raises
        ------
        SurfaceValidationError
            If ``history`` is empty or the arguments do not fit the model.
        SurfaceCalibrationError
            If the filter cannot produce a state it stands behind, or the
            forecast parameters are not finite.  A silently defaulted surface
            is never returned.

        Examples
        --------
        >>> from fast_vollib.surface import ForecastHorizon, SurfaceObservations
        >>> from fast_vollib.surface.fitting.state_space import StateSpaceForecaster
        >>> history = [SurfaceObservations(k=[0.0], T=[1.0], iv=[0.2])]
        >>> StateSpaceForecaster().forecast(history, ForecastHorizon(steps=1)).level
        0.2
        """
        model, state = self._project(history, horizon, rng)
        parameters = model.observation_matrix @ state.mean
        if not bool(np.all(np.isfinite(parameters))):
            raise SurfaceCalibrationError(
                f"The forecast parameter vector must be finite; got {parameters.tolist()}. "
                f"A surface is not rebuilt from parameters the filter could not produce."
            )
        return self.reconstruct_surface(parameters)

    def _project(
        self,
        history: Sequence["SurfaceObservations"],
        horizon: "ForecastHorizon",
        rng: "RNGInput",
    ) -> tuple[LinearGaussianModel, GaussianState]:
        """Filter the parameter path and apply the transition ``horizon.steps`` times."""
        if not isinstance(horizon, ForecastHorizon):
            raise SurfaceValidationError(
                f"horizon must be a ForecastHorizon; got {type(horizon).__name__}."
            )
        path = self.parameter_path(history, rng=rng)
        model = self._model_for(path.shape[1])
        initial = self._initial_for(model, path[0])
        state = kalman_filter(path, model, initial).terminal
        for _ in range(horizon.steps):
            state = kalman_predict(state, model)
        return model, state

    def _model_for(self, dimension: int) -> LinearGaussianModel:
        """The configured model, or the default random walk at ``dimension``."""
        if self.model is not None:
            if self.model.observation_dimension != dimension:
                raise SurfaceValidationError(
                    f"model must observe one row per extracted parameter ({dimension}); "
                    f"got {self.model.observation_dimension}. The parameter vector is "
                    f"the observation, so the two dimensions are the same number."
                )
            return self.model
        return LinearGaussianModel.random_walk(
            dimension,
            transition_variance=self.transition_variance,
            observation_variance=self.observation_variance,
        )

    def _initial_for(self, model: LinearGaussianModel, first: np.ndarray) -> GaussianState:
        """The state the filter starts from: the configured one, or the first fit."""
        if self.initial_state is not None:
            if self.initial_state.dimension != model.state_dimension:
                raise SurfaceValidationError(
                    f"initial_state must have the model's state dimension "
                    f"({model.state_dimension}); got {self.initial_state.dimension}."
                )
            return self.initial_state
        if model.state_dimension != model.observation_dimension:
            raise SurfaceValidationError(
                f"initial_state is required for a model whose state ({model.state_dimension}) "
                f"is not its observation ({model.observation_dimension}); got None. The "
                f"first parameter vector says nothing about the components it does not "
                f"observe -- a level does not imply a slope -- and guessing one would put "
                f"an unstated prior into every forecast."
            )
        return GaussianState.isotropic(first, self.initial_variance)


def _check_state(state: Any, name: str) -> None:
    """Reject anything that is not a :class:`GaussianState`."""
    if not isinstance(state, GaussianState):
        raise SurfaceValidationError(
            f"{name} must be a GaussianState; got {type(state).__name__}. The recursions "
            f"take the state as a value precisely so that it cannot be something else."
        )


def _check_model(model: Any, name: str) -> None:
    """Reject anything that is not a :class:`LinearGaussianModel`."""
    if not isinstance(model, LinearGaussianModel):
        raise SurfaceValidationError(
            f"{name} must be a LinearGaussianModel; got {type(model).__name__}."
        )
