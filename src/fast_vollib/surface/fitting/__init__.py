"""Surface-fitting algorithms: representations, calibrators, and forecasters.

Every algorithm here consumes
:class:`~fast_vollib.surface.observations.SurfaceObservations` and produces a
:class:`~fast_vollib.surface.protocols.DefiniteIVSurface`.  None of them knows
about a dataset, a split, a ticker, a padded tensor, a result directory, or a
checkpoint path: those belong to whatever runs the experiment, and letting them
in here is how a numerical library becomes a pipeline nobody else can reuse.

The module lives under ``surface`` rather than at the top level because
``fast_vollib.models`` already means *pricing* models -- Black, Black-Scholes,
Black-Scholes-Merton.  Overloading it with surface parameterizations would make
"which model" an ambiguous question in a package where the answer decides which
formula runs.

:func:`builtin_providers` is the single ordered list the capability registry
reads.  Adding an algorithm means adding one entry here, so the set of public
identifiers is decided in one place and cannot change with import order.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

from ..observations import SurfaceObservations
from .factors import (
    VALUE_SPACES,
    FactorIVSurface,
    FactorPCARecipe,
    FactorSurfaceCalibrator,
    SurfaceFactorBasis,
    fit_factor_basis,
)
from .heston import (
    HESTON_OBJECTIVES,
    HestonCalibrator,
    HestonIVSurface,
    HestonParameters,
)
from .parametric import (
    PHI_FAMILIES,
    SAFE_ETA,
    SVI_OBJECTIVES,
    HestonLikePhi,
    PowerLawPhi,
    SSVICalibrator,
    SSVISurface,
    SVICalibrator,
    SVIJumpWings,
    SVIParameters,
    SVISliceFit,
    SVISmile,
    SVISurface,
    svi_total_variance,
)
from .persistence import PersistenceForecaster
from .prior import FlatIVSurface, FlatVolatilityCalibrator
from .regularized import (
    MAX_DIFFERENCE_ORDER,
    LeastSquaresDiagnostics,
    TikhonovPenalty,
    couple_parameter_sequence,
    difference_matrix,
    solve_penalized_least_squares,
)
from .splines import (
    DEFAULT_MAX_INTERIOR_KNOTS_K,
    DEFAULT_SMOOTHING,
    SplineIVSurface,
    SplineSmileCalibrator,
    SplineSurfaceCalibrator,
)
from .state_space import (
    FilteredPath,
    GaussianState,
    KalmanUpdate,
    LinearGaussianModel,
    StateSpaceForecaster,
    kalman_filter,
    kalman_predict,
    kalman_smooth,
    kalman_update,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..capabilities import _Provider
    from ..protocols import DefiniteIVSurface, SurfaceCalibrator

__all__ = [
    "DEFAULT_MAX_INTERIOR_KNOTS_K",
    "DEFAULT_SMOOTHING",
    "HESTON_OBJECTIVES",
    "MAX_DIFFERENCE_ORDER",
    "PHI_FAMILIES",
    "SAFE_ETA",
    "SVI_OBJECTIVES",
    "VALUE_SPACES",
    "FactorIVSurface",
    "FactorPCARecipe",
    "FactorSurfaceCalibrator",
    "FilteredPath",
    "FlatIVSurface",
    "FlatVolatilityCalibrator",
    "GaussianState",
    "HestonCalibrator",
    "HestonIVSurface",
    "HestonLikePhi",
    "HestonParameters",
    "KalmanUpdate",
    "LeastSquaresDiagnostics",
    "LinearGaussianModel",
    "PersistenceForecaster",
    "PowerLawPhi",
    "SSVICalibrator",
    "SSVISurface",
    "SVICalibrator",
    "SVIJumpWings",
    "SVIParameters",
    "SVISliceFit",
    "SVISmile",
    "SVISurface",
    "SplineIVSurface",
    "SplineSmileCalibrator",
    "SplineSurfaceCalibrator",
    "StateSpaceForecaster",
    "SurfaceFactorBasis",
    "TikhonovPenalty",
    "builtin_providers",
    "couple_parameter_sequence",
    "difference_matrix",
    "fit_each",
    "fit_factor_basis",
    "kalman_filter",
    "kalman_predict",
    "kalman_smooth",
    "kalman_update",
    "solve_penalized_least_squares",
    "svi_total_variance",
]


def fit_each(
    calibrator: "SurfaceCalibrator", observations: SurfaceObservations
) -> dict[Any, "DefiniteIVSurface"]:
    """Calibrate one surface per ``surface_id`` in ``observations``.

    A calibrator fits *one* surface, because a fit that silently pooled several
    days would produce parameters belonging to none of them.  This helper does
    the loop explicitly, keyed by the label, so the caller can see that it
    happened.

    Examples
    --------
    >>> from fast_vollib.surface import SurfaceObservations
    >>> from fast_vollib.surface.fitting import FlatVolatilityCalibrator, fit_each
    >>> observations = SurfaceObservations(
    ...     k=[0.0, 0.0], T=[1.0, 1.0], iv=[0.2, 0.3], surface_id=["mon", "tue"]
    ... )
    >>> {label: round(surface.level, 6) for label, surface in
    ...  fit_each(FlatVolatilityCalibrator(), observations).items()}
    {'mon': 0.2, 'tue': 0.3}
    """
    return {label: calibrator.fit(subset) for label, subset in observations.surfaces()}


def builtin_providers() -> Iterator["_Provider"]:
    """The built-in algorithm table, in a fixed order.

    Read once by :mod:`fast_vollib.surface.capabilities` and cached there.  The
    order is the order algorithms are listed in, so it is chosen deliberately:
    baselines first, then parametric representations, then everything else.
    """
    from ..capabilities import BackendSupport, SurfaceAlgorithmSpec, _Provider

    yield _Provider(
        spec=SurfaceAlgorithmSpec(
            public_id="flat",
            display_name="Flat volatility",
            family="calibrator",
            output="definite",
            summary=(
                "One implied-volatility level fitted to every observation. The control "
                "baseline: arbitrage-free by construction, and a poor fit to any real "
                "smile, so it separates 'passes the hard checks' from 'describes the "
                "market'."
            ),
            implementation_version="1",
            configuration_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "objective": {
                        "type": "string",
                        "enum": ["implied_volatility", "variance"],
                        "description": "Whether the level averages sigma or sigma squared.",
                    },
                    "use_weights": {
                        "type": "boolean",
                        "description": "Honour the observations' weight column.",
                    },
                },
                "required": [],
            },
            support=BackendSupport(),
            supports_arbitrary_points=True,
        ),
        factory=FlatVolatilityCalibrator,
    )

    yield _Provider(
        spec=SurfaceAlgorithmSpec(
            public_id="svi",
            display_name="SVI (raw parameterization)",
            family="calibrator",
            output="definite",
            summary=(
                "One five-parameter raw SVI slice per observed maturity, fitted by bounded "
                "least squares in total variance. Slices are independent, so the fit is "
                "sharp per smile and carries no joint calendar constraint; the fit record "
                "reports the achieved minimum Durrleman g rather than claiming convexity."
            ),
            implementation_version="1",
            configuration_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "objective": {
                        "type": "string",
                        "enum": list(SVI_OBJECTIVES),
                        "description": "Space the least-squares residual is measured in.",
                    },
                    "butterfly_penalty": {
                        "type": "number",
                        "minimum": 0.0,
                        "description": "Weight on a soft relu(-g) residual; a penalty, not a guarantee.",
                    },
                    "n_starts": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Deterministic starting points, differing in the initial rho.",
                    },
                    "max_iterations": {"type": "integer", "minimum": 1},
                    "maturity_interpolation": {
                        "type": "string",
                        "enum": ["total_variance_linear", "none"],
                    },
                },
                "required": [],
            },
            support=BackendSupport(),
            references=(
                "Gatheral, J. (2004). A parsimonious arbitrage-free implied volatility "
                "parameterization.",
                "Gatheral, J. and Jacquier, A. (2014). Arbitrage-free SVI volatility surfaces.",
            ),
        ),
        factory=SVICalibrator,
    )

    yield _Provider(
        spec=SurfaceAlgorithmSpec(
            public_id="ssvi",
            display_name="SSVI (surface SVI)",
            family="calibrator",
            output="definite",
            summary=(
                "One ATM total-variance term structure, one correlation and one smoothing "
                "function fitted jointly across every maturity. theta is parameterized "
                "non-decreasing, so the fitted surface cannot exhibit calendar-spread "
                "arbitrage between its pillars -- at the cost of fitting any single smile "
                "less closely than an independent slice would."
            ),
            implementation_version="1",
            configuration_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "phi_family": {"type": "string", "enum": list(PHI_FAMILIES)},
                    "enforce_no_butterfly": {
                        "type": "boolean",
                        "description": (
                            "Bound the search so the first sufficient butterfly condition "
                            "holds at every maturity."
                        ),
                    },
                    "n_starts": {"type": "integer", "minimum": 1},
                    "max_iterations": {"type": "integer", "minimum": 1},
                },
                "required": [],
            },
            support=BackendSupport(),
            references=(
                "Gatheral, J. and Jacquier, A. (2014). Arbitrage-free SVI volatility surfaces.",
            ),
        ),
        factory=SSVICalibrator,
    )

    yield _Provider(
        spec=SurfaceAlgorithmSpec(
            public_id="spline",
            display_name="Penalized tensor-product B-spline",
            family="calibrator",
            output="definite",
            summary=(
                "A tensor-product B-spline in total variance, fitted by penalized least "
                "squares with a second-difference roughness penalty in each direction. "
                "Non-parametric: it has no shape prior, so it fits what is there and "
                "extrapolates nothing -- points outside the knot span come back invalid "
                "rather than continued by a cubic that diverges."
            ),
            implementation_version="1",
            configuration_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "degree_k": {"type": "integer", "minimum": 0, "maximum": 5},
                    "degree_t": {"type": "integer", "minimum": 0, "maximum": 5},
                    "n_interior_knots_k": {"type": "integer", "minimum": 0},
                    "n_interior_knots_t": {"type": "integer", "minimum": 0},
                    "smoothing_k": {"type": "number", "minimum": 0.0},
                    "smoothing_t": {"type": "number", "minimum": 0.0},
                    "use_weights": {"type": "boolean"},
                },
                "required": [],
            },
            support=BackendSupport(),
        ),
        factory=SplineSurfaceCalibrator,
    )

    yield _Provider(
        spec=SurfaceAlgorithmSpec(
            public_id="heston",
            display_name="Heston stochastic volatility",
            family="calibrator",
            output="definite",
            summary=(
                "Five parameters fitted to the whole surface at once, priced by Fourier "
                "inversion of the characteristic function and inverted with the Jaeckel "
                "solver. The only first-wave model whose surface comes from a genuine "
                "martingale measure, and the only one that declines a wing: a point whose "
                "vega cannot carry the pricer's absolute error is reported invalid rather "
                "than answered."
            ),
            implementation_version="1",
            configuration_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "objective": {"type": "string", "enum": list(HESTON_OBJECTIVES)},
                    "n_starts": {"type": "integer", "minimum": 1},
                    "max_iterations": {"type": "integer", "minimum": 1},
                    "n_nodes": {"type": "integer", "minimum": 8},
                    "diff_step": {
                        "type": "number",
                        "exclusiveMinimum": 0.0,
                        "exclusiveMaximum": 1.0,
                    },
                },
                "required": [],
            },
            support=BackendSupport(),
            references=(
                "Heston, S. (1993). A closed-form solution for options with stochastic volatility.",
                "Lewis, A. (2000). Option Valuation under Stochastic Volatility.",
                "Albrecher, H., Mayer, P., Schoutens, W. and Tistaert, J. (2007). "
                "The little Heston trap.",
                "Andersen, L. (2008). Simple and efficient simulation of the Heston "
                "stochastic volatility model.",
            ),
        ),
        factory=HestonCalibrator,
    )

    yield _Provider(
        spec=SurfaceAlgorithmSpec(
            public_id="factor-pca",
            display_name="Deterministic SVD factor model",
            family="calibrator",
            output="definite",
            summary=(
                "A mean surface plus principal components estimated from an ensemble by a "
                "deterministic SVD with a fixed sign convention, and a single surface "
                "projected onto that basis by least squares. Two phases with two different "
                "inputs, so build_algorithm returns a recipe carrying train() and "
                "calibrator(basis) rather than a calibrator that has nothing to project on."
            ),
            implementation_version="1",
            configuration_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "n_factors": {"type": "integer", "minimum": 1},
                    "value_space": {"type": "string", "enum": list(VALUE_SPACES)},
                    "policy": {
                        "type": "string",
                        "enum": ["total_variance", "implied_volatility", "nearest"],
                    },
                    "extrapolation": {
                        "type": "string",
                        "enum": ["invalid", "error", "clamp"],
                    },
                    "use_weights": {"type": "boolean"},
                },
                "required": [],
            },
            support=BackendSupport(),
            requires_training=True,
        ),
        factory=FactorPCARecipe,
    )

    yield _Provider(
        spec=SurfaceAlgorithmSpec(
            public_id="state-space",
            display_name="Kalman state-space parameter forecast",
            family="forecaster",
            output="definite",
            summary=(
                "Fits each element of a history, extracts a parameter vector, filters the "
                "resulting path with an explicit-state linear Gaussian model, and projects "
                "it forward. Returns the point forecast: the filter does produce a "
                "covariance over parameters, but turning that into a distribution over "
                "surfaces is a modelling choice this does not make."
            ),
            implementation_version="1",
            configuration_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "transition_variance": {"type": "number", "minimum": 0.0},
                    "observation_variance": {"type": "number", "minimum": 0.0},
                    "initial_variance": {"type": "number", "exclusiveMinimum": 0.0},
                },
                "required": [],
            },
            support=BackendSupport(),
            supports_temporal_context=True,
            supports_uncertainty=False,
        ),
        factory=StateSpaceForecaster,
    )

    yield _Provider(
        spec=SurfaceAlgorithmSpec(
            public_id="gaussian-field",
            display_name="Gaussian-field surface generator",
            family="generative",
            output="distribution",
            summary=(
                "Fits the context and multiplies the fitted total variance by a "
                "multiplicatively unbiased log-normal random field with a "
                "squared-exponential covariance. Draws are surfaces rather than point "
                "clouds, and its violation rate is controlled by one length scale, which "
                "is what makes it a usable check on the generative evaluation itself. It "
                "has no dynamics and ignores the horizon, and says so."
            ),
            implementation_version="1",
            configuration_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "volatility": {"type": "number", "exclusiveMinimum": 0.0},
                    "length_scale_k": {"type": "number", "exclusiveMinimum": 0.0},
                    "length_scale_T": {"type": "number", "exclusiveMinimum": 0.0},
                },
                "required": [],
            },
            support=BackendSupport(),
            supports_uncertainty=True,
        ),
        factory=_build_gaussian_field,
    )

    yield _Provider(
        spec=SurfaceAlgorithmSpec(
            public_id="persistence",
            display_name="Persistence (random walk)",
            family="forecaster",
            output="definite",
            summary=(
                "Forecasts the last observed surface unchanged, at every horizon. "
                "Implied-volatility surfaces are highly persistent, so this is a strong "
                "baseline rather than a straw man."
            ),
            implementation_version="1",
            configuration_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "base": {
                        "type": "string",
                        "description": (
                            "public_id of the calibrator that turns the last observations "
                            "into an evaluable surface."
                        ),
                    }
                },
                "required": [],
            },
            support=BackendSupport(),
            supports_temporal_context=True,
        ),
        factory=_build_persistence,
    )


def _build_gaussian_field(**config: Any) -> Any:
    """Construct the Gaussian-field generator over the default flat calibrator."""
    from ..generative.distributions import GaussianFieldSurfaceGenerator

    return GaussianFieldSurfaceGenerator(**config)


def _build_persistence(base: str = "flat") -> PersistenceForecaster:
    """Construct a persistence forecaster over the calibrator named ``base``."""
    from ..capabilities import build_algorithm

    return PersistenceForecaster(calibrator=build_algorithm(base))
