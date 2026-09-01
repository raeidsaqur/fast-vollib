"""One universe for implied-volatility surfaces: represent, fit, evaluate, check.

A surface model joins this package by producing a **definite surface** -- an
evaluable map from ``(k, T)`` points to implied volatilities -- or a
distribution over definite surfaces.  Everything else here consumes those two
outputs: the materializer puts one on a declared mesh, the arbitrage harness
checks that mesh, the metrics score it against observations, and the
differentiable penalty turns the same conditions into a training loss.  A
calibrated SVI slice, a Heston surface backed by characteristic-function
pricing, a conditioned network, and one draw of a generative model are all
definite surfaces, and none of the machinery downstream can tell them apart.

The canonical coordinates are forward log-moneyness ``k = log(K / F(T))`` and
year-fraction maturity ``T``, because that is where the no-arbitrage conditions
are stated.  Strikes, spot-moneyness, deltas, and day-count maturities are
converted by explicit adapters that record the market state they used.

Quick start
-----------
>>> import numpy as np
>>> from fast_vollib.surface import IVSurface, validate_surface
>>> k = np.linspace(-0.4, 0.4, 21)
>>> T = np.array([0.1, 0.25, 0.5, 1.0])
>>> iv = np.full((k.size, T.size), 0.2)          # flat, arbitrage-free
>>> surf = IVSurface.from_logmoneyness(k, T, iv)
>>> report = validate_surface(surf)
>>> report.passed
True

Fitting a surface and checking it on a declared mesh:

>>> from fast_vollib.surface import SurfaceGridSpec, SurfaceObservations, materialize_surface
>>> from fast_vollib.surface.fitting import FlatVolatilityCalibrator
>>> observations = SurfaceObservations(k=[-0.1, 0.0, 0.1], T=[1.0, 1.0, 1.0], iv=[0.22, 0.2, 0.24])
>>> fitted = FlatVolatilityCalibrator().fit(observations)
>>> grid = SurfaceGridSpec(k=np.linspace(-0.3, 0.3, 7), T=[0.25, 1.0])
>>> materialize_surface(fitted, grid).validate().passed
True

The same checks run differentiably on the torch / jax backend via
:func:`arbitrage_penalty`, which can be dropped into a generator's training
loss as a soft no-arbitrage constraint.

Importing this package pulls in numpy and scipy only.  Torch, JAX, matplotlib,
and every optional algorithm dependency are imported lazily, on the path that
needs them.
"""

from __future__ import annotations

from .capabilities import (
    AlgorithmAvailability,
    BackendSupport,
    SurfaceAlgorithmSpec,
    build_algorithm,
    capabilities_document,
    get_algorithm,
    list_algorithms,
)
from .errors import (
    MissingMarketStateError,
    SurfaceAlgorithmUnavailableError,
    SurfaceCalibrationError,
    SurfaceDomainError,
    SurfaceError,
    SurfaceTypeError,
    SurfaceValidationError,
)
from .evaluation import (
    MaturityEvaluation,
    RegionEvaluation,
    SurfaceEvaluation,
    VerificationLevel,
    evaluate_prediction,
)
from .grid import IVSurface, SurfaceSequence
from .gridspec import SurfaceGridSpec
from .market import SurfaceMarket
from .materialize import GridIVSurface, materialize_samples, materialize_surface
from .metrics import (
    DEFAULT_SAS_WEIGHTS,
    DEFAULT_TOLERANCE,
    validate_surface,
)
from .observations import SurfaceObservations, align_predictions
from .penalty import DEFAULT_PENALTY_WEIGHTS, arbitrage_penalty, penalty_from_surface
from .points import (
    CoordinateConvention,
    SurfacePoints,
    points_from_forward_delta,
    points_from_spot_moneyness,
    points_from_strikes,
)
from .prediction import SurfacePrediction, SurfaceSamples
from .protocols import (
    ConditionalSurfaceEstimator,
    DefiniteIVSurface,
    ForecastHorizon,
    GenerativeSurfaceModel,
    SurfaceCalibrator,
    SurfaceDistribution,
    SurfaceForecaster,
)
from .report import ArbitrageReport, ArbitrageViolation

__all__ = [
    "DEFAULT_PENALTY_WEIGHTS",
    "DEFAULT_SAS_WEIGHTS",
    "DEFAULT_TOLERANCE",
    "AlgorithmAvailability",
    "ArbitrageReport",
    "ArbitrageViolation",
    "BackendSupport",
    "ConditionalSurfaceEstimator",
    "CoordinateConvention",
    "DefiniteIVSurface",
    "ForecastHorizon",
    "GenerativeSurfaceModel",
    "GridIVSurface",
    "IVSurface",
    "MissingMarketStateError",
    "SurfaceAlgorithmSpec",
    "SurfaceAlgorithmUnavailableError",
    "SurfaceCalibrationError",
    "SurfaceCalibrator",
    "SurfaceDistribution",
    "SurfaceDomainError",
    "SurfaceError",
    "SurfaceEvaluation",
    "SurfaceForecaster",
    "SurfaceGridSpec",
    "SurfaceMarket",
    "SurfaceObservations",
    "SurfacePoints",
    "SurfacePrediction",
    "SurfaceSamples",
    "SurfaceSequence",
    "SurfaceTypeError",
    "SurfaceValidationError",
    "MaturityEvaluation",
    "RegionEvaluation",
    "VerificationLevel",
    "align_predictions",
    "arbitrage_penalty",
    "build_algorithm",
    "capabilities_document",
    "evaluate_prediction",
    "get_algorithm",
    "list_algorithms",
    "materialize_samples",
    "materialize_surface",
    "penalty_from_surface",
    "points_from_forward_delta",
    "points_from_spot_moneyness",
    "points_from_strikes",
    "validate_surface",
]
