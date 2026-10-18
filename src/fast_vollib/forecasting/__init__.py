"""Generic forecast targets and owned, retained draws on declared fixed grids."""

from .adapters import ProcessDistribution, SurfaceDistributionAdapter, returns_to_spot
from .draws import FixedGridDrawSet
from .errors import ForecastError, ForecastProjectionError, ForecastQueryError
from .protocols import ForecastDistribution, ForecastDrawSet, GenerativeForecastModel
from .query import ForecastQuery
from .results import (
    CurveBlock,
    EventBlock,
    ForecastMeta,
    ForecastResult,
    MarginalBlock,
    MarkedEvent,
    ScenarioView,
    SurfaceBlock,
    TrajectoryBlock,
)
from .targets import ForecastTarget, Quantity, ReturnSpec

__all__ = [
    "ForecastTarget",
    "Quantity",
    "ReturnSpec",
    "ForecastQuery",
    "ForecastDistribution",
    "ForecastDrawSet",
    "GenerativeForecastModel",
    "ForecastMeta",
    "ForecastResult",
    "TrajectoryBlock",
    "MarginalBlock",
    "SurfaceBlock",
    "CurveBlock",
    "EventBlock",
    "MarkedEvent",
    "ScenarioView",
    "FixedGridDrawSet",
    "ProcessDistribution",
    "SurfaceDistributionAdapter",
    "returns_to_spot",
    "ForecastError",
    "ForecastQueryError",
    "ForecastProjectionError",
]
