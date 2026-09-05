"""Scenario generation and explicit Monte Carlo valuation.

The layering rule mirrors the instruments package: a *process* holds dynamics
and parameters, a *simulation* turns a process and an explicit initial state
into paths, a *contract* holds terms, and a *payoff* maps state to a cashflow.
Nothing in that chain infers anything about another link, and no analytic route
ever silently becomes a simulated one or the other way round.

Importing this package pulls in neither torch, jax, numba, nor triton.
"""

from __future__ import annotations

from .discounting import (
    RULES,
    ConstantRateDiscounting,
    DiscountingRule,
    PathwiseShortRateDiscounting,
)
from .errors import (
    ScenarioMismatchError,
    SimulationError,
    SimulationValidationError,
    UnsupportedProcessError,
)
from .monte_carlo import MCResult, MonteCarloEngine
from .scenario import Scenario
from .simulate import simulate

__all__ = [
    "RULES",
    "ConstantRateDiscounting",
    "DiscountingRule",
    "MCResult",
    "MonteCarloEngine",
    "PathwiseShortRateDiscounting",
    "Scenario",
    "ScenarioMismatchError",
    "SimulationError",
    "SimulationValidationError",
    "UnsupportedProcessError",
    "simulate",
]
