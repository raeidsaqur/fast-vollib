"""Stochastic processes: dynamics and parameters, with no state of their own.

A process describes how a state variable evolves. It does not hold random
state, a path buffer, a device, or any knowledge of a contract, so sampling it
twice cannot change what it means and the same object can be reused across
scenarios.

:class:`GBM`, :class:`Heston` and :class:`CIRShortRate` are the processes this
library implements and makes numerical claims about. They differ in an important
way: GBM is sampled exactly on any grid, while the square-root diffusion that
Heston's variance and the CIR short rate both follow has no *elementary* exact
transition. Its two discretizations carry a bias their docstrings state rather
than gloss; ``CIRShortRate`` additionally offers ``exact_transition``, which is
exact at the grid points and, being a draw from the true law rather than a step,
still says nothing about the path between them. :class:`StochasticProcess` is
the structural contract :func:`fast_vollib.simulation.simulate` drives, so a
caller's own dynamics can be sampled through the same entry point.

Importing this package pulls in neither torch, jax, numba, nor triton.

Examples
--------
>>> import numpy as np
>>> from fast_vollib.processes import GBM
>>> paths = GBM.risk_neutral(rate=0.03, volatility=0.2).sample(
...     initial_state={"spot": 100.0},
...     time_grid=np.array([0.0, 0.5, 1.0]),
...     n_paths=4,
...     rng=0,
... )
>>> paths.shape
(4, 3, 1)
>>> bool(np.all(paths[:, 0, 0] == 100.0))
True
"""

from __future__ import annotations

from .base import StochasticProcess
from .bates import Bates
from .bcc97 import BCC97
from .cir import CIR_SCHEMES, CIRShortRate
from .components import (
    ConstantShortRate,
    ConstantVariance,
    HestonVariance,
    JumpComponent,
    LognormalJumps,
    NoJumps,
    ShortRateComponent,
    VarianceComponent,
)
from .gbm import GBM
from .heston import SCHEMES, Heston

__all__ = [
    "CIR_SCHEMES",
    "SCHEMES",
    "BCC97",
    "Bates",
    "CIRShortRate",
    "ConstantShortRate",
    "ConstantVariance",
    "GBM",
    "Heston",
    "HestonVariance",
    "JumpComponent",
    "LognormalJumps",
    "NoJumps",
    "ShortRateComponent",
    "StochasticProcess",
    "VarianceComponent",
]
