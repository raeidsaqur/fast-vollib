"""Stochastic processes: dynamics and parameters, with no state of their own.

A process describes how a state variable evolves. It does not hold random
state, a path buffer, a device, or any knowledge of a contract, so sampling it
twice cannot change what it means and the same object can be reused across
scenarios.

:class:`GBM` and :class:`Heston` are the processes this library implements and
makes numerical claims about. They differ in an important way: GBM is sampled
exactly on any grid, while Heston's square-root variance has no elementary exact
transition and every scheme carries a discretization bias, which its docstring
states rather than glosses. :class:`StochasticProcess` is the structural contract
:func:`fast_vollib.simulation.simulate` drives, so a caller's own dynamics can
be sampled through the same entry point.

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
from .gbm import GBM
from .heston import SCHEMES, Heston

__all__ = ["SCHEMES", "GBM", "Heston", "StochasticProcess"]
