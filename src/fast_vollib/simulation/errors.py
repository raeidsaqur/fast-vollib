"""Typed errors for the simulation layer.

The definitions live in a private module so that
:mod:`fast_vollib.processes` and the shared RNG adapter can raise them without
importing this package -- the dependency runs one way, from simulation down to
processes, and never back.  This module is where they are documented and the
name callers should import from.

Each error also subclasses the builtin a caller would naturally catch
(:class:`ValueError` for bad input, :class:`NotImplementedError` for an absent
capability), and :class:`SimulationError` catches the whole layer at once.

Examples
--------
>>> from fast_vollib.simulation import SimulationValidationError
>>> issubclass(SimulationValidationError, ValueError)
True
"""

from __future__ import annotations

from .._simulation_errors import (
    ScenarioMismatchError,
    SimulationError,
    SimulationValidationError,
    UnsupportedProcessError,
)

__all__ = [
    "ScenarioMismatchError",
    "SimulationError",
    "SimulationValidationError",
    "UnsupportedProcessError",
]
