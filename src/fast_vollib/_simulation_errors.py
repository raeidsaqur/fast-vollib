"""Typed failures for process sampling, simulation, and Monte Carlo pricing.

These live in a private, dependency-free module rather than in
:mod:`fast_vollib.simulation` so that every layer that can raise them --
:mod:`fast_vollib._random_api`, :mod:`fast_vollib.processes`, and
:mod:`fast_vollib.simulation` -- can import them without a package pointing
back up at the one above it.  :mod:`fast_vollib.simulation.errors` re-exports
them unchanged and is the public surface; nothing else about them is private.

The layer fails closed, like the instruments package it sits beside: an
unsupported process, an invalid grid, a mismatched underlier, a mixed array
namespace, or an RNG on the wrong device raises before a single random number
is drawn.  A sampler that quietly produced *some* paths for input it did not
understand would hand back numbers no caller could tell apart from the ones
they asked for.
"""

from __future__ import annotations

__all__ = [
    "ScenarioMismatchError",
    "SimulationError",
    "SimulationValidationError",
    "UnsupportedProcessError",
]


class SimulationError(Exception):
    """Base class for every failure raised by the simulation layer."""


class SimulationValidationError(SimulationError, ValueError):
    """An input to sampling is missing, malformed, or outside its domain.

    Covers time grids, path counts, initial states, process parameters, RNG
    types, mixed array namespaces, and device mismatches.  Raised before any
    sampling happens, so a partial result is never returned.
    """


class UnsupportedProcessError(SimulationError, NotImplementedError):
    """A process is unsupported or violates the structural sampling contract.

    Covers an engine that cannot drive the requested state model, missing
    protocol members, and a sampler returning the wrong shape, namespace,
    dtype, or device. The message names what the caller would have to change.
    """


class ScenarioMismatchError(SimulationError, ValueError):
    """A scenario does not describe the contract it was asked to evaluate.

    Raised when the underlier does not match, or when the simulated horizon is
    not the contract's maturity.  Either would otherwise produce a plausible
    number for a different instrument.
    """
