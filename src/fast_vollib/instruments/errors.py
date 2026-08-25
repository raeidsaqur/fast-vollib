"""Typed errors for the instruments package.

The layer fails closed.  When a request cannot be served exactly as written --
an unknown contract type, a model with no kernel, a solver that cannot produce
the requested gradients, a missing market input -- it raises one of the errors
below rather than substituting a different model, engine, solver, or backend.
Silent substitution is the failure mode this hierarchy exists to prevent: a
caller who asks for machine-precision gradients and receives a host-staged
approximation has no way to notice.

Every error also subclasses the builtin exception a caller would naturally
reach for (:class:`ValueError` for bad input, :class:`NotImplementedError` for
absent capability), so existing ``except`` clauses keep working, and
:class:`InstrumentError` catches the whole layer at once.
"""

from __future__ import annotations

__all__ = [
    "InstrumentError",
    "InstrumentValidationError",
    "MissingMarketInputError",
    "SerializationError",
    "UnsupportedInstrumentError",
    "UnsupportedModelError",
    "UnsupportedSolverError",
]


class InstrumentError(Exception):
    """Base class for every error raised by :mod:`fast_vollib.instruments`."""


class InstrumentValidationError(InstrumentError, ValueError):
    """A contract term is missing, malformed, or outside its admissible range.

    Raised at construction time, so an instrument object that exists is always
    a valid one.
    """


class UnsupportedInstrumentError(InstrumentError, NotImplementedError):
    """The operation has no implementation for this instrument type.

    Distinct from "unknown type": the message states whether the public
    registry contains the type and names the operation that is missing.
    """


class UnsupportedModelError(InstrumentError, NotImplementedError):
    """The requested pricing model is unknown, or unavailable for this request.

    Never resolved by choosing a different model.  The message names the models
    that would work.
    """


class UnsupportedSolverError(InstrumentError, NotImplementedError):
    """The requested implied-volatility route is unavailable.

    Covers a solver that cannot serve the requested model, and -- more
    importantly -- a differentiable route that is not installed or not defined
    for the requested backend.  Raising here is what keeps a request for native
    gradients from silently degrading into a host-staged numeric answer that
    carries no tape.
    """


class MissingMarketInputError(InstrumentError, ValueError):
    """A market input the requested operation needs was not supplied.

    The message names the missing field and the operation that required it.
    """


class SerializationError(InstrumentError, ValueError):
    """An instrument record could not be decoded, or is not encodable.

    The codec is strict: unknown schema versions, unknown instrument types, and
    unknown or missing fields are all errors rather than ignored input.
    """
