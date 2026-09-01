"""Typed errors for the surface layer.

The layer fails closed.  When a surface request cannot be served exactly as
written -- a market state that is needed and absent, a query point outside the
domain a fitted model is willing to speak for, an algorithm whose optional
dependency is not installed, a calibration that did not converge -- it raises
one of the errors below rather than substituting a default, extrapolating
silently, or returning a number with no standing.  Silent substitution is the
failure mode this hierarchy exists to prevent: a caller who receives an
implied volatility interpolated across a gap they did not know was there has
no way to notice that the number is an artifact of the grid rather than a
statement about the market.

Every error also subclasses the builtin exception a caller would naturally
reach for (:class:`ValueError` for bad input, :class:`NotImplementedError` for
absent capability), so existing ``except`` clauses keep working, and
:class:`SurfaceError` catches the whole layer at once.
"""

from __future__ import annotations

__all__ = [
    "MissingMarketStateError",
    "SurfaceAlgorithmUnavailableError",
    "SurfaceCalibrationError",
    "SurfaceDomainError",
    "SurfaceError",
    "SurfaceTypeError",
    "SurfaceValidationError",
]


class SurfaceError(Exception):
    """Base class for every error the surface layer raises."""


class SurfaceValidationError(SurfaceError, ValueError):
    """An input does not satisfy the contract of the value object it was given to.

    Raised on construction, before any numerics run, so a malformed coordinate
    or a mismatched array length is reported where it was introduced rather
    than as a shape error three layers down.
    """


class SurfaceTypeError(SurfaceError, TypeError):
    """A value is of a type the contract cannot carry.

    Distinct from :class:`SurfaceValidationError`, which reports a value of the
    right type in the wrong range.  A float where a label is required is the
    canonical case: it has no range problem, it simply does not survive a JSON
    round trip unchanged.
    """


class MissingMarketStateError(SurfaceError, ValueError):
    """A computation needs a :class:`~fast_vollib.surface.market.SurfaceMarket` and none was given.

    Never resolved by assuming a forward of 1.0 or a zero rate.  Price-space
    error, vega weighting, and strike conversion are statements about a
    specific market; inventing one produces a number that looks like a price
    and is not.
    """


class SurfaceDomainError(SurfaceError, ValueError):
    """A query point lies outside the domain the surface is defined on.

    Raised by definite surfaces that decline to extrapolate.  The message names
    the offending coordinate and the domain, so the caller can either restrict
    the query or opt in to an explicit extrapolation policy.
    """


class SurfaceCalibrationError(SurfaceError, RuntimeError):
    """A calibration finished without producing a surface a caller may use.

    Carries the optimizer's terminal status so the failure can be attributed:
    an objective that never decreased, an iteration limit, or a parameter set
    that left the admissible region.  A calibrator never returns a surface
    built from parameters it knows to be invalid.
    """


class SurfaceAlgorithmUnavailableError(SurfaceError, NotImplementedError):
    """A registered algorithm cannot run in this installation.

    Raised when an optional dependency, a compatible backend, or a required
    checkpoint is absent.  Never resolved by choosing a different algorithm;
    the message repeats the machine-readable reason carried by the algorithm's
    :class:`~fast_vollib.surface.capabilities.SurfaceAlgorithmSpec`.
    """
