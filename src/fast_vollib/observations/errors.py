"""Errors in observation identity, immutable content and revision selection."""

from __future__ import annotations


class ObservationError(ValueError):
    """Invalid or unresolved evidence; no guessed replacement is available."""


class ObservationConflictError(ObservationError):
    """An immutable identity was delivered with different content."""


class RevisionError(ObservationError):
    """A revision chain is missing, forked, cyclic or inconsistent."""
