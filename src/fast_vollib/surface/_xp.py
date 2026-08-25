"""Backward-compatible alias for :mod:`fast_vollib._array_api`.

The backend-neutral array namespace originated here, serving only the surface
arbitrage harness.  It is now shared library infrastructure (the instruments
payoff layer evaluates in the caller's native namespace through it), so the
implementation lives at :mod:`fast_vollib._array_api`.  This module re-exports
it unchanged so existing imports keep working.
"""

from __future__ import annotations

from .._array_api import ArrayNS, get_namespace, numpy_namespace

__all__ = ["ArrayNS", "get_namespace", "numpy_namespace"]
