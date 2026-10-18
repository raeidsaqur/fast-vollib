"""Explicit capability and domain errors for terminal P&L."""

from __future__ import annotations


class PnLError(ValueError):
    """Invalid valuation request."""


class PnLHorizonError(PnLError):
    """The requested target crosses an unsupported lifecycle boundary."""


class PnLAssumptionError(PnLError):
    """Required exercise, settlement or terminal assumptions are absent."""
