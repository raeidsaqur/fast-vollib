"""Typed refusals for unsupported terms and incomplete accounting evidence."""

from __future__ import annotations


class PortfolioError(ValueError):
    """Invalid portfolio terms or accounting inputs."""


class IncompleteBasisError(PortfolioError):
    """Entry cash cannot be reconstructed from the selected evidence."""


class UnsupportedContractError(PortfolioError):
    """The descriptor is retained, but its delivery terms cannot be valued."""
