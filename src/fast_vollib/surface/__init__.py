"""Generator-agnostic IV-surface arbitrage-evaluation harness.

A reusable, backend-pluggable layer that takes an *arbitrary* generated implied
-volatility surface on an *arbitrary* ``(log-moneyness × maturity)`` mesh and
returns calibrated, comparable arbitrage diagnostics — the packaged evaluator
that the generative-surface literature (VolGAN, deep-smoothing, VAE families)
otherwise re-derives inline and unnormalized.

Quick start
-----------
>>> import numpy as np
>>> from fast_vollib.surface import IVSurface, validate_surface
>>> k = np.linspace(-0.4, 0.4, 21)
>>> T = np.array([0.1, 0.25, 0.5, 1.0])
>>> iv = np.full((k.size, T.size), 0.2)          # flat, arbitrage-free
>>> surf = IVSurface.from_logmoneyness(k, T, iv)
>>> report = validate_surface(surf)
>>> report.passed
True

The same checks run differentiably on the torch / jax backend via
:func:`arbitrage_penalty`, which can be dropped into a generator's training
loss as a soft no-arbitrage constraint.
"""

from __future__ import annotations

from .grid import IVSurface, SurfaceSequence
from .metrics import (
    DEFAULT_SAS_WEIGHTS,
    DEFAULT_TOLERANCE,
    validate_surface,
)
from .penalty import DEFAULT_PENALTY_WEIGHTS, arbitrage_penalty, penalty_from_surface
from .report import ArbitrageReport, ArbitrageViolation

__all__ = [
    "IVSurface",
    "SurfaceSequence",
    "ArbitrageReport",
    "ArbitrageViolation",
    "validate_surface",
    "arbitrage_penalty",
    "penalty_from_surface",
    "DEFAULT_SAS_WEIGHTS",
    "DEFAULT_PENALTY_WEIGHTS",
    "DEFAULT_TOLERANCE",
]
