"""Shared fixtures for the surface-eval harness tests.

Provides analytic, arbitrage-free reference surfaces (flat-vol and SVI) and the
SVI closed-form ``w, w', w'', g`` used as oracles for the finite-difference
stencils.  Using *known-good* Gatheral–Jacquier SVI parameters (verified
``g ≥ 0`` on the fixture) avoids the common trap where a randomly-parametrized
SVI is itself arbitrageable.
"""

from __future__ import annotations

import numpy as np
import pytest

# Gatheral–Jacquier (2014) style arbitrage-free SVI raw parameters.
# Verified butterfly-free (g > 0) and calendar-free over the test mesh below.
SVI_PARAMS = dict(a=0.04, b=0.40, rho=-0.40, m=0.0, sigma=0.10)


def svi_total_variance(k: np.ndarray, *, a, b, rho, m, sigma) -> np.ndarray:
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma**2))


def svi_w_prime(k: np.ndarray, *, b, rho, m, sigma, **_) -> np.ndarray:
    return b * (rho + (k - m) / np.sqrt((k - m) ** 2 + sigma**2))


def svi_w_double_prime(k: np.ndarray, *, b, m, sigma, **_) -> np.ndarray:
    return b * sigma**2 / ((k - m) ** 2 + sigma**2) ** 1.5


def svi_g(k: np.ndarray, **p) -> np.ndarray:
    w = svi_total_variance(k, **p)
    wp = svi_w_prime(k, **p)
    wpp = svi_w_double_prime(k, **p)
    return (1 - k * wp / (2 * w)) ** 2 - (wp / 2) ** 2 * (0.25 + 1 / w) + wpp / 2


@pytest.fixture
def moneyness() -> np.ndarray:
    return np.linspace(-0.5, 0.5, 41)


@pytest.fixture
def maturities() -> np.ndarray:
    return np.array([0.1, 0.25, 0.5, 1.0, 2.0])


@pytest.fixture
def flat_iv(moneyness, maturities) -> np.ndarray:
    return np.full((moneyness.size, maturities.size), 0.2)


@pytest.fixture
def svi_iv(moneyness, maturities) -> np.ndarray:
    """An arbitrage-free SVI surface: total variance scaled linearly in T.

    ``w(k, T) = w_svi(k) · (T / T_max)`` is butterfly-free per slice (the shape
    is fixed) and monotone in T (positive scaling), so it is calendar-free.
    """
    w_shape = svi_total_variance(moneyness, **SVI_PARAMS)
    w = np.outer(w_shape, maturities / maturities[-1])
    return np.sqrt(w / maturities[None, :])
