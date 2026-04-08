"""
Vectorized Jäckel "Let's Be Rational" (2016) implied-volatility algorithm.
:copyright: © 2026 Raeid Saqur <raeid.saqur@maths.ox.ac.uk> — fast-vollib Python port.

The source code of LetsBeRational resides at www.jaeckel.org/LetsBeRational.7z .

Fully-vectorized NumPy implementation of Peter Jaeckel's Let's Be Rational algorithm.
Uses only scipy/numpy (already deps of fast-vollib).

Copyright © 2013-2014 Peter Jäckel — original algorithm & scalar reference.
Copyright © 2026 Raeid Saqur — fast-vollib Python port.
This vectorization: fast-vollib contributors.

Permission to use, copy, modify, and distribute this software is freely
granted, provided that this notice is preserved.
"""

from __future__ import annotations

import sys

import numpy as np
from scipy.special import erfcx as _sp_erfcx, ndtr as _sp_ndtr, ndtri as _sp_ndtri

# ── Floating-point constants ───────────────────────────────────────────────────
_DBL_EPSILON: float = sys.float_info.epsilon
_DBL_MIN: float = sys.float_info.min
_DBL_MAX: float = sys.float_info.max
_SQRT_DBL_EPSILON: float = _DBL_EPSILON**0.5
_FOURTH_ROOT_DBL_EPSILON: float = _SQRT_DBL_EPSILON**0.5
_EIGHTH_ROOT_DBL_EPSILON: float = _FOURTH_ROOT_DBL_EPSILON**0.5
_SIXTEENTH_ROOT_DBL_EPSILON: float = _EIGHTH_ROOT_DBL_EPSILON**0.5
_SQRT_DBL_MIN: float = _DBL_MIN**0.5

# Jäckel thresholds
_ASYM_THRESH: float = -10.0
_SMALL_T_THRESH: float = 2.0 * _SIXTEENTH_ROOT_DBL_EPSILON

# Mathematical constants
_ONE_OVER_SQRT2: float = 0.7071067811865475
_ONE_OVER_SQRT2PI: float = 0.39894228040143267
_SQRT2PI: float = 2.5066282746310005
_SQRTPI_OVER2: float = 1.2533141373155003
_SQRT3: float = 1.7320508075688772
_SQRT_ONE_OVER3: float = 0.5773502691896258
_TWO_PI_OVER_SQRT27: float = 1.2091995761561453  # 2π/√27
_PI_OVER6: float = 0.5235987755982989
_TWO_PI: float = 6.283185307179586476925286766559

# Rational-cubic control-parameter bounds
_RC_MIN: float = -(1.0 - _SQRT_DBL_EPSILON)
_RC_MAX: float = 2.0 / (_DBL_EPSILON * _DBL_EPSILON)


# ── Normalized Black call — helper branches ────────────────────────────────────


def _nb_asymptotic(h: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Region 1: 17th-order asymptotic expansion, valid for h ≪ -10."""
    e = (t / h) * (t / h)
    r = (h + t) * (h - t)  # h² - t²
    # Guard division by r (safe because Region 1 has |h| >> |t|)
    r_safe = np.where(r != 0.0, r, np.finfo(float).tiny)
    q = (h / r_safe) * (h / r_safe)
    eas = 2.0 + q * (
        -6.0e0
        - 2.0 * e
        + 3.0
        * q
        * (
            1.0e1
            + e * (2.0e1 + 2.0 * e)
            + 5.0
            * q
            * (
                -1.4e1
                + e * (-7.0e1 + e * (-4.2e1 - 2.0 * e))
                + 7.0
                * q
                * (
                    1.8e1
                    + e * (1.68e2 + e * (2.52e2 + e * (7.2e1 + 2.0 * e)))
                    + 9.0
                    * q
                    * (
                        -2.2e1
                        + e * (-3.3e2 + e * (-9.24e2 + e * (-6.6e2 + e * (-1.1e2 - 2.0 * e))))
                        + 1.1e1
                        * q
                        * (
                            2.6e1
                            + e
                            * (
                                5.72e2
                                + e
                                * (2.574e3 + e * (3.432e3 + e * (1.43e3 + e * (1.56e2 + 2.0 * e))))
                            )
                            + 1.3e1
                            * q
                            * (
                                -3.0e1
                                + e
                                * (
                                    -9.1e2
                                    + e
                                    * (
                                        -6.006e3
                                        + e
                                        * (
                                            -1.287e4
                                            + e
                                            * (-1.001e4 + e * (-2.73e3 + e * (-2.1e2 - 2.0 * e)))
                                        )
                                    )
                                )
                                + 1.5e1
                                * q
                                * (
                                    3.4e1
                                    + e
                                    * (
                                        1.36e3
                                        + e
                                        * (
                                            1.2376e4
                                            + e
                                            * (
                                                3.8896e4
                                                + e
                                                * (
                                                    4.862e4
                                                    + e
                                                    * (
                                                        2.4752e4
                                                        + e * (4.76e3 + e * (2.72e2 + 2.0 * e))
                                                    )
                                                )
                                            )
                                        )
                                    )
                                    + 1.7e1
                                    * q
                                    * (
                                        -3.8e1
                                        + e
                                        * (
                                            -1.938e3
                                            + e
                                            * (
                                                -2.3256e4
                                                + e
                                                * (
                                                    -1.00776e5
                                                    + e
                                                    * (
                                                        -1.84756e5
                                                        + e
                                                        * (
                                                            -1.51164e5
                                                            + e
                                                            * (
                                                                -5.4264e4
                                                                + e
                                                                * (
                                                                    -7.752e3
                                                                    + e * (-3.42e2 - 2.0 * e)
                                                                )
                                                            )
                                                        )
                                                    )
                                                )
                                            )
                                        )
                                        + 1.9e1
                                        * q
                                        * (
                                            4.2e1
                                            + e
                                            * (
                                                2.66e3
                                                + e
                                                * (
                                                    4.0698e4
                                                    + e
                                                    * (
                                                        2.3256e5
                                                        + e
                                                        * (
                                                            5.8786e5
                                                            + e
                                                            * (
                                                                7.05432e5
                                                                + e
                                                                * (
                                                                    4.0698e5
                                                                    + e
                                                                    * (
                                                                        1.08528e5
                                                                        + e
                                                                        * (
                                                                            1.197e4
                                                                            + e * (4.2e2 + 2.0 * e)
                                                                        )
                                                                    )
                                                                )
                                                            )
                                                        )
                                                    )
                                                )
                                            )
                                            + 2.1e1
                                            * q
                                            * (
                                                -4.6e1
                                                + e
                                                * (
                                                    -3.542e3
                                                    + e
                                                    * (
                                                        -6.7298e4
                                                        + e
                                                        * (
                                                            -4.90314e5
                                                            + e
                                                            * (
                                                                -1.63438e6
                                                                + e
                                                                * (
                                                                    -2.704156e6
                                                                    + e
                                                                    * (
                                                                        -2.288132e6
                                                                        + e
                                                                        * (
                                                                            -9.80628e5
                                                                            + e
                                                                            * (
                                                                                -2.01894e5
                                                                                + e
                                                                                * (
                                                                                    -1.771e4
                                                                                    + e
                                                                                    * (
                                                                                        -5.06e2
                                                                                        - 2.0 * e
                                                                                    )
                                                                                )
                                                                            )
                                                                        )
                                                                    )
                                                                )
                                                            )
                                                        )
                                                    )
                                                )
                                                + 2.3e1
                                                * q
                                                * (
                                                    5.0e1
                                                    + e
                                                    * (
                                                        4.6e3
                                                        + e
                                                        * (
                                                            1.0626e5
                                                            + e
                                                            * (
                                                                9.614e5
                                                                + e
                                                                * (
                                                                    4.08595e6
                                                                    + e
                                                                    * (
                                                                        8.9148e6
                                                                        + e
                                                                        * (
                                                                            1.04006e7
                                                                            + e
                                                                            * (
                                                                                6.53752e6
                                                                                + e
                                                                                * (
                                                                                    2.16315e6
                                                                                    + e
                                                                                    * (
                                                                                        3.542e5
                                                                                        + e
                                                                                        * (
                                                                                            2.53e4
                                                                                            + e
                                                                                            * (
                                                                                                6.0e2
                                                                                                + 2.0
                                                                                                * e
                                                                                            )
                                                                                        )
                                                                                    )
                                                                                )
                                                                            )
                                                                        )
                                                                    )
                                                                )
                                                            )
                                                        )
                                                    )
                                                    + 2.5e1
                                                    * q
                                                    * (
                                                        -5.4e1
                                                        + e
                                                        * (
                                                            -5.85e3
                                                            + e
                                                            * (
                                                                -1.6146e5
                                                                + e
                                                                * (
                                                                    -1.77606e6
                                                                    + e
                                                                    * (
                                                                        -9.37365e6
                                                                        + e
                                                                        * (
                                                                            -2.607579e7
                                                                            + e
                                                                            * (
                                                                                -4.01166e7
                                                                                + e
                                                                                * (
                                                                                    -3.476772e7
                                                                                    + e
                                                                                    * (
                                                                                        -1.687257e7
                                                                                        + e
                                                                                        * (
                                                                                            -4.44015e6
                                                                                            + e
                                                                                            * (
                                                                                                -5.9202e5
                                                                                                + e
                                                                                                * (
                                                                                                    -3.51e4
                                                                                                    + e
                                                                                                    * (
                                                                                                        -7.02e2
                                                                                                        - 2.0
                                                                                                        * e
                                                                                                    )
                                                                                                )
                                                                                            )
                                                                                        )
                                                                                    )
                                                                                )
                                                                            )
                                                                        )
                                                                    )
                                                                )
                                                            )
                                                        )
                                                        + 2.7e1
                                                        * q
                                                        * (
                                                            5.8e1
                                                            + e
                                                            * (
                                                                7.308e3
                                                                + e
                                                                * (
                                                                    2.3751e5
                                                                    + e
                                                                    * (
                                                                        3.12156e6
                                                                        + e
                                                                        * (
                                                                            2.003001e7
                                                                            + e
                                                                            * (
                                                                                6.919458e7
                                                                                + e
                                                                                * (
                                                                                    1.3572783e8
                                                                                    + e
                                                                                    * (
                                                                                        1.5511752e8
                                                                                        + e
                                                                                        * (
                                                                                            1.0379187e8
                                                                                            + e
                                                                                            * (
                                                                                                4.006002e7
                                                                                                + e
                                                                                                * (
                                                                                                    8.58429e6
                                                                                                    + e
                                                                                                    * (
                                                                                                        9.5004e5
                                                                                                        + e
                                                                                                        * (
                                                                                                            4.7502e4
                                                                                                            + e
                                                                                                            * (
                                                                                                                8.12e2
                                                                                                                + 2.0
                                                                                                                * e
                                                                                                            )
                                                                                                        )
                                                                                                    )
                                                                                                )
                                                                                            )
                                                                                        )
                                                                                    )
                                                                                )
                                                                            )
                                                                        )
                                                                    )
                                                                )
                                                            )
                                                            + 2.9e1
                                                            * q
                                                            * (
                                                                -6.2e1
                                                                + e
                                                                * (
                                                                    -8.99e3
                                                                    + e
                                                                    * (
                                                                        -3.39822e5
                                                                        + e
                                                                        * (
                                                                            -5.25915e6
                                                                            + e
                                                                            * (
                                                                                -4.032015e7
                                                                                + e
                                                                                * (
                                                                                    -1.6934463e8
                                                                                    + e
                                                                                    * (
                                                                                        -4.1250615e8
                                                                                        + e
                                                                                        * (
                                                                                            -6.0108039e8
                                                                                            + e
                                                                                            * (
                                                                                                -5.3036505e8
                                                                                                + e
                                                                                                * (
                                                                                                    -2.8224105e8
                                                                                                    + e
                                                                                                    * (
                                                                                                        -8.870433e7
                                                                                                        + e
                                                                                                        * (
                                                                                                            -1.577745e7
                                                                                                            + e
                                                                                                            * (
                                                                                                                -1.472562e6
                                                                                                                + e
                                                                                                                * (
                                                                                                                    -6.293e4
                                                                                                                    + e
                                                                                                                    * (
                                                                                                                        -9.3e2
                                                                                                                        - 2.0
                                                                                                                        * e
                                                                                                                    )
                                                                                                                )
                                                                                                            )
                                                                                                        )
                                                                                                    )
                                                                                                )
                                                                                            )
                                                                                        )
                                                                                    )
                                                                                )
                                                                            )
                                                                        )
                                                                    )
                                                                )
                                                                + 3.1e1
                                                                * q
                                                                * (
                                                                    6.6e1
                                                                    + e
                                                                    * (
                                                                        1.0912e4
                                                                        + e
                                                                        * (
                                                                            4.74672e5
                                                                            + e
                                                                            * (
                                                                                8.544096e6
                                                                                + e
                                                                                * (
                                                                                    7.71342e7
                                                                                    + e
                                                                                    * (
                                                                                        3.8707344e8
                                                                                        + e
                                                                                        * (
                                                                                            1.14633288e9
                                                                                            + e
                                                                                            * (
                                                                                                2.07431664e9
                                                                                                + e
                                                                                                * (
                                                                                                    2.33360622e9
                                                                                                    + e
                                                                                                    * (
                                                                                                        1.6376184e9
                                                                                                        + e
                                                                                                        * (
                                                                                                            7.0963464e8
                                                                                                            + e
                                                                                                            * (
                                                                                                                1.8512208e8
                                                                                                                + e
                                                                                                                * (
                                                                                                                    2.7768312e7
                                                                                                                    + e
                                                                                                                    * (
                                                                                                                        2.215136e6
                                                                                                                        + e
                                                                                                                        * (
                                                                                                                            8.184e4
                                                                                                                            + e
                                                                                                                            * (
                                                                                                                                1.056e3
                                                                                                                                + 2.0
                                                                                                                                * e
                                                                                                                            )
                                                                                                                        )
                                                                                                                    )
                                                                                                                )
                                                                                                            )
                                                                                                        )
                                                                                                    )
                                                                                                )
                                                                                            )
                                                                                        )
                                                                                    )
                                                                                )
                                                                            )
                                                                        )
                                                                    )
                                                                    + 3.3e1
                                                                    * (
                                                                        -7.0e1
                                                                        + e
                                                                        * (
                                                                            -1.309e4
                                                                            + e
                                                                            * (
                                                                                -6.49264e5
                                                                                + e
                                                                                * (
                                                                                    -1.344904e7
                                                                                    + e
                                                                                    * (
                                                                                        -1.4121492e8
                                                                                        + e
                                                                                        * (
                                                                                            -8.344518e8
                                                                                            + e
                                                                                            * (
                                                                                                -2.9526756e9
                                                                                                + e
                                                                                                * (
                                                                                                    -6.49588632e9
                                                                                                    + e
                                                                                                    * (
                                                                                                        -9.0751353e9
                                                                                                        + e
                                                                                                        * (
                                                                                                            -8.1198579e9
                                                                                                            + e
                                                                                                            * (
                                                                                                                -4.6399188e9
                                                                                                                + e
                                                                                                                * (
                                                                                                                    -1.6689036e9
                                                                                                                    + e
                                                                                                                    * (
                                                                                                                        -3.67158792e8
                                                                                                                        + e
                                                                                                                        * (
                                                                                                                            -4.707164e7
                                                                                                                            + e
                                                                                                                            * (
                                                                                                                                -3.24632e6
                                                                                                                                + e
                                                                                                                                * (
                                                                                                                                    -1.0472e5
                                                                                                                                    + e
                                                                                                                                    * (
                                                                                                                                        -1.19e3
                                                                                                                                        - 2.0
                                                                                                                                        * e
                                                                                                                                    )
                                                                                                                                )
                                                                                                                            )
                                                                                                                        )
                                                                                                                    )
                                                                                                                )
                                                                                                            )
                                                                                                        )
                                                                                                    )
                                                                                                )
                                                                                            )
                                                                                        )
                                                                                    )
                                                                                )
                                                                            )
                                                                        )
                                                                    )
                                                                    * q
                                                                )
                                                            )
                                                        )
                                                    )
                                                )
                                            )
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            )
        )
    )  # noqa: E501
    b = _ONE_OVER_SQRT2PI * np.exp(-0.5 * (h * h + t * t)) * (t / r_safe) * eas
    return np.abs(np.maximum(b, 0.0))


def _nb_small_t(h: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Region 2: 12th-order small-t expansion; valid for s/2 ≪ 1."""
    # a = 1 + h·Y(h) where Y(h) = Φ(h)/φ(h) = √(π/2)·erfcx(-h/√2)
    a = 1.0 + h * (_SQRTPI_OVER2 * _sp_erfcx(-_ONE_OVER_SQRT2 * h))
    w = t * t
    h2 = h * h
    expansion = (
        2
        * t
        * (
            a
            + w
            * (
                (-1 + 3 * a + a * h2) / 6
                + w
                * (
                    (-7 + 15 * a + h2 * (-1 + 10 * a + a * h2)) / 120
                    + w
                    * (
                        (-57 + 105 * a + h2 * (-18 + 105 * a + h2 * (-1 + 21 * a + a * h2))) / 5040
                        + w
                        * (
                            (
                                -561
                                + 945 * a
                                + h2
                                * (
                                    -285
                                    + 1260 * a
                                    + h2 * (-33 + 378 * a + h2 * (-1 + 36 * a + a * h2))
                                )
                            )
                            / 362880
                            + w
                            * (
                                (
                                    -6555
                                    + 10395 * a
                                    + h2
                                    * (
                                        -4680
                                        + 17325 * a
                                        + h2
                                        * (
                                            -840
                                            + 6930 * a
                                            + h2 * (-52 + 990 * a + h2 * (-1 + 55 * a + a * h2))
                                        )
                                    )
                                )
                                / 39916800
                                + (
                                    (
                                        -89055
                                        + 135135 * a
                                        + h2
                                        * (
                                            -82845
                                            + 270270 * a
                                            + h2
                                            * (
                                                -20370
                                                + 135135 * a
                                                + h2
                                                * (
                                                    -1926
                                                    + 25740 * a
                                                    + h2
                                                    * (-75 + 2145 * a + h2 * (-1 + 78 * a + a * h2))
                                                )
                                            )
                                        )
                                    )
                                    * w
                                )
                                / 6227020800.0
                            )
                        )
                    )
                )
            )
        )
    )
    b = _ONE_OVER_SQRT2PI * np.exp(-0.5 * (h * h + t * t)) * expansion
    return np.abs(np.maximum(b, 0.0))


def _nb_norm_cdf(x_neg: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Region 3: standard N(d1)·exp(x/2) − N(d2)/exp(x/2), first-term dominated."""
    h = x_neg / s
    t = 0.5 * s
    b_max = np.exp(0.5 * x_neg)
    b = _sp_ndtr(h + t) * b_max - _sp_ndtr(h - t) / b_max
    return np.abs(np.maximum(b, 0.0))


def _nb_erfcx(h: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Region 4 (general): erfcx-based formula — numerically stable for large |h|+|t|."""
    scale = 0.5 * np.exp(-0.5 * (h * h + t * t))
    diff = _sp_erfcx(-_ONE_OVER_SQRT2 * (h + t)) - _sp_erfcx(-_ONE_OVER_SQRT2 * (h - t))
    return np.abs(np.maximum(scale * diff, 0.0))


def _normalised_intrinsic_call(x: np.ndarray) -> np.ndarray:
    """Normalised intrinsic value of a call: max(exp(x/2) − exp(−x/2), 0)."""
    x = np.asarray(x, dtype=float)
    x2 = x * x
    # Use Taylor expansion near 0 to avoid cancellation
    near_zero = x2 < 98.0 * _FOURTH_ROOT_DBL_EPSILON
    taylor = x * (
        1.0
        + x2 * (1.0 / 24.0 + x2 * (1.0 / 1920.0 + x2 * (1.0 / 322560.0 + x2 * (1.0 / 92897280.0))))
    )
    b_max = np.exp(0.5 * x)
    exact = b_max - 1.0 / b_max
    return np.abs(np.maximum(np.where(near_zero, taylor, exact), 0.0))


def normalised_black_call(x: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Normalised Black call b(x, s) = exp(x/2)·Φ(x/s+s/2) − exp(−x/2)·Φ(x/s−s/2).

    Uses 4 numerical regions for full precision across the entire (x, s) plane.
    Masked computation: each branch is evaluated only for elements that need it,
    avoiding the cost of the large asymptotic polynomial on every call.
    """
    x = np.asarray(x, dtype=float)
    s = np.asarray(s, dtype=float)

    # For x > 0, call-put symmetry: b_call(x,s) = intrinsic_call(x) + b_call(-x,s)
    x_neg = -np.abs(x)  # Work with x ≤ 0 throughout

    # h = x_neg / s,  t = s/2
    s_safe = np.where(s > 0.0, s, np.finfo(float).tiny)
    h = x_neg / s_safe
    t = 0.5 * s

    # Region masks
    in_r1 = (x_neg < s * _ASYM_THRESH) & (
        0.5 * s * s + x_neg < s * (_SMALL_T_THRESH + _ASYM_THRESH)
    )
    in_r2 = ~in_r1 & (0.5 * s < _SMALL_T_THRESH)
    in_r3 = ~in_r1 & ~in_r2 & (x_neg + 0.5 * s * s > s * 0.85)
    # in_r4: everything else (erfcx — most common in the typical trading range)

    # Masked evaluation: compute each branch only where needed.
    # Avoids the expensive asymptotic polynomial (12ms/100k) being run for all elements
    # when Region 1 is rarely triggered (0% for typical K ∈ [0.5F, 2F], σ ∈ [10%, 150%]).
    scalar = x.ndim == 0
    if scalar:
        # Scalar path — no masking overhead
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            if in_r1:
                b = _nb_asymptotic(h, t)
            elif in_r2:
                b = _nb_small_t(h, t)
            elif in_r3:
                b = _nb_norm_cdf(x_neg, s_safe)
            else:
                b = _nb_erfcx(h, t)
    else:
        b = np.empty_like(s)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            if np.any(in_r1):
                b[in_r1] = _nb_asymptotic(h[in_r1], t[in_r1])
            in_r2_or_r1 = in_r1 | in_r2
            if np.any(in_r2):
                b[in_r2] = _nb_small_t(h[in_r2], t[in_r2])
            if np.any(in_r3):
                b[in_r3] = _nb_norm_cdf(x_neg[in_r3], s_safe[in_r3])
            in_r4 = ~in_r1 & ~in_r2 & ~in_r3
            # erfcx branch: always compute, even when in_r4 is empty,
            # because it's the dominant path (avoid an extra boolean allocation).
            b_r4 = _nb_erfcx(h, t)
            b[in_r4] = b_r4[in_r4]

    # Apply call-put symmetry for x > 0
    if np.any(x > 0.0):
        b = np.where(x > 0.0, _normalised_intrinsic_call(x) + b, b)

    return np.abs(np.maximum(b, 0.0))


def normalised_vega(x: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Derivative of normalised_black_call w.r.t. s: φ(x/s) · exp(−(x/s)²/2) · exp(−s²/8)."""
    x = np.asarray(x, dtype=float)
    s = np.asarray(s, dtype=float)
    ax = np.abs(x)
    safe_s = np.where(s > 0.0, s, np.finfo(float).tiny)
    atm = _ONE_OVER_SQRT2PI * np.exp(-0.125 * s * s)
    otm = _ONE_OVER_SQRT2PI * np.exp(-0.5 * (x / safe_s) ** 2 - 0.125 * s * s)
    zero_x = ax <= 0.0
    bad_s = (s <= 0.0) | (s <= ax * _SQRT_DBL_MIN)
    return np.where(zero_x, atm, np.where(bad_s, 0.0, otm))


# ── Rational-cubic interpolation ───────────────────────────────────────────────


def _rc_min_control(
    d_l: np.ndarray, d_r: np.ndarray, slope: np.ndarray, prefer_shape: bool
) -> np.ndarray:
    """Minimum control parameter ensuring shape preservation (vectorized)."""
    monotonic = (d_l * slope >= 0.0) & (d_r * slope >= 0.0)
    convex = (d_l <= slope) & (slope <= d_r)
    concave = (d_l >= slope) & (slope >= d_r)
    none = ~monotonic & ~convex & ~concave

    r1 = np.full_like(slope, -_DBL_MAX)
    r2 = np.full_like(slope, -_DBL_MAX)

    # r1: monotonicity condition (3.8)
    slope_safe = np.where(np.abs(slope) > 0.0, slope, np.finfo(float).tiny)
    r1_val = (d_r + d_l) / slope_safe
    r1 = np.where(monotonic & (np.abs(slope) > 0.0), r1_val, r1)
    if prefer_shape:
        r1 = np.where(monotonic & (np.abs(slope) <= 0.0), _RC_MAX, r1)

    # r2: convexity/concavity condition (3.18)
    drmd = d_r - d_l
    drms = d_r - slope
    smdi = slope - d_l
    denom_ok = (np.abs(smdi) > 0.0) & (np.abs(drms) > 0.0)
    r2_val = np.maximum(
        np.abs(drmd / np.where(drms != 0.0, drms, np.finfo(float).tiny)),
        np.abs(drmd / np.where(smdi != 0.0, smdi, np.finfo(float).tiny)),
    )
    r2 = np.where((convex | concave) & denom_ok, r2_val, r2)
    if prefer_shape:
        r2 = np.where((convex | concave) & ~denom_ok, _RC_MAX, r2)
        r2 = np.where(monotonic & ~(convex | concave) & prefer_shape, _RC_MAX, r2)

    # No shape constraints at all → minimum
    result = np.maximum(_RC_MIN, np.maximum(r1, r2))
    result = np.where(none, _RC_MIN, result)
    return result


def _rc_control_fit_right_2nd(
    x_l: np.ndarray,
    x_r: np.ndarray,
    y_l: np.ndarray,
    y_r: np.ndarray,
    d_l: np.ndarray,
    d_r: np.ndarray,
    d2r: np.ndarray,
    prefer_shape: bool,
) -> np.ndarray:
    """Control parameter matching the second derivative at the right endpoint."""
    h = x_r - x_l
    h_safe = np.where(h != 0.0, h, np.finfo(float).tiny)
    num = 0.5 * h * d2r + (d_r - d_l)
    den = d_r - (y_r - y_l) / h_safe
    r_raw = np.where(
        np.abs(num) < _DBL_MIN,
        0.0,
        np.where(np.abs(den) < _DBL_MIN, np.where(num > 0.0, _RC_MAX, _RC_MIN), num / den),
    )
    slope = (y_r - y_l) / h_safe
    r_min = _rc_min_control(d_l, d_r, slope, prefer_shape)
    return np.maximum(r_raw, r_min)


def _rc_control_fit_left_2nd(
    x_l: np.ndarray,
    x_r: np.ndarray,
    y_l: np.ndarray,
    y_r: np.ndarray,
    d_l: np.ndarray,
    d_r: np.ndarray,
    d2l: np.ndarray,
    prefer_shape: bool,
) -> np.ndarray:
    """Control parameter matching the second derivative at the left endpoint."""
    h = x_r - x_l
    h_safe = np.where(h != 0.0, h, np.finfo(float).tiny)
    num = 0.5 * h * d2l + (d_r - d_l)
    den = (y_r - y_l) / h_safe - d_l
    r_raw = np.where(
        np.abs(num) < _DBL_MIN,
        0.0,
        np.where(np.abs(den) < _DBL_MIN, np.where(num > 0.0, _RC_MAX, _RC_MIN), num / den),
    )
    slope = (y_r - y_l) / h_safe
    r_min = _rc_min_control(d_l, d_r, slope, prefer_shape)
    return np.maximum(r_raw, r_min)


def _rational_cubic_interp(
    x: np.ndarray,
    x_l: np.ndarray,
    x_r: np.ndarray,
    y_l: np.ndarray,
    y_r: np.ndarray,
    d_l: np.ndarray,
    d_r: np.ndarray,
    r: np.ndarray,
) -> np.ndarray:
    """Delbourgo-Gregory rational cubic interpolation (formula 2.4/2.5)."""
    h = x_r - x_l
    h_safe = np.where(np.abs(h) > 0.0, h, np.finfo(float).tiny)
    t = (x - x_l) / h_safe
    omt = 1.0 - t
    t2, omt2 = t * t, omt * omt

    # Full rational formula
    num = (
        y_r * t2 * t
        + (r * y_r - h_safe * d_r) * t2 * omt
        + (r * y_l + h_safe * d_l) * t * omt2
        + y_l * omt2 * omt
    )
    den = 1.0 + (r - 3.0) * t * omt
    rational = num / np.where(den != 0.0, den, np.finfo(float).tiny)

    # Linear interpolation for r ≥ RC_MAX
    linear = y_r * t + y_l * omt

    return np.where(np.abs(h) <= 0.0, 0.5 * (y_l + y_r), np.where(r >= _RC_MAX, linear, rational))


# ── Lower and upper map functions ──────────────────────────────────────────────


def _f_lower_map(x: np.ndarray, s: np.ndarray):
    """f, f', f'' of the lower-branch objective mapping."""
    ax = np.abs(x)
    safe_s = np.where(s > 0.0, s, np.finfo(float).tiny)
    z = _SQRT_ONE_OVER3 * ax / safe_s
    y = z * z
    s2 = s * s
    Phi = _sp_ndtr(-z)
    phi = _ONE_OVER_SQRT2PI * np.exp(-0.5 * z * z)

    safe_phi = np.where(phi > 0.0, phi, np.finfo(float).tiny)
    fpp = (
        _PI_OVER6
        * y
        / (s2 * safe_s)
        * Phi
        * (8 * _SQRT3 * s * ax + (3 * s2 * (s2 - 8) - 8 * x * x) * Phi / safe_phi)
        * np.exp(2 * y + 0.25 * s2)
    )

    Phi2 = Phi * Phi
    fp_full = _TWO_PI * y * Phi2 * np.exp(y + 0.125 * s * s)
    f_full = _TWO_PI_OVER_SQRT27 * ax * Phi2 * Phi

    s_below = s < _DBL_MIN
    x_below = ax < _DBL_MIN

    fp = np.where(s_below, 1.0, fp_full)
    f = np.where(s_below | x_below, 0.0, f_full)
    return f, fp, fpp


def _f_upper_map(x: np.ndarray, s: np.ndarray):
    """f, f', f'' of the upper-branch objective mapping."""
    ax = np.abs(x)
    safe_s = np.where(s > 0.0, s, np.finfo(float).tiny)
    f = _sp_ndtr(-0.5 * s)
    w = (x / safe_s) ** 2
    fp_full = -0.5 * np.exp(0.5 * w)
    fpp_full = _SQRTPI_OVER2 * np.exp(w + 0.125 * s * s) * w / safe_s
    x_below = ax < _DBL_MIN
    fp = np.where(x_below, -0.5, fp_full)
    fpp = np.where(x_below, 0.0, fpp_full)
    return f, fp, fpp


def _inverse_f_lower_map(x: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Inverse of the lower map: σ from f = (2π/√27)·|x|·Φ(−|x|/(√3σ))³."""
    ax = np.abs(x)
    ax_safe = np.where(ax > 0.0, ax, np.finfo(float).tiny)
    coeff = _TWO_PI_OVER_SQRT27 * ax_safe
    f_over_coeff = f / coeff
    # cube root (safe: f ≥ 0 and coeff > 0)
    cbrt = np.cbrt(np.maximum(f_over_coeff, 0.0))
    # ppf(cbrt) = inverse_norm_cdf(cbrt)
    z = _sp_ndtri(np.clip(cbrt, _DBL_MIN, 1.0 - _DBL_EPSILON))
    denom = np.where(np.abs(z) > 0.0, _SQRT3 * z, np.finfo(float).tiny)
    result = np.where(f < _DBL_MIN, 0.0, np.abs(x / denom))
    return result


def _inverse_f_upper_map(f: np.ndarray) -> np.ndarray:
    """Inverse of the upper map: σ = −2·Φ⁻¹(f)."""
    return -2.0 * _sp_ndtri(np.clip(f, _DBL_MIN, 1.0 - _DBL_EPSILON))


# ── Jäckel 4-branch rational initial guess ─────────────────────────────────────


def _jackel_initial_guess(
    beta: np.ndarray,
    x: np.ndarray,
    b_max: np.ndarray,
    s_c: np.ndarray,
    b_c: np.ndarray,
    v_c_safe: np.ndarray,
    s_l: np.ndarray,
    b_l: np.ndarray,
    v_l_safe: np.ndarray,
    s_h: np.ndarray,
    b_h: np.ndarray,
    v_h_safe: np.ndarray,
) -> np.ndarray:
    """Vectorized 4-branch rational initial guess for normalised IV.

    All boundary quantities (b_c, b_l, b_h, v_*) are passed in pre-computed
    so that jackel_iv_normalized can avoid redundant normalised_black_call calls.

    Zones:
      1  beta < b_l          → lower segment  (lower map + inverse)
      2  b_l ≤ beta < b_c    → lower-middle   (Delbourgo-Gregory in (b, σ))
      3  b_c ≤ beta ≤ b_h    → upper-middle   (Delbourgo-Gregory in (b, σ))
      4  beta > b_h           → upper segment  (upper map + inverse)
    """
    # Zone masks
    z1 = beta < b_l
    z2 = (beta >= b_l) & (beta < b_c)
    z4 = beta > b_h
    # z3 = ~z1 & ~z2 & ~z4

    # ── Zone 2: lower-middle segment ──────────────────────────────────────────
    r_lm = _rc_control_fit_right_2nd(
        b_l, b_c, s_l, s_c, 1.0 / v_l_safe, 1.0 / v_c_safe, np.zeros_like(beta), False
    )
    s_z2 = _rational_cubic_interp(beta, b_l, b_c, s_l, s_c, 1.0 / v_l_safe, 1.0 / v_c_safe, r_lm)

    # ── Zone 3: upper-middle segment ──────────────────────────────────────────
    r_hm = _rc_control_fit_left_2nd(
        b_c, b_h, s_c, s_h, 1.0 / v_c_safe, 1.0 / v_h_safe, np.zeros_like(beta), False
    )
    s_z3 = _rational_cubic_interp(beta, b_c, b_h, s_c, s_h, 1.0 / v_c_safe, 1.0 / v_h_safe, r_hm)

    # Dispatch starting from middle zones (common case: no z1 or z4 elements)
    s0 = np.where(z2, s_z2, s_z3)

    # ── Zone 1: lower segment — computed only when needed (uses ndtri via
    #    _inverse_f_lower_map, which is expensive; skip if no elements in z1) ──
    if np.any(z1):
        # Mask to z1 elements only to avoid wasting ndtri on the full array
        x1 = x[z1]
        beta1 = beta[z1]
        b_l1 = b_l[z1]
        s_l1 = s_l[z1]
        f_l1, df_l1, d2f_l1 = _f_lower_map(x1, s_l1)
        n1 = z1.sum()
        r_ll1 = _rc_control_fit_right_2nd(
            np.zeros(n1), b_l1, np.zeros(n1), f_l1, np.ones(n1), df_l1, d2f_l1, True
        )
        f_z1 = _rational_cubic_interp(
            beta1, np.zeros(n1), b_l1, np.zeros(n1), f_l1, np.ones(n1), df_l1, r_ll1
        )
        b_l1_safe = np.where(b_l1 > 0.0, b_l1, np.finfo(float).tiny)
        t_z1 = beta1 / b_l1_safe
        f_z1_qd = (f_l1 * t_z1 + b_l1 * (1.0 - t_z1)) * t_z1
        f_z1 = np.where(f_z1 > 0.0, f_z1, f_z1_qd)
        s_z1 = _inverse_f_lower_map(x1, f_z1)
        s0 = s0.copy()
        s0[z1] = s_z1

    # ── Zone 4: upper segment — computed only when needed ─────────────────────
    if np.any(z4):
        x4 = x[z4]
        beta4 = beta[z4]
        b_h4 = b_h[z4]
        b_max4 = b_max[z4]
        s_h4 = s_h[z4]
        f_h4, df_h4, d2f_h4 = _f_upper_map(x4, s_h4)
        n4 = z4.sum()
        r_hh4 = _rc_control_fit_left_2nd(
            b_h4, b_max4, f_h4, np.zeros(n4), df_h4, np.full(n4, -0.5), d2f_h4, True
        )
        f_z4 = _rational_cubic_interp(
            beta4, b_h4, b_max4, f_h4, np.zeros(n4), df_h4, np.full(n4, -0.5), r_hh4
        )
        h_span4 = b_max4 - b_h4
        h_safe4 = np.where(h_span4 > 0.0, h_span4, np.finfo(float).tiny)
        t_z4 = (beta4 - b_h4) / h_safe4
        f_z4_qd = (f_h4 * (1.0 - t_z4) + 0.5 * h_safe4 * t_z4) * (1.0 - t_z4)
        f_z4 = np.where(f_z4 > 0.0, f_z4, f_z4_qd)
        s_z4 = _inverse_f_upper_map(f_z4)
        s0 = s0.copy() if not np.any(z1) else s0
        s0[z4] = s_z4

    return s0


# ── Jäckel 3-branch Householder(3) iteration ──────────────────────────────────


def _householder_factor(newton: np.ndarray, halley: np.ndarray, hh3: np.ndarray) -> np.ndarray:
    return (1.0 + 0.5 * halley * newton) / (1.0 + newton * (halley + hh3 * newton / 6.0))


def jackel_iv_normalized(beta: np.ndarray, x: np.ndarray, n_iters: int = 2) -> np.ndarray:
    """Solve normalised IV: find σ such that normalised_black_call(x, σ) = β.

    Uses Jäckel's 4-branch rational initial guess + Householder(3) × n_iters
    iterations with a 3-branch objective function g(σ).

    Parameters
    ----------
    beta:   normalised price  = price / √(F·K)
    x:      log-moneyness     = ln(F/K)
    n_iters: number of Householder iterations (default 2, Jäckel's guarantee)

    Returns
    -------
    σ̂:  normalised implied volatility (denormalize: σ = σ̂ / √T)
    """
    beta = np.asarray(beta, dtype=float)
    x = np.asarray(x, dtype=float)

    b_max = np.exp(0.5 * x)

    # ── Degenerate cases ──────────────────────────────────────────────────────
    out = np.full_like(beta, 0.0)
    finite = np.isfinite(beta) & np.isfinite(x) & (beta > 0.0) & (beta < b_max)
    if not np.any(finite):
        return out

    # Branch boundaries — computed ONCE and shared between initial guess and
    # Householder objective dispatch (eliminates 3 redundant normalised_black_call calls).
    s_c = np.sqrt(np.abs(2.0 * x))
    b_c = normalised_black_call(x, s_c)
    v_c = normalised_vega(x, s_c)
    v_c_safe = np.where(v_c > 0.0, v_c, np.finfo(float).tiny)

    s_l = s_c - b_c / v_c_safe
    b_l = normalised_black_call(x, s_l)
    v_l = normalised_vega(x, s_l)
    v_l_safe = np.where(v_l > 0.0, v_l, np.finfo(float).tiny)

    s_h = np.where(v_c > _DBL_MIN, s_c + (b_max - b_c) / v_c_safe, s_c)
    b_h = normalised_black_call(x, s_h)
    v_h = normalised_vega(x, s_h)
    v_h_safe = np.where(v_h > 0.0, v_h, np.finfo(float).tiny)

    # Upper branch applies when beta > max(b_h, b_max/2)
    b_tilde_h = np.maximum(b_h, 0.5 * b_max)

    # ── Initial guess (4-branch) — pass precomputed boundary values ────────────
    s = _jackel_initial_guess(
        beta,
        x,
        b_max,
        s_c,
        b_c,
        v_c_safe,
        s_l,
        b_l,
        v_l_safe,
        s_h,
        b_h,
        v_h_safe,
    )
    s = np.where(s > 0.0, s, s_c)  # fallback: inflection point

    # ── Householder(3) × n_iters with 3-branch objective ──────────────────────
    use_lower = beta < b_l
    use_upper = beta > b_tilde_h
    # use_middle: all others

    for _ in range(n_iters):
        s_safe = np.where(s > 0.0, s, np.finfo(float).tiny)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            b = normalised_black_call(x, s_safe)
            bp = normalised_vega(x, s_safe)

        bp_safe = np.where(bp > 0.0, bp, np.finfo(float).tiny)
        b_safe = np.where(b > 0.0, b, np.finfo(float).tiny)

        x_over_s = x / s_safe
        xs2 = x_over_s / s_safe  # x/s²
        b_halley = x_over_s * x_over_s / s_safe - s_safe / 4.0  # (x/s)²/s = x²/s³ − s/4
        b_hh3 = b_halley * b_halley - 3.0 * xs2 * xs2 - 0.25  # halley² − 3(x/s²)² − 1/4

        # Middle branch: g = b − β
        newton_m = (beta - b) / bp_safe
        halley_m = b_halley
        hh3_m = b_hh3

        # Lower branch: g = 1/ln(b) − 1/ln(β)
        ln_b = np.log(b_safe)
        ln_beta = np.log(np.where(beta > 0.0, beta, np.finfo(float).tiny))
        bpob = bp / b_safe
        ln_b_safe = np.where(np.abs(ln_b) > 0.0, ln_b, np.finfo(float).tiny)
        newton_lo = (ln_beta - ln_b) * ln_b / ln_beta / bpob
        halley_lo = b_halley - bpob * (1.0 + 2.0 / ln_b_safe)
        hh3_lo = (
            b_hh3
            + 2.0 * bpob * bpob * (1.0 + 3.0 / ln_b_safe * (1.0 + 1.0 / ln_b_safe))
            - 3.0 * b_halley * bpob * (1.0 + 2.0 / ln_b_safe)
        )

        # Upper branch: g = ln((b_max − β) / (b_max − b))
        b_max_b = b_max - b
        b_max_b_safe = np.where(b_max_b > 0.0, b_max_b, np.finfo(float).tiny)
        b_max_beta = b_max - beta
        b_max_beta_safe = np.where(b_max_beta > 0.0, b_max_beta, np.finfo(float).tiny)
        g = np.log(b_max_beta_safe / b_max_b_safe)
        gp = bp / b_max_b_safe
        newton_hi = -g / gp
        halley_hi = b_halley + gp
        hh3_hi = b_hh3 + gp * (2.0 * gp + 3.0 * b_halley)

        # Dispatch
        newton = np.where(use_lower, newton_lo, np.where(use_upper, newton_hi, newton_m))
        halley = np.where(use_lower, halley_lo, np.where(use_upper, halley_hi, halley_m))
        hh3 = np.where(use_lower, hh3_lo, np.where(use_upper, hh3_hi, hh3_m))

        ds = newton * _householder_factor(newton, halley, hh3)
        ds = np.maximum(-0.5 * s_safe, ds)
        s = s_safe + ds

    out = np.where(finite, np.maximum(s, 0.0), 0.0)
    return out


# ── Full Black-76 IV (public API) ──────────────────────────────────────────────


def jackel_iv_black(
    price: np.ndarray,
    F: float | np.ndarray,
    K: np.ndarray,
    T: float | np.ndarray,
    is_call: bool | np.ndarray = True,
) -> np.ndarray:
    """Jäckel "Let's Be Rational" IV for Black-76 options.

    Parameters
    ----------
    price   : undiscounted option price
    F       : forward price
    K       : strike
    T       : time to expiry (years)
    is_call : True = call, False = put

    Returns
    -------
    σ  : annualised implied volatility (NaN for degenerate inputs)
    """
    price = np.asarray(price, dtype=float)
    F = np.asarray(F, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    is_call_arr = np.asarray(is_call, dtype=bool)

    sqrt_FK = np.sqrt(F * K)
    x = np.log(F / K)
    sqrt_T = np.sqrt(np.maximum(T, 0.0))

    # Put-call reduction and symmetry mapping to OTM call space.
    #
    # The Jäckel algorithm always solves: normalised_black_call(x_red, σ) = beta
    # where x_red ≤ 0 (OTM call equivalent).  The mapping is:
    #
    #   Call, OTM (x ≤ 0): x_red = x,  beta = price / √(FK)
    #   Call, ITM (x > 0): subtract call-intrinsic; x_red = -x (extrinsic = call(-x))
    #   Put,  OTM (x ≥ 0): normalised_put(x) = normalised_call(-x);  x_red = -x
    #   Put,  ITM (x < 0): subtract put-intrinsic; extrinsic = normalised_call(x); x_red = x
    #
    # In all four cases: x_red = np.where(x > 0, -x, x) = -|x|
    # Intrinsic is subtracted only when ITM: call ITM ↔ x>0, put ITM ↔ x<0.
    q = np.where(is_call_arr, 1.0, -1.0)
    intrinsic = np.abs(np.maximum(q * (F - K), 0.0))
    itm = q * x > 0.0  # call ITM: x>0; put ITM: x<0
    price_red = np.where(itm, np.abs(np.maximum(price - intrinsic, 0.0)), price)
    # Map all cases to OTM-call space: x_red ≤ 0 always.
    x_red = np.where(x > 0.0, -x, x)

    # Normalize
    beta = price_red / np.where(sqrt_FK > 0.0, sqrt_FK, np.finfo(float).tiny)

    # Solve in normalised space
    sigma_hat = jackel_iv_normalized(beta, x_red)

    # De-normalize: σ = σ̂ / √T
    sigma = np.where(sqrt_T > 0.0, sigma_hat / sqrt_T, 0.0)

    # Mark degenerate / infeasible cases as NaN
    bad = (price <= 0.0) | (T <= 0.0) | (F <= 0.0) | (K <= 0.0) | (sigma_hat <= 0.0)
    return np.where(bad, np.nan, sigma)
