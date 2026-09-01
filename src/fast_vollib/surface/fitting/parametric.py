"""SVI, SVI-JW, and SSVI: the parametric smiles the surface literature is written in.

Three representations of one object, and the conversions between them are exact.

**Raw SVI** (Gatheral) states a smile's total variance directly:

.. math::

    w(k) = a + b\\left(\\rho (k - m) + \\sqrt{(k - m)^2 + \\sigma^2}\\right).

Five parameters, two asymptotically linear wings with slopes :math:`b(1 \\pm \\rho)`, and a
hyperbola joining them over a width set by :math:`\\sigma`.  It is what a fitter
optimizes, and it is opaque: no single parameter means anything a trader would
recognize.

**SVI-JW** (jump-wings) restates the same five numbers as quantities that do:
ATM variance, ATM skew, the two wing slopes, and the minimum variance.  The map
is a bijection, implemented in both directions in closed form, and the round
trip is tested to float64 precision -- which is the only reason to trust an
algebraic inverse that nobody derives twice.

**SSVI** (Gatheral-Jacquier) ties the slices together through one ATM
total-variance term structure :math:`\\theta_T` and a smoothing function
:math:`\\varphi`:

.. math::

    w(k, \\theta) = \\frac{\\theta}{2}
        \\left(1 + \\rho\\varphi(\\theta) k
        + \\sqrt{(\\varphi(\\theta) k + \\rho)^2 + 1 - \\rho^2}\\right).

*A slice-by-slice fit has no term structure, and that is a modelling choice, not
an oversight.*  Independent SVI slices can cross in total variance and produce
calendar arbitrage between two maturities that are each individually fine.  SSVI
cannot, because :math:`\\theta_T` is fitted non-decreasing by construction.  The
cost is that SSVI has three global shape parameters where the slice fit has five
per maturity, so it fits any single smile worse.  Both are provided, and neither
is the default answer.

*A sufficient condition is not a certificate.*  :meth:`SSVISurface.sufficient_butterfly_condition`
reports the standard SSVI inequalities, which are sufficient and conservative.
The authority on whether a fitted surface is arbitrage-free here is the numerical
check -- :func:`~fast_vollib.surface.metrics.validate_surface` on a materialized
grid, or :meth:`SVIParameters.durrleman_g` on a stated set of points -- and the
docstrings say which is which.

Examples
--------
>>> from fast_vollib.surface.fitting import SVIParameters
>>> raw = SVIParameters(a=0.04, b=0.4, rho=-0.4, m=0.0, sigma=0.1)
>>> float(round(raw.total_variance(0.0), 12))
0.08
>>> round_trip = raw.to_jump_wings(1.0).to_raw(1.0)
>>> all(abs(getattr(round_trip, n) - getattr(raw, n)) < 1e-12 for n in raw.parameters())
True
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING, Any

import numpy as np

from ..._array_api import get_namespace
from ..errors import SurfaceCalibrationError, SurfaceValidationError
from ..points import SurfacePoints
from ..prediction import SurfacePrediction

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..market import SurfaceMarket
    from ..observations import SurfaceObservations
    from ..protocols import RNGInput

__all__ = [
    "PHI_FAMILIES",
    "SAFE_ETA",
    "SVI_OBJECTIVES",
    "HestonLikePhi",
    "PowerLawPhi",
    "SSVICalibrator",
    "SSVISurface",
    "SVICalibrator",
    "SVIJumpWings",
    "SVIParameters",
    "SVISliceFit",
    "SVISmile",
    "SVISurface",
    "svi_total_variance",
]

#: Objectives an SVI calibration can minimize.  Total variance is the default:
#: it is the quantity the model is linear-ish in and the one the calendar
#: condition is stated in, so a residual there is not reweighted by maturity.
SVI_OBJECTIVES = ("total_variance", "implied_volatility")

#: Smoothing-function families SSVI accepts.
PHI_FAMILIES = ("power_law", "heston_like")

#: The power-law ``eta`` bound that makes the first sufficient butterfly
#: condition hold at every maturity.  With
#: ``phi(theta) = eta / (theta^gamma (1 + theta)^(1 - gamma))`` the condition
#: ``theta phi (1 + |rho|) < 4`` reads
#: ``eta (theta / (1 + theta))^(1 - gamma) (1 + |rho|) < 4``; the moneyness
#: factor is strictly below one and ``1 + |rho| < 2``, so ``eta <= 2`` implies it
#: for every positive ``theta`` and every admissible ``rho``.  It bounds the
#: *search*, and the fitted surface still reports its achieved minimum ``g``.
SAFE_ETA = 2.0

#: Relative tolerance within which a query maturity counts as a slice's own.
#: A smile is defined at one maturity; asking it about another is a different
#: question, and answering it would be interpolation nobody asked for.
_MATURITY_RTOL = 1e-9

_MIN_SIGMA = 1e-8

#: Relative slack allowed on the minimum-total-variance check; see
#: :meth:`SVIParameters.__post_init__` for why it exists.
_VARIANCE_FLOOR_RTOL = 1e-12


def svi_total_variance(k, a, b, rho, m, sigma, xp=None):
    """Raw-SVI total variance, in the array namespace of ``k``.

    Written against the backend-neutral namespace rather than NumPy directly, so
    a torch or JAX caller gets a differentiable expression: the gradient of this
    with respect to ``(a, b, rho, m, sigma)`` is what a generator training under
    the arbitrage penalty needs, and a host round trip would break the tape.

    Parameters
    ----------
    k:
        Forward log-moneyness, any shape, in any supported backend.
    a, b, rho, m, sigma:
        Raw SVI parameters.  Scalars, or arrays broadcastable against ``k``.
    xp:
        Optional namespace override; inferred from ``k`` when omitted.

    Returns
    -------
    Total variance ``w(k)`` in the namespace of ``k``.

    Examples
    --------
    >>> import numpy as np
    >>> from fast_vollib.surface.fitting import svi_total_variance
    >>> float(round(svi_total_variance(np.array(0.0), 0.04, 0.4, -0.4, 0.0, 0.1), 12))
    0.08
    """
    xp = xp if xp is not None else get_namespace(k)
    y = k - m
    return a + b * (rho * y + xp.sqrt(y * y + sigma * sigma))


@dataclass(frozen=True, slots=True)
class SVIParameters:
    """The five raw SVI parameters of one smile.

    Parameters
    ----------
    a:
        Vertical level of the hyperbola.
    b:
        Wing scale, non-negative.  Wing slopes are ``b(1 - rho)`` on the left and
        ``b(1 + rho)`` on the right.
    rho:
        Wing asymmetry, strictly inside ``(-1, 1)``.
    m:
        Horizontal shift of the hyperbola.
    sigma:
        Curvature width, strictly positive.

    Raises
    ------
    SurfaceValidationError
        If ``b < 0``, ``|rho| >= 1``, ``sigma <= 0``, or the minimum total
        variance ``a + b sigma sqrt(1 - rho^2)`` is negative.  The last of these
        is not a stylistic constraint: a negative total variance has no implied
        volatility, so the parameters describe no smile at all.

    Notes
    -----
    Satisfying these bounds makes ``w >= 0`` everywhere.  It does **not** make
    the smile butterfly-free; that is :meth:`durrleman_g`, which is a function of
    ``k`` and can be negative for perfectly valid-looking parameters.

    Examples
    --------
    >>> from fast_vollib.surface.fitting import SVIParameters
    >>> raw = SVIParameters(a=0.04, b=0.4, rho=-0.4, m=0.0, sigma=0.1)
    >>> float(round(raw.minimum_total_variance, 6))
    0.076661
    """

    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def __post_init__(self) -> None:
        values = {}
        for name in ("a", "b", "rho", "m", "sigma"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise SurfaceValidationError(f"{name} must be finite; got {value!r}.")
            values[name] = value
        if values["b"] < 0.0:
            raise SurfaceValidationError(
                f"b must be non-negative; got {values['b']!r}. A negative wing scale "
                f"inverts both wings and describes a smile that falls away from the money."
            )
        if abs(values["rho"]) >= 1.0:
            raise SurfaceValidationError(
                f"rho must lie strictly inside (-1, 1); got {values['rho']!r}. At "
                f"|rho| = 1 one wing is flat and the hyperbola degenerates to a line."
            )
        if values["sigma"] <= 0.0:
            raise SurfaceValidationError(
                f"sigma must be strictly positive; got {values['sigma']!r}. At zero the "
                f"smile has a kink at k = m rather than a curvature."
            )
        minimum = values["a"] + values["b"] * values["sigma"] * math.sqrt(
            1.0 - values["rho"] * values["rho"]
        )
        # The reparameterization a = v_min - b sigma sqrt(1 - rho^2) reconstructs the
        # minimum by subtracting and re-adding the same quantity, so an exactly-zero
        # minimum can land a few ulps below it. The tolerance is that rounding, scaled
        # to the magnitudes involved -- not a licence for a negative variance.
        floor = -_VARIANCE_FLOOR_RTOL * max(abs(values["a"]), 1.0)
        if minimum < floor:
            raise SurfaceValidationError(
                f"The minimum total variance a + b sigma sqrt(1 - rho^2) must be "
                f"non-negative; got {minimum!r}. A negative total variance has no "
                f"implied volatility."
            )
        for name, value in values.items():
            object.__setattr__(self, name, value)

    # -- the smile -----------------------------------------------------------
    def total_variance(self, k: Any, xp: Any = None) -> Any:
        """``w(k)``, in the array namespace of ``k``."""
        return svi_total_variance(k, self.a, self.b, self.rho, self.m, self.sigma, xp)

    def implied_volatility(self, k: Any, T: float) -> Any:
        """``sigma(k) = sqrt(w(k) / T)`` at maturity ``T``."""
        if T <= 0.0:
            raise SurfaceValidationError(f"T must be strictly positive; got {T!r}.")
        return np.sqrt(np.asarray(self.total_variance(np.asarray(k)), dtype=np.float64) / T)

    def derivatives(self, k: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(w, w', w'')`` at ``k``, in closed form.

        Analytic rather than finite-differenced, so the butterfly diagnostic on a
        fitted slice is exact and a disagreement with the mesh-based check is
        attributable to the mesh rather than to the derivative estimate.

        With ``y = k - m`` and ``s = sqrt(y^2 + sigma^2)``:
        ``w' = b(rho + y/s)`` and ``w'' = b sigma^2 / s^3``.
        """
        y = np.asarray(k, dtype=np.float64) - self.m
        s = np.sqrt(y * y + self.sigma * self.sigma)
        w = self.a + self.b * (self.rho * y + s)
        w_prime = self.b * (self.rho + y / s)
        w_second = self.b * self.sigma * self.sigma / (s * s * s)
        return w, w_prime, w_second

    def durrleman_g(self, k: Any) -> np.ndarray:
        """Durrleman's ``g(k)`` for this slice, from the closed-form derivatives.

        ``g(k) >= 0`` everywhere is equivalent to a non-negative risk-neutral
        density, so a negative value locates butterfly arbitrage exactly rather
        than to the resolution of a mesh.
        """
        w, w_prime, w_second = self.derivatives(k)
        k_array = np.asarray(k, dtype=np.float64)
        term1 = (1.0 - k_array * w_prime / (2.0 * w)) ** 2
        term2 = (w_prime * 0.5) ** 2 * (0.25 + 1.0 / w)
        return term1 - term2 + 0.5 * w_second

    @property
    def minimum_total_variance(self) -> float:
        """``a + b sigma sqrt(1 - rho^2)``, the smallest ``w`` the slice attains."""
        return self.a + self.b * self.sigma * math.sqrt(1.0 - self.rho * self.rho)

    @property
    def minimum_at(self) -> float:
        """The log-moneyness where ``w`` attains its minimum."""
        return self.m - self.rho * self.sigma / math.sqrt(1.0 - self.rho * self.rho)

    # -- conversions ---------------------------------------------------------
    def to_jump_wings(self, T: float) -> SVIJumpWings:
        """The jump-wings restatement of this slice at maturity ``T``."""
        if T <= 0.0:
            raise SurfaceValidationError(f"T must be strictly positive; got {T!r}.")
        root = math.sqrt(self.m * self.m + self.sigma * self.sigma)
        w_atm = self.a + self.b * (-self.rho * self.m + root)
        if w_atm <= 0.0:
            raise SurfaceValidationError(
                f"The ATM total variance is {w_atm!r}, which is not positive, so the "
                f"jump-wings parameters -- all of which are scaled by sqrt(w) -- do not exist."
            )
        sqrt_w = math.sqrt(w_atm)
        return SVIJumpWings(
            v=w_atm / T,
            psi=0.5 * self.b / sqrt_w * (self.rho - self.m / root),
            p=self.b * (1.0 - self.rho) / sqrt_w,
            c=self.b * (1.0 + self.rho) / sqrt_w,
            v_tilde=self.minimum_total_variance / T,
        )

    def parameters(self) -> dict[str, float]:
        """The parameters as a JSON-safe mapping."""
        return {
            "a": self.a,
            "b": self.b,
            "rho": self.rho,
            "m": self.m,
            "sigma": self.sigma,
        }


@dataclass(frozen=True, slots=True)
class SVIJumpWings:
    """The same smile, in parameters that mean something.

    Attributes
    ----------
    v : float
        ATM variance, ``w(0) / T``.
    psi : float
        ATM skew, ``w'(0) / (2 sqrt(w(0)))``.
    p : float
        Slope of the left (put) wing in ``sqrt(w)`` units, ``b(1 - rho)/sqrt(w(0))``.
    c : float
        Slope of the right (call) wing, ``b(1 + rho)/sqrt(w(0))``.
    v_tilde : float
        Minimum variance, ``min_k w(k) / T``.

    Notes
    -----
    Each of the five has an independent operational meaning, and each is checked
    against that meaning in the tests rather than only against the inverse map --
    a pair of mutually inverse but equally wrong formulas would round-trip
    perfectly.
    """

    v: float
    psi: float
    p: float
    c: float
    v_tilde: float

    def __post_init__(self) -> None:
        for name in ("v", "psi", "p", "c", "v_tilde"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise SurfaceValidationError(f"{name} must be finite; got {value!r}.")
            object.__setattr__(self, name, value)
        if self.v <= 0.0:
            raise SurfaceValidationError(
                f"v must be strictly positive; got {self.v!r}. It is an ATM variance."
            )
        if self.v_tilde < 0.0:
            raise SurfaceValidationError(
                f"v_tilde must be non-negative; got {self.v_tilde!r}. It is the smallest "
                f"variance the smile attains."
            )
        if self.p < 0.0 or self.c < 0.0:
            raise SurfaceValidationError(
                f"Wing slopes must be non-negative; got p={self.p!r}, c={self.c!r}. A "
                f"negative wing slope makes total variance fall without bound."
            )

    def to_raw(self, T: float) -> SVIParameters:
        """Invert to raw SVI at maturity ``T``.

        The inverse of :meth:`SVIParameters.to_jump_wings`, in closed form.  With
        ``w = v T``:

        ``b = sqrt(w)(c + p)/2``, ``rho = 1 - p sqrt(w)/b``,
        ``beta = rho - 2 psi sqrt(w)/b`` (which is ``m / sqrt(m^2 + sigma^2)``),
        ``alpha = beta / sqrt(1 - beta^2) = m / sigma``, then
        ``sigma = (v - v_tilde) T / (b [sqrt(1 + alpha^2) - rho alpha - sqrt(1 - rho^2)])``,
        ``m = alpha sigma`` and ``a = v_tilde T - b sigma sqrt(1 - rho^2)``.

        Raises
        ------
        SurfaceValidationError
            If the parameters are mutually inconsistent -- a zero wing sum, a
            ``|beta| >= 1``, or a vanishing denominator, each of which means no
            raw slice reproduces them.
        """
        if T <= 0.0:
            raise SurfaceValidationError(f"T must be strictly positive; got {T!r}.")
        w = self.v * T
        sqrt_w = math.sqrt(w)
        b = 0.5 * sqrt_w * (self.c + self.p)
        if b <= 0.0:
            raise SurfaceValidationError(
                f"The wing slopes sum to {self.c + self.p!r}, giving b = {b!r}. A smile "
                f"with no wings is flat, which raw SVI represents only in the limit."
            )
        rho = 1.0 - self.p * sqrt_w / b
        if abs(rho) >= 1.0:
            raise SurfaceValidationError(
                f"The wing slopes imply rho = {rho!r}, which is outside (-1, 1); one of "
                f"p and c is zero, so the smile has a single wing."
            )
        beta = rho - 2.0 * self.psi * sqrt_w / b
        if abs(beta) >= 1.0:
            raise SurfaceValidationError(
                f"The ATM skew implies m / sqrt(m^2 + sigma^2) = {beta!r}, which must lie "
                f"strictly inside (-1, 1); the skew is too steep for these wing slopes."
            )
        alpha = beta / math.sqrt(1.0 - beta * beta)
        denominator = math.sqrt(1.0 + alpha * alpha) - rho * alpha - math.sqrt(1.0 - rho * rho)
        gap = (self.v - self.v_tilde) * T
        if abs(denominator) <= 1e-14:
            if abs(gap) > 1e-14:
                raise SurfaceValidationError(
                    "The ATM variance exceeds the minimum variance but the shape "
                    "parameters imply they coincide; the jump-wings parameters are "
                    "mutually inconsistent."
                )
            sigma = _MIN_SIGMA
        else:
            sigma = gap / (b * denominator)
        if sigma <= 0.0:
            raise SurfaceValidationError(
                f"The implied sigma is {sigma!r}, which is not positive; v must exceed "
                f"v_tilde for a smile with curvature."
            )
        m = alpha * sigma
        a = self.v_tilde * T - b * sigma * math.sqrt(1.0 - rho * rho)
        return SVIParameters(a=a, b=b, rho=rho, m=m, sigma=sigma)

    def parameters(self) -> dict[str, float]:
        """The parameters as a JSON-safe mapping."""
        return {
            "v": self.v,
            "psi": self.psi,
            "p": self.p,
            "c": self.c,
            "v_tilde": self.v_tilde,
        }


@dataclass(frozen=True, slots=True)
class SVISliceFit:
    """What the optimizer reported about one calibrated slice.

    Attributes
    ----------
    maturity : float
    parameters : SVIParameters
    cost : float
        Half the sum of squared residuals at the solution, as SciPy defines it.
    optimality : float
        First-order optimality measure the solver terminated on.
    status : int
        SciPy's termination status; positive means a convergence criterion was met.
    message : str
    n_observations : int
    n_function_evaluations : int
    min_durrleman_g : float
        The smallest ``g`` on the observed moneyness range.  Negative means the
        fitted slice is butterfly-arbitrageable there, and the fit says so rather
        than the caller having to discover it downstream.
    """

    maturity: float
    parameters: SVIParameters
    cost: float
    optimality: float
    status: int
    message: str
    n_observations: int
    n_function_evaluations: int
    min_durrleman_g: float

    def to_dict(self) -> dict[str, Any]:
        """The diagnostics as a JSON-safe mapping."""
        return {
            "maturity": float(self.maturity),
            "parameters": self.parameters.parameters(),
            "cost": float(self.cost),
            "optimality": float(self.optimality),
            "status": int(self.status),
            "message": self.message,
            "n_observations": int(self.n_observations),
            "n_function_evaluations": int(self.n_function_evaluations),
            "min_durrleman_g": float(self.min_durrleman_g),
        }


@dataclass(frozen=True, slots=True)
class SVISmile:
    """One SVI slice, evaluable at its own maturity.

    A smile is a statement about one expiry.  Asked about another maturity it
    reports the point invalid rather than reusing its total variance there, which
    would assert a flat term structure nobody fitted.

    Examples
    --------
    >>> from fast_vollib.surface import SurfacePoints
    >>> from fast_vollib.surface.fitting import SVIParameters, SVISmile
    >>> smile = SVISmile(parameters=SVIParameters(0.04, 0.4, -0.4, 0.0, 0.1), maturity=1.0)
    >>> prediction = smile.evaluate(SurfacePoints(k=[0.0, 0.0], T=[1.0, 2.0]))
    >>> prediction.valid.tolist()
    [True, False]
    """

    parameters: SVIParameters
    maturity: float

    def __post_init__(self) -> None:
        if not isinstance(self.parameters, SVIParameters):
            raise SurfaceValidationError(
                f"parameters must be SVIParameters; got {type(self.parameters).__name__}."
            )
        maturity = float(self.maturity)
        if not math.isfinite(maturity) or maturity <= 0.0:
            raise SurfaceValidationError(
                f"maturity must be finite and strictly positive; got {self.maturity!r}."
            )
        object.__setattr__(self, "maturity", maturity)

    def evaluate(
        self, points: SurfacePoints, *, market: "SurfaceMarket | None" = None
    ) -> SurfacePrediction:
        """Implied volatility at ``points`` whose maturity is this slice's own."""
        del market
        if not isinstance(points, SurfacePoints):
            raise SurfaceValidationError(
                f"points must be a SurfacePoints; got {type(points).__name__}."
            )
        on_slice = np.isclose(points.T, self.maturity, rtol=_MATURITY_RTOL, atol=0.0)
        w = np.asarray(self.parameters.total_variance(points.k), dtype=np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            iv = np.sqrt(np.where(w > 0.0, w, np.nan) / self.maturity)
        return SurfacePrediction(
            points=points, iv=np.where(on_slice, iv, np.nan), valid=on_slice & np.isfinite(iv)
        )

    def parameters_dict(self) -> dict[str, Any]:
        """The slice as a JSON-safe mapping."""
        return {"maturity": self.maturity, **self.parameters.parameters()}


@dataclass(frozen=True, slots=True)
class SVISurface:
    """A term structure of SVI slices, with a declared maturity policy.

    Parameters
    ----------
    maturities:
        Strictly increasing maturities, one per slice.
    slices:
        The raw parameters at each maturity.
    maturity_interpolation:
        ``'total_variance_linear'`` (default) interpolates ``w`` linearly in ``T``
        between adjacent slices, which preserves calendar monotonicity when the
        slices are themselves ordered; ``'none'`` answers only at the fitted
        maturities.  Interpolating implied volatility instead is not offered:
        it does not preserve that ordering, and the resulting calendar violations
        would be attributed to the fit.
    diagnostics:
        Optional per-slice optimizer records.

    Notes
    -----
    Independent slices carry no joint constraint, so nothing here guarantees the
    surface is calendar-arbitrage-free.  Use
    :func:`~fast_vollib.surface.metrics.validate_surface` on a materialized grid,
    or fit :class:`SSVISurface`, which is monotone by construction.
    """

    maturities: tuple[float, ...]
    slices: tuple[SVIParameters, ...]
    maturity_interpolation: str = "total_variance_linear"
    diagnostics: tuple[SVISliceFit, ...] = ()

    def __post_init__(self) -> None:
        maturities = tuple(float(value) for value in self.maturities)
        slices = tuple(self.slices)
        if len(maturities) != len(slices):
            raise SurfaceValidationError(
                f"maturities and slices must have the same length; got "
                f"{len(maturities)} and {len(slices)}."
            )
        if not maturities:
            raise SurfaceValidationError("An SVI surface needs at least one slice.")
        if any(not math.isfinite(value) or value <= 0.0 for value in maturities):
            raise SurfaceValidationError("Every maturity must be finite and strictly positive.")
        if len(maturities) > 1 and any(
            later <= earlier for earlier, later in zip(maturities, maturities[1:])
        ):
            raise SurfaceValidationError(
                f"maturities must be strictly increasing; got {list(maturities)}."
            )
        if any(not isinstance(item, SVIParameters) for item in slices):
            raise SurfaceValidationError("Every slice must be an SVIParameters.")
        if self.maturity_interpolation not in ("total_variance_linear", "none"):
            raise SurfaceValidationError(
                f"maturity_interpolation must be 'total_variance_linear' or 'none'; "
                f"got {self.maturity_interpolation!r}."
            )
        object.__setattr__(self, "maturities", maturities)
        object.__setattr__(self, "slices", slices)
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))

    def total_variance(self, k: Any, T: Any) -> np.ndarray:
        """``w(k, T)`` under the declared maturity policy; ``NaN`` where undefined."""
        k_array = np.asarray(k, dtype=np.float64)
        T_array = np.asarray(T, dtype=np.float64)
        pillars = np.asarray(self.maturities, dtype=np.float64)
        slice_w = np.stack(
            [np.asarray(item.total_variance(k_array), dtype=np.float64) for item in self.slices]
        )
        if self.maturity_interpolation == "none" or pillars.size == 1:
            index = np.full(T_array.shape, -1, dtype=np.intp)
            for position, pillar in enumerate(pillars):
                index = np.where(
                    np.isclose(T_array, pillar, rtol=_MATURITY_RTOL, atol=0.0), position, index
                )
            out = np.full(T_array.shape, np.nan)
            for position in range(pillars.size):
                mask = index == position
                out[mask] = slice_w[position][mask]
            return out
        upper = np.clip(np.searchsorted(pillars, T_array, side="left"), 1, pillars.size - 1)
        lower = upper - 1
        span = pillars[upper] - pillars[lower]
        weight = (T_array - pillars[lower]) / span
        blended = np.empty(T_array.shape, dtype=np.float64)
        for position in range(pillars.size - 1):
            mask = lower == position
            blended[mask] = (1.0 - weight[mask]) * slice_w[position][mask] + weight[mask] * slice_w[
                position + 1
            ][mask]
        outside = (T_array < pillars[0]) | (T_array > pillars[-1])
        return np.where(outside, np.nan, blended)

    def evaluate(
        self, points: SurfacePoints, *, market: "SurfaceMarket | None" = None
    ) -> SurfacePrediction:
        """Implied volatility at ``points``; points outside the fitted term are invalid."""
        del market
        if not isinstance(points, SurfacePoints):
            raise SurfaceValidationError(
                f"points must be a SurfacePoints; got {type(points).__name__}."
            )
        w = self.total_variance(points.k, points.T)
        with np.errstate(invalid="ignore", divide="ignore"):
            iv = np.sqrt(np.where(w > 0.0, w, np.nan) / points.T)
        return SurfacePrediction(points=points, iv=iv)

    def parameters(self) -> dict[str, Any]:
        """The surface as a JSON-safe mapping."""
        return {
            "maturity_interpolation": self.maturity_interpolation,
            "slices": [
                {"maturity": maturity, **item.parameters()}
                for maturity, item in zip(self.maturities, self.slices)
            ],
        }


@dataclass(frozen=True, slots=True)
class SVICalibrator:
    """Fits one raw SVI slice per observed maturity.

    Parameters
    ----------
    objective:
        One of :data:`SVI_OBJECTIVES`.
    butterfly_penalty:
        Weight on a soft ``relu(-g)`` residual evaluated on the observed
        moneyness range.  Zero (the default) fits the quotes and *reports* the
        resulting minimum ``g``; a positive weight trades fit quality for
        convexity.  It is a penalty, not a guarantee, and the fit record says so
        by carrying the achieved minimum rather than a boolean.
    n_starts:
        Number of deterministic starting points, differing in the initial
        ``rho``.  The SVI objective is not convex and a single start finds a
        local minimum whose identity depends on the data; the starts are fixed
        rather than random so two runs on the same quotes agree exactly.
    max_iterations:
        Cap on solver function evaluations per start.

    Notes
    -----
    The optimizer works in a reparameterization ``(v_min, b, rho, m, sigma)``
    where ``v_min = a + b sigma sqrt(1 - rho^2) >= 0`` is the minimum total
    variance.  Bounding ``v_min`` below at zero makes ``w >= 0`` automatic, so no
    iterate is ever a set of parameters that describes no smile -- which a bound
    on ``a`` alone would not achieve.

    Satisfies :class:`~fast_vollib.surface.protocols.SurfaceCalibrator`.
    Calibration is host-side float64 through SciPy; the fitted surface evaluates
    in any backend.
    """

    objective: str = "total_variance"
    butterfly_penalty: float = 0.0
    n_starts: int = 3
    max_iterations: int = 2000
    maturity_interpolation: str = "total_variance_linear"
    maturity_decimals: int = 9

    def __post_init__(self) -> None:
        if self.objective not in SVI_OBJECTIVES:
            raise SurfaceValidationError(
                f"objective must be one of {SVI_OBJECTIVES}; got {self.objective!r}."
            )
        if self.butterfly_penalty < 0.0:
            raise SurfaceValidationError(
                f"butterfly_penalty must be non-negative; got {self.butterfly_penalty!r}."
            )
        if self.n_starts < 1:
            raise SurfaceValidationError(f"n_starts must be at least 1; got {self.n_starts}.")
        if self.max_iterations < 1:
            raise SurfaceValidationError(
                f"max_iterations must be at least 1; got {self.max_iterations}."
            )

    def fit(self, observations: "SurfaceObservations", *, rng: "RNGInput" = None) -> SVISurface:
        """Calibrate one slice per maturity present in ``observations``.

        Raises
        ------
        SurfaceCalibrationError
            If a maturity carries fewer than five usable quotes -- five
            parameters cannot be identified from four points -- or if the solver
            terminates without a usable parameter set.
        """
        del rng  # deterministic starts
        buckets = _by_maturity(observations, self.maturity_decimals)
        if not buckets:
            raise SurfaceCalibrationError(
                "No usable observation: every implied volatility is missing, so there is "
                "no slice to fit."
            )
        maturities: list[float] = []
        slices: list[SVIParameters] = []
        records: list[SVISliceFit] = []
        for maturity, k, iv, weight in buckets:
            record = self._fit_slice(maturity, k, iv, weight)
            maturities.append(maturity)
            slices.append(record.parameters)
            records.append(record)
        return SVISurface(
            maturities=tuple(maturities),
            slices=tuple(slices),
            maturity_interpolation=self.maturity_interpolation,
            diagnostics=tuple(records),
        )

    def _fit_slice(
        self, maturity: float, k: np.ndarray, iv: np.ndarray, weight: np.ndarray
    ) -> SVISliceFit:
        from scipy.optimize import least_squares

        if k.size < 5:
            raise SurfaceCalibrationError(
                f"Maturity {maturity!r} carries {k.size} usable quote(s); raw SVI has five "
                f"parameters and cannot be identified from fewer than five points."
            )
        w = iv * iv * maturity
        sqrt_weight = np.sqrt(weight)
        k_span = float(np.max(k) - np.min(k)) or 1.0
        penalty_grid = np.linspace(float(np.min(k)), float(np.max(k)), 41)
        n_residuals = k.size + (penalty_grid.size if self.butterfly_penalty > 0.0 else 0)

        def unpack(x: np.ndarray) -> SVIParameters:
            v_min, b, rho, m, sigma = x
            a = v_min - b * sigma * math.sqrt(max(1.0 - rho * rho, 0.0))
            return SVIParameters(a=a, b=b, rho=rho, m=m, sigma=sigma)

        def residuals(x: np.ndarray) -> np.ndarray:
            try:
                parameters = unpack(x)
            except SurfaceValidationError:  # pragma: no cover - bounds keep us inside
                return np.full(n_residuals, 1e6)
            model_w = np.asarray(parameters.total_variance(k), dtype=np.float64)
            if self.objective == "total_variance":
                base = sqrt_weight * (model_w - w)
            else:
                model_iv = np.sqrt(np.maximum(model_w, 0.0) / maturity)
                base = sqrt_weight * (model_iv - iv)
            if self.butterfly_penalty <= 0.0:
                return base
            g = parameters.durrleman_g(penalty_grid)
            return np.concatenate([base, math.sqrt(self.butterfly_penalty) * np.maximum(-g, 0.0)])

        bounds = (
            np.array([0.0, 0.0, -0.999, float(np.min(k)) - k_span, _MIN_SIGMA]),
            np.array(
                [
                    10.0 * float(np.max(w)) + 1.0,
                    10.0,
                    0.999,
                    float(np.max(k)) + k_span,
                    10.0 * k_span,
                ]
            ),
        )
        best = None
        for start in _starting_points(k, w, self.n_starts, k_span):
            clipped = np.clip(start, bounds[0], bounds[1])
            result = least_squares(
                residuals,
                clipped,
                bounds=bounds,
                method="trf",
                x_scale="jac",
                xtol=1e-14,
                ftol=1e-14,
                gtol=1e-14,
                max_nfev=self.max_iterations,
            )
            if best is None or result.cost < best.cost:
                best = result
        assert best is not None  # n_starts >= 1
        parameters = unpack(best.x)
        return SVISliceFit(
            maturity=maturity,
            parameters=parameters,
            cost=float(best.cost),
            optimality=float(best.optimality),
            status=int(best.status),
            message=str(best.message),
            n_observations=int(k.size),
            n_function_evaluations=int(best.nfev),
            min_durrleman_g=float(np.min(parameters.durrleman_g(penalty_grid))),
        )


def _starting_points(
    k: np.ndarray, w: np.ndarray, n_starts: int, k_span: float
) -> list[np.ndarray]:
    """Deterministic starting points in ``(v_min, b, rho, m, sigma)``.

    Fixed rather than random, so two calibrations of the same quotes agree
    exactly.  They differ in the initial ``rho``, which is the parameter the
    objective is least convex in.
    """
    w_min = float(np.min(w))
    m0 = float(k[int(np.argmin(w))])
    sigma0 = max(0.25 * k_span, _MIN_SIGMA)
    slope = (float(np.max(w)) - w_min) / max(k_span, 1e-12)
    b0 = max(slope, 1e-4)
    rhos = np.linspace(-0.5, 0.5, n_starts) if n_starts > 1 else np.array([0.0])
    return [np.array([max(w_min, 0.0), b0, float(rho), m0, sigma0]) for rho in rhos]


def _by_maturity(
    observations: "SurfaceObservations", decimals: int
) -> list[tuple[float, np.ndarray, np.ndarray, np.ndarray]]:
    """Group usable observations into ``(maturity, k, iv, weight)`` buckets, ascending."""
    observed = ~np.isnan(observations.iv)
    if not bool(np.any(observed)):
        return []
    bucket = np.round(observations.T, decimals)
    weights = (
        observations.weight
        if observations.weight is not None
        else np.ones(observations.n, dtype=np.float64)
    )
    out = []
    for value in np.unique(bucket[observed]):
        mask = observed & (bucket == value)
        order = np.argsort(observations.k[mask], kind="stable")
        out.append(
            (
                float(value),
                observations.k[mask][order],
                observations.iv[mask][order],
                weights[mask][order],
            )
        )
    return out


# --- SSVI --------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PowerLawPhi:
    """``phi(theta) = eta / (theta^gamma (1 + theta)^(1 - gamma))``.

    The standard SSVI smoothing function.  It behaves like ``eta theta^-gamma``
    for small ``theta`` and like ``eta / theta`` for large ``theta``, which keeps
    the long-dated wings from flattening to nothing.
    """

    eta: float
    gamma: float = 0.5

    def __post_init__(self) -> None:
        eta = float(self.eta)
        gamma = float(self.gamma)
        if not math.isfinite(eta) or eta <= 0.0:
            raise SurfaceValidationError(f"eta must be finite and strictly positive; got {eta!r}.")
        if not math.isfinite(gamma) or not (0.0 < gamma < 1.0):
            raise SurfaceValidationError(
                f"gamma must lie strictly inside (0, 1); got {gamma!r}. At the endpoints "
                f"the smoothing function loses one of its two asymptotic regimes."
            )
        object.__setattr__(self, "eta", eta)
        object.__setattr__(self, "gamma", gamma)

    def __call__(self, theta: Any) -> Any:
        theta_array = np.asarray(theta, dtype=np.float64)
        return self.eta / (theta_array**self.gamma * (1.0 + theta_array) ** (1.0 - self.gamma))

    def parameters(self) -> dict[str, Any]:
        """The smoothing function as a JSON-safe mapping."""
        return {"family": "power_law", "eta": self.eta, "gamma": self.gamma}


@dataclass(frozen=True, slots=True)
class HestonLikePhi:
    """``phi(theta) = (1 - (1 - e^{-lam theta}) / (lam theta)) / (lam theta)``.

    The shape the Heston model's own smile takes in the small-vol-of-vol limit,
    offered because a benchmark that generates data from Heston should be able to
    fit it with a smoothing function of the right family rather than only with a
    power law that must approximate it.
    """

    lam: float

    def __post_init__(self) -> None:
        lam = float(self.lam)
        if not math.isfinite(lam) or lam <= 0.0:
            raise SurfaceValidationError(f"lam must be finite and strictly positive; got {lam!r}.")
        object.__setattr__(self, "lam", lam)

    def __call__(self, theta: Any) -> Any:
        x = self.lam * np.asarray(theta, dtype=np.float64)
        return (1.0 - (1.0 - np.exp(-x)) / x) / x

    def parameters(self) -> dict[str, Any]:
        """The smoothing function as a JSON-safe mapping."""
        return {"family": "heston_like", "lam": self.lam}


@dataclass(frozen=True, slots=True)
class SSVISurface:
    """A whole surface from one ATM term structure, one ``rho``, and one ``phi``.

    Parameters
    ----------
    maturities:
        Strictly increasing maturities carrying the ``theta`` pillars.
    theta:
        ATM total variance at each maturity, strictly positive and
        **non-decreasing** -- which is exactly the necessary condition for
        calendar-spread arbitrage freedom, so the constraint lives in the type
        rather than in a check somebody has to remember to run.
    rho:
        Global correlation, strictly inside ``(-1, 1)``.
    phi:
        The smoothing function.
    maturity_interpolation:
        How ``theta`` is read between pillars; linear interpolation preserves
        monotonicity, which is why it is the only option offered.

    Notes
    -----
    ``w(k, theta) = (theta/2)(1 + rho phi k + sqrt((phi k + rho)^2 + 1 - rho^2))``,
    and ``w(0, theta) = theta`` exactly, which is what makes ``theta`` the ATM
    total variance rather than a shape parameter.

    Each slice is a raw SVI slice: see :meth:`slice_parameters`.  That
    equivalence is an exact algebraic identity and is tested as one.
    """

    maturities: tuple[float, ...]
    theta: tuple[float, ...]
    rho: float
    phi: Any
    maturity_interpolation: str = "linear"
    diagnostics: Any = None

    def __post_init__(self) -> None:
        maturities = tuple(float(value) for value in self.maturities)
        theta = tuple(float(value) for value in self.theta)
        if len(maturities) != len(theta):
            raise SurfaceValidationError(
                f"maturities and theta must have the same length; got {len(maturities)} "
                f"and {len(theta)}."
            )
        if not maturities:
            raise SurfaceValidationError("An SSVI surface needs at least one maturity.")
        if any(not math.isfinite(value) or value <= 0.0 for value in maturities):
            raise SurfaceValidationError("Every maturity must be finite and strictly positive.")
        if len(maturities) > 1 and any(
            later <= earlier for earlier, later in zip(maturities, maturities[1:])
        ):
            raise SurfaceValidationError(
                f"maturities must be strictly increasing; got {list(maturities)}."
            )
        if any(not math.isfinite(value) or value <= 0.0 for value in theta):
            raise SurfaceValidationError("Every theta must be finite and strictly positive.")
        if len(theta) > 1 and any(later < earlier for earlier, later in zip(theta, theta[1:])):
            raise SurfaceValidationError(
                f"theta must be non-decreasing in maturity; got {list(theta)}. A falling "
                f"ATM total variance is calendar-spread arbitrage, not a shape."
            )
        rho = float(self.rho)
        if not math.isfinite(rho) or abs(rho) >= 1.0:
            raise SurfaceValidationError(f"rho must lie strictly inside (-1, 1); got {rho!r}.")
        if not callable(self.phi):
            raise SurfaceValidationError(f"phi must be callable; got {type(self.phi).__name__}.")
        if self.maturity_interpolation != "linear":
            raise SurfaceValidationError(
                f"maturity_interpolation must be 'linear'; got "
                f"{self.maturity_interpolation!r}. Linear interpolation of theta preserves "
                f"the monotonicity the calendar condition needs."
            )
        object.__setattr__(self, "maturities", maturities)
        object.__setattr__(self, "theta", theta)
        object.__setattr__(self, "rho", rho)

    def theta_at(self, T: Any) -> np.ndarray:
        """The ATM total variance at ``T``; ``NaN`` outside the fitted term."""
        T_array = np.asarray(T, dtype=np.float64)
        pillars = np.asarray(self.maturities, dtype=np.float64)
        values = np.asarray(self.theta, dtype=np.float64)
        if pillars.size == 1:
            return np.where(
                np.isclose(T_array, pillars[0], rtol=_MATURITY_RTOL, atol=0.0),
                values[0],
                np.nan,
            )
        outside = (T_array < pillars[0]) | (T_array > pillars[-1])
        return np.where(outside, np.nan, np.interp(T_array, pillars, values))

    def total_variance(self, k: Any, T: Any) -> np.ndarray:
        """``w(k, T)``; ``NaN`` outside the fitted term."""
        theta = self.theta_at(T)
        phi = np.asarray(self.phi(np.where(np.isnan(theta), 1.0, theta)), dtype=np.float64)
        k_array = np.asarray(k, dtype=np.float64)
        x = phi * k_array + self.rho
        w = (
            0.5
            * theta
            * (1.0 + self.rho * phi * k_array + np.sqrt(x * x + 1.0 - self.rho * self.rho))
        )
        return np.where(np.isnan(theta), np.nan, w)

    def slice_parameters(self, T: float) -> SVIParameters:
        """This surface's slice at ``T``, as raw SVI parameters.

        The identity is exact:

        ``a = theta(1 - rho^2)/2``, ``b = theta phi / 2``, ``rho`` unchanged,
        ``m = -rho / phi``, ``sigma = sqrt(1 - rho^2) / phi``.

        Derived by completing the square in the SSVI form, so an SSVI surface can
        be handed to anything that speaks raw SVI without a refit.
        """
        theta = float(self.theta_at(np.asarray(T, dtype=np.float64)))
        if not math.isfinite(theta):
            raise SurfaceValidationError(
                f"T = {T!r} lies outside the fitted term [{self.maturities[0]}, "
                f"{self.maturities[-1]}], so this surface has no slice there."
            )
        phi = float(np.asarray(self.phi(theta)))
        one_minus = 1.0 - self.rho * self.rho
        return SVIParameters(
            a=0.5 * theta * one_minus,
            b=0.5 * theta * phi,
            rho=self.rho,
            m=-self.rho / phi,
            sigma=math.sqrt(one_minus) / phi,
        )

    def sufficient_butterfly_condition(self, T: float) -> dict[str, Any]:
        """The standard SSVI sufficient conditions at ``T``, reported as numbers.

        The two inequalities are ``theta phi (1 + |rho|) < 4`` and
        ``theta phi^2 (1 + |rho|) <= 4``.  They are **sufficient**, not necessary,
        and conservative: a surface that fails them may still be butterfly-free.

        The authority on a fitted surface is the numerical check --
        :meth:`SVIParameters.durrleman_g` on the slice, or
        :func:`~fast_vollib.surface.metrics.validate_surface` on a materialized
        grid.  This method exists so a calibrator can constrain itself to the
        safe region, not so a report can claim a guarantee it did not verify.
        """
        theta = float(self.theta_at(np.asarray(T, dtype=np.float64)))
        phi = float(np.asarray(self.phi(theta)))
        first = theta * phi * (1.0 + abs(self.rho))
        second = theta * phi * phi * (1.0 + abs(self.rho))
        return {
            "maturity": float(T),
            "theta": theta,
            "phi": phi,
            "theta_phi_term": first,
            "theta_phi_squared_term": second,
            "satisfied": bool(first < 4.0 and second <= 4.0),
        }

    def evaluate(
        self, points: SurfacePoints, *, market: "SurfaceMarket | None" = None
    ) -> SurfacePrediction:
        """Implied volatility at ``points``; points outside the fitted term are invalid."""
        del market
        if not isinstance(points, SurfacePoints):
            raise SurfaceValidationError(
                f"points must be a SurfacePoints; got {type(points).__name__}."
            )
        w = self.total_variance(points.k, points.T)
        with np.errstate(invalid="ignore", divide="ignore"):
            iv = np.sqrt(np.where(w > 0.0, w, np.nan) / points.T)
        return SurfacePrediction(points=points, iv=iv)

    def parameters(self) -> dict[str, Any]:
        """The surface as a JSON-safe mapping."""
        return {
            "rho": self.rho,
            "phi": self.phi.parameters() if hasattr(self.phi, "parameters") else None,
            "maturities": list(self.maturities),
            "theta": list(self.theta),
            "maturity_interpolation": self.maturity_interpolation,
        }


@dataclass(frozen=True, slots=True)
class SSVICalibrator:
    """Fits one SSVI surface jointly across every observed maturity.

    Parameters
    ----------
    phi_family:
        One of :data:`PHI_FAMILIES`.
    enforce_no_butterfly:
        When true (the default), the optimizer is bounded so that the sufficient
        butterfly conditions hold at every fitted maturity.  It is a constraint
        on the search, and the fitted surface still reports its achieved minimum
        ``g``: the constraint makes arbitrage unlikely, and the report is what
        establishes it did not occur.
    n_starts:
        Deterministic starting points, differing in the initial ``rho``.
    max_iterations:
        Cap on solver function evaluations per start.

    Notes
    -----
    ``theta`` is parameterized as a strictly positive first pillar plus
    non-negative increments, so every iterate is non-decreasing and the fitted
    surface cannot exhibit calendar-spread arbitrage between its pillars.  That
    is the structural advantage over independent SVI slices, and it is the reason
    to accept a worse fit to any single smile.

    Satisfies :class:`~fast_vollib.surface.protocols.SurfaceCalibrator`.
    """

    phi_family: str = "power_law"
    enforce_no_butterfly: bool = True
    n_starts: int = 3
    max_iterations: int = 4000
    maturity_decimals: int = 9

    def __post_init__(self) -> None:
        if self.phi_family not in PHI_FAMILIES:
            raise SurfaceValidationError(
                f"phi_family must be one of {PHI_FAMILIES}; got {self.phi_family!r}."
            )
        if self.n_starts < 1:
            raise SurfaceValidationError(f"n_starts must be at least 1; got {self.n_starts}.")
        if self.max_iterations < 1:
            raise SurfaceValidationError(
                f"max_iterations must be at least 1; got {self.max_iterations}."
            )

    def fit(self, observations: "SurfaceObservations", *, rng: "RNGInput" = None) -> SSVISurface:
        """Calibrate ``theta``, ``rho`` and the smoothing function to ``observations``.

        Raises
        ------
        SurfaceCalibrationError
            If there is no usable observation, or fewer usable quotes than free
            parameters.
        """
        del rng  # deterministic starts
        from scipy.optimize import least_squares

        buckets = _by_maturity(observations, self.maturity_decimals)
        if not buckets:
            raise SurfaceCalibrationError(
                "No usable observation: every implied volatility is missing, so there is "
                "no surface to fit."
            )
        maturities = np.array([item[0] for item in buckets], dtype=np.float64)
        k = np.concatenate([item[1] for item in buckets])
        iv = np.concatenate([item[2] for item in buckets])
        weight = np.concatenate([item[3] for item in buckets])
        index = np.concatenate(
            [
                np.full(item[1].size, position, dtype=np.intp)
                for position, item in enumerate(buckets)
            ]
        )
        maturity_per_point = maturities[index]
        w = iv * iv * maturity_per_point
        sqrt_weight = np.sqrt(weight)
        n_pillars = maturities.size
        n_free = n_pillars + (3 if self.phi_family == "power_law" else 2)
        if k.size < n_free:
            raise SurfaceCalibrationError(
                f"SSVI has {n_free} free parameters at {n_pillars} maturities and there "
                f"are {k.size} usable quotes; the fit is not identified."
            )

        # theta is the ATM total variance, so seed it from the quote nearest the
        # money rather than from the smile's minimum, which sits off the money
        # whenever the skew is non-zero.
        atm_w = np.array(
            [
                max(float(item[2][int(np.argmin(np.abs(item[1])))] ** 2 * item[0]), 1e-8)
                for item in buckets
            ],
            dtype=np.float64,
        )

        def unpack(x: np.ndarray) -> SSVISurface:
            theta = np.cumsum(np.concatenate([[x[0]], x[1:n_pillars]]))
            if self.phi_family == "power_law":
                phi: Any = PowerLawPhi(eta=x[n_pillars], gamma=x[n_pillars + 1])
                rho = x[n_pillars + 2]
            else:
                phi = HestonLikePhi(lam=x[n_pillars])
                rho = x[n_pillars + 1]
            return SSVISurface(
                maturities=tuple(maturities), theta=tuple(theta), rho=float(rho), phi=phi
            )

        def residuals(x: np.ndarray) -> np.ndarray:
            try:
                surface = unpack(x)
            except SurfaceValidationError:  # pragma: no cover - bounds keep us inside
                return np.full(k.size, 1e6)
            model_w = surface.total_variance(k, maturity_per_point)
            model_w = np.where(np.isnan(model_w), 1e6, model_w)
            return sqrt_weight * (model_w - w)

        lower = [1e-10] + [0.0] * (n_pillars - 1)
        upper = [10.0] * n_pillars
        if self.phi_family == "power_law":
            eta_cap = SAFE_ETA if self.enforce_no_butterfly else 50.0
            gamma_cap = 0.5 if self.enforce_no_butterfly else 0.999999
            lower += [1e-6, 1e-6, -0.999]
            upper += [eta_cap, gamma_cap, 0.999]
            starts_tail = [[1.0, 0.5, rho] for rho in _start_rhos(self.n_starts)]
        else:
            lower += [1e-6, -0.999]
            upper += [50.0, 0.999]
            starts_tail = [[1.0, rho] for rho in _start_rhos(self.n_starts)]
        bounds = (np.array(lower), np.array(upper))

        increments = np.concatenate([[atm_w[0]], np.maximum(np.diff(atm_w), 0.0)])
        best = None
        for tail in starts_tail:
            start = np.clip(np.concatenate([increments, tail]), bounds[0], bounds[1])
            result = least_squares(
                residuals,
                start,
                bounds=bounds,
                method="trf",
                x_scale="jac",
                xtol=1e-14,
                ftol=1e-14,
                gtol=1e-14,
                max_nfev=self.max_iterations,
            )
            if best is None or result.cost < best.cost:
                best = result
        assert best is not None  # n_starts >= 1
        surface = unpack(best.x)
        grid = np.linspace(float(np.min(k)), float(np.max(k)), 41)
        min_g = min(
            float(np.min(surface.slice_parameters(maturity).durrleman_g(grid)))
            for maturity in maturities
        )
        object.__setattr__(
            surface,
            "diagnostics",
            {
                "cost": float(best.cost),
                "optimality": float(best.optimality),
                "status": int(best.status),
                "message": str(best.message),
                "n_observations": int(k.size),
                "n_function_evaluations": int(best.nfev),
                "min_durrleman_g": min_g,
            },
        )
        return surface


def _start_rhos(n_starts: int) -> list[float]:
    """Deterministic initial correlations, spread over the admissible range."""
    if n_starts == 1:
        return [-0.5]
    return [float(value) for value in np.linspace(-0.8, 0.2, n_starts)]
