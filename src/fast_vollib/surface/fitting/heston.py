"""Heston as a surface: parameters, an evaluable surface, and a calibrator.

A Heston parameter set induces an implied-volatility surface, and this module is
the bridge from one to the other.  Prices come from the characteristic-function
integral in :mod:`fast_vollib.pricing.heston`; implied volatilities come from
inverting those prices with the Jaeckel solver the rest of the library uses.
Neither computation is re-derived here.

*The surface needs no market state, and that is a property of the coordinates.*
In forward log-moneyness the undiscounted call per unit forward,
``c/F = C(k, T; v0, kappa, theta, xi, rho)``, depends on no level and no rate:
the forward divides out of both the price and the strike.  So
:class:`HestonIVSurface` prices at ``F = 1``, ``K = e^k``, inverts, and returns an
implied volatility that is correct against any forward curve.  A ``market``
argument is accepted and ignored, rather than being required for a computation
that does not use it.

*A calibrated Heston surface is arbitrage-free in the model and not necessarily
on a mesh.*  The model's prices come from a genuine martingale measure, so the
continuous surface admits no arbitrage by construction.  A finite grid sampled
from it can still trip a discrete butterfly test at the wings, where the density
is small and the second divided difference is dominated by inversion noise.  The
report says which of the two it measured, and this module never upgrades the
model's guarantee into a claim about the grid.

Examples
--------
>>> from fast_vollib.surface import SurfacePoints
>>> from fast_vollib.surface.fitting import HestonIVSurface, HestonParameters
>>> surface = HestonIVSurface(
...     parameters=HestonParameters(v0=0.04, kappa=2.0, theta=0.04, vol_of_vol=0.3, rho=-0.7)
... )
>>> prediction = surface.evaluate(SurfacePoints(k=[-0.1, 0.0, 0.1], T=[1.0, 1.0, 1.0]))
>>> bool(prediction.iv[0] > prediction.iv[2])  # a negative correlation skews the smile
True
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING, Any

import numpy as np

from ...pricing.heston import DEFAULT_QUADRATURE_NODES, heston_price
from ..errors import SurfaceCalibrationError, SurfaceValidationError
from ..points import SurfacePoints
from ..prediction import SurfacePrediction

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ...processes.heston import Heston
    from ..market import SurfaceMarket
    from ..observations import SurfaceObservations
    from ..protocols import RNGInput

__all__ = ["HESTON_OBJECTIVES", "HestonCalibrator", "HestonIVSurface", "HestonParameters"]

#: Spaces a Heston calibration can measure its residual in.  Implied volatility
#: is the default because it weights the wings comparably to the money; a price
#: residual is dominated by at-the-money options, where the vega is largest.
HESTON_OBJECTIVES = ("implied_volatility", "price")

_MIN_MATURITY = 1e-8

#: Absolute error, per unit forward, that the Fourier price carries in the wings.
#: :func:`fast_vollib.pricing.heston` computes a call as ``F`` minus a quantity
#: that approaches ``F``, so the answer has absolute rather than relative
#: accuracy however many quadrature nodes are used.  The constant is measured:
#: across maturities from a month to three years and log-moneyness out to +/-1.2,
#: the implied-volatility disagreement between a 768-node and a 4096-node
#: evaluation, multiplied by the Black vega at that point, is between 5e-13 and
#: 2.7e-12 -- flat, which is what an absolute price error looks like after
#: inversion.  3e-12 covers the observed range.
PRICE_NOISE_PER_UNIT_FORWARD = 3e-12

#: Default ceiling on the implied volatility's inversion uncertainty.  A point
#: whose vega is too small to carry :data:`PRICE_NOISE_PER_UNIT_FORWARD` within
#: this band is reported invalid rather than returned with a fabricated wing
#: value -- which is precisely where a surface fit would otherwise be judged.
DEFAULT_IV_UNCERTAINTY_TOLERANCE = 1e-6

_INV_SQRT_2PI = 0.3989422804014327


@dataclass(frozen=True, slots=True)
class HestonParameters:
    """The five Heston parameters, validated as a set.

    Parameters
    ----------
    v0:
        Initial variance, strictly positive.
    kappa:
        Mean-reversion speed, strictly positive.
    theta:
        Long-run variance, strictly positive.
    vol_of_vol:
        Volatility of variance, strictly positive.
    rho:
        Correlation, strictly inside ``(-1, 1)``.

    Notes
    -----
    The Feller condition is reported, not enforced -- see
    :class:`~fast_vollib.processes.Heston` for why.

    Examples
    --------
    >>> from fast_vollib.surface.fitting import HestonParameters
    >>> HestonParameters(v0=0.04, kappa=2.0, theta=0.04, vol_of_vol=0.3, rho=-0.7).satisfies_feller
    True
    """

    v0: float
    kappa: float
    theta: float
    vol_of_vol: float
    rho: float

    def __post_init__(self) -> None:
        values = {}
        for name in ("v0", "kappa", "theta", "vol_of_vol", "rho"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise SurfaceValidationError(f"{name} must be finite; got {value!r}.")
            values[name] = value
        for name in ("v0", "kappa", "theta", "vol_of_vol"):
            if values[name] <= 0.0:
                raise SurfaceValidationError(
                    f"{name} must be strictly positive; got {values[name]!r}. At zero the "
                    f"model degenerates and its characteristic function is undefined."
                )
        if abs(values["rho"]) >= 1.0:
            raise SurfaceValidationError(
                f"rho must lie strictly inside (-1, 1); got {values['rho']!r}. At |rho| = 1 "
                f"the two drivers are the same Brownian motion and the model has one factor."
            )
        for name, value in values.items():
            object.__setattr__(self, name, value)

    @property
    def feller_ratio(self) -> float:
        """``2 kappa theta / xi^2``.  Above one the variance never reaches zero."""
        return 2.0 * self.kappa * self.theta / (self.vol_of_vol * self.vol_of_vol)

    @property
    def satisfies_feller(self) -> bool:
        """Whether :attr:`feller_ratio` exceeds one."""
        return bool(self.feller_ratio > 1.0)

    def as_mapping(self) -> dict[str, float]:
        """The parameters as the keyword arguments the pricer takes."""
        return {
            "v0": self.v0,
            "kappa": self.kappa,
            "theta": self.theta,
            "vol_of_vol": self.vol_of_vol,
            "rho": self.rho,
        }

    def parameters(self) -> dict[str, float]:
        """The parameters as a JSON-safe mapping, with the Feller diagnostics."""
        return {
            **self.as_mapping(),
            "feller_ratio": self.feller_ratio,
            "satisfies_feller": self.satisfies_feller,
        }

    def to_process(self, *, drift: float = 0.0) -> "Heston":
        """The corresponding :class:`~fast_vollib.processes.Heston` process.

        The same parameters describe the dynamics and the surface, so a benchmark
        can simulate and price from one object rather than keeping two in sync.
        """
        from ...processes.heston import Heston

        return Heston(
            kappa=self.kappa,
            theta=self.theta,
            vol_of_vol=self.vol_of_vol,
            rho=self.rho,
            drift=drift,
        )


@dataclass(frozen=True, slots=True)
class HestonIVSurface:
    """The implied-volatility surface a Heston parameter set induces.

    Parameters
    ----------
    parameters:
        The five Heston parameters.
    formulation:
        Which Fourier formulation prices the options; see
        :data:`fast_vollib.pricing.FORMULATIONS`.
    n_nodes:
        Quadrature node count.

    Notes
    -----
    Satisfies :class:`~fast_vollib.surface.protocols.DefiniteIVSurface`.
    Evaluation is host-side float64: the quadrature and the Jaeckel inversion are
    both NumPy, and the capability metadata says so rather than advertising a
    backend that would stage through host memory anyway.

    A point whose price falls outside the no-arbitrage band -- which happens deep
    in the wings, where the price underflows to its intrinsic value -- inverts to
    ``NaN`` and is reported invalid.  That is the honest answer: there is no
    implied volatility for a price at the boundary, and returning the last
    resolvable one would put a fabricated number on the wing where every fit is
    judged.
    """

    parameters: HestonParameters
    formulation: str = "lewis"
    n_nodes: int = DEFAULT_QUADRATURE_NODES
    iv_uncertainty_tolerance: float = DEFAULT_IV_UNCERTAINTY_TOLERANCE

    def __post_init__(self) -> None:
        if not isinstance(self.parameters, HestonParameters):
            raise SurfaceValidationError(
                f"parameters must be HestonParameters; got {type(self.parameters).__name__}."
            )
        tolerance = float(self.iv_uncertainty_tolerance)
        if not math.isfinite(tolerance) or tolerance <= 0.0:
            raise SurfaceValidationError(
                f"iv_uncertainty_tolerance must be finite and strictly positive; got "
                f"{self.iv_uncertainty_tolerance!r}."
            )
        object.__setattr__(self, "iv_uncertainty_tolerance", tolerance)

    def total_variance(self, k: Any, T: Any) -> np.ndarray:
        """``w(k, T) = sigma(k, T)^2 T`` implied by the model."""
        iv = self.implied_volatility(k, T)
        return iv * iv * np.asarray(T, dtype=np.float64)

    def implied_volatility(self, k: Any, T: Any) -> np.ndarray:
        """Black implied volatility at ``(k, T)``; ``NaN`` where the price does not invert."""
        from ...jackel.jackel_iv import jackel_iv_black

        k_array = np.asarray(k, dtype=np.float64)
        T_array = np.asarray(T, dtype=np.float64)
        if k_array.shape != T_array.shape:
            k_array, T_array = np.broadcast_arrays(k_array, T_array)
        strikes = np.exp(k_array)
        forwards = np.ones_like(strikes)
        usable = np.isfinite(k_array) & np.isfinite(T_array) & (T_array > _MIN_MATURITY)
        prices = np.zeros_like(strikes)
        if bool(np.any(usable)):
            prices[usable] = heston_price(
                forward=forwards[usable],
                strike=strikes[usable],
                maturity=T_array[usable],
                is_call=True,
                formulation=self.formulation,
                n_nodes=self.n_nodes,
                v0=self.parameters.v0,
                kappa=self.parameters.kappa,
                theta=self.parameters.theta,
                vol_of_vol=self.parameters.vol_of_vol,
                rho=self.parameters.rho,
            )
        with np.errstate(all="ignore"):
            iv = jackel_iv_black(prices, forwards, strikes, T_array, is_call=True)
        resolved = usable & np.isfinite(iv) & (iv > 0.0)
        with np.errstate(all="ignore"):
            uncertainty = np.where(
                resolved,
                PRICE_NOISE_PER_UNIT_FORWARD / _black_vega(k_array, T_array, iv),
                np.inf,
            )
        return np.where(resolved & (uncertainty <= self.iv_uncertainty_tolerance), iv, np.nan)

    def inversion_uncertainty(self, k: Any, T: Any) -> np.ndarray:
        """How much implied volatility the pricer's absolute error is worth here.

        ``PRICE_NOISE_PER_UNIT_FORWARD / vega``.  This is the quantity
        :meth:`implied_volatility` thresholds on, exposed so a caller can see why
        a wing was declined rather than inferring it from a ``NaN``.
        """
        from ...jackel.jackel_iv import jackel_iv_black

        k_array = np.asarray(k, dtype=np.float64)
        T_array = np.asarray(T, dtype=np.float64)
        if k_array.shape != T_array.shape:
            k_array, T_array = np.broadcast_arrays(k_array, T_array)
        strikes = np.exp(k_array)
        forwards = np.ones_like(strikes)
        prices = heston_price(
            forward=forwards,
            strike=strikes,
            maturity=T_array,
            is_call=True,
            formulation=self.formulation,
            n_nodes=self.n_nodes,
            v0=self.parameters.v0,
            kappa=self.parameters.kappa,
            theta=self.parameters.theta,
            vol_of_vol=self.parameters.vol_of_vol,
            rho=self.parameters.rho,
        )
        with np.errstate(all="ignore"):
            iv = jackel_iv_black(prices, forwards, strikes, T_array, is_call=True)
            return PRICE_NOISE_PER_UNIT_FORWARD / _black_vega(k_array, T_array, iv)

    def evaluate(
        self, points: SurfacePoints, *, market: "SurfaceMarket | None" = None
    ) -> SurfacePrediction:
        """Implied volatility at ``points``.

        ``market`` is accepted and ignored: in forward log-moneyness the model's
        implied volatility does not depend on the forward level or the rate.
        """
        del market
        if not isinstance(points, SurfacePoints):
            raise SurfaceValidationError(
                f"points must be a SurfacePoints; got {type(points).__name__}."
            )
        return SurfacePrediction(points=points, iv=self.implied_volatility(points.k, points.T))

    def parameters_dict(self) -> dict[str, Any]:
        """The surface as a JSON-safe mapping."""
        return {
            **self.parameters.parameters(),
            "formulation": self.formulation,
            "n_nodes": int(self.n_nodes),
        }


@dataclass(frozen=True, slots=True)
class HestonFit:
    """What the optimizer reported about a Heston calibration."""

    parameters: HestonParameters
    cost: float
    optimality: float
    status: int
    message: str
    n_observations: int
    n_function_evaluations: int

    def to_dict(self) -> dict[str, Any]:
        """The diagnostics as a JSON-safe mapping."""
        return {
            "parameters": self.parameters.parameters(),
            "cost": float(self.cost),
            "optimality": float(self.optimality),
            "status": int(self.status),
            "message": self.message,
            "n_observations": int(self.n_observations),
            "n_function_evaluations": int(self.n_function_evaluations),
        }


@dataclass(frozen=True, slots=True)
class HestonCalibrator:
    """Fits five Heston parameters to a whole surface at once.

    Parameters
    ----------
    objective:
        One of :data:`HESTON_OBJECTIVES`.
    n_starts:
        Deterministic starting points.  The Heston objective has well-known local
        minima -- a low-vol-of-vol, high-mean-reversion fit and a high-vol-of-vol,
        low-mean-reversion one can describe the same smile almost equally well --
        so several starts are the difference between a fit and a coincidence.
        They are fixed rather than random, so two runs on the same quotes agree
        exactly.
    max_iterations:
        Cap on solver function evaluations per start.
    n_nodes:
        Quadrature nodes used during the fit and carried into the fitted surface.
    diff_step:
        Relative step for the solver's finite-difference Jacobian.  The default
        is deliberately much larger than SciPy's, and the difference decides
        whether the calibration works at all.  Each residual here is a numerical
        quadrature followed by a root-find, so it carries a noise floor around
        1e-10; SciPy's default relative step of about 1.5e-8 differences two
        values whose difference *is* that noise, and the resulting Jacobian
        points in an arbitrary direction.  A step of 1e-4 is far above the noise
        and far below the curvature scale of the objective, and it moves
        parameter recovery on noiseless data from three significant figures to
        seven.

    Notes
    -----
    Satisfies :class:`~fast_vollib.surface.protocols.SurfaceCalibrator`.
    Calibration is host-side float64 and costs one Fourier quadrature per
    maturity per function evaluation, so it is seconds rather than milliseconds;
    the capability metadata says so.
    """

    objective: str = "implied_volatility"
    n_starts: int = 4
    max_iterations: int = 800
    n_nodes: int = DEFAULT_QUADRATURE_NODES
    diff_step: float = 1e-4

    def __post_init__(self) -> None:
        if self.objective not in HESTON_OBJECTIVES:
            raise SurfaceValidationError(
                f"objective must be one of {HESTON_OBJECTIVES}; got {self.objective!r}."
            )
        if self.n_starts < 1:
            raise SurfaceValidationError(f"n_starts must be at least 1; got {self.n_starts}.")
        if self.max_iterations < 1:
            raise SurfaceValidationError(
                f"max_iterations must be at least 1; got {self.max_iterations}."
            )
        if not (0.0 < float(self.diff_step) < 1.0):
            raise SurfaceValidationError(
                f"diff_step must lie strictly inside (0, 1); got {self.diff_step!r}."
            )

    def fit(
        self, observations: "SurfaceObservations", *, rng: "RNGInput" = None
    ) -> HestonIVSurface:
        """Calibrate the five parameters to ``observations``.

        Raises
        ------
        SurfaceCalibrationError
            If there is no usable observation, or fewer than five of them: five
            parameters cannot be identified from four points.
        """
        del rng  # deterministic starts
        from scipy.optimize import least_squares

        observed = ~np.isnan(observations.iv)
        if not bool(np.any(observed)):
            raise SurfaceCalibrationError(
                "No usable observation: every implied volatility is missing, so there is "
                "no surface to fit."
            )
        k = observations.k[observed]
        T = observations.T[observed]
        iv = observations.iv[observed]
        weight = (
            observations.weight[observed]
            if observations.weight is not None
            else np.ones(k.size, dtype=np.float64)
        )
        if k.size < 5:
            raise SurfaceCalibrationError(
                f"Heston has five parameters and there are {k.size} usable quote(s); the "
                f"fit is not identified."
            )
        sqrt_weight = np.sqrt(weight)
        strikes = np.exp(k)
        forwards = np.ones_like(strikes)
        target_price = (
            None
            if self.objective == "implied_volatility"
            else _black_call(forwards, strikes, T, iv)
        )

        def residuals(x: np.ndarray) -> np.ndarray:
            try:
                parameters = HestonParameters(
                    v0=x[0], kappa=x[1], theta=x[2], vol_of_vol=x[3], rho=x[4]
                )
            except SurfaceValidationError:  # pragma: no cover - bounds keep us inside
                return np.full(k.size, 1e3)
            surface = HestonIVSurface(parameters=parameters, n_nodes=self.n_nodes)
            if self.objective == "price":
                model = heston_price(
                    forward=forwards,
                    strike=strikes,
                    maturity=T,
                    is_call=True,
                    n_nodes=self.n_nodes,
                    v0=parameters.v0,
                    kappa=parameters.kappa,
                    theta=parameters.theta,
                    vol_of_vol=parameters.vol_of_vol,
                    rho=parameters.rho,
                )
                return sqrt_weight * (model - target_price)
            model_iv = surface.implied_volatility(k, T)
            return sqrt_weight * np.where(np.isfinite(model_iv), model_iv - iv, 1.0)

        atm = float(np.median(iv)) ** 2
        bounds = (
            np.array([1e-6, 1e-3, 1e-6, 1e-3, -0.999]),
            np.array([4.0, 50.0, 4.0, 10.0, 0.999]),
        )
        best = None
        for start in _starting_points(atm, self.n_starts):
            result = least_squares(
                residuals,
                np.clip(start, bounds[0], bounds[1]),
                bounds=bounds,
                method="trf",
                x_scale="jac",
                diff_step=self.diff_step,
                xtol=1e-12,
                ftol=1e-12,
                gtol=1e-12,
                max_nfev=self.max_iterations,
            )
            if best is None or result.cost < best.cost:
                best = result
        assert best is not None  # n_starts >= 1
        parameters = HestonParameters(
            v0=best.x[0], kappa=best.x[1], theta=best.x[2], vol_of_vol=best.x[3], rho=best.x[4]
        )
        return HestonIVSurface(parameters=parameters, n_nodes=self.n_nodes)


def _starting_points(atm_variance: float, n_starts: int) -> list[np.ndarray]:
    """Deterministic starts spanning the two basins the objective is known to have.

    A fast-reverting, low-vol-of-vol parameter set and a slow-reverting,
    high-vol-of-vol one produce similar smiles at a single maturity, so a single
    start finds whichever basin it happened to begin in.
    """
    level = max(atm_variance, 1e-4)
    templates = [
        (level, 2.0, level, 0.4, -0.6),
        (level, 8.0, level, 1.0, -0.3),
        (level, 0.5, level, 0.2, -0.8),
        (level, 4.0, level * 1.5, 0.7, 0.0),
        (level, 1.0, level, 0.6, -0.7),
        (level, 15.0, level, 2.0, -0.5),
    ]
    return [np.array(templates[index % len(templates)]) for index in range(n_starts)]


def _black_vega(k: np.ndarray, T: np.ndarray, iv: np.ndarray) -> np.ndarray:
    """Black vega per unit forward, ``n(d1) sqrt(T)``, in forward log-moneyness."""
    w = iv * iv * T
    sqrt_w = np.sqrt(w)
    d1 = (-k + 0.5 * w) / sqrt_w
    return _INV_SQRT_2PI * np.exp(-0.5 * d1 * d1) * np.sqrt(T)


def _black_call(
    forward: np.ndarray, strike: np.ndarray, maturity: np.ndarray, iv: np.ndarray
) -> np.ndarray:
    """Undiscounted Black call prices, through the library's own kernel."""
    from ..._array_api import numpy_namespace
    from ..transforms import undiscounted_call

    xp = numpy_namespace()
    k = np.log(strike / forward)
    return undiscounted_call(k, iv * iv * maturity, forward, xp)
