"""Heston stochastic volatility: dynamics and parameters, and an honest sampler.

The variance follows a square-root diffusion and the spot is driven by a
Brownian motion correlated with it:

.. math::

    dS_t &= (r - q) S_t\\, dt + \\sqrt{v_t}\\, S_t\\, dW^1_t, \\\\
    dv_t &= \\kappa(\\theta - v_t)\\, dt + \\xi \\sqrt{v_t}\\, dW^2_t, \\qquad
    d\\langle W^1, W^2\\rangle_t = \\rho\\, dt.

*Unlike geometric Brownian motion, this cannot be sampled exactly on a grid, and
the sampler says so.*  :class:`~fast_vollib.processes.GBM` uses the closed-form
transition, so its only discretization is how often the path is observed.  The
square-root variance has no elementary exact transition, so every scheme here
carries a bias that shrinks with the step size and does not vanish at any finite
one.  Two are provided and the choice is a parameter, because a Monte Carlo
price that disagrees with the characteristic-function price by a tenth of a
volatility point is a discretization artifact and a caller has to be able to
tell that from a bug.

``'quadratic_exponential'`` (the default) is Andersen's QE scheme: it matches the
first two moments of the exact non-central chi-squared transition and switches
to an exponential-with-atom approximation where that distribution is nearly
degenerate, then advances the log-spot with the martingale-corrected
representation that eliminates the leading discretization error in the
correlation.  ``'full_truncation_euler'`` is the Euler scheme with the variance
truncated at zero inside the drift and the diffusion, which is the least biased
of the naive fixes and is here as a simple, transparent comparison.

Randomness is drawn entirely through :func:`fast_vollib._random_api.standard_normal`.
The QE scheme needs a uniform in one of its two branches, and takes it as
:math:`\\Phi(Z)` of a normal from the same stream rather than opening a second
one -- which keeps the whole path reproducible from a single generator, and keeps
the antithetic pairing meaningful.

Examples
--------
>>> import numpy as np
>>> from fast_vollib.processes import Heston
>>> process = Heston(kappa=2.0, theta=0.04, vol_of_vol=0.3, rho=-0.7, drift=0.0)
>>> process.state_names
('spot', 'variance')
>>> paths = process.sample(
...     initial_state={"spot": 100.0, "variance": 0.04},
...     time_grid=np.array([0.0, 0.5, 1.0]),
...     n_paths=8,
...     rng=20260831,
... )
>>> paths.shape
(8, 3, 2)
>>> bool(np.all(paths[:, 0, 0] == 100.0))
True
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Any, ClassVar, Mapping

from .._array_api import get_namespace
from .._random_api import (
    random_stream,
    resolve_device,
    resolve_dtype,
    resolve_namespace,
    standard_normal,
)
from .._simulation_errors import SimulationValidationError
from ._stochastic_volatility import (
    full_truncation_euler_step,
    quadratic_exponential_step_with_spot,
)
from .gbm import _validate_parameter

__all__ = ["SCHEMES", "Heston"]

#: Discretization schemes :meth:`Heston.sample` accepts.  Neither is exact; see
#: the module docstring for what each one biases and why.
SCHEMES = ("quadratic_exponential", "full_truncation_euler")

#: The QE switching threshold.  Andersen's analysis puts any value in [1, 2]
#: within the scheme's accuracy; 1.5 is the value the scheme is usually reported
#: with, and it is exposed so a caller can see it rather than discover it.


@dataclass(frozen=True, slots=True)
class Heston:
    """Heston dynamics: five parameters, and no state of its own.

    Parameters
    ----------
    kappa : float or scalar array
        Mean-reversion speed of the variance, strictly positive.
    theta : float or scalar array
        Long-run variance, strictly positive.
    vol_of_vol : float or scalar array
        :math:`\\xi`, the volatility of the variance, strictly positive.  At
        zero the variance is deterministic and the model degenerates to
        Black-Scholes with a time-dependent volatility; that limit is a useful
        test and a useless model, so it is excluded here and reached instead by
        taking the limit in a test.
    rho : float or scalar array
        Correlation between the two drivers, strictly inside ``(-1, 1)``.
    drift : float or scalar array
        The log-spot drift :math:`r - q`, per year.  Whatever measure it
        expresses is the caller's; see :meth:`risk_neutral`.

    Notes
    -----
    The Feller condition :math:`2\\kappa\\theta > \\xi^2` keeps the variance
    strictly positive.  It is **not** enforced: real calibrations routinely
    violate it, and refusing those parameters would make the library unable to
    represent the surfaces it exists to fit.  :attr:`feller_ratio` reports it so
    a caller can decide, and both samplers stay well defined when it fails.

    Parameters are stored exactly as passed, so a torch tensor with
    ``requires_grad=True`` stays that tensor.

    Examples
    --------
    >>> from fast_vollib.processes import Heston
    >>> process = Heston(kappa=2.0, theta=0.04, vol_of_vol=0.3, rho=-0.7)
    >>> float(round(process.feller_ratio, 6))
    1.777778
    >>> process.satisfies_feller
    True
    """

    kappa: Any
    theta: Any
    vol_of_vol: Any
    rho: Any
    drift: Any = 0.0

    #: The two state variables this process evolves, in the order they occupy
    #: the last axis of a sample.
    state_names: ClassVar[tuple[str, ...]] = ("spot", "variance")

    def __post_init__(self) -> None:
        _validate_parameter(self.kappa, field="kappa", positive=True)
        _validate_parameter(self.theta, field="theta", positive=True)
        _validate_parameter(self.vol_of_vol, field="vol_of_vol", positive=True)
        _validate_parameter(self.rho, field="rho")
        _validate_parameter(self.drift, field="drift")
        rho = _concrete(self.rho)
        if rho is not None and abs(rho) >= 1.0:
            raise SimulationValidationError(
                f"rho must lie strictly inside (-1, 1); got {rho!r}. At |rho| = 1 the two "
                f"drivers are the same Brownian motion and the model has one factor."
            )

    @classmethod
    def risk_neutral(
        cls,
        *,
        rate: Any,
        kappa: Any,
        theta: Any,
        vol_of_vol: Any,
        rho: Any,
        dividend_yield: Any = 0.0,
    ) -> "Heston":
        """A process whose log-spot drift is ``rate - dividend_yield``.

        A convenience for the common case, and nothing more.  It records a
        modelling decision the caller made rather than making one.
        """
        _validate_parameter(rate, field="rate")
        _validate_parameter(dividend_yield, field="dividend_yield")
        return cls(
            kappa=kappa,
            theta=theta,
            vol_of_vol=vol_of_vol,
            rho=rho,
            drift=rate - dividend_yield,
        )

    def params(self) -> Mapping[str, Any]:
        """The five parameters, as the original objects."""
        return MappingProxyType(
            {
                "kappa": self.kappa,
                "theta": self.theta,
                "vol_of_vol": self.vol_of_vol,
                "rho": self.rho,
                "drift": self.drift,
            }
        )

    @property
    def feller_ratio(self) -> float:
        """``2 kappa theta / xi^2``.  Above one the variance never reaches zero."""
        kappa = _concrete(self.kappa)
        theta = _concrete(self.theta)
        xi = _concrete(self.vol_of_vol)
        if kappa is None or theta is None or xi is None:  # pragma: no cover - traced values
            return float("nan")
        return 2.0 * kappa * theta / (xi * xi)

    @property
    def satisfies_feller(self) -> bool:
        """Whether :attr:`feller_ratio` exceeds one."""
        return bool(self.feller_ratio > 1.0)

    def integrated_variance_mean(self, T: float, v0: float) -> float:
        """``E[int_0^T v_s ds] = theta T + (v0 - theta)(1 - e^{-kappa T}) / kappa``.

        Exact, and the reason a discretization bias can be measured rather than
        guessed at: the sampler's mean integrated variance has a closed form to
        converge to.
        """
        kappa = float(self.kappa)
        theta = float(self.theta)
        return theta * T + (float(v0) - theta) * (1.0 - math.exp(-kappa * T)) / kappa

    def sample(
        self,
        *,
        initial_state: Mapping[str, Any],
        time_grid: Any,
        n_paths: int,
        rng: Any,
        antithetic: bool = False,
        scheme: str = "quadratic_exponential",
    ) -> Any:
        """Paths shaped ``(n_paths, n_times, 2)`` -- spot first, then variance.

        Parameters
        ----------
        initial_state : Mapping
            Must contain ``"spot"`` (strictly positive) and ``"variance"``
            (non-negative).
        time_grid : array-like
            Increasing times starting at zero.  Column zero of every path is the
            initial state exactly.
        n_paths : int
        rng : object
            A generator, PRNG key, or integer seed, per
            :mod:`fast_vollib._random_api`.
        antithetic : bool, default False
            Draw half the normals and mirror them.  The mirroring is applied to
            the whole draw, so it flips the variance innovation as well as the
            spot's -- which is what makes the pair a genuine antithetic pair for
            a correlated model.
        scheme : str, default 'quadratic_exponential'
            One of :data:`SCHEMES`.

        Returns
        -------
        array
            In the namespace, device, and dtype the inputs selected.

        Raises
        ------
        SimulationValidationError
            On an unknown scheme, a missing or invalid initial state.
        """
        if scheme not in SCHEMES:
            raise SimulationValidationError(f"scheme must be one of {SCHEMES}; got {scheme!r}.")
        spot = _validate_parameter(
            _require(initial_state, "spot"), field="initial_state['spot']", positive=True
        )
        variance = _validate_parameter(
            _require(initial_state, "variance"),
            field="initial_state['variance']",
            non_negative=True,
        )
        inputs = {
            "initial_state['spot']": spot,
            "initial_state['variance']": variance,
            "kappa": self.kappa,
            "theta": self.theta,
            "vol_of_vol": self.vol_of_vol,
            "rho": self.rho,
            "drift": self.drift,
            "time_grid": time_grid,
            "rng": rng,
        }
        namespace = resolve_namespace(inputs)
        device = resolve_device(namespace, inputs)
        dtype = resolve_dtype(namespace, inputs)
        stream = random_stream(rng, namespace=namespace, device=device, dtype=dtype)

        xp = get_namespace(time_grid, spot, variance, self.kappa, self.theta)
        steps = time_grid[1:] - time_grid[:-1]
        n_steps = len(steps)
        # Two independent normals per step: one drives the variance transition,
        # one the conditionally independent part of the log-spot.
        normals = standard_normal(stream, (int(n_paths), n_steps, 2), antithetic=antithetic)

        # The running quantity is the log *return*, not the log spot, so column
        # zero is ``spot * exp(0)`` -- the initial state bit for bit, rather than
        # ``exp(log(spot))``, which is a value the sampler arrived at.
        log_return = xp.zeros((int(n_paths),), like=normals)
        v = xp.zeros((int(n_paths),), like=normals) + xp.asarray(variance, like=normals)
        spot_value = xp.asarray(spot, like=normals)
        spots = [spot_value + log_return]
        variances = [v]
        advance = _qe_step if scheme == "quadratic_exponential" else _full_truncation_euler_step
        for index in range(n_steps):
            dt = steps[index]
            log_return, v = advance(self, xp, log_return, v, dt, normals[:, index, :])
            spots.append(spot_value * xp.exp(log_return))
            variances.append(v)
        return xp.stack([xp.stack(spots, axis=1), xp.stack(variances, axis=1)], axis=2)


def _require(initial_state: Mapping[str, Any], name: str) -> Any:
    try:
        return initial_state[name]
    except (KeyError, TypeError):
        raise SimulationValidationError(
            f"Heston evolves 'spot' and 'variance' states, which the initial state must "
            f"supply; {name!r} is missing from "
            f"{sorted(initial_state) if hasattr(initial_state, 'keys') else '?'}."
        ) from None


def _concrete(value: Any) -> float | None:
    from .._array_api import concrete_float

    return concrete_float(value)


def _qe_step(process: Heston, xp: Any, log_spot: Any, v: Any, dt: Any, normals: Any) -> Any:
    """Heston's own parameters, into the shared quadratic-exponential step."""
    return quadratic_exponential_step_with_spot(
        xp,
        log_spot,
        v,
        kappa=xp.asarray(process.kappa, like=v),
        theta=xp.asarray(process.theta, like=v),
        xi=xp.asarray(process.vol_of_vol, like=v),
        rho=xp.asarray(process.rho, like=v),
        drift=xp.asarray(process.drift, like=v),
        dt=xp.asarray(dt, like=v),
        z_variance=normals[:, 0],
        z_spot=normals[:, 1],
    )


def _full_truncation_euler_step(
    process: Heston, xp: Any, log_spot: Any, v: Any, dt: Any, normals: Any
) -> Any:
    """Heston's own parameters, into the shared full-truncation Euler step."""
    return full_truncation_euler_step(
        xp,
        log_spot,
        v,
        kappa=xp.asarray(process.kappa, like=v),
        theta=xp.asarray(process.theta, like=v),
        xi=xp.asarray(process.vol_of_vol, like=v),
        rho=xp.asarray(process.rho, like=v),
        drift=xp.asarray(process.drift, like=v),
        dt=xp.asarray(dt, like=v),
        z_variance=normals[:, 0],
        z_orthogonal=normals[:, 1],
    )
