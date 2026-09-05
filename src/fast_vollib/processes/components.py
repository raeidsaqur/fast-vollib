"""The three axes a model in this library varies along, as named values.

Bakshi, Cao and Chen (1997) is not one model but a lattice: stochastic
volatility, jumps, and stochastic interest rates, each of which can be switched
off, and the well-known models are the corners.  Writing that as six classes
would mean six copies of the same sampler; writing it as one class with boolean
flags would mean a configuration that says ``jumps=True`` and then holds jump
parameters nobody validated.

So each axis is a *component*: a small frozen value that either carries the
parameters of a feature or is a marker saying the feature is absent.  A
configuration is then a tuple of components, every parameter it holds has been
validated, and every parameter it does not hold cannot be read.

Two rules make that work.

**The unions are closed.**  ``VarianceComponent``, ``JumpComponent`` and
``ShortRateComponent`` list every admissible type, and a sampler checks
membership before it touches arithmetic or randomness.  A duck-typed
look-alike is refused at construction rather than half-way through a hundred
thousand paths.

**A constant marker stores no level.**  :class:`ConstantVariance` does not hold
the variance and :class:`ConstantShortRate` does not hold the rate; both come
from ``initial_state``, which is where every other state variable comes from.
An API able to hold two disagreeing values for one quantity will eventually hold
two disagreeing values for one quantity.

Naming
------
:class:`LognormalJumps` is the Merton (1976) law under a name that says what the
law *is* rather than whose model it appeared in.  There is no second
``MertonJumps`` and no ``MertonJumpDiffusion`` facade: Merton's model is
``Bates(variance=ConstantVariance(), jumps=LognormalJumps(...))``, and the
reduction is an evaluation of the same code path rather than a separate
implementation that has to be kept in agreement.

References
----------
Merton, R. C. (1976). Option pricing when underlying stock returns are
discontinuous. *Journal of Financial Economics* 3(1-2), 125-144.

Bates, D. S. (1996). Jumps and stochastic volatility: exchange rate processes
implicit in Deutsche Mark options. *Review of Financial Studies* 9(1), 69-107.

Bakshi, G., Cao, C., Chen, Z. (1997). Empirical performance of alternative
option pricing models. *Journal of Finance* 52(5), 2003-2049.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Union

from .._array_api import concrete_float
from .._jump_law import mean_relative_jump
from .._simulation_errors import SimulationValidationError
from .cir import CIRShortRate
from .gbm import _validate_parameter

__all__ = [
    "CIRShortRate",
    "ConstantShortRate",
    "ConstantVariance",
    "HestonVariance",
    "JumpComponent",
    "LognormalJumps",
    "NoJumps",
    "ShortRateComponent",
    "VarianceComponent",
]


@dataclass(frozen=True, slots=True)
class HestonVariance:
    """A square-root variance correlated with the spot.

    The same four parameters :class:`~fast_vollib.processes.Heston` holds, with
    the same validation and the same meanings, and deliberately *not* the
    drift: a drift belongs to the spot, and a variance component that carried
    one could be combined with a facade that carried another.

    Parameters
    ----------
    kappa, theta, vol_of_vol : float or scalar array
        Strictly positive. ``vol_of_vol`` is excluded at zero for the reason
        :class:`~fast_vollib.processes.Heston` gives: the variance is then
        deterministic but generally time-dependent. :class:`ConstantVariance`
        represents only a constant variance, not a mean-reverting deterministic path.
    rho : float or scalar array
        Strictly inside ``(-1, 1)``.

    Examples
    --------
    >>> from fast_vollib.processes import HestonVariance
    >>> component = HestonVariance(kappa=2.0, theta=0.04, vol_of_vol=0.3, rho=-0.7)
    >>> float(round(component.feller_ratio, 6))
    1.777778
    """

    kappa: Any
    theta: Any
    vol_of_vol: Any
    rho: Any

    def __post_init__(self) -> None:
        _validate_parameter(self.kappa, field="variance.kappa", positive=True)
        _validate_parameter(self.theta, field="variance.theta", positive=True)
        _validate_parameter(self.vol_of_vol, field="variance.vol_of_vol", positive=True)
        _validate_parameter(self.rho, field="variance.rho")
        rho = concrete_float(self.rho)
        if rho is not None and abs(rho) >= 1.0:
            raise SimulationValidationError(
                f"variance.rho must lie strictly inside (-1, 1); got {rho!r}. At "
                f"|rho| = 1 the two drivers are the same Brownian motion and the model "
                f"has one factor."
            )

    def params(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "kappa": self.kappa,
                "theta": self.theta,
                "vol_of_vol": self.vol_of_vol,
                "rho": self.rho,
            }
        )

    @property
    def feller_ratio(self) -> float:
        """``2 kappa theta / vol_of_vol^2``.  Above one the variance stays positive."""
        kappa = concrete_float(self.kappa)
        theta = concrete_float(self.theta)
        xi = concrete_float(self.vol_of_vol)
        if kappa is None or theta is None or xi is None:  # pragma: no cover - traced
            return float("nan")
        return 2.0 * kappa * theta / (xi * xi)

    @property
    def satisfies_feller(self) -> bool:
        """Whether :attr:`feller_ratio` exceeds one.  Reported, never enforced."""
        return bool(self.feller_ratio > 1.0)


@dataclass(frozen=True, slots=True)
class ConstantVariance:
    """A variance that does not move, held at ``initial_state["variance"]``.

    Stores no level, so a configuration cannot hold one value here and a
    different one in the initial state.  The spot is then exactly log-normal
    and its transition is sampled exactly on any grid.

    Examples
    --------
    >>> from fast_vollib.processes import ConstantVariance
    >>> ConstantVariance() == ConstantVariance()
    True
    >>> dict(ConstantVariance().params())
    {}
    """

    def params(self) -> Mapping[str, Any]:
        return MappingProxyType({})


@dataclass(frozen=True, slots=True)
class LognormalJumps:
    """Compound Poisson jumps whose multiplier ``1 + J`` is log-normal.

    Parameters
    ----------
    jump_intensity : float or scalar array
        Expected jumps per year, non-negative. **Zero is admissible** and is
        the reduction: at zero intensity the jump contribution is zero. The
        jump draw block is still consumed; use :class:`NoJumps` to omit it.
    mean_log_jump : float or scalar array
        Mean of ``ln(1 + J)``, finite, any sign.
    jump_volatility : float or scalar array
        Standard deviation of ``ln(1 + J)``, non-negative. Zero is a jump of a
        fixed proportional size, which is a model rather than a degenerate case.

    Notes
    -----
    ``1 + J`` is log-normal, so a jump can never take the spot to or below
    zero -- which is what makes the log-spot representation exact rather than
    an approximation that needs a floor.

    The compound Poisson is genuine: the number of jumps in a step is Poisson
    and the aggregate log jump is ``K * mean_log_jump + jump_volatility *
    sqrt(K) * Z``, which is the exact conditional law of a sum of ``K``
    independent normals. It is never an at-most-one-jump Bernoulli
    approximation, which would misprice short maturities at high intensity.

    Examples
    --------
    >>> from fast_vollib.processes import LognormalJumps
    >>> jumps = LognormalJumps(
    ...     jump_intensity=0.5, mean_log_jump=-0.05, jump_volatility=0.2
    ... )
    >>> float(round(jumps.mean_relative_jump, 8))
    -0.02955447
    >>> float(round(jumps.drift_compensator, 8))
    -0.01477723
    """

    jump_intensity: Any
    mean_log_jump: Any
    jump_volatility: Any

    def __post_init__(self) -> None:
        _validate_parameter(self.jump_intensity, field="jumps.jump_intensity", non_negative=True)
        _validate_parameter(self.mean_log_jump, field="jumps.mean_log_jump")
        _validate_parameter(self.jump_volatility, field="jumps.jump_volatility", non_negative=True)

    def params(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "jump_intensity": self.jump_intensity,
                "mean_log_jump": self.mean_log_jump,
                "jump_volatility": self.jump_volatility,
            }
        )

    @property
    def mean_relative_jump(self) -> Any:
        r"""``E[J] = exp(m + delta^2 / 2) - 1``, BCC97's :math:`\mu_J`.

        Written once and used twice, which is the point of putting it here.
        The sampler subtracts ``jump_intensity * mean_relative_jump`` from the
        drift so the discounted spot is a martingale, and the characteristic
        function carries ``-i u lambda mu_J T`` for the same reason. If the two
        were spelled separately they would disagree in a sign or in the last
        bits, and a Fourier-versus-Monte-Carlo test would be the first place
        anyone found out.
        """
        return mean_relative_jump(self.mean_log_jump, self.jump_volatility)

    @property
    def drift_compensator(self) -> Any:
        """``jump_intensity * mean_relative_jump``, subtracted from the drift."""
        return self.jump_intensity * self.mean_relative_jump


@dataclass(frozen=True, slots=True)
class NoJumps:
    """No jump component.  Both jump quantities are exactly zero.

    Exactly, and it matters: the compensator is subtracted from the drift and
    added into the characteristic function's exponent, so a value that was
    merely very small would move every price in the last digits and make the
    reduction tests approximate.

    Examples
    --------
    >>> from fast_vollib.processes import NoJumps
    >>> NoJumps().mean_relative_jump, NoJumps().drift_compensator
    (0.0, 0.0)
    """

    def params(self) -> Mapping[str, Any]:
        return MappingProxyType({})

    @property
    def mean_relative_jump(self) -> float:
        return 0.0

    @property
    def drift_compensator(self) -> float:
        return 0.0


@dataclass(frozen=True, slots=True)
class ConstantShortRate:
    """A short rate that does not move, held at ``initial_state["short_rate"]``.

    Stores no level, for the same reason :class:`ConstantVariance` does not.

    Examples
    --------
    >>> from fast_vollib.processes import ConstantShortRate
    >>> dict(ConstantShortRate().params())
    {}
    """

    def params(self) -> Mapping[str, Any]:
        return MappingProxyType({})


#: Every variance model a configuration may hold.
VarianceComponent = Union[HestonVariance, ConstantVariance]

#: Every jump model a configuration may hold.
JumpComponent = Union[LognormalJumps, NoJumps]

#: Every short-rate model a configuration may hold.
#:
#: :class:`~fast_vollib.processes.CIRShortRate` appears here as a component as
#: well as being a process in its own right; it is the same object, and a rate
#: model is exactly the thing that can be simulated alone or driven inside a
#: larger one.
ShortRateComponent = Union[CIRShortRate, ConstantShortRate]

#: The concrete types of each union, for the membership checks a sampler makes
#: before it touches arithmetic or randomness.
VARIANCE_TYPES: tuple[type, ...] = (HestonVariance, ConstantVariance)
JUMP_TYPES: tuple[type, ...] = (LognormalJumps, NoJumps)
SHORT_RATE_TYPES: tuple[type, ...] = (CIRShortRate, ConstantShortRate)


def validate_component(value: Any, *, field: str, admissible: tuple[type, ...]) -> Any:
    """``value`` is one of ``admissible``, or an error naming what is.

    An exact-type check by ``isinstance`` against a closed tuple, matching the
    registry's rule for instruments: an object that merely looks like a
    component is refused, because a sampler reads fields by name and would
    otherwise produce numbers from something nobody validated.
    """
    if not isinstance(value, admissible):
        names = ", ".join(t.__name__ for t in admissible)
        raise SimulationValidationError(
            f"{field} must be one of {names}; got {type(value).__name__}. The set is "
            f"closed: a sampler reads a component's fields by name, so an object that "
            f"merely resembles one would produce numbers from parameters nothing "
            f"validated."
        )
    return value
