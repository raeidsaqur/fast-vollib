"""How a simulated payoff is carried back to today.

Discounting is the one step of a Monte Carlo valuation that is usually left
implicit, and leaving it implicit is exactly what stops a stochastic-rate model
from working.  Under a deterministic rate the discount factor is a constant, so
it commutes with the expectation and can be applied to the average at the end.
Under a stochastic one it cannot: the payoff and the discount factor are
functions of the same path, and

.. math:: E\\!\\left[e^{-\\int_0^T r_u du} X_T\\right]
    \\neq E\\!\\left[e^{-\\int_0^T r_u du}\\right] E[X_T]

whenever the two are correlated -- which generally happens under BCC97 because
the rate drives the spot's drift.  A ``P(0,T)``-times-expected-payoff shortcut
is therefore not an approximation to be offered with a warning; it is a
different quantity, and it is not offered.

Making the rule an argument is what lets the difference be *stated*.  The two
implementations here are the two things a caller might mean, they are named,
and neither is inferred from the presence of a state variable.

The quadrature is chosen, not guessed
-------------------------------------
:class:`PathwiseShortRateDiscounting` integrates the sampled rate over the
grid, and the grid is finite, so the integral is approximate however it is
taken.  ``rule`` names the approximation rather than defaulting to whichever is
more accurate, because the rules use different interpolants and a caller
comparing runs needs to know which one produced a number.  Neither is applied
to a grid it cannot see: the rule reads the times it was given and nothing
else.

What is *not* approximate is the sampling of the rate at the grid points, which
is a separate error with a separate cause.  ``CIRShortRate``'s
``exact_transition`` removes it entirely and leaves this one untouched, which
is what makes the two measurable apart.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Protocol, Sequence, runtime_checkable

from .._array_api import concrete_float, get_namespace
from .._random_api import namespace_of
from .errors import SimulationValidationError

__all__ = [
    "RULES",
    "ConstantRateDiscounting",
    "DiscountingRule",
    "PathwiseShortRateDiscounting",
    "constant_rate_factor",
]

#: The quadratures :class:`PathwiseShortRateDiscounting` implements.
RULES = ("trapezoid", "left_riemann")


@runtime_checkable
class DiscountingRule(Protocol):
    """What a caller supplies to say how a simulated cashflow is discounted.

    Structural, like :class:`fast_vollib.rates.DiscountCurve` and
    :class:`fast_vollib.processes.StochasticProcess`: an object with this one
    method works, with no registration and no subclassing.

    Notes
    -----
    ``isinstance`` against this protocol checks that the method exists, not
    that it behaves.
    """

    def discount_factors(self, *, states: Any, time_grid: Any, state_names: Sequence[str]) -> Any:
        """One discount factor per path, shaped ``(n_paths,)``.

        Parameters
        ----------
        states : array
            Simulated paths shaped ``(n_paths, n_times, n_state)``.
        time_grid : array
            The times those states were sampled at, starting at zero.
        state_names : Sequence[str]
            What the trailing axis holds, in order -- the process's own
            ``state_names``.  A rule that needs a particular state finds it by
            name here rather than by a position it would have to assume.
        """
        ...  # pragma: no cover - protocol declaration


def constant_rate_factor(rate: Any, maturity: Any) -> Any:
    """``exp(-rate * maturity)``, by the route the rest of the library takes.

    Deliberately identical, operation for operation, to what
    :func:`fast_vollib.simulation.monte_carlo._discount` does with its rate --
    including the choice of :func:`math.exp` for a backend-neutral rate, which
    is not a stylistic preference.  ``math.exp`` and ``numpy.exp`` disagree in
    the last bit at some arguments, so computing the same factor "the obvious
    way" here would make a caller who moved from the engine's own discounting
    to an explicit :class:`ConstantRateDiscounting` see their price change in
    the last digits for no reason they could name.

    A native rate keeps its autograd graph and goes through the namespace's
    exponential; a Python one has no graph to keep, and ``math.exp`` avoids
    materializing a zero-dimensional array whose default precision would
    promote a single-precision payoff.
    """
    if namespace_of(rate) is None:
        return math.exp(-float(rate) * float(maturity))
    ns = get_namespace(rate)
    return ns.exp(-rate * maturity)


@dataclass(frozen=True, slots=True)
class ConstantRateDiscounting:
    """``exp(-rate * T)`` at the horizon the grid ends at.

    Today's behaviour, given a name so it can be chosen instead of assumed.

    Parameters
    ----------
    rate : float or array
        Continuously compounded, matching
        :class:`fast_vollib.rates.FlatDiscountCurve` and
        ``SurfaceMarket.discount_at``.  May be negative.

    Notes
    -----
    ``T`` is the last entry of the grid, which is the only horizon a rule with
    this signature can see.  It is not necessarily the contract's maturity: a
    caller-supplied grid is accepted when it ends *within tolerance* of
    maturity, so a rule reading the grid and an engine reading the contract can
    differ in the last bits.  That is a real difference and it is documented
    rather than papered over -- pass a grid that ends exactly at maturity and
    the two agree to the bit.

    Examples
    --------
    >>> import numpy as np
    >>> from fast_vollib.simulation.discounting import ConstantRateDiscounting
    >>> states = np.zeros((3, 2, 1))
    >>> rule = ConstantRateDiscounting(rate=0.05)
    >>> factors = rule.discount_factors(
    ...     states=states, time_grid=np.array([0.0, 2.0]), state_names=("spot",)
    ... )
    >>> factors.shape
    (3,)
    >>> bool(np.all(factors == np.exp(-0.10)))
    True
    """

    rate: Any

    def discount_factors(self, *, states: Any, time_grid: Any, state_names: Sequence[str]) -> Any:
        _validate_shapes(states, time_grid)
        xp = get_namespace(states)
        factor = constant_rate_factor(self.rate, time_grid[-1])
        return xp.zeros((_n_paths(states),), like=states) + factor


@dataclass(frozen=True, slots=True)
class PathwiseShortRateDiscounting:
    """``exp(-int_0^T r_u du)`` from a simulated short rate.

    Parameters
    ----------
    state_name : str, default 'short_rate'
        Which state of the sampled paths is the short rate.  Looked up in the
        ``state_names`` it is handed; a name that is not there is an error
        naming the ones that are, never a fallback to the first column.
    rule : {'trapezoid', 'left_riemann'}, default 'trapezoid'
        The quadrature. The difference between the two rules is a useful
        discretization diagnostic, not a guaranteed error estimate or a
        convergence-order claim for stochastic paths.  ``left_riemann`` is here to make that measurement
        possible, not as a recommendation.

    Notes
    -----
    Both rules are exact for a rate that is constant in time, and neither is
    exact for one that is not.  The rate is a *sampled* path, so even the
    trapezoid rule is integrating a piecewise-linear interpolant of something
    that is nowhere differentiable; refining the grid reduces the error and no
    order is claimed for it here.

    Examples
    --------
    >>> import numpy as np
    >>> from fast_vollib.simulation.discounting import PathwiseShortRateDiscounting
    >>> rates = np.full((2, 3, 1), 0.04)
    >>> rule = PathwiseShortRateDiscounting()
    >>> factors = rule.discount_factors(
    ...     states=rates,
    ...     time_grid=np.array([0.0, 0.5, 1.0]),
    ...     state_names=("short_rate",),
    ... )
    >>> bool(np.allclose(factors, np.exp(-0.04)))
    True
    """

    state_name: str = "short_rate"
    rule: str = "trapezoid"

    def __post_init__(self) -> None:
        if not isinstance(self.state_name, str) or not self.state_name:
            raise SimulationValidationError(
                f"state_name must be a non-empty string naming a simulated state; got "
                f"{self.state_name!r}."
            )
        if self.rule not in RULES:
            raise SimulationValidationError(
                f"rule must be one of {RULES}; got {self.rule!r}. These are the supported "
                f"quadrature rules."
            )

    def discount_factors(self, *, states: Any, time_grid: Any, state_names: Sequence[str]) -> Any:
        index = self._index_of(state_names)
        _validate_shapes(states, time_grid)
        xp = get_namespace(states)
        rate = states[:, :, index]
        steps = time_grid[1:] - time_grid[:-1]
        if self.rule == "trapezoid":
            contribution = 0.5 * (rate[:, :-1] + rate[:, 1:]) * steps
        else:
            contribution = rate[:, :-1] * steps
        return xp.exp(-xp.sum(contribution, axis=1))

    def _index_of(self, state_names: Sequence[str]) -> int:
        """Where the short rate sits on the trailing axis.

        Resolved before any arithmetic, so a mismatched process fails naming
        both sides rather than silently discounting by a spot.
        """
        names = tuple(state_names)
        if self.state_name not in names:
            raise SimulationValidationError(
                f"{self.state_name!r} is not among the simulated states {names}. A "
                f"discounting rule will not fall back to another column: discounting a "
                f"payoff by a spot instead of a rate produces a number, and the number "
                f"is not a price."
            )
        return names.index(self.state_name)


def _n_paths(states: Any) -> int:
    return int(states.shape[0])


def _validate_shapes(states: Any, time_grid: Any) -> None:
    """The horizon check, run before any factor is formed.

    Cheap, and it catches the two mistakes that would otherwise produce a
    plausible number: paths and times that came from different runs, and a grid
    that does not start at the valuation date, which would silently discount
    over the wrong interval.
    """
    shape = getattr(states, "shape", None)
    if shape is None or len(shape) != 3:
        raise SimulationValidationError(
            f"states must be shaped (n_paths, n_times, n_state); got "
            f"{shape if shape is not None else type(states).__name__}."
        )
    times = getattr(time_grid, "shape", None)
    if times is None or len(times) != 1:
        raise SimulationValidationError(f"time_grid must be one-dimensional; got shape {times}.")
    if times[0] != shape[1]:
        raise SimulationValidationError(
            f"time_grid has {times[0]} times but the paths have {shape[1]}. They "
            f"describe different runs."
        )
    if shape[1] < 2:
        raise SimulationValidationError(
            "Discounting needs at least two times: a horizon is an interval, and a "
            "single point does not describe one."
        )
    # The value check runs whenever the number can be read eagerly and is
    # skipped for a tracer, which has no value to check -- the same split
    # ``_validate_parameter`` makes for process parameters. The shape checks
    # above are static and always run, so a traced call is still guarded
    # against the mistake that actually happens: paths and times from
    # different runs.
    start = concrete_float(time_grid[0])
    if start is not None and start != 0.0:
        raise SimulationValidationError(
            f"time_grid must start at the valuation date, t=0; got {start!r}. An "
            f"integral taken from a later start would discount over the wrong "
            f"interval and still return a plausible factor."
        )
