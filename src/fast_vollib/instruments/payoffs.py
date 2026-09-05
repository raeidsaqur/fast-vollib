"""Terminal and path payoffs, evaluated in the caller's array namespace.

A payoff is a pure map from the state it requires to a cashflow: terminal
contracts consume a terminal underlying value, while path-dependent contracts
consume a validated Scenario bridge.  It does not discount, simulate, or
consult market inputs -- those are separate concerns, and folding any of them
in here would make the function untestable against a hand-computed value.

The implementation matters as much as the formula.  Evaluation runs through
:mod:`fast_vollib._array_api` in whatever namespace the input arrives in, so a
torch tensor stays a torch tensor on its own device with its autograd tape
intact, and a jax array stays a jax array that ``jit`` and ``grad`` can trace.
No *computed* value stages through host memory: the cashflow you get back was
built entirely in the caller's namespace, which is the difference between a
payoff you can put in a loss function and one you can only print.

Two payoffs do read a single boolean back.  Geometric averaging and realized
variance take logarithms, so both check that the path is strictly positive
before evaluating, and reading that one flag costs a device synchronization per
call.  It is a domain check, not arithmetic: the flag decides whether to raise
and never enters the result or its graph.  The alternative is a silent
all-``NaN`` price with no indication of which path caused it.

This is deliberately unlike the pricing adapters.  Those wrap fused kernels
that move data host-to-device by design and therefore do not carry gradients --
see :func:`fast_vollib.instruments.capabilities` for which routes do.

Dispatch is on the instrument class, through an explicit table.  Payoffs are
not callables stored on contracts and not expressions parsed from strings: a
serialized instrument must not be able to make the library execute something.

Examples
--------
>>> import numpy as np
>>> from fast_vollib.instruments import EuropeanOption, payoff
>>> call = EuropeanOption(underlier="ACME", option_type="call", strike=100.0, maturity=1.0)
>>> payoff(call, np.array([90.0, 100.0, 110.0]))
array([ 0.,  0., 10.])
>>> short_ten = EuropeanOption(
...     underlier="ACME", option_type="put", strike=100.0, maturity=1.0, notional=-10.0,
... )
>>> payoff(short_ten, np.array([90.0, 110.0]))
array([-100.,   -0.])
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, TypeVar, overload

from .._array_api import concrete_bool, get_namespace
from .base import Instrument
from .enums import AveragingMethod, OptionType, PayoffRequirement, StrikeConvention
from .errors import InstrumentValidationError, UnsupportedInstrumentError
from .exotics import (
    AsianOption,
    BarrierOption,
    BinaryOption,
    LookbackOption,
    VarianceSwap,
)
from .fixed_income import FixedIncomeSecurity
from .forwards import Forward, Future
from .options import EuropeanOption
from .registry import instrument_types

__all__ = ["payoff", "payoff_requirement"]

_ArrayT = TypeVar("_ArrayT")


class _ScenarioInput(ABC):
    """What a payoff needs from a simulated scenario, and nothing more.

    A path-dependent payoff needs a whole trajectory rather than a terminal
    state, and it must be able to refuse a trajectory that belongs to a
    different instrument.  Both of those live in
    :mod:`fast_vollib.simulation`, which imports this package -- so importing
    it back to recognize a ``Scenario`` would make the dependency circular.

    This class is the seam.  ``simulation.Scenario`` implements it, dispatch
    here recognizes the abstract type, and the checks that need to know what a
    scenario *is* stay on the scenario.  It is deliberately private: it is an
    implementation bridge with three members, not a public protocol inviting
    third-party scenario types, and nothing about it is a compatibility
    promise.

    A bare array is not a scenario and never becomes one by accident.  A
    two-dimensional path matrix carries no underlier, no time grid, and no way
    to tell whether its horizon is the contract's maturity, so a payoff handed
    one refuses it rather than computing a plausible number.
    """

    __slots__ = ()

    @abstractmethod
    def _validate_for_contract(self, instrument: Instrument) -> None:
        """Raise unless this scenario describes ``instrument``'s underlier and horizon."""

    @abstractmethod
    def _state_path(self, name: str = "spot") -> Any:
        """The ``(n_paths, n_times)`` trajectory of one state variable."""

    @abstractmethod
    def _path_time_grid(self) -> Any:
        """The observation times, starting at zero and ending at the horizon."""


def _require_payoff_support(instrument: Instrument) -> None:
    """Raise the canonical refusal if this type has no payoff evaluator at all.

    Used by scenario dispatch so that "an asset is not a contract" is reported
    as such, rather than as a mismatch between an asset and a trajectory.
    """
    if type(instrument) not in _PAYOFFS and type(instrument) not in _PATH_PAYOFFS:
        raise UnsupportedInstrumentError(_no_payoff_message(instrument))


def _reject_scenario_for_terminal_payoff(instrument: Instrument) -> None:
    raise InstrumentValidationError(
        f"{type(instrument).__name__} has a terminal payoff and takes the underlier's "
        f"value at maturity, not a whole scenario. Evaluate it with "
        f"scenario.payoff(instrument), which reads the terminal state after checking "
        f"that the scenario describes this contract."
    )


def _forward_payoff(instrument: Forward, terminal_state: Any) -> Any:
    return instrument.notional * (terminal_state - instrument.delivery_price)


def _future_payoff(instrument: Future, terminal_state: Any) -> Any:
    return instrument.notional * (terminal_state - instrument.contract_price)


def _european_option_payoff(instrument: EuropeanOption, terminal_state: Any) -> Any:
    ns = get_namespace(terminal_state)
    if instrument.option_type is OptionType.CALL:
        intrinsic = terminal_state - instrument.strike
    else:
        intrinsic = instrument.strike - terminal_state
    # ``clip(x, 0, None)`` rather than ``maximum(x, 0.0)``: the latter has to
    # materialize a zero, and a zero built at the namespace's default precision
    # promotes a float32 input to float64. Clipping keeps the caller's dtype in
    # all three namespaces.
    return instrument.notional * ns.clip(intrinsic, 0.0, None)


def _indicator(ns: Any, condition: Any, reference: Any) -> Any:
    """1 where ``condition`` holds and 0 elsewhere, in ``reference``'s form.

    Built from ``reference`` rather than from two literals so the result keeps
    the caller's dtype, device, and autograd graph. Selecting between bare
    Python constants would detach the output, and a caller differentiating a
    digital payoff would then get ``None`` where the correct answer is a
    gradient of exactly zero.
    """
    zero = reference * 0.0
    return ns.where(condition, zero + 1.0, zero)


def _binary_option_payoff(instrument: BinaryOption, terminal_state: Any) -> Any:
    ns = get_namespace(terminal_state)
    # Strict on both sides: at the strike exactly, a call and a put both pay
    # nothing, so the two can never both pay at a level the market treats as
    # unresolved.
    if instrument.option_type is OptionType.CALL:
        in_the_money = terminal_state > instrument.strike
    else:
        in_the_money = terminal_state < instrument.strike
    scale = ns.scalar(instrument.notional * instrument.cash_amount, like=terminal_state)
    return scale * _indicator(ns, in_the_money, terminal_state)


def _require_positive_path(ns: Any, values: Any, *, operation: str) -> None:
    """Refuse a trajectory a multiplicative payoff is undefined on.

    A geometric average takes logarithms, so a non-positive state has no answer
    rather than a large one. Returning ``NaN`` would push the failure into a
    reduction, where it becomes an all-``NaN`` price with no indication of
    which path caused it.

    The check reads a concrete flag and is skipped under a JAX trace, which has
    none; a traced call carries the domain as a precondition, exactly as
    parameter validation does.
    """
    if concrete_bool(ns.all(values > 0.0)) is False:
        raise InstrumentValidationError(
            f"{operation} needs strictly positive underlying states; this path reaches "
            f"zero or below, where the payoff is undefined rather than large."
        )


def _asian_average(instrument: AsianOption, ns: Any, fixings: Any) -> Any:
    if instrument.averaging_method is AveragingMethod.GEOMETRIC:
        _require_positive_path(ns, fixings, operation="Geometric averaging")
        return ns.exp(ns.mean(ns.log(fixings), axis=1))
    return ns.mean(fixings, axis=1)


def _asian_option_payoff(instrument: AsianOption, scenario: _ScenarioInput) -> Any:
    spot = scenario._state_path("spot")
    ns = get_namespace(spot)
    # Fixings exclude the valuation date: S_0 is known when the contract is
    # written, so averaging it in would make part of the payoff a constant the
    # contract never agreed to.
    average = _asian_average(instrument, ns, spot[:, 1:])
    terminal = spot[:, -1]
    if instrument.strike_convention is StrikeConvention.FIXED:
        reference, level = average, instrument.strike
    else:
        reference, level = terminal, average
    if instrument.option_type is OptionType.CALL:
        intrinsic = reference - level
    else:
        intrinsic = level - reference
    return instrument.notional * ns.clip(intrinsic, 0.0, None)


_PAYOFFS: dict[type[Instrument], Callable[[Any, Any], Any]] = {
    Forward: _forward_payoff,
    Future: _future_payoff,
    EuropeanOption: _european_option_payoff,
    BinaryOption: _binary_option_payoff,
}


def _european_intrinsic(option_type: OptionType, ns: Any, terminal: Any, strike: float) -> Any:
    intrinsic = terminal - strike if option_type is OptionType.CALL else strike - terminal
    return ns.clip(intrinsic, 0.0, None)


def _barrier_option_payoff(instrument: BarrierOption, scenario: _ScenarioInput) -> Any:
    spot = scenario._state_path("spot")
    ns = get_namespace(spot)
    # Monitoring is discrete and inclusive: an observation *at* the barrier is a
    # touch, and both ends of the grid are observations. Coarsening the grid
    # therefore prices a differently monitored contract rather than
    # approximating a continuously monitored one, and no bridge correction is
    # applied to pretend otherwise.
    barrier = instrument.barrier
    if instrument.barrier_type.is_up:
        touched = ns.amax(spot, axis=1) >= barrier
    else:
        touched = ns.amin(spot, axis=1) <= barrier
    alive = touched if instrument.barrier_type.knocks_in else ns.logical_not(touched)
    terminal = spot[:, -1]
    intrinsic = _european_intrinsic(instrument.option_type, ns, terminal, instrument.strike)
    return instrument.notional * intrinsic * _indicator(ns, alive, terminal)


def _lookback_option_payoff(instrument: LookbackOption, scenario: _ScenarioInput) -> Any:
    spot = scenario._state_path("spot")
    ns = get_namespace(spot)
    # Both ends included: a running extreme is a property of the whole path
    # rather than a set of agreed fixings, so dropping either end would discard
    # an observation the contract monitors.
    highest = ns.amax(spot, axis=1)
    lowest = ns.amin(spot, axis=1)
    terminal = spot[:, -1]
    if instrument.strike_convention is StrikeConvention.FIXED:
        strike = instrument.strike
        if instrument.option_type is OptionType.CALL:
            intrinsic = ns.clip(highest - strike, 0.0, None)
        else:
            intrinsic = ns.clip(strike - lowest, 0.0, None)
    elif instrument.option_type is OptionType.CALL:
        # Non-negative by construction: the terminal value is one of the
        # observations the minimum was taken over.
        intrinsic = terminal - lowest
    else:
        intrinsic = highest - terminal
    return instrument.notional * intrinsic


def _variance_swap_payoff(instrument: VarianceSwap, scenario: _ScenarioInput) -> Any:
    spot = scenario._state_path("spot")
    ns = get_namespace(spot)
    _require_positive_path(ns, spot, operation="Realized variance")
    log_returns = ns.log(spot[:, 1:] / spot[:, :-1])
    horizon = scenario._path_time_grid()[-1]
    # Sum of squared log returns over the year fraction. No sample-mean
    # subtraction -- the contract pays on the sum, not on a statistical
    # variance estimate -- and no factor of 252: dividing by T annualizes
    # already, and for daily observations 1/T is exactly the familiar 252/n.
    realized = ns.sum(log_returns * log_returns, axis=1) / horizon
    return instrument.notional * (realized - instrument.strike_variance)


#: Contracts whose payoff needs the whole trajectory rather than its last point.
_PATH_PAYOFFS: dict[type[Instrument], Callable[[Any, Any], Any]] = {
    AsianOption: _asian_option_payoff,
    BarrierOption: _barrier_option_payoff,
    LookbackOption: _lookback_option_payoff,
    VarianceSwap: _variance_swap_payoff,
}


@overload
def payoff(
    instrument: Forward | Future | EuropeanOption | BinaryOption, terminal_state: _ArrayT
) -> _ArrayT: ...


@overload
def payoff(
    instrument: AsianOption | BarrierOption | LookbackOption | VarianceSwap,
    terminal_state: _ScenarioInput,
) -> Any: ...


@overload
def payoff(instrument: Instrument, terminal_state: Any) -> Any: ...


def payoff(instrument: Instrument, terminal_state: Any) -> Any:
    """The undiscounted cashflow of ``instrument``, given the state it needs.

    Parameters
    ----------
    instrument : Instrument
        A registered contract with a terminal or path payoff evaluator.
    terminal_state : array-like or Scenario
        For a terminal contract, the underlier's value at maturity: a scalar,
        NumPy array, torch tensor, or jax array. For a path-dependent contract,
        a :class:`~fast_vollib.simulation.Scenario`; a bare path matrix is
        refused because it carries no underlier or horizon. The result stays in
        the input state's namespace, dtype, and device with its tape preserved.

    Returns
    -------
    array-like
        Cashflow, scaled by the contract's ``notional``.  Undiscounted: this is
        the value at maturity, not at valuation.

    Raises
    ------
    UnsupportedInstrumentError
        If the type has no payoff. The message distinguishes an unregistered
        type from a registered type without an evaluator.

    Notes
    -----
    A :class:`~fast_vollib.instruments.Future` evaluates to the same terminal
    cashflow as the equivalent :class:`~fast_vollib.instruments.Forward` here.
    Daily variation margin is not modelled; the types remain distinct because
    their economic cashflow timing differs.

    Examples
    --------
    >>> import numpy as np
    >>> from fast_vollib.instruments import Forward, payoff
    >>> payoff(Forward(underlier="CL", delivery_price=75.0, maturity=0.25), 80.0)
    5.0

    The tape survives, which is what makes a payoff usable inside a loss:

    >>> torch = __import__("importlib").import_module("torch")  # doctest: +SKIP
    >>> spot = torch.tensor([110.0], requires_grad=True)        # doctest: +SKIP
    >>> payoff(call, spot).sum().backward()                     # doctest: +SKIP
    """
    path_evaluator = _PATH_PAYOFFS.get(type(instrument))
    if path_evaluator is not None:
        if not isinstance(terminal_state, _ScenarioInput):
            raise InstrumentValidationError(_needs_scenario_message(instrument, terminal_state))
        # Scenario-owned: the underlier and the horizon are checked before any
        # arithmetic, so a trajectory of the wrong asset or the wrong length
        # cannot produce a number that looks like a price.
        terminal_state._validate_for_contract(instrument)
        return path_evaluator(instrument, terminal_state)

    evaluator = _PAYOFFS.get(type(instrument))
    if evaluator is None:
        raise UnsupportedInstrumentError(_no_payoff_message(instrument))
    if isinstance(terminal_state, _ScenarioInput):
        _reject_scenario_for_terminal_payoff(instrument)
    return evaluator(instrument, terminal_state)


def payoff_requirement(instrument: Instrument | type[Instrument]) -> PayoffRequirement | None:
    """What state a payoff evaluator needs for this instrument type.

    Returns ``None`` for types with no payoff at all.  An engine should consult
    this before simulating, so that an incompatible contract is rejected before
    the expensive part rather than after it.

    Examples
    --------
    >>> from fast_vollib.instruments import Asset, EuropeanOption, payoff_requirement
    >>> payoff_requirement(EuropeanOption).value
    'terminal'
    >>> payoff_requirement(Asset) is None
    True
    """
    cls = instrument if isinstance(instrument, type) else type(instrument)
    for info in instrument_types().values():
        if info.python_type is cls:
            return info.payoff_requirement
    raise UnsupportedInstrumentError(
        f"{cls.__name__} is not an instrument type known to fast_vollib.instruments."
    )


def _needs_scenario_message(instrument: Instrument, given: Any) -> str:
    return (
        f"{type(instrument).__name__} has a path-dependent payoff and needs a simulated "
        f"scenario, not a {type(given).__name__}. A bare array carries no underlier, no "
        f"observation times, and no way to tell whether its horizon is this contract's "
        f"maturity, so evaluating one would produce a number for an unknown instrument. "
        f"Build a scenario with fast_vollib.simulation.simulate or Scenario.from_states."
    )


def _no_payoff_message(instrument: Instrument) -> str:
    cls = type(instrument)
    if isinstance(instrument, FixedIncomeSecurity):
        return (
            f"{cls.__name__} has dated cashflows rather than a payoff: its payments "
            f"happen at several times, and payoff() maps the state at one horizon to "
            f"one amount. Read the schedule with cashflows(instrument), and value it "
            f"with fast_vollib.pricing.present_value(instrument, discount_curve=...)."
        )
    recognized = any(info.python_type is cls for info in instrument_types().values())
    if recognized:
        return (
            f"{cls.__name__} is a recognized instrument type but has no payoff: it "
            f"describes an underlier rather than a contract with a cashflow."
        )
    terminal = ", ".join(sorted(t.__name__ for t in _PAYOFFS))
    path = ", ".join(sorted(t.__name__ for t in _PATH_PAYOFFS))
    return (
        f"{cls.__name__} is not an instrument type known to fast_vollib.instruments. "
        f"Types with a terminal payoff: {terminal}. Types with a path payoff: {path}."
    )
