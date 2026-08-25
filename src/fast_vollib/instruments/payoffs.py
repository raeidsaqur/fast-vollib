"""Terminal payoffs, evaluated in the caller's own array namespace.

A payoff is a pure map from terminal underlying state to a cashflow.  It does
not discount, does not simulate, and does not consult market inputs -- those
are separate concerns, and folding any of them in here would make the function
untestable against a hand-computed value.

The implementation matters as much as the formula.  Evaluation runs through
:mod:`fast_vollib._array_api` in whatever namespace the input arrives in, so a
torch tensor stays a torch tensor on its own device with its autograd tape
intact, and a jax array stays a jax array that ``jit`` and ``grad`` can trace.
Nothing here calls ``.numpy()``, ``.detach()``, ``.cpu()``, or otherwise stages
through host memory: that is the difference between a payoff you can put in a
loss function and one you can only print.

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

from typing import Any, Callable, TypeVar

from .._array_api import get_namespace
from .base import Instrument
from .enums import OptionType, PayoffRequirement
from .errors import UnsupportedInstrumentError
from .forwards import Forward, Future
from .options import EuropeanOption
from .registry import instrument_types

__all__ = ["payoff", "payoff_requirement"]

_ArrayT = TypeVar("_ArrayT")


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


_PAYOFFS: dict[type[Instrument], Callable[[Any, Any], Any]] = {
    Forward: _forward_payoff,
    Future: _future_payoff,
    EuropeanOption: _european_option_payoff,
}


def payoff(instrument: Instrument, terminal_state: _ArrayT) -> _ArrayT:
    """The cashflow of ``instrument`` at maturity, given the terminal state.

    Parameters
    ----------
    instrument : Instrument
        A contract whose :class:`~fast_vollib.instruments.PayoffRequirement` is
        ``TERMINAL``.
    terminal_state : array-like
        The underlier's value at maturity.  A scalar, a NumPy array, a torch
        tensor, or a jax array; the result comes back in the same namespace,
        with the same dtype and device, and with the autograd tape preserved.
        Ordinary broadcasting applies, so one contract evaluates against a
        whole vector of terminal states.

    Returns
    -------
    array-like
        Cashflow, scaled by the contract's ``notional``.  Undiscounted: this is
        the value at maturity, not at valuation.

    Raises
    ------
    UnsupportedInstrumentError
        If the type has no terminal payoff. The message distinguishes an
        unregistered type from a registered type without a payoff evaluator.

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
    evaluator = _PAYOFFS.get(type(instrument))
    if evaluator is None:
        raise UnsupportedInstrumentError(_no_payoff_message(instrument))
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


def _no_payoff_message(instrument: Instrument) -> str:
    cls = type(instrument)
    recognized = any(info.python_type is cls for info in instrument_types().values())
    if recognized:
        return (
            f"{cls.__name__} is a recognized instrument type but has no payoff: it "
            f"describes an underlier rather than a contract with a cashflow."
        )
    known = ", ".join(sorted(t.__name__ for t in _PAYOFFS))
    return (
        f"{cls.__name__} is not an instrument type known to fast_vollib.instruments. "
        f"Types with a terminal payoff: {known}."
    )
