"""What can actually be done with each instrument type.

A pricing library's most dangerous answer is a plausible number computed by a
route the caller did not ask for.  This module is the machine-readable record
that prevents it: for each instrument type it states exactly which pricing
models have kernels, which implied-volatility solvers can invert which models,
and -- separately, because it is not the same question -- which
``(model, solver, backend)`` combinations return a result with gradients
attached.

Two distinctions are load-bearing.

*Implied volatility is not a boolean.*  The two solvers have different
accuracy, different edge-case behaviour, and only one of them has a
differentiable route.  A caller dispatching on "does IV work here" would be
asking a question whose answer cannot be acted on, so the capability is a
mapping from model to the set of solvers that serve it.

*A native tensor is not a gradient.*  ``return_native=True`` returns a torch or
jax array from any route, including ones that staged through host memory and
dropped the tape on the way.  ``native_autodiff`` therefore records the exact
combinations where differentiating the output is meaningful, and nothing else
in this module implies it.

Capability sets are computed against what is installed: a differentiable route
whose optional backend is absent is not advertised, and requesting it raises
:class:`~fast_vollib.instruments.UnsupportedSolverError` rather than quietly
returning a host-staged answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib.util
from types import MappingProxyType
from typing import TYPE_CHECKING, Callable, Mapping

from .base import Asset, Instrument
from .enums import IVSolver, PricingModel
from .errors import UnsupportedInstrumentError
from .exotics import (
    AsianOption,
    BarrierOption,
    BinaryOption,
    LookbackOption,
    VarianceSwap,
)
from .forwards import Forward, Future
from .options import EuropeanOption

if TYPE_CHECKING:
    from ..types import BackendLiteral

__all__ = ["CapabilitySet", "capabilities"]

#: Every model the analytic kernels implement.
_ALL_MODELS: frozenset[PricingModel] = frozenset(PricingModel)

#: Both solvers invert all three model conventions.
_ALL_SOLVERS: frozenset[IVSolver] = frozenset(IVSolver)

#: Backends whose native arrays the Jäckel solver can differentiate through.
_AUTODIFF_BACKENDS: tuple[tuple[str, "BackendLiteral"], ...] = (("torch", "torch"), ("jax", "jax"))


@dataclass(frozen=True, slots=True)
class CapabilitySet:
    """The operations available for one instrument type.

    Attributes
    ----------
    payoff : bool
        Whether :func:`~fast_vollib.instruments.payoff` can evaluate a cashflow
        for this type, from either terminal state or a Scenario as declared by
        its payoff requirement.
    price : frozenset[PricingModel]
        Models with an analytic pricing kernel.  Empty means
        :func:`~fast_vollib.instruments.price_instrument` raises.
    greeks : frozenset[PricingModel]
        Models with an analytic Greeks kernel.
    implied_volatility : Mapping[PricingModel, frozenset[IVSolver]]
        Which solvers can invert which models.  A model absent from the mapping
        has no inversion route at all.
    native_autodiff : frozenset[tuple[PricingModel, IVSolver, str]]
        ``(model, solver, backend)`` triples for which an
        implied-volatility call on native input with ``return_native=True``
        returns a value carrying gradients.  Host-formatted output is never in
        this set: formatting terminates the tape.
    simulate : bool
        Whether :class:`fast_vollib.simulation.MonteCarloEngine` has a route
        for this type. Type-level: an *instance* is additionally eligible only
        with a strictly positive maturity, which ``MonteCarloEngine.supports``
        applies and is authoritative for an actual request.
    simulation_autodiff : frozenset[BackendLiteral]
        Backends on which a simulated valuation returns a result that still
        carries an autograd graph. Recorded against what is installed and
        demonstrated by the test suite.

        This is tape retention, and only that. A digital or barrier payoff
        keeps a graph whose pathwise derivative is zero almost everywhere and
        undefined at the boundary, so a gradient that exists is not
        automatically a Greek worth acting on.

    Notes
    -----
    Read as a value, not used as a dictionary key: the ``implied_volatility``
    mapping makes instances unhashable.
    """

    payoff: bool = False
    price: frozenset[PricingModel] = frozenset()
    greeks: frozenset[PricingModel] = frozenset()
    implied_volatility: Mapping[PricingModel, frozenset[IVSolver]] = field(
        default_factory=lambda: MappingProxyType({})
    )
    native_autodiff: frozenset[tuple[PricingModel, IVSolver, BackendLiteral]] = frozenset()
    simulate: bool = False
    simulation_autodiff: frozenset[BackendLiteral] = frozenset()

    def supports_price(self, model: PricingModel) -> bool:
        """Whether ``model`` has an analytic pricing kernel for this type."""
        return model in self.price

    def supports_greeks(self, model: PricingModel) -> bool:
        """Whether ``model`` has an analytic Greeks kernel for this type."""
        return model in self.greeks

    def solvers_for(self, model: PricingModel) -> frozenset[IVSolver]:
        """The solvers that can invert ``model``; empty if none can."""
        return self.implied_volatility.get(model, frozenset())

    def supports_native_autodiff(self, model: PricingModel, solver: IVSolver, backend: str) -> bool:
        """Whether that exact combination returns gradient-carrying output."""
        return (model, solver, backend) in self.native_autodiff

    def supports_simulation_autodiff(self, backend: str) -> bool:
        """Whether a simulated valuation on ``backend`` keeps its autograd graph."""
        return backend in self.simulation_autodiff


def _module_installed(name: str) -> bool:
    """Whether an optional backend is importable, without importing it.

    ``find_spec`` walks the finders and stops; it does not execute the module.
    That matters here: this package must not pull torch or jax into a process
    that only wanted to describe an option.
    """
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):  # pragma: no cover - malformed installation
        return False


def _native_autodiff_combinations() -> frozenset[tuple[PricingModel, IVSolver, BackendLiteral]]:
    combinations = set()
    for module_name, backend in _AUTODIFF_BACKENDS:
        if not _module_installed(module_name):
            continue
        for model in PricingModel:
            combinations.add((model, IVSolver.JACKEL, backend))
    return frozenset(combinations)


def _simulation_backends() -> frozenset[BackendLiteral]:
    """Installed backends whose arrays a simulated valuation keeps a graph on."""
    backends: set[BackendLiteral] = set()
    for module_name, backend in _AUTODIFF_BACKENDS:
        if _module_installed(module_name):
            backends.add(backend)
    return frozenset(backends)


def _european_option_capabilities() -> CapabilitySet:
    return CapabilitySet(
        payoff=True,
        price=_ALL_MODELS,
        greeks=_ALL_MODELS,
        implied_volatility=MappingProxyType({model: _ALL_SOLVERS for model in PricingModel}),
        native_autodiff=_native_autodiff_combinations(),
        simulate=True,
        simulation_autodiff=_simulation_backends(),
    )


def _forward_capabilities() -> CapabilitySet:
    """A payoff and a simulated valuation, but no analytic pricing adapter.

    Pricing a forward is discounting, not a Black formula, and this layer adds
    no math of its own -- so the honest capability is "payoff yes, price no",
    and :func:`~fast_vollib.instruments.price_instrument` says so by name
    instead of returning an option value.
    """
    return CapabilitySet(payoff=True, simulate=True, simulation_autodiff=_simulation_backends())


def _simulated_only_capabilities() -> CapabilitySet:
    """A payoff and a simulated valuation, with no analytic kernel in this library.

    Closed forms exist in the literature for several of these types. They are
    not implemented here, and the capability set says so rather than letting a
    caller assume that "there is a formula" means "this library evaluates it".
    """
    return CapabilitySet(payoff=True, simulate=True, simulation_autodiff=_simulation_backends())


def _future_capabilities() -> CapabilitySet:
    """A terminal payoff, and deliberately no valuation route at all.

    A future's terminal formula coincides with a forward's, so a simulated
    price would look right. Its economics are a stream of daily variation
    margin, which is not modelled, so the capability says no rather than
    letting the coincidence stand in for the contract.
    """
    return CapabilitySet(payoff=True)


def _no_capabilities() -> CapabilitySet:
    """An asset is a description of an underlier, not a contract to evaluate."""
    return CapabilitySet()


_CAPABILITY_BUILDERS: dict[type[Instrument], Callable[[], CapabilitySet]] = {
    EuropeanOption: _european_option_capabilities,
    BinaryOption: _simulated_only_capabilities,
    AsianOption: _simulated_only_capabilities,
    BarrierOption: _simulated_only_capabilities,
    LookbackOption: _simulated_only_capabilities,
    VarianceSwap: _simulated_only_capabilities,
    Forward: _forward_capabilities,
    Future: _future_capabilities,
    Asset: _no_capabilities,
}


def capabilities(instrument_type: type[Instrument] | Instrument) -> CapabilitySet:
    """The capability set for an instrument type or instance.

    Parameters
    ----------
    instrument_type : type or Instrument
        A registered instrument class, or an instance of one.

    Returns
    -------
    CapabilitySet

    Raises
    ------
    UnsupportedInstrumentError
        If the type is not part of the public registry.

    Examples
    --------
    >>> from fast_vollib.instruments import EuropeanOption, IVSolver, PricingModel, capabilities
    >>> caps = capabilities(EuropeanOption)
    >>> caps.supports_price(PricingModel.BLACK_SCHOLES)
    True
    >>> IVSolver.JACKEL in caps.solvers_for(PricingModel.BLACK)
    True

    A linear contract is recognized but has no analytic kernel here, and the
    capability set says so rather than an adapter guessing:

    >>> from fast_vollib.instruments import Forward
    >>> capabilities(Forward).payoff, sorted(capabilities(Forward).price)
    (True, [])
    """
    cls = instrument_type if isinstance(instrument_type, type) else type(instrument_type)
    builder = _CAPABILITY_BUILDERS.get(cls)
    if builder is None:
        raise UnsupportedInstrumentError(
            f"{cls.__name__} is not an instrument type known to fast_vollib.instruments. "
            f"Known types: {', '.join(sorted(t.__name__ for t in _CAPABILITY_BUILDERS))}."
        )
    return builder()
