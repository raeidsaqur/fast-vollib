"""Adapters from contracts to the existing kernels.

These functions add no mathematics.  Every number they return comes from
``fast_black``, ``fast_black_scholes``, ``fast_black_scholes_merton``,
``get_all_greeks``, ``fast_implied_volatility``, or the Jäckel solvers -- the
same public entry points a caller would use directly.  What the adapters do is
read contract terms out of an instrument or a batch, hand them to the right
kernel, and scale by notional.

That is a deliberate constraint, not modesty.  A second implementation of the
Black formula living behind an object API would drift from the first one, and
the drift would show up as a small pricing difference that is hard to diagnose.
Tests assert exact array equality against direct kernel calls, which is only
achievable because the adapters delegate.

Two rules govern dispatch, both aimed at the same failure mode.

*Nothing is inferred.*  ``model`` is always an explicit argument.  An equity
underlier does not imply Black-Scholes over Black-Scholes-Merton -- that
depends on whether a dividend yield is being modelled, which is a modelling
decision, not a property of the asset.  So the asset class may validate a
choice; it never makes one.  The same holds for the IV solver and the backend.

*Nothing falls back.*  A request that cannot be served exactly raises.  There
is no case in which asking for machine-precision gradients yields a host-staged
approximation, or asking for a barrier price yields a European one.

Differentiability
-----------------
The price and Greeks adapters are **not** differentiable, and this module does
not pretend otherwise.  The fused kernels stage data host-to-device by design;
``return_native=True`` gives back a torch or jax array, but the tape is gone.
The one gradient-carrying route here is
:func:`implied_volatility_instrument` with ``solver="jackel"`` on native input
with ``return_native=True``, which reaches the implicit-function-theorem
wrappers in :mod:`fast_vollib.jackel`.  Requesting host or formatted output
terminates the gradient path as part of the documented API contract.
:func:`fast_vollib.instruments.capabilities` records exactly which
``(model, solver, backend)`` triples carry gradients.

Examples
--------
>>> import numpy as np
>>> from fast_vollib.instruments import (
...     EuropeanOption, VanillaMarketInputs, price_instrument,
... )
>>> option = EuropeanOption(underlier="ACME", option_type="call", strike=100.0, maturity=1.0)
>>> market = VanillaMarketInputs(underlying=100.0, rate=0.02, volatility=0.2)
>>> price = price_instrument(option, market, model="black_scholes")
>>> float(np.round(price[0], 6))
8.916037
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np

from .. import backends
from .._array_api import get_namespace
from ..api import get_all_greeks
from ..config import get_backend
from ..implied_volatility import fast_implied_volatility, fast_implied_volatility_black
from ..models import fast_black, fast_black_scholes, fast_black_scholes_merton
from ..utils.broadcast import maybe_format_data_and_broadcast, preprocess_flags
from ..utils.formatting import format_greeks_output, format_named_output
from ..utils.validation import validate_data
from .base import Instrument
from .batch import EuropeanOptionBatch
from .capabilities import capabilities
from .enums import IVSolver, PricingModel
from .errors import (
    UnsupportedInstrumentError,
    UnsupportedModelError,
    UnsupportedSolverError,
)
from .fixed_income import FixedIncomeSecurity
from .options import EuropeanOption
from .registry import instrument_types

if TYPE_CHECKING:
    from ..types import (
        BackendLiteral,
        IVSolverLiteral,
        ModelLiteral,
        OnErrorLiteral,
        ReturnAsLiteral,
    )
    from .market import VanillaMarketInputs

__all__ = [
    "greeks_instrument",
    "implied_volatility_instrument",
    "price_instrument",
]

#: Backends the Jäckel solver has a full-model wrapper for.
_JACKEL_BACKENDS = ("numpy", "torch", "jax")


class _Terms(NamedTuple):
    """The contract terms a kernel needs, in the kernel's own conventions."""

    flag: Any
    strike: Any
    maturity: Any
    notional: Any


def _resolve_model(model: PricingModel | ModelLiteral) -> PricingModel:
    if isinstance(model, PricingModel):
        return model
    if isinstance(model, str):
        try:
            return PricingModel(model)
        except ValueError:
            pass
    valid = ", ".join(repr(member.value) for member in PricingModel)
    raise UnsupportedModelError(
        f"Unknown pricing model {model!r}. fast-vollib implements {valid}. "
        f"The model is always explicit; it is never inferred from the instrument."
    )


def _resolve_solver(solver: IVSolver | IVSolverLiteral) -> IVSolver:
    if isinstance(solver, IVSolver):
        return solver
    if isinstance(solver, str):
        try:
            return IVSolver(solver)
        except ValueError:
            pass
    valid = ", ".join(repr(member.value) for member in IVSolver)
    raise UnsupportedSolverError(
        f"Unknown implied-volatility solver {solver!r}. Available solvers: {valid}."
    )


def _terms_of(instrument_or_batch: object) -> _Terms:
    if isinstance(instrument_or_batch, EuropeanOption):
        option = instrument_or_batch
        return _Terms(option.flag, option.strike, option.maturity, option.notional)
    if isinstance(instrument_or_batch, EuropeanOptionBatch):
        batch = instrument_or_batch
        return _Terms(batch.flag, batch.strike, batch.maturity, batch.notional)
    raise UnsupportedInstrumentError(_no_adapter_message(instrument_or_batch))


def _no_adapter_message(instrument_or_batch: object) -> str:
    cls = type(instrument_or_batch)
    if isinstance(instrument_or_batch, FixedIncomeSecurity):
        return (
            f"{cls.__name__} is a fixed-income security and is not valued by an "
            f"option-pricing model: it has a schedule of dated payments and a discount "
            f"curve, not a volatility. Use "
            f"fast_vollib.pricing.present_value(instrument, discount_curve=...)."
        )
    recognized = any(info.python_type is cls for info in instrument_types().values())
    if recognized:
        return (
            f"{cls.__name__} is a recognized instrument type, but fast-vollib implements "
            f"no analytic pricing kernel for it. The analytic adapters serve "
            f"EuropeanOption and EuropeanOptionBatch; use payoff() for a terminal "
            f"cashflow, and check capabilities({cls.__name__}) for what is available."
        )
    return (
        f"{cls.__name__} is not an instrument type known to fast_vollib.instruments. "
        f"The analytic adapters accept EuropeanOption and EuropeanOptionBatch."
    )


def _capability_type(instrument_or_batch: object) -> type[Instrument]:
    """The contract class whose capabilities govern this request.

    A batch is a container of European options, so it inherits their
    capabilities rather than having a separate entry.
    """
    if isinstance(instrument_or_batch, EuropeanOptionBatch):
        return EuropeanOption
    if isinstance(instrument_or_batch, Instrument):
        return type(instrument_or_batch)
    raise UnsupportedInstrumentError(_no_adapter_message(instrument_or_batch))


def _check_model_capability(
    instrument_or_batch: object, model: PricingModel, *, operation: str
) -> None:
    caps = capabilities(_capability_type(instrument_or_batch))
    available = caps.price if operation == "price" else caps.greeks
    if model not in available:
        supported = ", ".join(sorted(repr(m.value) for m in available)) or "no model"
        raise UnsupportedModelError(
            f"{_capability_type(instrument_or_batch).__name__} cannot be evaluated with "
            f"model {model.value!r} for {operation}. Supported: {supported}."
        )


def _notional_is_unit(notional: Any) -> bool:
    return bool(np.all(np.asarray(notional, dtype=np.float64) == 1.0))


def _scale(values: np.ndarray, notional: Any) -> np.ndarray:
    if _notional_is_unit(notional):
        return values
    return values * np.asarray(notional, dtype=np.float64)


def _finalize(
    values: np.ndarray,
    *,
    name: str,
    return_as: ReturnAsLiteral,
    backend_name: str,
    return_native: bool,
) -> Any:
    """Format exactly as the wrapped kernels do, so parity is bit-for-bit."""
    if return_native and backend_name in {"torch", "jax"}:
        return backends.get_module(backend_name).to_native(values)
    if return_as == "numpy":
        return values
    return format_named_output(values, name, return_as)


def price_instrument(
    instrument_or_batch: Instrument | EuropeanOptionBatch,
    market: VanillaMarketInputs,
    *,
    model: PricingModel | ModelLiteral,
    backend: BackendLiteral = "auto",
    return_native: bool = False,
    return_as: ReturnAsLiteral = "numpy",
) -> Any:
    """Price a contract or a batch against market inputs under an explicit model.

    Parameters
    ----------
    instrument_or_batch : EuropeanOption or EuropeanOptionBatch
        A batch is one vectorized kernel call, never a loop.
    market : VanillaMarketInputs
        Must carry ``underlying``, ``rate``, and ``volatility``; and
        ``dividend_yield`` for Black-Scholes-Merton.
    model : PricingModel or {"black", "black_scholes", "black_scholes_merton"}
        Required and explicit.  Under ``"black"``, ``market.underlying`` is
        read as a forward; under the other two, as a spot.
    backend : {"auto", "numpy", "torch", "jax", "numba"}, default "auto"
    return_native : bool, default False
        Return a torch or jax array instead of NumPy.  A native container is
        **not** a gradient: this path stages through host memory.
    return_as : {"numpy", "dataframe", "series", "dict", "json"}, default "numpy"
        Defaults to ``"numpy"`` rather than the functional API's
        ``"dataframe"``, because an adapter's caller usually wants values.

    Returns
    -------
    Price, scaled by the contract's notional.

    Raises
    ------
    UnsupportedInstrumentError
        If the type has no analytic pricing kernel.
    UnsupportedModelError
        If the model is unknown, or unavailable for this type.
    MissingMarketInputError
        Naming the market field that was needed.

    Examples
    --------
    >>> import numpy as np
    >>> from fast_vollib.instruments import (
    ...     EuropeanOptionBatch, VanillaMarketInputs, price_instrument,
    ... )
    >>> batch = EuropeanOptionBatch.from_arrays(
    ...     option_type=["c", "p"], strike=[100.0, 100.0], maturity=1.0, underlier="ACME",
    ... )
    >>> market = VanillaMarketInputs(underlying=100.0, rate=0.0, volatility=0.2)
    >>> np.round(price_instrument(batch, market, model="black"), 6).tolist()
    [7.965567, 7.965567]
    """
    resolved = _resolve_model(model)
    terms = _terms_of(instrument_or_batch)
    _check_model_capability(instrument_or_batch, resolved, operation="price")

    underlying = market.require("underlying", operation="price")
    rate = market.require("rate", operation="price")
    volatility = market.require("volatility", operation="price")

    # Formatting is applied once, after the notional scaling, so every branch
    # asks the kernel for raw NumPy.
    if resolved is PricingModel.BLACK:
        values = fast_black(
            terms.flag,
            underlying,
            terms.strike,
            terms.maturity,
            rate,
            volatility,
            return_as="numpy",
            backend=backend,
            return_native=False,
        )
    elif resolved is PricingModel.BLACK_SCHOLES:
        values = fast_black_scholes(
            terms.flag,
            underlying,
            terms.strike,
            terms.maturity,
            rate,
            volatility,
            return_as="numpy",
            backend=backend,
            return_native=False,
        )
    else:
        dividend_yield = market.require(
            "dividend_yield", operation="price under black_scholes_merton"
        )
        values = fast_black_scholes_merton(
            terms.flag,
            underlying,
            terms.strike,
            terms.maturity,
            rate,
            volatility,
            dividend_yield,
            return_as="numpy",
            backend=backend,
            return_native=False,
        )

    return _finalize(
        _scale(values, terms.notional),
        name="Price",
        return_as=return_as,
        backend_name=get_backend(backend),
        return_native=return_native,
    )


def greeks_instrument(
    instrument_or_batch: Instrument | EuropeanOptionBatch,
    market: VanillaMarketInputs,
    *,
    model: PricingModel | ModelLiteral,
    backend: BackendLiteral = "auto",
    return_native: bool = False,
    return_as: ReturnAsLiteral = "numpy",
) -> Any:
    """Delta, gamma, theta, rho, and vega, in one kernel pass.

    Parameters are as :func:`price_instrument`.  Every Greek is scaled by the
    contract's notional.

    Returns
    -------
    dict or pandas.DataFrame
        Keyed by Greek name; the container follows ``return_as``.

    Notes
    -----
    Not differentiable.  These are analytic Greeks from a fused kernel, not
    derivatives of a traced graph.

    Examples
    --------
    >>> import numpy as np
    >>> from fast_vollib.instruments import (
    ...     EuropeanOption, VanillaMarketInputs, greeks_instrument,
    ... )
    >>> option = EuropeanOption(underlier="ACME", option_type="call", strike=100.0, maturity=1.0)
    >>> market = VanillaMarketInputs(underlying=100.0, rate=0.0, volatility=0.2)
    >>> greeks = greeks_instrument(option, market, model="black_scholes")
    >>> float(np.round(greeks["delta"][0], 6))
    0.539828
    """
    resolved = _resolve_model(model)
    terms = _terms_of(instrument_or_batch)
    _check_model_capability(instrument_or_batch, resolved, operation="greeks")

    underlying = market.require("underlying", operation="compute Greeks")
    rate = market.require("rate", operation="compute Greeks")
    volatility = market.require("volatility", operation="compute Greeks")
    # Only Black-Scholes-Merton consumes a dividend yield. The other two models
    # get None rather than whatever the market object happens to carry, so that
    # a market snapshot with a yield in it cannot change a Black-76 or
    # Black-Scholes answer through a back door.
    dividend_yield = (
        market.require("dividend_yield", operation="compute Greeks under black_scholes_merton")
        if resolved is PricingModel.BLACK_SCHOLES_MERTON
        else None
    )

    data = get_all_greeks(
        terms.flag,
        underlying,
        terms.strike,
        terms.maturity,
        rate,
        volatility,
        dividend_yield,
        model=resolved.value,
        return_as="numpy",
        backend=backend,
        return_native=False,
    )
    scaled = {name: _scale(value, terms.notional) for name, value in data.items()}

    backend_name = get_backend(backend)
    if return_native and backend_name in {"torch", "jax"}:
        module = backends.get_module(backend_name)
        return {name: module.to_native(value) for name, value in scaled.items()}
    return format_greeks_output(scaled, return_as)


def _jackel_backend_module(backend_name: str, *, requested: BackendLiteral) -> Any:
    """The Jäckel full-model wrapper for a backend, or a typed refusal.

    Never substitutes: a solver that cannot serve the requested backend is an
    error, because silently answering with a different one is exactly the
    behaviour the capability system exists to prevent.
    """
    if backend_name in _JACKEL_BACKENDS:
        import importlib

        return importlib.import_module(f"..jackel.{backend_name}_backend", __spec__.parent)

    available = ", ".join(_JACKEL_BACKENDS)
    detail = (
        f"Request solver='halley' for the {backend_name!r} backend, or a backend the "
        f"Jäckel solver supports; the solver is never substituted silently."
        if requested != "auto"
        else (
            f"backend='auto' resolved to {backend_name!r} here. Pass an explicit "
            f"backend the Jäckel solver supports, or solver='halley'."
        )
    )
    raise UnsupportedSolverError(
        f"The Jäckel solver has no {backend_name!r} implementation "
        f"(available: {available}). {detail}"
    )


def _native_jackel_route(namespace_name: str) -> Any:
    """The differentiable Jäckel wrapper for a native namespace."""
    from .. import jackel

    if namespace_name == "torch":
        wrapper = jackel.implied_volatility_autograd
    else:
        wrapper = jackel.implied_volatility_autograd_jax
    if wrapper is None:
        raise UnsupportedSolverError(
            f"A differentiable Jäckel implied-volatility route was requested for "
            f"{namespace_name!r} input, but fast_vollib.jackel could not load its "
            f"{namespace_name} wrapper -- the optional dependency is not installed. "
            f"No substitution is made: a host-staged result would return a "
            f"{namespace_name} array with no gradients attached, which is worse than "
            f"an error. Install the '{namespace_name}' extra, or pass "
            f"return_native=False to accept a host result."
        )
    return wrapper


def implied_volatility_instrument(
    instrument_or_batch: Instrument | EuropeanOptionBatch,
    market: VanillaMarketInputs,
    *,
    model: PricingModel | ModelLiteral,
    solver: IVSolver | IVSolverLiteral = IVSolver.JACKEL,
    backend: BackendLiteral = "auto",
    return_native: bool = False,
    return_as: ReturnAsLiteral = "numpy",
    on_error: OnErrorLiteral = "warn",
) -> Any:
    """Invert observed prices to implied volatility.

    Parameters
    ----------
    instrument_or_batch : EuropeanOption or EuropeanOptionBatch
    market : VanillaMarketInputs
        Must carry ``underlying``, ``rate``, and ``price``; and
        ``dividend_yield`` for Black-Scholes-Merton.  ``market.price`` is the
        price of the whole position and is divided by the contract's notional
        before inversion.
    model : PricingModel or ModelLiteral
        Required and explicit.
    solver : IVSolver or {"jackel", "halley"}, default ``"jackel"``
        ``"jackel"`` is the machine-precision "Let's Be Rational" solver and
        the only route with gradient support.  ``"halley"`` is the library's
        Halley-with-bisection solver.  The solver is never substituted.
    backend : {"auto", "numpy", "torch", "jax", "numba"}, default "auto"
        The Jäckel solver implements ``numpy``, ``torch``, and ``jax``;
        requesting ``numba`` with it raises rather than falling back.
    return_native : bool, default False
        With ``solver="jackel"`` and torch or jax market inputs, this selects
        the differentiable route and the result carries gradients.  With
        NumPy inputs, or with ``solver="halley"``, it only changes the
        container -- a native array is not evidence of a gradient.
    return_as : {"numpy", "dataframe", "series"}, default "numpy"
    on_error : {"warn", "raise", "ignore"}, default "warn"
        How to report prices that cannot be inverted.  Not consulted on the
        differentiable route: those wrappers signal an invalid domain with
        ``NaN`` in both the forward and the backward pass, which is their
        documented contract and is preserved rather than reshaped.

    Returns
    -------
    Implied volatility, one value per contract.

    Raises
    ------
    UnsupportedSolverError
        If the solver cannot serve the requested backend, or a differentiable
        route was requested and its optional dependency is missing.
    UnsupportedInstrumentError, UnsupportedModelError, MissingMarketInputError

    Notes
    -----
    The raw ``jackel_iv_black`` solver consumes an *undiscounted* Black-76
    price and a forward.  Market prices are discounted, so this adapter always
    goes through the full-model wrappers, which perform the discount and
    forward conversion for the model actually requested.  Feeding a discounted
    Black-Scholes price to the raw solver would produce a plausible, wrong
    volatility; that path does not exist here.

    Gradients, when the differentiable route is taken, are with respect to
    price, spot or forward, strike, maturity, rate, and -- under
    Black-Scholes-Merton -- dividend yield, via the implicit function theorem
    applied to the pricing equation.  The wrappers' invalid-domain and low-vega
    behaviour is preserved rather than normalized to another solver's policy.
    Requesting host or formatted output ends the gradient path.

    Examples
    --------
    >>> import numpy as np
    >>> from fast_vollib.instruments import (
    ...     EuropeanOption, VanillaMarketInputs, implied_volatility_instrument,
    ...     price_instrument,
    ... )
    >>> option = EuropeanOption(underlier="ACME", option_type="call", strike=100.0, maturity=1.0)
    >>> quoted = VanillaMarketInputs(underlying=100.0, rate=0.02, volatility=0.25)
    >>> price = price_instrument(option, quoted, model="black_scholes")
    >>> observed = VanillaMarketInputs(underlying=100.0, rate=0.02, price=price)
    >>> iv = implied_volatility_instrument(option, observed, model="black_scholes")
    >>> float(np.round(iv[0], 8))
    0.25
    """
    resolved = _resolve_model(model)
    resolved_solver = _resolve_solver(solver)
    terms = _terms_of(instrument_or_batch)

    cls = (
        EuropeanOption
        if isinstance(instrument_or_batch, EuropeanOptionBatch)
        else type(instrument_or_batch)
    )
    solvers = capabilities(cls).solvers_for(resolved)
    if resolved_solver not in solvers:
        available = ", ".join(sorted(repr(s.value) for s in solvers)) or "no solver"
        raise UnsupportedSolverError(
            f"{cls.__name__} cannot be inverted with solver {resolved_solver.value!r} "
            f"under model {resolved.value!r}. Available: {available}."
        )

    underlying = market.require("underlying", operation="invert implied volatility")
    rate = market.require("rate", operation="invert implied volatility")
    observed = market.require("price", operation="invert implied volatility")
    # As in the pricing adapter: only Black-Scholes-Merton reads a dividend
    # yield. Passing one through for the other models would make the two
    # solvers disagree -- the Jäckel wrappers use q in their below-intrinsic
    # check even where the forward ignores it.
    dividend_yield = (
        market.require(
            "dividend_yield",
            operation="invert implied volatility under black_scholes_merton",
        )
        if resolved is PricingModel.BLACK_SCHOLES_MERTON
        else None
    )

    # An observed price is the price of the position; the kernels invert a unit
    # contract. Divide first, in the caller's namespace so a native route keeps
    # its tape.
    if not _notional_is_unit(terms.notional):
        namespace = get_namespace(observed)
        observed = observed / namespace.asarray(
            np.asarray(terms.notional, dtype=np.float64), like=observed
        )

    if resolved_solver is IVSolver.HALLEY:
        return _halley_implied_volatility(
            terms,
            observed=observed,
            underlying=underlying,
            rate=rate,
            dividend_yield=dividend_yield,
            model=resolved,
            backend=backend,
            return_native=return_native,
            return_as=return_as,
            on_error=on_error,
        )
    return _jackel_implied_volatility(
        terms,
        observed=observed,
        underlying=underlying,
        rate=rate,
        dividend_yield=dividend_yield,
        model=resolved,
        backend=backend,
        return_native=return_native,
        return_as=return_as,
        on_error=on_error,
    )


def _halley_implied_volatility(
    terms: _Terms,
    *,
    observed: Any,
    underlying: Any,
    rate: Any,
    dividend_yield: Any,
    model: PricingModel,
    backend: BackendLiteral,
    return_native: bool,
    return_as: ReturnAsLiteral,
    on_error: OnErrorLiteral,
) -> Any:
    if model is PricingModel.BLACK:
        # Note the argument order of this entry point: (price, F, K, r, t, flag).
        return fast_implied_volatility_black(
            observed,
            underlying,
            terms.strike,
            rate,
            terms.maturity,
            terms.flag,
            on_error=on_error,
            return_as=return_as,
            backend=backend,
            return_native=return_native,
        )
    return fast_implied_volatility(
        observed,
        underlying,
        terms.strike,
        terms.maturity,
        rate,
        terms.flag,
        q=dividend_yield,
        model=model.value,
        on_error=on_error,
        return_as=return_as,
        backend=backend,
        return_native=return_native,
    )


def _jackel_implied_volatility(
    terms: _Terms,
    *,
    observed: Any,
    underlying: Any,
    rate: Any,
    dividend_yield: Any,
    model: PricingModel,
    backend: BackendLiteral,
    return_native: bool,
    return_as: ReturnAsLiteral,
    on_error: OnErrorLiteral,
) -> Any:
    namespace = get_namespace(observed, underlying, rate)

    if return_native and namespace.name in {"torch", "jax"}:
        if backend not in {"auto", namespace.name}:
            raise UnsupportedSolverError(
                f"A differentiable Jäckel route was requested for {namespace.name!r} "
                f"input, but backend={backend!r} was also requested. Those cannot both "
                f"be honoured, and neither is silently dropped: pass backend='auto' or "
                f"backend={namespace.name!r}, or set return_native=False."
            )
        wrapper = _native_jackel_route(namespace.name)
        return wrapper(
            observed,
            underlying,
            terms.strike,
            terms.maturity,
            rate,
            terms.flag,
            dividend_yield,
            model=model.value,
        )

    backend_name = get_backend(backend)
    module = _jackel_backend_module(backend_name, requested=backend)

    # Mirror fast_implied_volatility's preprocessing exactly. The Jäckel
    # wrappers take raw arrays, so the broadcasting and validation contract has
    # to be reproduced here or the two solvers would accept different inputs.
    flag = preprocess_flags(terms.flag)
    if model is PricingModel.BLACK_SCHOLES_MERTON:
        price, spot, strike, maturity, rates, yields, flag = maybe_format_data_and_broadcast(
            observed, underlying, terms.strike, terms.maturity, rate, dividend_yield, flag
        )
        validate_data(price, spot, strike, maturity, rates, yields)
    else:
        price, spot, strike, maturity, rates, flag = maybe_format_data_and_broadcast(
            observed, underlying, terms.strike, terms.maturity, rate, flag
        )
        validate_data(price, spot, strike, maturity, rates)
        yields = None

    values = module.implied_volatility(
        model.value, price, spot, strike, maturity, rates, flag, q=yields, on_error=on_error
    )
    return _finalize(
        values,
        name="IV",
        return_as=return_as,
        backend_name=backend_name,
        return_native=return_native,
    )
