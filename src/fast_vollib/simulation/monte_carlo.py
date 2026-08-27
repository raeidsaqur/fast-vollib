"""Explicit Monte Carlo valuation.

The engine is concrete and it is asked for by name.  There is no registry that
picks it, no analytic adapter that falls back to it when a closed form is
missing, and no path by which it substitutes for one when a closed form exists.
A caller who wants a simulated price says so; a caller who wants the Black
formula calls :func:`~fast_vollib.instruments.price_instrument`.  The two never
stand in for each other, because a number that came from a different method
than the one requested is indistinguishable from the one that was asked for.

What the engine supplies and what it refuses to supply
-----------------------------------------------------
It supplies exactly three things the process does not: the initial spot, the
discount factor, and the estimator.  It does **not** supply a measure.
``market.rate`` discounts and nothing else; it is never used to rewrite a
drift.  Simulating under a physical drift and discounting at the risk-free rate
produces a number, and the number is not a price -- so a risk-neutral valuation
requires a process the caller made risk-neutral, for which
:meth:`~fast_vollib.processes.GBM.risk_neutral` exists.  ``market.volatility``
is ignored outright: volatility is a process parameter here, and reading both
would let two disagreeing values silently pick one.

Everything is validated before a single path is drawn.  An unsupported type, a
zero maturity, a multi-state process, a grid that does not end at maturity, a
mixed array namespace, an unusable RNG: each raises with the sampling budget
untouched, because discovering the problem after a hundred thousand paths is
merely expensive, and discovering it *never* is worse.

Zero maturity and futures
-------------------------
A contract expiring now has a payoff but no path: a strictly increasing grid
with at least two points cannot both start and end at zero.  Rather than
simulate a fake step, the engine refuses and points at
:func:`~fast_vollib.instruments.payoff`, which answers the question exactly.

:class:`~fast_vollib.instruments.Future` is refused for a different reason.  Its
terminal payoff formula coincides with a forward's, so pricing it here would
return a plausible number -- but a future's economics are a stream of daily
variation margin, which this library does not model.  The types stay distinct
and so does the refusal.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, ClassVar

import numpy as np

from .._array_api import concrete_float, get_namespace
from .._random_api import (
    _is_floating,
    namespace_of,
    random_stream,
    resolve_device,
    resolve_dtype,
    resolve_namespace,
)
from .._simulation_errors import SimulationValidationError, UnsupportedProcessError
from ..instruments import (
    Instrument,
    InstrumentRef,
    PayoffRequirement,
    UnsupportedInstrumentError,
    payoff,
    payoff_requirement,
)
from ..instruments.exotics import (
    AsianOption,
    BarrierOption,
    BinaryOption,
    LookbackOption,
    VarianceSwap,
)
from ..instruments.forwards import Forward, Future
from ..instruments.options import EuropeanOption
from .scenario import dtype_epsilon, horizon_tolerance
from .simulate import simulate

__all__ = ["MCResult", "MonteCarloEngine"]


@dataclass(frozen=True, slots=True)
class MCResult:
    """One Monte Carlo valuation, with the uncertainty that came with it.

    Attributes
    ----------
    price : float or array
        The discounted sample mean. Native when ``return_native=True``.
    stderr : float or array
        The standard error *of that mean*, so ``price ± 2 * stderr`` is roughly
        a 95% interval for the value the estimator converges to. It says
        nothing about model error.
    n_paths : int
        Paths simulated.
    effective_samples : int
        Independent samples the standard error is computed from. Equal to
        ``n_paths`` for ordinary sampling and half of it under antithetic
        sampling, where the two halves are a matched pair rather than two
        independent draws.
    """

    price: Any
    stderr: Any
    n_paths: int
    effective_samples: int


@dataclass(frozen=True, slots=True)
class MonteCarloEngine:
    """Prices a contract by simulating its underlier and averaging the payoff.

    Parameters
    ----------
    antithetic : bool, default False
        Draw half the paths and mirror them, then average each pair before
        estimating. Whether that reduces variance depends on the payoff; the
        estimator is correct either way and nothing here claims it always
        helps.

    Examples
    --------
    >>> from fast_vollib.instruments import EuropeanOption, VanillaMarketInputs
    >>> from fast_vollib.processes import GBM
    >>> from fast_vollib.simulation import MonteCarloEngine
    >>> option = EuropeanOption(
    ...     underlier="ACME", option_type="call", strike=100.0, maturity=1.0,
    ... )
    >>> market = VanillaMarketInputs(underlying=100.0, rate=0.02)
    >>> result = MonteCarloEngine().price(
    ...     option, market,
    ...     process=GBM.risk_neutral(rate=0.02, volatility=0.2),
    ...     n_paths=4096, n_steps=1, rng=0,
    ... )
    >>> result.effective_samples
    4096
    >>> abs(result.price - 8.916037) < 5 * result.stderr
    True
    """

    antithetic: bool = False

    #: Types the engine has a route for. Membership is exact: a subclass the
    #: registry does not know about is refused rather than treated as its base.
    SUPPORTED_TYPES: ClassVar[tuple[type[Instrument], ...]] = (
        EuropeanOption,
        Forward,
        BinaryOption,
        AsianOption,
        BarrierOption,
        LookbackOption,
        VarianceSwap,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.antithetic, (bool, np.bool_)):
            raise SimulationValidationError(
                f"antithetic must be a bool; got {type(self.antithetic).__name__}. A "
                f"truthy value must not silently change the estimator."
            )
        object.__setattr__(self, "antithetic", bool(self.antithetic))

    def supports(self, instrument: Instrument | type[Instrument]) -> bool:
        """Whether this engine can price the type, or this particular instance.

        Given a *class*, answers whether a route exists at all. Given an
        *instance*, also applies the eligibility a route depends on: a
        simulated valuation needs a positive maturity, because a grid that both
        starts and ends at zero does not exist.

        Examples
        --------
        >>> from fast_vollib.instruments import EuropeanOption, Future
        >>> from fast_vollib.simulation import MonteCarloEngine
        >>> engine = MonteCarloEngine()
        >>> engine.supports(EuropeanOption)
        True
        >>> expiring = EuropeanOption(
        ...     underlier="ACME", option_type="call", strike=100.0, maturity=0.0,
        ... )
        >>> engine.supports(expiring)
        False
        >>> engine.supports(Future)
        False
        """
        cls = instrument if isinstance(instrument, type) else type(instrument)
        if cls not in self.SUPPORTED_TYPES:
            return False
        if isinstance(instrument, type):
            return True
        return _maturity_of(instrument) > 0.0

    def price(
        self,
        instrument: Instrument,
        market: Any,
        *,
        process: Any,
        n_paths: int,
        rng: Any,
        time_grid: Any = None,
        n_steps: int | None = None,
        return_native: bool = False,
    ) -> MCResult:
        """Simulate ``instrument``'s underlier and average its discounted payoff.

        Parameters
        ----------
        instrument : Instrument
            A supported type with a strictly positive maturity and scalar terms.
        market : VanillaMarketInputs
            ``underlying`` is the spot the simulation starts from and ``rate``
            is the continuously compounded discount rate. ``volatility`` is not
            read: the process owns it.
        process : StochasticProcess
            Must evolve exactly ``("spot",)``. The engine never inspects or
            rewrites its drift.
        n_paths : int
            At least 2, or an even number of at least 4 with ``antithetic``:
            a standard error needs more than one sample, and an antithetic
            standard error needs more than one pair.
        rng : object
            A generator, PRNG key, or integer seed for the inferred backend.
        time_grid : array-like, optional
            Explicit observation times. Must end at the contract's maturity.
        n_steps : int, optional
            Build an evenly spaced grid over ``[0, maturity]`` instead. Exactly
            one of ``time_grid`` and ``n_steps`` is supplied; supplying both,
            or neither, is an error rather than a precedence rule to memorize.
        return_native : bool, default False
            Return the price and standard error as backend arrays with their
            dtype, device, and autograd graph intact. The default extracts
            Python floats, which ends the graph. No *computed* value reaches
            the host before that point: the arithmetic stays in the backend
            from the first draw to the last reduction. Validation is a separate
            matter and does read a few scalars -- the grid endpoint, the rate,
            the spot -- to enforce domains before sampling; those never enter
            the numerical graph.

        Returns
        -------
        MCResult

        Raises
        ------
        UnsupportedInstrumentError
            For a type with no route, or an instance that is not eligible.
        UnsupportedProcessError
            For a process this engine cannot supply the initial state of.
        MissingMarketInputError
            Naming the market field that was needed.
        SimulationValidationError
            For any invalid grid, path count, scalar shape, namespace, or RNG.
        """
        maturity = self._validated_instrument(instrument)
        requirement = self._validated_requirement(instrument)
        self._validate_process(process)

        spot = market.require("underlying", operation="price by simulation")
        rate = market.require("rate", operation="discount a simulated payoff")
        _require_scalar(spot, field="market.underlying")
        _require_scalar(rate, field="market.rate")
        _require_finite(rate, field="market.rate")
        _require_positive(spot, field="market.underlying")
        for name, value in process.params().items():
            _require_scalar(value, field=f"process.{name}")

        paths = self._validated_path_count(n_paths)
        inputs: dict[str, Any] = {
            "market.underlying": spot,
            "market.rate": rate,
            "rng": rng,
            "time_grid": time_grid,
        }
        for name, value in process.params().items():
            inputs[f"process.{name}"] = value
        namespace = resolve_namespace(inputs)
        device = resolve_device(namespace, inputs)
        dtype = resolve_dtype(namespace, inputs)

        grid = self._validated_grid(
            time_grid,
            n_steps,
            maturity=maturity,
            namespace=namespace,
            device=device,
            dtype=dtype,
        )
        # Building the stream here rather than leaving it to the sampler is what
        # makes "no paths are drawn on a bad request" true for the RNG as well:
        # a seed builds a throwaway generator, and a supplied one is inspected
        # without being advanced.
        random_stream(rng, namespace=namespace, device=device, dtype=dtype)

        scenario = simulate(
            _underlier_of(instrument),
            process,
            initial_state={"spot": spot},
            time_grid=grid,
            n_paths=paths,
            rng=rng,
            antithetic=self.antithetic,
        )

        if requirement is PayoffRequirement.TERMINAL:
            cashflow = payoff(instrument, scenario.terminal("spot"))
        else:
            cashflow = payoff(instrument, scenario)

        discounted = _discount(cashflow, rate=rate, maturity=maturity)
        return self._estimate(discounted, n_paths=paths, return_native=return_native)

    # --- validation -----------------------------------------------------------

    def _validated_instrument(self, instrument: Instrument) -> float:
        cls = type(instrument)
        if cls is Future:
            raise UnsupportedInstrumentError(
                "A Future is not priced by simulation here. Its terminal payoff formula "
                "matches a Forward's, so this engine would return a plausible number, but "
                "a future's economics are a stream of daily variation margin that this "
                "library does not model. Use payoff() for the terminal cashflow, or a "
                "Forward if a single settlement at maturity is what you mean."
            )
        if cls not in self.SUPPORTED_TYPES:
            supported = ", ".join(sorted(t.__name__ for t in self.SUPPORTED_TYPES))
            raise UnsupportedInstrumentError(
                f"MonteCarloEngine has no route for {cls.__name__}. It prices: {supported}."
            )
        maturity = _maturity_of(instrument)
        if maturity <= 0.0:
            raise UnsupportedInstrumentError(
                f"{cls.__name__} matures at t={maturity!r}, so there is no path to "
                f"simulate: a time grid is strictly increasing and cannot both start and "
                f"end at zero. The contract is still valid -- evaluate it exactly with "
                f"payoff(instrument, terminal_state)."
            )
        return maturity

    def _validated_requirement(self, instrument: Instrument) -> PayoffRequirement:
        requirement = payoff_requirement(instrument)
        if requirement is None:  # pragma: no cover - unreachable for supported types
            raise UnsupportedInstrumentError(
                f"{type(instrument).__name__} has no payoff, so there is nothing to average."
            )
        return requirement

    def _validate_process(self, process: Any) -> None:
        names = getattr(process, "state_names", None)
        if names != ("spot",):
            raise UnsupportedProcessError(
                f"MonteCarloEngine drives a single-factor spot process, but "
                f"{type(process).__name__} evolves {names!r}. Its public inputs carry one "
                f"initial state -- market.underlying -- so it cannot supply the rest. "
                f"simulate() itself is not restricted this way."
            )
        if not callable(getattr(process, "params", None)):
            raise UnsupportedProcessError(
                f"{type(process).__name__} does not expose params(); see "
                f"fast_vollib.processes.StochasticProcess for the contract."
            )

    def _validated_path_count(self, n_paths: Any) -> int:
        # ``np.integer`` too, matching ``simulate()``: an integer read out of a
        # NumPy array is an integer, and refusing it here while accepting it one
        # layer down would be an accident rather than a stricter contract.
        if isinstance(n_paths, bool) or not isinstance(n_paths, (int, np.integer)):
            raise SimulationValidationError(
                f"n_paths must be an integer; got {type(n_paths).__name__}."
            )
        n_paths = int(n_paths)
        if self.antithetic:
            if n_paths < 4 or n_paths % 2 != 0:
                raise SimulationValidationError(
                    f"Antithetic pricing needs an even n_paths of at least 4: the standard "
                    f"error is computed from n_paths / 2 matched pairs, and one pair has no "
                    f"spread. Got {n_paths}."
                )
        elif n_paths < 2:
            raise SimulationValidationError(
                f"Pricing needs at least 2 paths; a standard error is undefined for one "
                f"sample. Got {n_paths}."
            )
        return n_paths

    def _validated_grid(
        self,
        time_grid: Any,
        n_steps: int | None,
        *,
        maturity: float,
        namespace: str,
        device: Any,
        dtype: Any,
    ) -> Any:
        if (time_grid is None) == (n_steps is None):
            raise SimulationValidationError(
                "Supply exactly one of time_grid and n_steps. Both would need a rule for "
                "which wins, and neither leaves the observation schedule undefined -- and "
                "for a path-dependent payoff the schedule is part of the contract's "
                "meaning, not a tuning knob."
            )
        if n_steps is not None:
            if (
                isinstance(n_steps, bool)
                or not isinstance(n_steps, (int, np.integer))
                or n_steps < 1
            ):
                raise SimulationValidationError(
                    f"n_steps must be a positive integer; got {n_steps!r}."
                )
            return _even_grid(
                maturity, int(n_steps), namespace=namespace, device=device, dtype=dtype
            )
        end = concrete_float(_last(time_grid))
        # The epsilon of the *resolved* dtype, not the incoming grid's: simulate
        # brings the grid to the resolved precision, so that is what the
        # scenario will hold and what its own horizon check will use. Reading
        # only the incoming dtype let a float64 grid pass here and be refused by
        # the scenario after every path had already been drawn.
        tolerance = horizon_tolerance(
            maturity, epsilon=max(dtype_epsilon(time_grid), dtype_epsilon(dtype))
        )
        if end is not None and abs(end - maturity) > tolerance:
            raise SimulationValidationError(
                f"time_grid ends at t={end!r} but the contract matures at t={maturity!r}. "
                f"The engine does not extend or truncate a grid: the schedule is the "
                f"caller's statement of when the underlier is observed."
            )
        return time_grid

    # --- the estimator --------------------------------------------------------

    def _estimate(self, discounted: Any, *, n_paths: int, return_native: bool) -> MCResult:
        """Sample mean and its standard error, in the payoff's own namespace.

        Under antithetic sampling the two halves are one sample, not two: path
        ``i`` and path ``i + n/2`` share their randomness by construction, so
        averaging each pair first is what makes the remaining values
        independent. The standard error then divides by the number of *pairs*,
        which is what ``effective_samples`` reports.
        """
        ns = get_namespace(discounted)
        if self.antithetic:
            half = n_paths // 2
            samples = 0.5 * (discounted[:half] + discounted[half:])
            effective = half
        else:
            samples = discounted
            effective = n_paths
        price = ns.mean(samples)
        stderr = ns.std(samples, ddof=1) / math.sqrt(effective)
        if not return_native:
            price = _to_host(price)
            stderr = _to_host(stderr)
        return MCResult(
            price=price,
            stderr=stderr,
            n_paths=n_paths,
            effective_samples=effective,
        )


# --- helpers ------------------------------------------------------------------


def _underlier_of(instrument: Instrument) -> InstrumentRef:
    """The single underlier reference of a supported contract.

    ``underliers`` is declared on ``Derivative`` rather than on ``Instrument``;
    every type this engine supports is one, and the support check has already
    run by the time this is reached.
    """
    underliers: tuple[InstrumentRef, ...] = getattr(instrument, "underliers")
    return underliers[0]


def _maturity_of(instrument: Instrument) -> float:
    maturity = getattr(instrument, "maturity", None)
    if maturity is None:  # pragma: no cover - every supported type has one
        raise UnsupportedInstrumentError(
            f"{type(instrument).__name__} has no maturity to simulate to."
        )
    return float(maturity)


def _require_scalar(value: Any, *, field: str) -> None:
    """Establish that ``value`` is one real number, before anything reads it.

    Screening the *type* first is what makes the value checks below meaningful.
    ``concrete_float`` returns ``None`` both for a JAX tracer, which has no
    value to check, and for anything NumPy cannot read as a number -- a string,
    a ``Decimal``, an object array. Treating the second like the first let a
    rate of ``"inf"`` pass validation and reach ``exp(-inf)``, so a price came
    back as exactly 0.0 with no error anywhere: a plausible number produced by
    a route nobody asked for, which is the failure this layer exists to
    prevent.
    """
    if isinstance(value, (bool, np.bool_)):
        raise SimulationValidationError(f"{field} must be a real number, not a bool.")
    if isinstance(value, (complex, np.complexfloating)):
        raise SimulationValidationError(f"{field} must be a real number, not complex.")
    if isinstance(value, (list, tuple)):
        raise SimulationValidationError(
            f"{field} must be a scalar; got a {type(value).__name__}. This engine prices "
            f"one contract against one market state."
        )
    ndim = getattr(value, "ndim", None)
    if ndim is not None and ndim != 0:
        raise SimulationValidationError(
            f"{field} must be a scalar; got an array with shape "
            f"{getattr(value, 'shape', '?')}. This engine prices one contract against one "
            f"market state; a zero-dimensional array counts as a scalar."
        )
    namespace = namespace_of(value)
    if namespace is not None:
        dtype = getattr(value, "dtype", None)
        if dtype is not None and not _is_floating(namespace, dtype):
            raise SimulationValidationError(
                f"{field} must be a real floating-point value; got dtype {dtype}."
            )
    elif not isinstance(value, (int, float, np.integer, np.floating)):
        raise SimulationValidationError(
            f"{field} must be a real number or a scalar array; got "
            f"{type(value).__name__}. A value the library cannot read as a number is "
            f"refused rather than skipped, so it cannot reach the sampler."
        )


def _require_finite(value: Any, *, field: str) -> None:
    as_float = concrete_float(value)
    if as_float is not None and not math.isfinite(as_float):
        raise SimulationValidationError(f"{field} must be finite; got {as_float!r}.")


def _require_positive(value: Any, *, field: str) -> None:
    as_float = concrete_float(value)
    if as_float is None:
        return
    if not math.isfinite(as_float) or as_float <= 0.0:
        raise SimulationValidationError(
            f"{field} must be a finite, strictly positive spot; got {as_float!r}."
        )


def _even_grid(maturity: float, n_steps: int, *, namespace: str, device: Any, dtype: Any) -> Any:
    """``n_steps + 1`` evenly spaced times from 0 to maturity, in ``namespace``.

    Built in the inferred namespace rather than in NumPy: a NumPy grid handed
    to a torch simulation would be a second array namespace, which the sampler
    refuses -- correctly, and unhelpfully if the engine caused it.
    """
    if namespace == "torch":
        import torch

        return torch.linspace(
            0.0,
            maturity,
            n_steps + 1,
            dtype=dtype if dtype is not None else torch.float64,
            device=device,
        )
    if namespace == "jax":
        import jax.numpy as jnp

        if dtype is not None:
            return jnp.linspace(0.0, maturity, n_steps + 1, dtype=dtype)
        return jnp.linspace(0.0, maturity, n_steps + 1)
    return np.linspace(0.0, maturity, n_steps + 1, dtype=dtype if dtype is not None else np.float64)


def _last(time_grid: Any) -> Any:
    try:
        return time_grid[-1]
    except (TypeError, IndexError, KeyError):
        raise SimulationValidationError(
            f"time_grid must be an indexable sequence of times; got {type(time_grid).__name__}."
        ) from None


def _to_host(value: Any) -> float:
    """The one host conversion in the call, made explicit.

    Detaching first rather than letting ``float()`` do it implicitly: the graph
    ends here by design, and torch is right to warn about a conversion that
    looks accidental.
    """
    detach = getattr(value, "detach", None)
    return float(detach() if callable(detach) else value)


def _discount(cashflow: Any, *, rate: Any, maturity: float) -> Any:
    """Present value of a cashflow at maturity, in the cashflow's namespace.

    A native rate keeps its autograd graph, so a gradient of the price with
    respect to the discount rate is available. A Python rate has no graph to
    keep, and ``math.exp`` avoids materializing a zero-dimensional array whose
    default precision would promote a single-precision payoff.
    """
    if namespace_of(rate) is None:
        return math.exp(-float(rate) * maturity) * cashflow
    ns = get_namespace(cashflow, rate)
    return ns.exp(-rate * maturity) * cashflow
