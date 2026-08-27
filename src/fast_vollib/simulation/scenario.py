"""A simulated trajectory: an execution value, not an instrument.

A :class:`Scenario` is what came out of one call to
:func:`~fast_vollib.simulation.simulate`.  It is deliberately *not* a contract
and not something an asset holds: contracts are terms, and an asset that
acquired a ``.paths`` buffer would mean something different after every run.
So a scenario is passed alongside a contract, exactly as market inputs are, and
nothing stores it.

It is also not serializable.  A record of terms can be written down and read
back; a hundred thousand simulated paths with a device, a dtype, and possibly
an autograd graph attached cannot, and pretending otherwise would produce
records nobody can reproduce.  Encoding one raises.

Ownership
---------
NumPy buffers a caller passes to :meth:`Scenario.from_states` are copied and
then marked read-only, so a later edit to the caller's array cannot change what
a scenario means.  Buffers :func:`~fast_vollib.simulation.simulate` allocates
are marked read-only in place -- they were never anyone else's -- so simulation
pays for exactly one allocation.

Native state buffers are stored as they arrive, undetached and uncopied,
because detaching would silently cut the autograd tape that is the reason for
using them.  A native grid is converted only when needed to match the states'
dtype.  A torch state tensor therefore remains mutable by whoever else holds
it.  The scenario's own attributes are frozen and it has no mutating methods;
that is the guarantee, and it is not the same as the array being immutable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .._array_api import concrete_bool, concrete_float, get_namespace
from .._random_api import namespace_of, resolve_device, resolve_dtype, resolve_namespace
from .._simulation_errors import ScenarioMismatchError, SimulationValidationError
from ..instruments import Instrument, InstrumentRef, PayoffRequirement, payoff, payoff_requirement
from ..instruments._validate import coerce_underlier
from ..instruments.payoffs import _require_payoff_support, _ScenarioInput

__all__ = ["Scenario"]

#: Absolute tolerance floor for matching a simulated horizon to a maturity.
_HORIZON_ATOL = 1e-12

#: How many units in the last place of the grid's own dtype to allow on top.
#: Four covers the accumulated rounding of building an evenly spaced grid and
#: reading back its endpoint.
_HORIZON_ULPS = 4.0


def horizon_tolerance(maturity: float, *, epsilon: float = 0.0) -> float:
    """How far a simulated horizon may sit from a contract maturity.

    Relative at ordinary sizes and absolute near zero, so a one-day maturity is
    not held to a tolerance that floating point cannot deliver for a thirty
    year one.

    ``epsilon`` is the machine epsilon of the dtype the *grid* is stored in,
    and widens the relative term when that is coarser than double precision.
    Without it a single-precision simulation could not price a path-dependent
    contract at all: a maturity of 0.1 is not representable in binary32, so the
    stored horizon differs from the contract's by about 1.5e-9 -- three orders
    of magnitude outside a tolerance written for float64, and nothing to do
    with the contract being wrong. Double precision is unaffected: four of its
    ulps are smaller than the floor, so the rule below reduces to the original
    one there.
    """
    relative = max(_HORIZON_ATOL, _HORIZON_ULPS * epsilon)
    return max(_HORIZON_ATOL, relative * abs(maturity))


def dtype_epsilon(value: Any) -> float:
    """Machine epsilon of a floating dtype, or of a value's, or 0.0.

    Accepts either an array-like carrying a ``dtype`` or a dtype itself, so a
    caller that has resolved a dtype but not yet built the buffer can ask.
    """
    dtype = getattr(value, "dtype", value)
    if dtype is None:
        return 0.0
    if type(dtype).__module__.partition(".")[0] == "torch":
        import torch

        return float(torch.finfo(dtype).eps)
    try:
        return float(np.finfo(dtype).eps)
    except (TypeError, ValueError):  # pragma: no cover - non-floating dtype
        return 0.0


@dataclass(frozen=True, slots=True, eq=False, init=False, repr=False)
class Scenario(_ScenarioInput):
    """Simulated state trajectories for one underlier on one time grid.

    Attributes
    ----------
    underlier : InstrumentRef
        What was simulated.  A payoff refuses a contract on anything else.
    time_grid : array
        Observation times, ``(n_times,)``, starting at zero.
    states : array
        ``(n_paths, n_times, n_state)``, in the namespace, device, and dtype of
        the simulation that produced it.
    state_names : tuple[str, ...]
        Names of the trailing axis, in order.

    Notes
    -----
    Equality is identity: two scenarios drawn from the same seed hold equal
    numbers but are different execution values, and a container of arrays has
    no single defensible ``==``.  Identity hashing follows, so a scenario can
    key a cache.

    Examples
    --------
    >>> import numpy as np
    >>> from fast_vollib.simulation import Scenario
    >>> scenario = Scenario.from_states(
    ...     "ACME",
    ...     time_grid=[0.0, 0.5, 1.0],
    ...     states=np.array([[[100.0], [110.0], [120.0]]]),
    ... )
    >>> scenario.n_paths, scenario.n_steps
    (1, 2)
    >>> scenario.terminal("spot")
    array([120.])
    """

    underlier: InstrumentRef
    time_grid: Any
    states: Any
    state_names: tuple[str, ...]
    #: The final grid point as a Python float, read once at construction, or
    #: ``None`` when it was built under a trace and has no concrete value.
    #: Cached so that evaluating a payoff performs no device synchronization:
    #: the horizon check would otherwise pull one scalar back from the GPU on
    #: every call.
    _horizon: float | None
    #: Machine epsilon of the grid's dtype, so the horizon check is held to a
    #: tolerance the stored precision can actually deliver.
    _epsilon: float

    def __init__(
        self,
        underlier: Any,
        *,
        time_grid: Any,
        states: Any,
        state_names: tuple[str, ...] = ("spot",),
    ) -> None:
        """Validate, normalize, and freeze; see :meth:`from_states`."""
        reference = coerce_underlier(underlier)
        names = _validated_state_names(state_names)
        namespace = resolve_namespace({"time_grid": time_grid, "states": states})
        device = resolve_device(namespace, {"time_grid": time_grid, "states": states})
        dtype = resolve_dtype(namespace, {"time_grid": time_grid, "states": states})

        # The states are normalized first and left at their own dtype: they are
        # the large buffer, and a native one is stored without a copy. The grid
        # then follows them, so a scenario never carries two precisions.
        buffer = _normalize(states, namespace=namespace, device=device, dtype=dtype, field="states")
        grid = _normalize(
            time_grid,
            namespace=namespace,
            device=device,
            dtype=getattr(buffer, "dtype", dtype),
            field="time_grid",
            cast=True,
        )
        validate_time_grid(grid)
        _validate_states(buffer, n_times=_length(grid), n_state=len(names))

        object.__setattr__(self, "underlier", reference)
        object.__setattr__(self, "time_grid", _freeze(grid))
        object.__setattr__(self, "states", _freeze(buffer))
        object.__setattr__(self, "state_names", names)
        object.__setattr__(self, "_horizon", concrete_float(grid[-1]))
        object.__setattr__(self, "_epsilon", dtype_epsilon(grid))

    @classmethod
    def from_states(
        cls,
        underlier: Any,
        *,
        time_grid: Any,
        states: Any,
        state_names: tuple[str, ...] = ("spot",),
    ) -> "Scenario":
        """Build a scenario from state trajectories a caller already has.

        Parameters
        ----------
        underlier : InstrumentRef or Asset or str
            Normalized to a reference, as everywhere else in the library.
        time_grid : array-like
            One dimension, at least two strictly increasing finite points,
            starting at exactly zero.
        states : array-like
            ``(n_paths, n_times, n_state)``, real and floating.  Python
            sequences normalize into the namespace the other input selected;
            bool and complex state are refused.
        state_names : tuple[str, ...], default ``("spot",)``
            Non-empty, unique, and as long as the trailing axis.

        Returns
        -------
        Scenario

        Raises
        ------
        SimulationValidationError
            For any shape, dtype, ordering, naming, or namespace problem.

        Examples
        --------
        >>> import numpy as np
        >>> from fast_vollib.simulation import Scenario
        >>> paths = np.array([[[100.0, 0.04], [105.0, 0.05]]])
        >>> two_factor = Scenario.from_states(
        ...     "ACME", time_grid=[0.0, 1.0], states=paths,
        ...     state_names=("spot", "variance"),
        ... )
        >>> two_factor.state("variance").shape
        (1, 2)
        """
        return cls(underlier, time_grid=time_grid, states=states, state_names=state_names)

    @classmethod
    def _from_owned_buffers(
        cls,
        underlier: InstrumentRef,
        *,
        time_grid: Any,
        states: Any,
        state_names: tuple[str, ...],
    ) -> "Scenario":
        """Take ownership of freshly allocated buffers, validating shape only.

        Used by :func:`~fast_vollib.simulation.simulate`, which has already
        validated the grid and allocated both arrays itself.  Copying them
        again would double the peak memory of every simulation to protect
        against an alias that does not exist.
        """
        _validate_states(states, n_times=_length(time_grid), n_state=len(state_names))
        scenario = object.__new__(cls)
        object.__setattr__(scenario, "underlier", underlier)
        object.__setattr__(scenario, "time_grid", _freeze(time_grid))
        object.__setattr__(scenario, "states", _freeze(states))
        object.__setattr__(scenario, "state_names", tuple(state_names))
        object.__setattr__(scenario, "_horizon", concrete_float(time_grid[-1]))
        object.__setattr__(scenario, "_epsilon", dtype_epsilon(time_grid))
        return scenario

    def __repr__(self) -> str:
        shown = "?" if self._horizon is None else f"{self._horizon:g}"
        return (
            f"Scenario(underlier={self.underlier.identifier!r}, n_paths={self.n_paths}, "
            f"n_steps={self.n_steps}, state_names={self.state_names!r}, horizon={shown})"
        )

    # --- access ---------------------------------------------------------------

    @property
    def n_paths(self) -> int:
        """How many trajectories were simulated."""
        return int(self.states.shape[0])

    @property
    def n_steps(self) -> int:
        """How many intervals the grid has: one fewer than its points."""
        return int(self.states.shape[1]) - 1

    @property
    def spot(self) -> Any:
        """The ``(n_paths, n_times)`` spot trajectory.

        Raises
        ------
        SimulationValidationError
            If this scenario has no ``"spot"`` state.
        """
        return self.state("spot")

    def state(self, name: str) -> Any:
        """One state variable's ``(n_paths, n_times)`` trajectory."""
        try:
            index = self.state_names.index(name)
        except ValueError:
            raise SimulationValidationError(
                f"This scenario has no {name!r} state; it holds "
                f"{', '.join(repr(n) for n in self.state_names)}."
            ) from None
        return self.states[:, :, index]

    def terminal(self, name: str = "spot") -> Any:
        """One state variable's ``(n_paths,)`` value at the horizon."""
        return self.state(name)[:, -1]

    def payoff(self, instrument: Instrument) -> Any:
        """The undiscounted cashflow of ``instrument`` along these paths.

        Checks that the scenario describes this contract -- same underlier,
        horizon equal to its maturity -- before any arithmetic, then routes on
        what the payoff needs: a terminal contract is evaluated at the horizon,
        a path-dependent one against the whole trajectory.

        Raises
        ------
        ScenarioMismatchError
            If the underlier or the horizon does not match.
        UnsupportedInstrumentError
            If the type has no payoff at all.
        """
        requirement = payoff_requirement(instrument)
        _require_payoff_support(instrument)
        self._validate_for_contract(instrument)
        if requirement is PayoffRequirement.TERMINAL:
            return payoff(instrument, self.terminal("spot"))
        return payoff(instrument, self)

    # --- the private bridge ---------------------------------------------------

    def _state_path(self, name: str = "spot") -> Any:
        return self.state(name)

    def _path_time_grid(self) -> Any:
        return self.time_grid

    def _validate_for_contract(self, instrument: Instrument) -> None:
        """Refuse a contract this scenario does not describe.

        Two independent checks, both of which would otherwise produce a number
        that looks like a price: the trajectory has to be of the right
        underlier, and it has to end at the contract's maturity rather than
        before or after it.
        """
        self._check_underlier(instrument)
        self._check_horizon(instrument)

    def _check_underlier(self, instrument: Instrument) -> None:
        underliers: tuple[InstrumentRef, ...] = getattr(instrument, "underliers", ())
        if len(underliers) != 1:  # pragma: no cover - every payoff type has exactly one
            raise ScenarioMismatchError(
                f"{type(instrument).__name__} references {len(underliers)} underliers; a "
                f"scenario describes one. Multi-underlier payoffs are not supported."
            )
        contract_ref = underliers[0]
        if contract_ref.identifier != self.underlier.identifier:
            raise ScenarioMismatchError(
                f"This scenario simulates {self.underlier.identifier!r} but the contract "
                f"is written on {contract_ref.identifier!r}."
            )
        for field in ("asset_class", "currency"):
            mine = getattr(self.underlier, field)
            theirs = getattr(contract_ref, field)
            if mine is not None and theirs is not None and mine != theirs:
                raise ScenarioMismatchError(
                    f"This scenario's underlier has {field}={mine!r} but the contract's "
                    f"has {field}={theirs!r}. Where both specify one they must agree."
                )

    def _check_horizon(self, instrument: Instrument) -> None:
        maturity = getattr(instrument, "maturity", None)
        if maturity is None:  # pragma: no cover - every payoff type declares one
            raise ScenarioMismatchError(
                f"{type(instrument).__name__} has no maturity, so no scenario horizon "
                f"can be checked against it."
            )
        if maturity <= 0.0:
            raise ScenarioMismatchError(
                f"A simulated horizon is strictly positive, so it cannot equal a "
                f"maturity of {maturity!r}. Evaluate an expiring contract with "
                f"payoff(instrument, terminal_state) instead."
            )
        horizon = self._horizon
        if horizon is None:
            return
        if abs(horizon - maturity) > horizon_tolerance(maturity, epsilon=self._epsilon):
            raise ScenarioMismatchError(
                f"This scenario ends at t={horizon!r} but the contract matures at "
                f"t={maturity!r}. A payoff is never evaluated on a truncated or "
                f"overlong horizon."
            )


# --- validation helpers -------------------------------------------------------


def _validated_state_names(state_names: Any) -> tuple[str, ...]:
    if isinstance(state_names, str) or not isinstance(state_names, (tuple, list)):
        raise SimulationValidationError(
            f"state_names must be a tuple of names; got {type(state_names).__name__}."
        )
    names = tuple(state_names)
    if not names:
        raise SimulationValidationError("state_names must name at least one state.")
    for name in names:
        if not isinstance(name, str) or not name.strip():
            raise SimulationValidationError(
                f"Every state name must be a non-empty string; got {name!r}."
            )
    if len(set(names)) != len(names):
        raise SimulationValidationError(f"State names must be unique; got {names!r}.")
    return names


def validate_time_grid(grid: Any) -> None:
    """Check a simulation time grid, eagerly where a trace permits it.

    Shape is static and always checked.  The value checks -- finite, starting
    at exactly zero, strictly increasing -- read concrete numbers and are
    skipped under a JAX trace, which has none; a traced call carries them as a
    precondition instead.
    """
    ndim = getattr(grid, "ndim", None)
    if ndim != 1:
        raise SimulationValidationError(
            f"time_grid must be one-dimensional; got shape {getattr(grid, 'shape', '?')}."
        )
    if _length(grid) < 2:
        raise SimulationValidationError(
            "time_grid needs at least two points: a start and something to step to."
        )
    ns = get_namespace(grid)
    if concrete_bool(ns.all(ns.isfinite(grid))) is False:
        raise SimulationValidationError("time_grid must be finite at every point.")
    start = concrete_float(grid[0])
    if start is not None and start != 0.0:
        raise SimulationValidationError(
            f"time_grid must start at exactly 0.0, the valuation date; got {start!r}. "
            f"Times are measured from valuation, as maturities are."
        )
    if concrete_bool(ns.all(grid[1:] > grid[:-1])) is False:
        raise SimulationValidationError(
            "time_grid must be strictly increasing; repeated or out-of-order times "
            "would make a step of zero or negative length."
        )


def _validate_states(states: Any, *, n_times: int, n_state: int) -> None:
    shape = getattr(states, "shape", None)
    if shape is None or len(shape) != 3:
        raise SimulationValidationError(
            f"states must be shaped (n_paths, n_times, n_state); got shape {shape}."
        )
    if shape[0] < 1:
        raise SimulationValidationError("states must hold at least one path.")
    if shape[1] != n_times:
        raise SimulationValidationError(f"states has {shape[1]} times but the grid has {n_times}.")
    if shape[2] != n_state:
        raise SimulationValidationError(
            f"states has {shape[2]} state variables but {n_state} were named."
        )


def _normalize(
    value: Any, *, namespace: str, device: Any, dtype: Any, field: str, cast: bool = False
) -> Any:
    """Bring ``value`` into ``namespace``, copying anything the caller owns.

    With ``cast``, a native input is also brought to ``dtype``. That is used
    for the time grid, which must agree with the states it indexes; the states
    themselves are never cast, because silently doubling a caller's buffer to
    match a grid would be the more expensive of the two mistakes.
    """
    if namespace_of(value) == namespace:
        if cast:
            _require_real_numeric(value, namespace=namespace, field=field)
        else:
            _require_real_floating(value, namespace=namespace, field=field)
        if namespace == "numpy":
            return np.array(value, dtype=dtype if cast else None, copy=True)
        if cast and dtype is not None and getattr(value, "dtype", None) != dtype:
            return value.astype(dtype) if namespace == "jax" else value.to(dtype)
        return value
    if namespace_of(value) is not None:  # pragma: no cover - resolve_namespace refuses first
        raise SimulationValidationError(f"{field} is not in the {namespace} namespace.")
    return _from_sequence(value, namespace=namespace, device=device, dtype=dtype, field=field)


def _from_sequence(value: Any, *, namespace: str, device: Any, dtype: Any, field: str) -> Any:
    """Bring a Python sequence into ``namespace`` -- if it is real and numeric.

    The check runs on what the caller passed, not on the buffer it converts to:
    building first would turn ``[[[True], [False]]]`` into ones and zeros and
    then find a floating dtype and accept it. A mask silently read as a
    trajectory of 1.0 and 0.0 is exactly the kind of plausible answer this
    layer exists to refuse.
    """
    _reject_non_real_sequence(value, field=field)
    built: Any
    if namespace == "torch":
        import torch

        built = torch.as_tensor(
            value, dtype=dtype if dtype is not None else torch.float64, device=device
        )
    elif namespace == "jax":
        import jax.numpy as jnp

        built = jnp.asarray(value, dtype=dtype) if dtype is not None else jnp.asarray(value)
    else:
        built = np.array(value, dtype=dtype if dtype is not None else np.float64)
    _require_real_floating(built, namespace=namespace, field=field)
    return built


def _reject_non_real_sequence(value: Any, *, field: str) -> None:
    try:
        as_array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise SimulationValidationError(
            f"{field} must be an array or a nested numeric sequence; got "
            f"{type(value).__name__} ({exc})."
        ) from None
    dtype = as_array.dtype
    if np.issubdtype(dtype, np.bool_) or np.issubdtype(dtype, np.complexfloating):
        raise SimulationValidationError(
            f"{field} must be real and floating-point; got a sequence of dtype {dtype}. "
            f"Boolean and complex state are refused rather than cast -- a "
            f"mask converted to ones and zeros would look like a trajectory."
        )
    if not np.issubdtype(dtype, np.number):
        raise SimulationValidationError(
            f"{field} must be numeric; got a sequence of dtype {dtype}."
        )


def _require_real_floating(value: Any, *, namespace: str, field: str) -> None:
    from .._random_api import _is_floating

    dtype = getattr(value, "dtype", None)
    if dtype is None or not _is_floating(namespace, dtype):
        raise SimulationValidationError(
            f"{field} must be real and floating-point; got dtype {dtype}. Boolean, "
            f"integer, and complex state are refused rather than cast."
        )


def _require_real_numeric(value: Any, *, namespace: str, field: str) -> None:
    """Require a native value that can be safely cast to a floating grid."""
    dtype = getattr(value, "dtype", None)
    valid = False
    if namespace == "torch":
        import torch

        if dtype is not None:
            probe = torch.empty((), dtype=dtype)
            valid = dtype is not torch.bool and not torch.is_complex(probe)
    elif namespace == "jax":
        import jax.numpy as jnp

        if dtype is not None:
            valid = bool(
                jnp.issubdtype(dtype, jnp.number) and not jnp.issubdtype(dtype, jnp.complexfloating)
            )
    elif dtype is not None:
        valid = bool(
            np.issubdtype(dtype, np.number) and not np.issubdtype(dtype, np.complexfloating)
        )
    if not valid:
        raise SimulationValidationError(
            f"{field} must be real and numeric; got dtype {dtype}. Boolean and complex "
            f"grids are refused rather than cast."
        )


def _freeze(value: Any) -> Any:
    """Mark a NumPy buffer read-only; leave native tensors as they are."""
    if isinstance(value, np.ndarray):
        value.setflags(write=False)
    return value


def _length(grid: Any) -> int:
    return int(grid.shape[0])
