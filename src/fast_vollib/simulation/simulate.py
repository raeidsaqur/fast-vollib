"""Turning a process and an explicit initial state into a scenario.

``simulate`` is the only place where dynamics, an initial state, a time grid,
and randomness meet, and it is deliberately narrow about what it knows.  It
does not know what contract the paths are for, and does not check any maturity:
a scenario is worth generating on its own, and the same one may be evaluated
against several contracts.  Matching a horizon to a maturity is the payoff's
concern, and :class:`~fast_vollib.simulation.Scenario` owns that check.

The function is pure.  It returns a new scenario every call and attaches
nothing to the asset, the reference, or the process -- no cached paths, no
device, no "last result".  Re-running a simulation therefore cannot change what
any of them mean, which is the property that makes a contract safe to keep in a
book for years.

Output is native.  There is no ``return_as`` and no host formatting: a scenario
that arrived on a GPU with an autograd graph would lose both to a conversion,
and the caller who wanted a NumPy array can ask for one at the point they
actually need it.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .._random_api import (
    _is_floating,
    namespace_of,
    normalize_device,
    resolve_device,
    resolve_dtype,
    resolve_namespace,
)
from .._simulation_errors import SimulationValidationError, UnsupportedProcessError
from ..instruments._validate import coerce_underlier
from .scenario import Scenario, validate_time_grid

__all__ = ["simulate"]


def simulate(
    underlier: Any,
    process: Any,
    *,
    initial_state: Any,
    time_grid: Any,
    n_paths: int,
    rng: Any,
    antithetic: bool = False,
) -> Scenario:
    """Sample ``process`` from ``initial_state`` over ``time_grid``.

    Parameters
    ----------
    underlier : InstrumentRef or Asset or str
        What is being simulated.  Recorded on the scenario so a payoff can
        refuse a contract written on something else.
    process : StochasticProcess
        Anything exposing ``state_names``, ``params()``, and ``sample()``.
    initial_state : Mapping or float
        The state at ``t = 0``, keyed by state name.  A bare number is
        shorthand for a one-state process and is read as that state's value.
        Mandatory: a simulation that invented its own starting point would be
        answering a question nobody asked.
    time_grid : array-like
        One-dimensional, finite, starting at exactly zero, with at least two
        strictly increasing points.  Irregular spacing is fine.
    n_paths : int
        At least one.  With ``antithetic``, an even number of at least two.
    rng : object
        A generator, PRNG key, or integer seed; see
        :mod:`fast_vollib._random_api` for what each backend accepts.
    antithetic : bool, default False
        Draw half the paths and mirror them.  This is a *sampling* choice; the
        variance it removes, if any, depends on the payoff, and nothing here
        claims it always helps.

    Returns
    -------
    Scenario
        Native, in the namespace, device, and dtype the inputs selected.

    Raises
    ------
    SimulationValidationError
        For any invalid grid, path count, initial state, RNG, or namespace.
    UnsupportedProcessError
        If ``process`` does not implement the sampling contract.

    Examples
    --------
    >>> import numpy as np
    >>> from fast_vollib.processes import GBM
    >>> from fast_vollib.simulation import simulate
    >>> scenario = simulate(
    ...     "ACME",
    ...     GBM.risk_neutral(rate=0.03, volatility=0.2),
    ...     initial_state=100.0,
    ...     time_grid=np.linspace(0.0, 1.0, 5),
    ...     n_paths=1000,
    ...     rng=0,
    ... )
    >>> scenario.n_paths, scenario.n_steps
    (1000, 4)
    >>> bool(np.all(scenario.spot[:, 0] == 100.0))
    True
    """
    if not isinstance(antithetic, (bool, np.bool_)):
        raise SimulationValidationError(
            f"antithetic must be a bool; got {type(antithetic).__name__}. A truthy value "
            f"must not silently change the sampling scheme."
        )
    antithetic = bool(antithetic)
    reference = coerce_underlier(underlier)
    state_names = _state_names_of(process)
    state = _normalize_initial_state(initial_state, state_names)
    paths = _validated_path_count(n_paths, antithetic=antithetic)

    inputs: dict[str, Any] = {f"initial_state[{k!r}]": v for k, v in state.items()}
    inputs["time_grid"] = time_grid
    inputs["rng"] = rng
    for name, value in _process_params(process).items():
        inputs[f"process.{name}"] = value

    namespace = resolve_namespace(inputs)
    device = resolve_device(namespace, inputs)
    dtype = resolve_dtype(namespace, inputs)
    grid = _normalized_grid(time_grid, namespace=namespace, device=device, dtype=dtype)
    validate_time_grid(grid)

    states = _sample(
        process,
        initial_state=state,
        time_grid=grid,
        n_paths=paths,
        rng=rng,
        antithetic=antithetic,
    )
    _check_sample(
        process,
        states,
        time_grid=grid,
        n_paths=paths,
        n_times=int(grid.shape[0]),
        namespace=namespace,
    )
    return Scenario._from_owned_buffers(
        reference, time_grid=grid, states=states, state_names=state_names
    )


def _state_names_of(process: Any) -> tuple[str, ...]:
    names = getattr(process, "state_names", None)
    if (
        not isinstance(names, tuple)
        or not names
        or not all(isinstance(n, str) and n.strip() for n in names)
    ):
        raise UnsupportedProcessError(
            f"{type(process).__name__} does not declare state_names as a non-empty tuple "
            f"of non-empty strings, so simulate() cannot label what it evolves."
        )
    if len(set(names)) != len(names):
        raise UnsupportedProcessError(
            f"{type(process).__name__} declares duplicate state names {names!r}."
        )
    if not callable(getattr(process, "sample", None)):
        raise UnsupportedProcessError(
            f"{type(process).__name__} has no sample() method; see "
            f"fast_vollib.processes.StochasticProcess for the contract."
        )
    return names


def _process_params(process: Any) -> Mapping[str, Any]:
    params = getattr(process, "params", None)
    if not callable(params):
        return {}
    resolved = params()
    if not hasattr(resolved, "items"):  # pragma: no cover - malformed process
        return {}
    return resolved


def _normalize_initial_state(initial_state: Any, state_names: tuple[str, ...]) -> dict[str, Any]:
    if initial_state is None:
        raise SimulationValidationError(
            "initial_state is required. A simulation starts where the caller says it "
            "starts; there is no default spot."
        )
    if not hasattr(initial_state, "keys"):
        if len(state_names) != 1:
            raise SimulationValidationError(
                f"A bare initial_state is shorthand for a one-state process, but this "
                f"one evolves {state_names!r}. Pass a mapping naming each state."
            )
        return {state_names[0]: initial_state}
    state = {str(key): value for key, value in initial_state.items()}
    missing = [name for name in state_names if name not in state]
    if missing:
        raise SimulationValidationError(
            f"initial_state is missing {', '.join(repr(m) for m in missing)}; this "
            f"process evolves {state_names!r}."
        )
    unexpected = sorted(set(state) - set(state_names))
    if unexpected:
        raise SimulationValidationError(
            f"initial_state carries {', '.join(repr(u) for u in unexpected)}, which this "
            f"process does not evolve. It evolves {state_names!r}."
        )
    return state


def _validated_path_count(n_paths: Any, *, antithetic: bool) -> int:
    if isinstance(n_paths, bool) or not isinstance(n_paths, (int, np.integer)):
        raise SimulationValidationError(
            f"n_paths must be an integer; got {type(n_paths).__name__}."
        )
    paths = int(n_paths)
    if antithetic:
        if paths < 2 or paths % 2 != 0:
            raise SimulationValidationError(
                f"Antithetic sampling draws matched pairs, so n_paths must be even and "
                f"at least 2; got {paths}."
            )
    elif paths < 1:
        raise SimulationValidationError(f"n_paths must be at least 1; got {paths}.")
    return paths


def _normalized_grid(time_grid: Any, *, namespace: str, device: Any, dtype: Any) -> Any:
    """The grid in the selected namespace and dtype, as a buffer we own.

    A NumPy grid is copied even when its dtype already matches: ``asarray``
    may hand back the caller's own array, and freezing that would make *their*
    array read-only as a side effect of calling this function.

    The grid is also brought to the *resolved* dtype rather than kept at
    whatever it arrived in. A single-precision grid alongside a double-
    precision spot would otherwise produce double-precision paths beside a
    single-precision schedule -- one scenario carrying two precisions, which no
    caller asked for and neither of which describes the whole object.
    """
    if namespace == "numpy":
        return np.array(time_grid, dtype=dtype if dtype is not None else np.float64)
    if namespace == "torch":
        import torch

        return torch.as_tensor(
            time_grid, dtype=dtype if dtype is not None else torch.float64, device=device
        )
    import jax.numpy as jnp

    return jnp.asarray(time_grid, dtype=dtype) if dtype is not None else jnp.asarray(time_grid)


def _sample(
    process: Any,
    *,
    initial_state: Mapping[str, Any],
    time_grid: Any,
    n_paths: int,
    rng: Any,
    antithetic: bool,
) -> Any:
    return process.sample(
        initial_state=initial_state,
        time_grid=time_grid,
        n_paths=n_paths,
        rng=rng,
        antithetic=antithetic,
    )


def _check_sample(
    process: Any,
    states: Any,
    *,
    time_grid: Any,
    n_paths: int,
    n_times: int,
    namespace: str,
) -> None:
    """Validate the output side of the structural process contract."""
    shape = getattr(states, "shape", None)
    expected = (n_paths, n_times, len(process.state_names))
    if shape is None or tuple(shape) != expected:
        raise UnsupportedProcessError(
            f"{type(process).__name__}.sample returned shape {shape}, but simulate() "
            f"asked for {expected} -- (n_paths, n_times, n_state)."
        )

    actual_namespace = namespace_of(states)
    if actual_namespace != namespace:
        raise UnsupportedProcessError(
            f"{type(process).__name__}.sample returned {actual_namespace or 'a non-native'} "
            f"array, but the simulation inputs selected {namespace}. A structural process "
            f"must return states in the inferred namespace; simulate() never moves its "
            f"output between backends."
        )

    state_dtype = getattr(states, "dtype", None)
    grid_dtype = getattr(time_grid, "dtype", None)
    if state_dtype is None or not _is_floating(namespace, state_dtype):
        raise UnsupportedProcessError(
            f"{type(process).__name__}.sample returned states with dtype {state_dtype}; "
            f"a process must return real floating-point states."
        )
    if grid_dtype is not None and state_dtype != grid_dtype:
        raise UnsupportedProcessError(
            f"{type(process).__name__}.sample returned states with dtype {state_dtype}, "
            f"but the inferred simulation dtype is {grid_dtype}. A Scenario carries one "
            f"precision, and simulate() does not cast a process's path buffer."
        )

    if namespace == "torch":
        state_device = normalize_device(states.device)
        grid_device = normalize_device(time_grid.device)
        if state_device != grid_device:
            raise UnsupportedProcessError(
                f"{type(process).__name__}.sample returned states on {state_device}, but "
                f"the simulation grid is on {grid_device}. A Scenario occupies one torch "
                f"device."
            )
