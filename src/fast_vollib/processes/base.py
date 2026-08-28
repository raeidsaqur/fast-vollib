"""The structural contract a stochastic process satisfies.

A process holds *dynamics and parameters*, and nothing else: no random state,
no path history, no device, no market data, and no knowledge of any contract.
That is what makes it reusable -- the same ``GBM(drift=0.07, volatility=0.2)``
describes a stress scenario and, with a different drift, a risk-neutral one,
and neither meaning is smuggled in by the sampler.

The protocol is structural rather than a base class to inherit from. A caller
with their own dynamics can pass an object that implements these three members
and :func:`fast_vollib.simulation.simulate` will drive it, but the numerical
claims this library makes are about the processes it ships -- currently
:class:`~fast_vollib.processes.GBM`.

Examples
--------
>>> from fast_vollib.processes import GBM, StochasticProcess
>>> isinstance(GBM(0.05, 0.2), StochasticProcess)
True
"""

from __future__ import annotations

from typing import Any, Mapping, Protocol, runtime_checkable

__all__ = ["StochasticProcess"]


@runtime_checkable
class StochasticProcess(Protocol):
    """What :func:`~fast_vollib.simulation.simulate` requires of a process.

    Attributes
    ----------
    state_names : tuple[str, ...]
        The state variables this process evolves, in the order they occupy the
        last axis of a sample.  A one-factor spot process is ``("spot",)``.

    Notes
    -----
    ``isinstance`` against this protocol checks that the three members exist,
    not that they behave.  It is a convenience for error messages, not a
    verification.
    """

    state_names: tuple[str, ...]

    def params(self) -> Mapping[str, Any]:
        """The parameters, as the caller supplied them.

        Returns a read-only mapping holding the *original* objects, so a torch
        tensor an optimizer is stepping stays the same tensor rather than a
        detached copy of its value.
        """
        ...  # pragma: no cover - protocol declaration

    def sample(
        self,
        *,
        initial_state: Mapping[str, Any],
        time_grid: Any,
        n_paths: int,
        rng: Any,
        antithetic: bool = False,
    ) -> Any:
        """Paths shaped ``(n_paths, n_times, n_state)`` in the inferred namespace.

        ``n_times`` is the length of ``time_grid``, whose first entry is the
        valuation date, so column zero of every path is the initial state.
        """
        ...  # pragma: no cover - protocol declaration
