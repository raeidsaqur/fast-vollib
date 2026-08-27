"""Geometric Brownian motion, sampled exactly on an arbitrary grid.

The log of the process is a Brownian motion with drift, so its transition over
one step is known in closed form:

.. math::

    \\log S_{i+1} = \\log S_i
        + \\left(\\mu - \\tfrac{1}{2}\\sigma^2\\right)\\Delta t_i
        + \\sigma \\sqrt{\\Delta t_i}\\, Z_i,
        \\qquad Z_i \\sim \\mathcal{N}(0, 1).

Sampling uses that transition directly rather than an Euler step of the SDE.
The distinction matters: an Euler discretization has a step-size bias that
shows up as a small, grid-dependent pricing error, and a caller comparing a
Monte Carlo European price against the analytic one would have to decide
whether the difference was bias or noise.  Here there is no bias to attribute:
the grid controls only how often the path is *observed*, which is exactly what
a discretely monitored contract needs.  The grid may be irregular; each step
uses its own :math:`\\Delta t_i`.

Two consequences follow from that and are tested:

*The first state is the initial state, exactly.*  Column zero is
``spot * exp(0)``, not a value the sampler drifted to.

*Zero volatility gives the deterministic path.*  With :math:`\\sigma = 0` the
diffusion term is exactly zero and the result is :math:`S_0 e^{\\mu t}`.

Measure is the caller's choice, never the sampler's.  :meth:`GBM.risk_neutral`
is a convenience that computes ``drift = rate - dividend_yield``; it is not
policy, and no engine in this library rewrites a drift to make a price look
risk-neutral.  Discounting a physical-measure simulation at the risk-free rate
produces a number, and the number is not a price.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, ClassVar, Mapping

import numpy as np

from .._array_api import concrete_float, get_namespace
from .._random_api import (
    namespace_of,
    random_stream,
    resolve_device,
    resolve_dtype,
    resolve_namespace,
    standard_normal,
)
from .._simulation_errors import SimulationValidationError

__all__ = ["GBM"]

#: Real scalar types accepted for a process parameter. ``bool`` is excluded
#: above; a NumPy scalar is accepted because it is a number, and rejecting
#: ``np.float32(0.2)`` while accepting ``np.float64(0.2)`` -- which subclasses
#: ``float`` and would otherwise slip through -- would be an accident of
#: inheritance rather than a rule.
_REAL_SCALARS = (int, float, np.integer, np.floating)


def _validate_parameter(
    value: Any, *, field: str, non_negative: bool = False, positive: bool = False
) -> Any:
    """Check a process parameter without consuming or replacing it.

    Static checks -- is it a bool, a complex number, a non-scalar, an integer
    tensor -- always run.  The value check runs whenever the number can be read
    eagerly and is skipped for a JAX tracer, which has no value to check; see
    :func:`fast_vollib._array_api.concrete_float`.  The object handed in is
    returned unchanged, so an optimizer keeps its own tensor.
    """
    if isinstance(value, (bool, np.bool_)):
        raise SimulationValidationError(f"{field} must be a real number, not a bool.")
    if isinstance(value, (complex, np.complexfloating)):
        raise SimulationValidationError(f"{field} must be a real number, not complex.")

    namespace = namespace_of(value)
    if namespace is not None:
        ndim = getattr(value, "ndim", None)
        if ndim is None or ndim != 0:
            raise SimulationValidationError(
                f"{field} must be a scalar; got an array with shape "
                f"{getattr(value, 'shape', '?')}. A process holds one parameter per "
                f"name, not a term structure."
            )
        dtype = getattr(value, "dtype", None)
        if dtype is not None and not _is_floating_dtype(namespace, dtype):
            raise SimulationValidationError(
                f"{field} must be a floating-point value; got dtype {dtype}. An "
                f"integer parameter would silently truncate the arithmetic it enters."
            )
    elif not isinstance(value, _REAL_SCALARS):
        raise SimulationValidationError(
            f"{field} must be a real number or a scalar array; got {type(value).__name__}."
        )

    as_float = concrete_float(value)
    if as_float is not None:
        if as_float != as_float or as_float in (float("inf"), float("-inf")):
            raise SimulationValidationError(f"{field} must be finite; got {as_float!r}.")
        if non_negative and as_float < 0.0:
            raise SimulationValidationError(f"{field} must be non-negative; got {as_float!r}.")
        if positive and as_float <= 0.0:
            raise SimulationValidationError(
                f"{field} must be strictly positive; got {as_float!r}. Geometric "
                f"Brownian motion is multiplicative, so a non-positive start has no "
                f"path -- the log of it does not exist."
            )
    return value


def _is_floating_dtype(namespace: str, dtype: Any) -> bool:
    from .._random_api import _is_floating

    return _is_floating(namespace, dtype)


@dataclass(frozen=True, slots=True)
class GBM:
    """One-factor geometric Brownian motion with constant coefficients.

    Parameters
    ----------
    drift : float or scalar array
        The log-price drift :math:`\\mu`, per year. Finite. Whatever measure it
        expresses is the caller's; see :meth:`risk_neutral`.
    volatility : float or scalar array
        :math:`\\sigma`, per year. Finite and non-negative. Zero is admissible
        and gives the deterministic path.

    Notes
    -----
    Parameters are stored exactly as passed. A torch tensor with
    ``requires_grad=True`` stays that tensor, so a gradient taken through a
    simulated price reaches it.

    Examples
    --------
    >>> from fast_vollib.processes import GBM
    >>> GBM(0.05, 0.2).state_names
    ('spot',)
    >>> GBM.risk_neutral(rate=0.04, volatility=0.2, dividend_yield=0.01).drift
    0.03
    """

    drift: Any
    volatility: Any

    #: The single state variable this process evolves.
    state_names: ClassVar[tuple[str, ...]] = ("spot",)

    def __post_init__(self) -> None:
        _validate_parameter(self.drift, field="drift")
        _validate_parameter(self.volatility, field="volatility", non_negative=True)

    @classmethod
    def risk_neutral(cls, rate: Any, volatility: Any, dividend_yield: Any = 0.0) -> "GBM":
        """A process whose drift is ``rate - dividend_yield``.

        A convenience for the common case, and nothing more. It records a
        modelling decision the caller made rather than making one: an engine
        that discounted at ``market.rate`` would not turn an arbitrary process
        into a risk-neutral one, and this library never tries.

        The subtraction happens in the caller's namespace, so a differentiable
        ``rate`` stays connected to the resulting drift.
        """
        _validate_parameter(rate, field="rate")
        _validate_parameter(dividend_yield, field="dividend_yield")
        return cls(rate - dividend_yield, volatility)

    def params(self) -> Mapping[str, Any]:
        """The two parameters, as the original objects."""
        return MappingProxyType({"drift": self.drift, "volatility": self.volatility})

    def sample(
        self,
        *,
        initial_state: Mapping[str, Any],
        time_grid: Any,
        n_paths: int,
        rng: Any,
        antithetic: bool = False,
    ) -> Any:
        """Exact GBM paths shaped ``(n_paths, n_times, 1)``.

        Parameters
        ----------
        initial_state : Mapping
            Must contain ``"spot"``, strictly positive.
        time_grid : array-like
            Increasing times starting at zero; validated by
            :func:`~fast_vollib.simulation.simulate` before it gets here.
        n_paths : int
        rng : object
            A generator, PRNG key, or integer seed, per
            :mod:`fast_vollib._random_api`.
        antithetic : bool, default False
            Draw ``n_paths / 2`` normals and mirror them.

        Returns
        -------
        array
            In the namespace, device, and dtype the inputs selected.
        """
        spot = _validate_parameter(
            _require_spot(initial_state), field="initial_state['spot']", positive=True
        )
        inputs = {
            "initial_state['spot']": spot,
            "drift": self.drift,
            "volatility": self.volatility,
            "time_grid": time_grid,
            "rng": rng,
        }
        namespace = resolve_namespace(inputs)
        device = resolve_device(namespace, inputs)
        dtype = resolve_dtype(namespace, inputs)
        stream = random_stream(rng, namespace=namespace, device=device, dtype=dtype)

        ns = get_namespace(time_grid, spot, self.drift, self.volatility)
        steps = time_grid[1:] - time_grid[:-1]
        normals = standard_normal(stream, (int(n_paths), len(steps)), antithetic=antithetic)

        variance = self.volatility * self.volatility
        log_increments = (self.drift - 0.5 * variance) * steps + self.volatility * ns.sqrt(
            steps
        ) * normals
        # Column zero is the initial state itself: the running log-return starts
        # at an exact zero, so ``spot * exp(0)`` is ``spot`` bit for bit rather
        # than a value the sampler arrived at.
        start = ns.zeros_like(log_increments[:, :1])
        cumulative = ns.concatenate((start, ns.cumsum(log_increments, axis=1)), axis=1)
        return (spot * ns.exp(cumulative))[..., None]


def _require_spot(initial_state: Mapping[str, Any]) -> Any:
    try:
        return initial_state["spot"]
    except (KeyError, TypeError):
        raise SimulationValidationError(
            f"GBM evolves a 'spot' state, which the initial state must supply; got "
            f"keys {sorted(initial_state) if hasattr(initial_state, 'keys') else '?'}."
        ) from None
