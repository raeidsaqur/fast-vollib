r"""Stochastic volatility with jumps, as a configuration rather than a class.

:class:`Bates` is the spot half of the BCC97 lattice: a variance component and a
jump component, over one spot.  Switching either component off is a *reduction*,
not a different implementation --

===========================  ==============  ==========================
``variance``                 ``jumps``       The model this is
===========================  ==============  ==========================
:class:`HestonVariance`      :class:`NoJumps`        Heston (1993)
:class:`HestonVariance`      :class:`LognormalJumps` Bates (1996), SVJ
:class:`ConstantVariance`    :class:`NoJumps`        Black-Scholes-Merton
:class:`ConstantVariance`    :class:`LognormalJumps` Merton (1976)
===========================  ==============  ==========================

-- and each of those is tested against the library's *existing* implementation
of the same model rather than against a second copy written here.

The draw-order contract
-----------------------
What makes a reduction checkable at all is that the randomness does not move
when a component is switched.  Blocks are drawn in a fixed order, into fixed
slots:

``Block 1``  the diffusion normals, shaped ``(n_paths, n_steps, 2)`` -- **always,
             and by the identical call** :class:`~fast_vollib.processes.Heston`
             makes.  :class:`ConstantVariance` leaves column 0 unused rather
             than drawing a narrower block, because a block whose shape depended
             on the configuration would make every reduction a different stream.

``Block 2``  the jump counts and their normals, drawn **only** for
             :class:`LognormalJumps`.

On NumPy and torch the blocks come from the advancing generator in that order,
so block 1 is invariant because it is first.  On JAX they come from
``split(stream, 3)`` by slot, reserving an unused rate slot to match
:class:`~fast_vollib.processes.BCC97`. This avoids relying on prefix stability
across different split sizes. The slots themselves, and the steps they feed,
live in :mod:`fast_vollib.processes._lattice`, so the two facades share one
implementation rather than two that agree.

The consequence, which is tested: switching ``HestonVariance <-> ConstantVariance``
or ``LognormalJumps <-> NoJumps`` leaves block 1 bitwise identical, and
``LognormalJumps(jump_intensity=0)`` produces paths bitwise equal to
``NoJumps``.

Compensation
------------
``drift`` means what :attr:`fast_vollib.processes.Heston.drift` means:
:math:`r - q`, *before* jump compensation.  The sampler subtracts
``jumps.drift_compensator`` itself, so a caller who switches jumps on does not
have to remember to adjust the drift, and the same expression is the one the
characteristic function carries.

References
----------
Bates, D. S. (1996). Jumps and stochastic volatility: exchange rate processes
implicit in Deutsche Mark options. *Review of Financial Studies* 9(1), 69-107.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, ClassVar, Mapping

from .._array_api import get_namespace
from .._random_api import (
    random_stream,
    resolve_device,
    resolve_dtype,
    resolve_namespace,
    split,
    standard_normal,
)
from .._simulation_errors import SimulationValidationError
from ._lattice import (
    DIFFUSION_SLOT as _DIFFUSION_SLOT,
    JUMP_SLOT as _JUMP_SLOT,
    N_SLOTS as _N_SLOTS,
    advance_spot_and_variance,
    aggregate_jump,
    draw_jump_block,
)
from .components import (
    JUMP_TYPES,
    VARIANCE_TYPES,
    JumpComponent,
    VarianceComponent,
    validate_component,
)
from .gbm import _validate_parameter
from .heston import SCHEMES

__all__ = ["Bates"]

#: How many slots this configuration reserves.
#:
#: The slots themselves are named in :mod:`fast_vollib.processes._lattice`, and
#: are fixed by *role* rather than by which components happen to be active, so
#: that switching a component off does not renumber the blocks after it.
#: :class:`~fast_vollib.processes.BCC97` adds slot 2 for rates without
#: disturbing either of these.


@dataclass(frozen=True, slots=True)
class Bates:
    """A spot driven by a variance component and a jump component.

    Parameters
    ----------
    variance : HestonVariance or ConstantVariance
    jumps : LognormalJumps or NoJumps
    drift : float or scalar array, default 0.0
        The log-spot drift :math:`r - q` per year, **before** jump
        compensation. Whatever measure it expresses is the caller's; see
        :meth:`risk_neutral`.

    Notes
    -----
    ``state_names`` is ``("spot", "variance")`` for *every* configuration,
    including :class:`ConstantVariance`, whose variance column holds its
    initial value at every time. A scenario's shape therefore does not change
    when a model is reduced, so the same downstream code reads all four
    corners of the lattice.

    Examples
    --------
    >>> import numpy as np
    >>> from fast_vollib.processes import Bates, HestonVariance, LognormalJumps
    >>> process = Bates(
    ...     variance=HestonVariance(kappa=2.0, theta=0.04, vol_of_vol=0.3, rho=-0.7),
    ...     jumps=LognormalJumps(
    ...         jump_intensity=0.5, mean_log_jump=-0.05, jump_volatility=0.2
    ...     ),
    ...     drift=0.03,
    ... )
    >>> process.state_names
    ('spot', 'variance')
    >>> sorted(process.params())  # doctest: +NORMALIZE_WHITESPACE
    ['drift', 'jumps.jump_intensity', 'jumps.jump_volatility', 'jumps.mean_log_jump',
     'variance.kappa', 'variance.rho', 'variance.theta', 'variance.vol_of_vol']
    >>> paths = process.sample(
    ...     initial_state={"spot": 100.0, "variance": 0.04},
    ...     time_grid=np.array([0.0, 0.5, 1.0]),
    ...     n_paths=4,
    ...     rng=0,
    ... )
    >>> paths.shape
    (4, 3, 2)
    >>> bool(np.all(paths[:, 0, 0] == 100.0))
    True
    """

    variance: VarianceComponent
    jumps: JumpComponent
    drift: Any = 0.0

    #: Fixed across every configuration; see the class notes.
    state_names: ClassVar[tuple[str, ...]] = ("spot", "variance")

    def __post_init__(self) -> None:
        validate_component(self.variance, field="variance", admissible=VARIANCE_TYPES)
        validate_component(self.jumps, field="jumps", admissible=JUMP_TYPES)
        _validate_parameter(self.drift, field="drift")

    @classmethod
    def risk_neutral(
        cls,
        *,
        rate: Any,
        variance: VarianceComponent,
        jumps: JumpComponent,
        dividend_yield: Any = 0.0,
    ) -> "Bates":
        """A process whose log-spot drift is ``rate - dividend_yield``.

        A convenience for the common case, and nothing more: it records a
        modelling decision the caller made rather than making one. The jump
        compensation is *not* applied here, because the sampler applies it --
        applying it twice is the mistake this arrangement is designed to make
        impossible.
        """
        _validate_parameter(rate, field="rate")
        _validate_parameter(dividend_yield, field="dividend_yield")
        return cls(variance=variance, jumps=jumps, drift=rate - dividend_yield)

    def params(self) -> Mapping[str, Any]:
        """Every scalar parameter, flat, under a dotted key.

        Flat because the engine and :func:`~fast_vollib.simulation.simulate`
        validate each value as a scalar and resolve a namespace from it; a
        component object among them is a hard error today and would be a
        confusing one. Dotted because ``kappa`` alone would not say which
        component it came from once a second component has one.

        The objects are the originals, so a torch tensor an optimizer is
        stepping stays that tensor.
        """
        flat: dict[str, Any] = {}
        for prefix, component in (("variance", self.variance), ("jumps", self.jumps)):
            for name, value in component.params().items():
                flat[f"{prefix}.{name}"] = value
        flat["drift"] = self.drift
        return MappingProxyType(flat)

    def sample(
        self,
        *,
        initial_state: Mapping[str, Any],
        time_grid: Any,
        n_paths: int,
        rng: Any,
        antithetic: bool = False,
        scheme: str = "quadratic_exponential",
    ) -> Any:
        """Paths shaped ``(n_paths, n_times, 2)`` -- spot first, then variance.

        Parameters
        ----------
        initial_state : Mapping
            ``"spot"`` (strictly positive) and ``"variance"`` (non-negative).
            The variance is required even for :class:`ConstantVariance`, which
            is where its level comes from.
        time_grid : array-like
            Increasing times starting at zero. Column zero of every path is the
            initial state exactly.
        n_paths : int
        rng : object
            A generator, PRNG key, or integer seed.
        antithetic : bool, default False
            Mirror the diffusion normals; jump counts and their normals are
            *duplicated* rather than mirrored, per
            :func:`fast_vollib._random_api.poisson`.
        scheme : str, default 'quadratic_exponential'
            One of :data:`fast_vollib.processes.SCHEMES`. Ignored by
            :class:`ConstantVariance`, whose transition is exact and has no
            scheme to choose -- the argument is still validated, so a typo is
            not silently accepted.

        Returns
        -------
        array
            In the namespace, device, and dtype the inputs selected.

        Raises
        ------
        SimulationValidationError
            On an unknown scheme, a missing or invalid initial state, or a
            component outside its closed union.
        """
        if scheme not in SCHEMES:
            raise SimulationValidationError(f"scheme must be one of {SCHEMES}; got {scheme!r}.")
        spot = _validate_parameter(
            _require(initial_state, "spot"), field="initial_state['spot']", positive=True
        )
        variance = _validate_parameter(
            _require(initial_state, "variance"),
            field="initial_state['variance']",
            non_negative=True,
        )

        inputs: dict[str, Any] = {
            "initial_state['spot']": spot,
            "initial_state['variance']": variance,
            "time_grid": time_grid,
            "rng": rng,
        }
        for name, value in self.params().items():
            inputs[f"process.{name}"] = value
        namespace = resolve_namespace(inputs)
        device = resolve_device(namespace, inputs)
        dtype = resolve_dtype(namespace, inputs)
        stream = random_stream(rng, namespace=namespace, device=device, dtype=dtype)

        xp = get_namespace(time_grid, spot, variance, *self.params().values())
        steps = time_grid[1:] - time_grid[:-1]
        n_steps = len(steps)
        paths = int(n_paths)

        # Slots, not "one child per active block": switching a component off
        # must not renumber the blocks after it.
        streams = split(stream, _N_SLOTS)
        normals = standard_normal(
            streams[_DIFFUSION_SLOT], (paths, n_steps, 2), antithetic=antithetic
        )
        counts, jump_normals = self._jump_block(
            streams[_JUMP_SLOT],
            xp,
            normals,
            paths=paths,
            n_steps=n_steps,
            steps=steps,
            antithetic=antithetic,
        )

        # The running quantity is the log *return*, so column zero is the
        # initial spot bit for bit rather than ``exp(log(spot))``.
        log_return = xp.zeros((paths,), like=normals)
        v = xp.zeros((paths,), like=normals) + xp.asarray(variance, like=normals)
        spot_value = xp.asarray(spot, like=normals)
        spots = [spot_value + log_return]
        variances = [v]

        effective_drift = xp.asarray(self.drift, like=normals) - xp.asarray(
            self.jumps.drift_compensator, like=normals
        )
        for index in range(n_steps):
            dt = xp.asarray(steps[index], like=normals)
            log_return, v = self._advance(
                xp,
                log_return,
                v,
                dt=dt,
                drift=effective_drift,
                normals=normals[:, index, :],
                scheme=scheme,
            )
            if counts is not None:
                log_return = log_return + self._jump(xp, counts[:, index], jump_normals[:, index])
            spots.append(spot_value * xp.exp(log_return))
            variances.append(v)
        return xp.stack([xp.stack(spots, axis=1), xp.stack(variances, axis=1)], axis=2)

    # --- the blocks ------------------------------------------------------------

    def _jump_block(
        self,
        stream: Any,
        xp: Any,
        like: Any,
        *,
        paths: int,
        n_steps: int,
        steps: Any,
        antithetic: bool,
    ) -> tuple[Any, Any]:
        """The shared jump block; see :func:`._lattice.draw_jump_block`."""
        return draw_jump_block(
            self.jumps,
            stream,
            xp,
            like,
            paths=paths,
            n_steps=n_steps,
            steps=steps,
            antithetic=antithetic,
        )

    def _jump(self, xp: Any, counts: Any, normals: Any) -> Any:
        """The shared aggregate jump; see :func:`._lattice.aggregate_jump`."""
        return aggregate_jump(self.jumps, xp, counts, normals)

    def _advance(
        self,
        xp: Any,
        log_return: Any,
        v: Any,
        *,
        dt: Any,
        drift: Any,
        normals: Any,
        scheme: str,
    ) -> tuple[Any, Any]:
        """The shared step; see :func:`._lattice.advance_spot_and_variance`."""
        return advance_spot_and_variance(
            self.variance, xp, log_return, v, dt=dt, drift=drift, normals=normals, scheme=scheme
        )


def _require(initial_state: Mapping[str, Any], name: str) -> Any:
    try:
        return initial_state[name]
    except (KeyError, TypeError):
        raise SimulationValidationError(
            f"Bates evolves 'spot' and 'variance' states, which the initial state must "
            f"supply; {name!r} is missing from "
            f"{sorted(initial_state) if hasattr(initial_state, 'keys') else '?'}."
        ) from None
