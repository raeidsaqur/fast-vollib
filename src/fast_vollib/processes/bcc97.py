r"""Bakshi, Cao and Chen (1997): stochastic volatility, jumps, and stochastic rates.

:class:`BCC97` is :class:`~fast_vollib.processes.Bates` with the short rate
promoted from a number to a state.  The lattice is the same one, with a third
component and a third slot, so every row of the model table is a *configuration*
rather than a separate implementation --

=========================  =====================  ======================  =====================
``variance``               ``jumps``              ``rates``               The model this is
=========================  =====================  ======================  =====================
:class:`ConstantVariance`  :class:`NoJumps`       :class:`ConstantShortRate`  Black-Scholes-Merton
:class:`HestonVariance`    :class:`NoJumps`       :class:`ConstantShortRate`  Heston (1993)
:class:`ConstantVariance`  :class:`LognormalJumps` :class:`ConstantShortRate` Merton (1976)
:class:`HestonVariance`    :class:`LognormalJumps` :class:`ConstantShortRate` Bates (1996), SVJ
:class:`HestonVariance`    :class:`NoJumps`       :class:`CIRShortRate`       SVSI
:class:`HestonVariance`    :class:`LognormalJumps` :class:`CIRShortRate`      BCC97, SVSI-J
=========================  =====================  ======================  =====================

-- and with :class:`ConstantShortRate` the spot and variance columns come back
**bitwise equal** to the corresponding :class:`Bates` configuration, on every
backend, because nothing is drawn from the rate slot and the drift reduces
term for term.

Risk-neutral by construction
----------------------------
Unlike :class:`Bates`, this process has no ``drift`` field to hold a measure:
the drift *is* the state.

.. math::

    \frac{dS_t}{S_t} = (r_t - q - \lambda\mu_J)\,dt + \sqrt{v_t}\,dW^S_t
                       + (e^{Y}-1)\,dN_t .

``dividend_yield`` is the only thing a caller supplies, and the jump
compensation is subtracted by the sampler as it is for :class:`Bates`.

The rate quadrature is the discounting quadrature
-------------------------------------------------
This is the one place where a discretization choice is not free.  In
continuous time :math:`E[e^{-\int_0^T r}S_T] = S_0e^{-qT}`; in discrete time it
holds only if the rate increment the spot's drift uses is *the same number* the
discount factor integrates.  So the step uses
:math:`\tfrac{1}{2}(r_k + r_{k+1})\Delta`, which is exactly what
:class:`~fast_vollib.simulation.PathwiseShortRateDiscounting` with
``rule="trapezoid"`` accumulates, and the martingale identity is tested with
that pairing.  Discounting the same paths with ``rule="left_riemann"`` is a
*different quantity*, not a worse approximation of this one; it is available
because measuring the difference is how the remaining discretization error is
seen, and it is not what the drift uses.

The whole rate path is sampled before the spot loop begins, so the step that
runs at time :math:`t_k` already knows :math:`r_{k+1}`.  That is a property of
the rate driver being independent of the spot, not an anticipation of the
future: no spot draw enters the rate path.

The independence assumption
---------------------------
``rho`` correlates the spot and the variance.  The rate driver is
*independent* of both, which is what makes the transform factorize in
:mod:`fast_vollib.pricing.bcc97` and is the assumption BCC97 themselves make.
They call it severe, and report that relaxing it did not improve empirical
performance.  It is recorded here because it is a modelling decision the
library makes on the caller's behalf and cannot be switched off.

References
----------
Bakshi, G., Cao, C., Chen, Z. (1997). Empirical Performance of Alternative
Option Pricing Models. *Journal of Finance* 52(5), 2003-2049.
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
    RATE_SLOT as _RATE_SLOT,
    advance_spot_and_variance,
    aggregate_jump,
    draw_jump_block,
)
from .cir import CIR_SCHEMES, CIRShortRate
from .components import (
    JUMP_TYPES,
    SHORT_RATE_TYPES,
    VARIANCE_TYPES,
    JumpComponent,
    ShortRateComponent,
    VarianceComponent,
    validate_component,
)
from .gbm import _validate_parameter
from .heston import SCHEMES

__all__ = ["BCC97"]

#: How many slots this configuration reserves; see
#: :mod:`fast_vollib.processes._lattice` for what each one holds.


@dataclass(frozen=True, slots=True)
class BCC97:
    """A spot driven by a variance, a jump, and a short-rate component.

    Parameters
    ----------
    variance : HestonVariance or ConstantVariance
    jumps : LognormalJumps or NoJumps
    rates : CIRShortRate or ConstantShortRate
        The short rate is a *state*, so its initial level comes from
        ``initial_state["short_rate"]`` rather than from the component --
        :class:`ConstantShortRate` stores nothing at all.
    dividend_yield : float or scalar array, default 0.0
        Continuous, subtracted from the short rate in the drift.

    Notes
    -----
    ``state_names`` is ``("spot", "variance", "short_rate")`` for *every*
    configuration, including the constant ones, whose columns hold their
    initial value at every time. A scenario's shape therefore does not change
    when a model is reduced.

    Examples
    --------
    >>> import numpy as np
    >>> from fast_vollib.processes import (
    ...     BCC97, CIRShortRate, HestonVariance, LognormalJumps,
    ... )
    >>> process = BCC97(
    ...     variance=HestonVariance(kappa=2.0, theta=0.04, vol_of_vol=0.3, rho=-0.7),
    ...     jumps=LognormalJumps(
    ...         jump_intensity=0.5, mean_log_jump=-0.05, jump_volatility=0.2
    ...     ),
    ...     rates=CIRShortRate(kappa=0.3, theta=0.04, volatility=0.1),
    ...     dividend_yield=0.01,
    ... )
    >>> process.state_names
    ('spot', 'variance', 'short_rate')
    >>> sorted(process.params())  # doctest: +NORMALIZE_WHITESPACE
    ['dividend_yield', 'jumps.jump_intensity', 'jumps.jump_volatility',
     'jumps.mean_log_jump', 'rates.kappa', 'rates.theta', 'rates.volatility',
     'variance.kappa', 'variance.rho', 'variance.theta', 'variance.vol_of_vol']
    >>> paths = process.sample(
    ...     initial_state={"spot": 100.0, "variance": 0.04, "short_rate": 0.03},
    ...     time_grid=np.array([0.0, 0.5, 1.0]),
    ...     n_paths=4,
    ...     rng=0,
    ... )
    >>> paths.shape
    (4, 3, 3)
    >>> bool(np.all(paths[:, 0, 2] == 0.03))
    True
    """

    variance: VarianceComponent
    jumps: JumpComponent
    rates: ShortRateComponent
    dividend_yield: Any = 0.0

    #: Fixed across every configuration; see the class notes.
    state_names: ClassVar[tuple[str, ...]] = ("spot", "variance", "short_rate")

    def __post_init__(self) -> None:
        validate_component(self.variance, field="variance", admissible=VARIANCE_TYPES)
        validate_component(self.jumps, field="jumps", admissible=JUMP_TYPES)
        validate_component(self.rates, field="rates", admissible=SHORT_RATE_TYPES)
        _validate_parameter(self.dividend_yield, field="dividend_yield")

    def params(self) -> Mapping[str, Any]:
        """Every scalar parameter, flat, under a dotted key.

        Flat and dotted for the reason :meth:`fast_vollib.processes.Bates.params`
        is: the engine and :func:`~fast_vollib.simulation.simulate` validate each
        value as a scalar, and ``kappa`` alone would not say whether it came from
        the variance or the rate.

        The objects are the originals, so a torch tensor an optimizer is
        stepping stays that tensor.
        """
        flat: dict[str, Any] = {}
        for prefix, component in (
            ("variance", self.variance),
            ("jumps", self.jumps),
            ("rates", self.rates),
        ):
            for name, value in component.params().items():
                flat[f"{prefix}.{name}"] = value
        flat["dividend_yield"] = self.dividend_yield
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
        rate_scheme: str = "quadratic_exponential",
    ) -> Any:
        """Paths shaped ``(n_paths, n_times, 3)`` -- spot, variance, short rate.

        Parameters
        ----------
        initial_state : Mapping
            ``"spot"`` (strictly positive), ``"variance"`` (non-negative), and
            ``"short_rate"`` (non-negative). All three are required for every
            configuration, including the constant ones, which is where their
            level comes from.
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
            The **variance** scheme, one of
            :data:`fast_vollib.processes.SCHEMES`.
        rate_scheme : str, default 'quadratic_exponential'
            The **short-rate** scheme, one of
            :data:`fast_vollib.processes.CIR_SCHEMES`. A separate argument
            rather than a shared one because the two state variables do not
            offer the same set: ``exact_transition`` exists for the square-root
            rate and not for the variance, whose spot coupling has no exact
            joint transition. Both are validated even when the corresponding
            component is constant, so a typo is never silently accepted.

        Returns
        -------
        array
            In the namespace, device, and dtype the inputs selected.

        Raises
        ------
        SimulationValidationError
            On an unknown scheme, a missing or invalid initial state, or a
            component outside its closed union.
        UnsupportedProcessError
            For ``rate_scheme='exact_transition'`` on torch, or with parameters
            this library declines to branch on; raised before any draw, so a
            refusal never leaves a caller's generator advanced.
        """
        if scheme not in SCHEMES:
            raise SimulationValidationError(f"scheme must be one of {SCHEMES}; got {scheme!r}.")
        if rate_scheme not in CIR_SCHEMES:
            raise SimulationValidationError(
                f"rate_scheme must be one of {CIR_SCHEMES}; got {rate_scheme!r}."
            )
        spot = _validate_parameter(
            _require(initial_state, "spot"), field="initial_state['spot']", positive=True
        )
        variance = _validate_parameter(
            _require(initial_state, "variance"),
            field="initial_state['variance']",
            non_negative=True,
        )
        short_rate = _validate_parameter(
            _require(initial_state, "short_rate"),
            field="initial_state['short_rate']",
            non_negative=True,
        )

        inputs: dict[str, Any] = {
            "initial_state['spot']": spot,
            "initial_state['variance']": variance,
            "initial_state['short_rate']": short_rate,
            "time_grid": time_grid,
            "rng": rng,
        }
        for name, value in self.params().items():
            inputs[f"process.{name}"] = value
        namespace = resolve_namespace(inputs)

        # The rate component's refusals fire here, before the stream exists, so
        # they cost nothing and advance nothing. They are the component's own
        # checks rather than a copy of them.
        degrees_of_freedom = None
        if isinstance(self.rates, CIRShortRate):
            degrees_of_freedom = self.rates._precheck(
                namespace, time_grid=time_grid, antithetic=antithetic, scheme=rate_scheme
            )

        device = resolve_device(namespace, inputs)
        dtype = resolve_dtype(namespace, inputs)
        stream = random_stream(rng, namespace=namespace, device=device, dtype=dtype)

        xp = get_namespace(time_grid, spot, variance, short_rate, *self.params().values())
        steps = time_grid[1:] - time_grid[:-1]
        n_steps = len(steps)
        paths = int(n_paths)

        streams = split(stream, _N_SLOTS)
        # Slot 0 and slot 1 are the identical calls ``Bates`` makes, in the
        # identical order, which is what makes the constant-rate reduction
        # bitwise rather than merely close.
        normals = standard_normal(
            streams[_DIFFUSION_SLOT], (paths, n_steps, 2), antithetic=antithetic
        )
        counts, jump_normals = draw_jump_block(
            self.jumps,
            streams[_JUMP_SLOT],
            xp,
            normals,
            paths=paths,
            n_steps=n_steps,
            steps=steps,
            antithetic=antithetic,
        )
        rates = self._rate_columns(
            streams[_RATE_SLOT],
            xp,
            normals,
            rate=short_rate,
            steps=steps,
            n_steps=n_steps,
            paths=paths,
            antithetic=antithetic,
            scheme=rate_scheme,
            degrees_of_freedom=degrees_of_freedom,
        )

        # The running quantity is the log *return*, so column zero is the
        # initial spot bit for bit rather than ``exp(log(spot))``.
        log_return = xp.zeros((paths,), like=normals)
        v = xp.zeros((paths,), like=normals) + xp.asarray(variance, like=normals)
        spot_value = xp.asarray(spot, like=normals)
        spots = [spot_value + log_return]
        variances = [v]

        dividend_yield = xp.asarray(self.dividend_yield, like=normals)
        compensator = xp.asarray(self.jumps.drift_compensator, like=normals)
        for index in range(n_steps):
            dt = xp.asarray(steps[index], like=normals)
            # The trapezoid, matching PathwiseShortRateDiscounting's default --
            # see the module note on why this is not a free choice.
            average_rate = 0.5 * (rates[index] + rates[index + 1])
            # Grouped as ``(r - q) - compensator``, which is the association
            # ``Bates`` uses, so a constant rate reduces to it bit for bit.
            drift = (average_rate - dividend_yield) - compensator
            log_return, v = advance_spot_and_variance(
                self.variance,
                xp,
                log_return,
                v,
                dt=dt,
                drift=drift,
                normals=normals[:, index, :],
                scheme=scheme,
            )
            if counts is not None:
                log_return = log_return + aggregate_jump(
                    self.jumps, xp, counts[:, index], jump_normals[:, index]
                )
            spots.append(spot_value * xp.exp(log_return))
            variances.append(v)
        return xp.stack(
            [
                xp.stack(spots, axis=1),
                xp.stack(variances, axis=1),
                xp.stack(rates, axis=1),
            ],
            axis=2,
        )

    # --- the rate block --------------------------------------------------------

    def _rate_columns(
        self,
        stream: Any,
        xp: Any,
        like: Any,
        *,
        rate: Any,
        steps: Any,
        n_steps: int,
        paths: int,
        antithetic: bool,
        scheme: str,
        degrees_of_freedom: float | None,
    ) -> list[Any]:
        """``n_steps + 1`` columns of the short rate, one per grid time.

        For :class:`~fast_vollib.processes.CIRShortRate` this is that process's
        own sampler, called rather than reimplemented: the short rate under
        BCC97 is the same process, sampled by the same code, and the only thing
        that differs is which stream it was handed.

        For :class:`~fast_vollib.processes.ConstantShortRate` nothing is drawn
        at all -- the slot stays untouched, which is what leaves the spot and
        jump blocks bitwise equal to a :class:`~fast_vollib.processes.Bates`
        run on every backend rather than only on JAX.
        """
        if isinstance(self.rates, CIRShortRate):
            return self.rates._sample_columns(
                xp,
                stream,
                rate=rate,
                steps=steps,
                n_steps=n_steps,
                n_paths=paths,
                antithetic=antithetic,
                scheme=scheme,
                degrees_of_freedom=degrees_of_freedom,
            )
        column = xp.zeros((paths,), like=like) + xp.asarray(rate, like=like)
        return [column for _ in range(n_steps + 1)]


def _require(initial_state: Mapping[str, Any], name: str) -> Any:
    try:
        return initial_state[name]
    except (KeyError, TypeError):
        raise SimulationValidationError(
            f"BCC97 evolves 'spot', 'variance' and 'short_rate' states, which the initial "
            f"state must supply; {name!r} is missing from "
            f"{sorted(initial_state) if hasattr(initial_state, 'keys') else '?'}."
        ) from None
