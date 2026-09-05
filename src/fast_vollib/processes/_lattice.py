"""The block layout a configured lattice draws in, and the steps it takes.

:class:`~fast_vollib.processes.Bates` and :class:`~fast_vollib.processes.BCC97`
are the same lattice with a different number of factors, and the reductions
between them are only checkable if that is true *in the code* rather than in
two implementations that happen to agree.  So the blocks are drawn here, once,
and both facades call these functions.

Slots, not "one child per active block"
---------------------------------------
Blocks occupy fixed positions.  A configuration that switches a component off
does not renumber the blocks after it, and a configuration that adds one does
not disturb the blocks before it:

===============  =========================================================
``DIFFUSION``    the spot/variance normals, ``(n_paths, n_steps, 2)``,
                 drawn for **every** configuration -- a constant variance
                 leaves column 0 unused rather than narrowing the block
``JUMP``         Poisson counts and their normals, drawn only for
                 :class:`~fast_vollib.processes.LognormalJumps`
``RATE``         the short-rate path, drawn only for
                 :class:`~fast_vollib.processes.CIRShortRate`
===============  =========================================================

On NumPy and torch the blocks come from one advancing generator in that order.
On JAX both facades split into the same three slots, reserving the rate slot
even in Bates. No prefix-stability assumption is made about split(k, n) across
different n: that property depends on the JAX PRNG implementation and settings.
Jump counts and jump-size normals use separate children of the jump slot.
"""

from __future__ import annotations

from typing import Any

from .._random_api import poisson as _poisson_draws, split, standard_normal
from ._stochastic_volatility import (
    constant_variance_step,
    full_truncation_euler_step,
    quadratic_exponential_step_with_spot,
)
from .components import ConstantVariance, HestonVariance, LognormalJumps

__all__ = [
    "DIFFUSION_SLOT",
    "JUMP_SLOT",
    "RATE_SLOT",
    "N_SLOTS",
    "advance_spot_and_variance",
    "aggregate_jump",
    "draw_jump_block",
]

#: Which split slot each block of draws comes from. See the module docstring.
DIFFUSION_SLOT = 0
JUMP_SLOT = 1
RATE_SLOT = 2
N_SLOTS = 3


def draw_jump_block(
    jumps: Any,
    stream: Any,
    xp: Any,
    like: Any,
    *,
    paths: int,
    n_steps: int,
    steps: Any,
    antithetic: bool,
) -> tuple[Any, Any]:
    """Poisson counts and their normals, or ``(None, None)`` for no jumps.

    Both are whole blocks drawn up front: the number of jumps in a step and
    their sizes depend on nothing the path has done, so there is no reason to
    draw them a step at a time and every reason not to.
    """
    if not isinstance(jumps, LognormalJumps):
        return None, None
    intensity = xp.asarray(jumps.jump_intensity, like=like)
    # One rate per step, read from the grid, so a non-uniform grid is handled
    # by construction rather than by assuming a constant ``dt``. Shape
    # ``(n_steps,)`` broadcasts against ``(paths, n_steps)``.
    rates = intensity * xp.asarray(steps, like=like)
    count_stream, normal_stream = split(stream, 2)
    counts = _poisson_draws(count_stream, (paths, n_steps), rates)
    normals = standard_normal(normal_stream, (paths, n_steps), antithetic=antithetic)
    return counts, normals


def aggregate_jump(jumps: Any, xp: Any, counts: Any, normals: Any) -> Any:
    """The aggregate log jump of one step: ``K m + delta sqrt(K) Z``.

    The exact conditional law of a sum of ``K`` independent normals, which is
    what makes the compound Poisson genuine rather than an at-most-one-jump
    approximation.  At ``K = 0`` it is exactly zero, and adding exact zero to a
    log return leaves it bit for bit unchanged -- which is what makes a zero
    intensity reduce to no jumps at all.
    """
    assert isinstance(jumps, LognormalJumps)  # only called for that branch
    mean = xp.asarray(jumps.mean_log_jump, like=counts)
    volatility = xp.asarray(jumps.jump_volatility, like=counts)
    return counts * mean + volatility * xp.sqrt(counts) * normals


def advance_spot_and_variance(
    variance: Any,
    xp: Any,
    log_return: Any,
    v: Any,
    *,
    dt: Any,
    drift: Any,
    normals: Any,
    scheme: str,
) -> tuple[Any, Any]:
    """One step of the log return and the variance, by the variance component.

    ``drift`` may be a scalar or one value per path; it is only ever used
    elementwise, which is what lets a stochastic-rate configuration pass the
    step's own realized rate without a second step function.
    """
    if isinstance(variance, ConstantVariance):
        # Column 0 of the diffusion block is deliberately unused; see the
        # module note on why the block is not narrowed.
        return constant_variance_step(xp, log_return, v, drift=drift, dt=dt, z_spot=normals[:, 1])
    assert isinstance(variance, HestonVariance)  # closed union
    shared = {
        "kappa": xp.asarray(variance.kappa, like=v),
        "theta": xp.asarray(variance.theta, like=v),
        "xi": xp.asarray(variance.vol_of_vol, like=v),
        "rho": xp.asarray(variance.rho, like=v),
        "drift": drift,
        "dt": dt,
    }
    if scheme == "quadratic_exponential":
        return quadratic_exponential_step_with_spot(
            xp, log_return, v, z_variance=normals[:, 0], z_spot=normals[:, 1], **shared
        )
    return full_truncation_euler_step(
        xp, log_return, v, z_variance=normals[:, 0], z_orthogonal=normals[:, 1], **shared
    )
