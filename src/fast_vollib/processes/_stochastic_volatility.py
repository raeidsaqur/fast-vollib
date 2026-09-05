"""The correlated spot-and-variance step, shared by every model that has one.

:class:`~fast_vollib.processes.Heston` is this transition and nothing else.
:class:`~fast_vollib.processes.Bates` is this transition plus a jump block, and
BCC97 is Bates with the drift driven by a simulated rate.  All three must
produce *the same* diffusion innovations from the same draws, or a reduction
test can only ever be statistical -- and a statistical test cannot tell a
correct sampler from one that reordered its arithmetic.

So the transition lives here once, parameterized rather than reading a
process's fields, and every model passes its own parameters in.  What stays with
each model is what each model actually owns: Heston owns its five fields, Bates
owns a variance *component* and a jump component, and neither owns the step.

Compatibility
-------------
The shared transition retains the Heston operation order. Parameters arrive
already converted by the caller. Seeded references check the parameter grid
with bounded platform rounding; same-environment model reductions are exact.

References
----------
Andersen, L. (2008). Simple and efficient simulation of the Heston stochastic
volatility model. *Journal of Computational Finance* 11(3), 1-42.

Lord, R., Koekkoek, R., van Dijk, D. (2010). A comparison of biased simulation
schemes for stochastic volatility models. *Quantitative Finance* 10(2), 177-194.
"""

from __future__ import annotations

from typing import Any

from ._square_root import full_truncation_step, quadratic_exponential_step

__all__ = ["full_truncation_euler_step", "quadratic_exponential_step_with_spot"]


def quadratic_exponential_step_with_spot(
    xp: Any,
    log_spot: Any,
    v: Any,
    *,
    kappa: Any,
    theta: Any,
    xi: Any,
    rho: Any,
    drift: Any,
    dt: Any,
    z_variance: Any,
    z_spot: Any,
) -> tuple[Any, Any]:
    """One Andersen quadratic-exponential step of ``(log spot, variance)``.

    The variance transition is
    :func:`fast_vollib.processes._square_root.quadratic_exponential_step`, which
    this model shares with the CIR short rate; what is the stochastic-volatility
    model's own is everything below it.

    The log-spot uses the martingale-corrected representation obtained by
    substituting ``sqrt(v) dW^2 = (dv - kappa(theta - v)dt) / xi``, which removes
    the leading correlation bias that a naive Euler step leaves behind.

    Parameters
    ----------
    xp : ArrayNS
    log_spot, v : array
        The running log-return and variance, shaped ``(n_paths,)``.
    kappa, theta, xi, rho, drift, dt : array
        Already converted into ``xp`` by the caller.
    z_variance, z_spot : array
        The two normals of this step, one per path.

    Returns
    -------
    tuple
        ``(log_spot_next, v_next)``.
    """
    v_next = quadratic_exponential_step(xp, v, kappa=kappa, theta=theta, xi=xi, dt=dt, z=z_variance)

    gamma = 0.5
    k0 = -rho * kappa * theta * dt / xi
    k1 = gamma * dt * (kappa * rho / xi - 0.5) - rho / xi
    k2 = gamma * dt * (kappa * rho / xi - 0.5) + rho / xi
    k3 = gamma * dt * (1.0 - rho * rho)
    k4 = k3
    variance_term = xp.maximum(k3 * v + k4 * v_next, xp.asarray(0.0, like=v))
    log_next = log_spot + drift * dt + k0 + k1 * v + k2 * v_next + xp.sqrt(variance_term) * z_spot
    return log_next, v_next


def full_truncation_euler_step(
    xp: Any,
    log_spot: Any,
    v: Any,
    *,
    kappa: Any,
    theta: Any,
    xi: Any,
    rho: Any,
    drift: Any,
    dt: Any,
    z_variance: Any,
    z_orthogonal: Any,
) -> tuple[Any, Any]:
    """One full-truncation Euler step of ``(log spot, variance)``.

    The variance transition is
    :func:`fast_vollib.processes._square_root.full_truncation_step`, which also
    hands back the truncated variance and its root: the log-spot below needs
    both, and recomputing them would be a second chance to get them wrong.

    Returns
    -------
    tuple
        ``(log_spot_next, v_next)``.
    """
    v_next, v_plus, root = full_truncation_step(
        xp, v, kappa=kappa, theta=theta, xi=xi, dt=dt, z=z_variance
    )
    z_spot = (
        rho * z_variance
        + xp.sqrt(xp.maximum(1.0 - rho * rho, xp.asarray(0.0, like=rho))) * z_orthogonal
    )
    log_next = log_spot + (drift - 0.5 * v_plus) * dt + root * z_spot
    return log_next, v_next


def constant_variance_step(
    xp: Any,
    log_spot: Any,
    v: Any,
    *,
    drift: Any,
    dt: Any,
    z_spot: Any,
) -> tuple[Any, Any]:
    """One exact log-normal step at a variance that does not move.

    Not an approximation of the two above: with ``v`` constant the log-spot
    transition has a closed form and this is it, so a constant-variance
    configuration is sampled exactly on any grid, exactly as
    :class:`~fast_vollib.processes.GBM` is.

    ``v`` is returned unchanged, so the variance column of a sample holds its
    initial value at every time rather than being absent -- which is what keeps
    a scenario's shape the same across model reductions.
    """
    log_next = log_spot + (drift - 0.5 * v) * dt + xp.sqrt(v * dt) * z_spot
    return log_next, v
