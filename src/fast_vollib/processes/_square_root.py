"""The square-root transition, shared by every model that has one.

:class:`~fast_vollib.processes.Heston` evolves a variance and
:class:`~fast_vollib.processes.CIRShortRate` evolves a short rate, and the two
are the same stochastic differential equation:

.. math:: dx_t = \\kappa(\\theta - x_t)\\,dt + \\xi\\sqrt{x_t}\\,dW_t

They differ only in what the state is called and in what else is driven by the
same Brownian motion.  Discretizing it is where the difficulty lives -- the
diffusion coefficient is not Lipschitz at zero, an Euler step walks the state
negative, and the square root of a negative number ends the simulation -- so
having one copy of the answer rather than two is worth more than the lines it
saves.

Nothing here knows about a spot, a correlation, or a drift.  Those belong to the
model that owns the second factor, and Heston applies them to the value these
functions return.

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

__all__ = ["PSI_CRITICAL", "full_truncation_step", "quadratic_exponential_step"]

#: Andersen's switching rule between the two quadratic-exponential branches.
#: Any value in ``[1, 2]`` works; 1.5 is the midpoint he recommends.
PSI_CRITICAL = 1.5

#: A floor that keeps a division defined without perturbing any value a caller
#: would notice. It is smaller than any variance or rate a model produces.
_EPS = 1e-300


def quadratic_exponential_step(
    xp: Any, x: Any, *, kappa: Any, theta: Any, xi: Any, dt: Any, z: Any
) -> Any:
    """One Andersen quadratic-exponential step of the square-root state.

    The transition matches the first two moments of the exact non-central
    chi-squared law.  Where that law is close to a squared normal
    (``psi <= PSI_CRITICAL``) it is approximated by ``a(b + Z)^2``; where it is
    close to degenerate at zero it is approximated by an exponential with an
    atom at zero, sampled by inverting the distribution function at
    ``U = Phi(Z)``.

    Parameters
    ----------
    xp : ArrayNS
        The namespace every operand already belongs to.
    x : array
        The state at the start of the step, shaped ``(n_paths,)``.
    kappa, theta, xi, dt : array
        Already converted into ``xp`` by the caller, which is what keeps the
        arithmetic identical to the pre-extraction code.
    z : array
        One standard normal per path.

    Returns
    -------
    array
        The state at the end of the step, floored at zero.
    """
    decay = xp.exp(-kappa * dt)
    m = theta + (x - theta) * decay
    s2 = x * xi * xi * decay * (1.0 - decay) / kappa + theta * xi * xi * (1.0 - decay) ** 2 / (
        2.0 * kappa
    )
    m_safe = xp.maximum(m, xp.asarray(_EPS, like=m))
    psi = s2 / (m_safe * m_safe)

    # Quadratic branch: x' = a (b + Z)^2.
    psi_safe = xp.maximum(psi, xp.asarray(1e-12, like=psi))
    inverse = 2.0 / psi_safe
    radicand = xp.maximum(inverse * (inverse - 1.0), xp.asarray(0.0, like=psi))
    b2 = inverse - 1.0 + xp.sqrt(radicand)
    b = xp.sqrt(xp.maximum(b2, xp.asarray(0.0, like=b2)))
    a = m_safe / (1.0 + b2)
    quadratic = a * (b + z) ** 2

    # Exponential branch: an atom at zero of mass p, then an exponential tail.
    p = (psi_safe - 1.0) / (psi_safe + 1.0)
    p = xp.clip(p, 0.0, 1.0 - 1e-12)
    beta = (1.0 - p) / m_safe
    u = xp.normcdf(z)
    tail = (
        xp.log(
            xp.maximum(
                (1.0 - p) / xp.maximum(1.0 - u, xp.asarray(_EPS, like=u)), xp.asarray(1.0, like=u)
            )
        )
        / beta
    )
    exponential = xp.where(u <= p, xp.asarray(0.0, like=u), tail)

    x_next = xp.where(psi <= PSI_CRITICAL, quadratic, exponential)
    return xp.maximum(x_next, xp.asarray(0.0, like=x_next))


def full_truncation_step(
    xp: Any, x: Any, *, kappa: Any, theta: Any, xi: Any, dt: Any, z: Any
) -> tuple[Any, Any, Any]:
    """One full-truncation Euler step of the square-root state.

    The state is truncated at zero wherever it enters the coefficients, but the
    state itself is allowed to go negative and is truncated again on the next
    step.  Of the naive fixes for the square-root diffusion this is the one with
    the smallest bias (Lord et al. 2010), and it is here as a transparent
    comparison for the quadratic-exponential scheme rather than as a
    recommendation.

    Returns
    -------
    x_next : array
        The state at the end of the step, which may be negative.
    x_plus : array
        The truncated state the coefficients used.
    root : array
        ``sqrt(x_plus * dt)``, returned because the model driving a second
        factor needs the same value and recomputing it would be a second
        chance to get it wrong.
    """
    x_plus = xp.maximum(x, xp.asarray(0.0, like=x))
    root = xp.sqrt(x_plus * dt)
    x_next = x + kappa * (theta - x_plus) * dt + xi * root * z
    return x_next, x_plus, root
