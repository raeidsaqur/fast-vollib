"""A generator can train against the arbitrage penalty: gradients reach its parameters.

The claim the surface layer makes to anyone building a generative model is that
the same no-arbitrage conditions the report checks can be *trained against*, on
the caller's own backend, without a host round trip that would cut the tape.
These tests are that claim, exercised end to end: a differentiable surface
parameterization, evaluated in the caller's namespace, into the existing
:func:`~fast_vollib.surface.penalty.arbitrage_penalty`, and back to the
parameters by autograd.

The parameterization used is SVI, through
:func:`~fast_vollib.surface.fitting.svi_total_variance`, which is written against
the backend-neutral namespace precisely so it can serve here.  A network's output
layer would sit in exactly the same place.
"""

from __future__ import annotations

import numpy as np
import pytest

from fast_vollib.surface import arbitrage_penalty
from fast_vollib.surface.fitting import svi_total_variance

K = np.linspace(-0.4, 0.4, 21)
MATURITIES = np.array([0.25, 0.5, 1.0])

#: An arbitrageable starting point: steep wings and a narrow waist, which is what
#: gives the penalty something to push against.
ARBITRAGEABLE = {"a": 0.002, "b": 0.9, "rho": -0.85, "m": 0.0, "sigma": 0.02}


def _numpy_penalty(**parameters) -> float:
    """The same penalty on the numpy path, as the value the backends must match."""
    w = np.stack([svi_total_variance(K, **parameters) * 1.0 for _ in MATURITIES], axis=1) * (
        MATURITIES[None, :] / MATURITIES[-1]
    )
    iv = np.sqrt(np.maximum(w, 1e-12) / MATURITIES[None, :])
    return float(arbitrage_penalty(iv, K, MATURITIES, 1.0, 0.0))


# --- torch ---------------------------------------------------------------------


def test_torch_gradients_reach_the_surface_parameters() -> None:
    torch = pytest.importorskip("torch", reason="torch not installed")

    k = torch.as_tensor(K, dtype=torch.float64)
    T = torch.as_tensor(MATURITIES, dtype=torch.float64)
    b = torch.tensor(ARBITRAGEABLE["b"], dtype=torch.float64, requires_grad=True)
    sigma = torch.tensor(ARBITRAGEABLE["sigma"], dtype=torch.float64, requires_grad=True)

    smile = svi_total_variance(
        k, ARBITRAGEABLE["a"], b, ARBITRAGEABLE["rho"], ARBITRAGEABLE["m"], sigma
    )
    w = smile[:, None] * (T[None, :] / T[-1])
    iv = torch.sqrt(torch.clamp(w, min=1e-12) / T[None, :])
    penalty = arbitrage_penalty(iv, k, T, 1.0, 0.0)

    assert float(penalty.detach()) > 0.0, "the starting point must violate something"
    penalty.backward()
    assert b.grad is not None and sigma.grad is not None
    assert torch.isfinite(b.grad) and torch.isfinite(sigma.grad)
    # A wider waist is a more convex smile, so the penalty falls as sigma grows.
    assert float(sigma.grad) < 0.0


def test_a_torch_gradient_step_reduces_the_penalty() -> None:
    """The end-to-end claim: this is usable as a training loss, not just callable."""
    torch = pytest.importorskip("torch", reason="torch not installed")

    k = torch.as_tensor(K, dtype=torch.float64)
    T = torch.as_tensor(MATURITIES, dtype=torch.float64)
    sigma = torch.tensor(ARBITRAGEABLE["sigma"], dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.SGD([sigma], lr=1e-4)

    def loss() -> "torch.Tensor":
        smile = svi_total_variance(
            k,
            ARBITRAGEABLE["a"],
            ARBITRAGEABLE["b"],
            ARBITRAGEABLE["rho"],
            ARBITRAGEABLE["m"],
            sigma,
        )
        w = smile[:, None] * (T[None, :] / T[-1])
        iv = torch.sqrt(torch.clamp(w, min=1e-12) / T[None, :])
        return arbitrage_penalty(iv, k, T, 1.0, 0.0)

    before = float(loss().detach())
    for _ in range(50):
        optimizer.zero_grad()
        value = loss()
        value.backward()
        optimizer.step()
    after = float(loss().detach())
    assert after < before, (before, after)


def test_the_torch_penalty_agrees_with_the_numpy_one() -> None:
    torch = pytest.importorskip("torch", reason="torch not installed")

    k = torch.as_tensor(K, dtype=torch.float64)
    T = torch.as_tensor(MATURITIES, dtype=torch.float64)
    smile = svi_total_variance(
        k,
        ARBITRAGEABLE["a"],
        ARBITRAGEABLE["b"],
        ARBITRAGEABLE["rho"],
        ARBITRAGEABLE["m"],
        ARBITRAGEABLE["sigma"],
    )
    w = smile[:, None] * (T[None, :] / T[-1])
    iv = torch.sqrt(torch.clamp(w, min=1e-12) / T[None, :])
    # Cross-backend float64 parity: the same expression, the same arithmetic.
    assert (
        abs(float(arbitrage_penalty(iv, k, T, 1.0, 0.0)) - _numpy_penalty(**ARBITRAGEABLE)) < 1e-12
    )


# --- jax -----------------------------------------------------------------------


def test_jax_gradients_reach_the_surface_parameters() -> None:
    jax = pytest.importorskip("jax", reason="jax not installed")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    k = jnp.asarray(K)
    T = jnp.asarray(MATURITIES)

    def loss(sigma):
        smile = svi_total_variance(
            k,
            ARBITRAGEABLE["a"],
            ARBITRAGEABLE["b"],
            ARBITRAGEABLE["rho"],
            ARBITRAGEABLE["m"],
            sigma,
        )
        w = smile[:, None] * (T[None, :] / T[-1])
        iv = jnp.sqrt(jnp.clip(w, 1e-12) / T[None, :])
        return arbitrage_penalty(iv, k, T, 1.0, 0.0)

    value, gradient = jax.value_and_grad(loss)(ARBITRAGEABLE["sigma"])
    assert float(value) > 0.0
    assert np.isfinite(float(gradient))
    assert float(gradient) < 0.0


def test_the_jax_penalty_agrees_with_the_numpy_one() -> None:
    jax = pytest.importorskip("jax", reason="jax not installed")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    k = jnp.asarray(K)
    T = jnp.asarray(MATURITIES)
    smile = svi_total_variance(
        k,
        ARBITRAGEABLE["a"],
        ARBITRAGEABLE["b"],
        ARBITRAGEABLE["rho"],
        ARBITRAGEABLE["m"],
        ARBITRAGEABLE["sigma"],
    )
    w = smile[:, None] * (T[None, :] / T[-1])
    iv = jnp.sqrt(jnp.clip(w, 1e-12) / T[None, :])
    assert (
        abs(float(arbitrage_penalty(iv, k, T, 1.0, 0.0)) - _numpy_penalty(**ARBITRAGEABLE)) < 1e-12
    )


# --- the honest label -----------------------------------------------------------


def test_a_penalized_surface_is_labelled_a_penalty_and_not_a_guarantee() -> None:
    """Training against the penalty is a different claim from passing the check.

    The vocabulary exists so that a model trained this way can say so without the
    report reading as an arbitrage-free certificate, which is what a single
    boolean would eventually be taken for.
    """
    from fast_vollib.surface import VerificationLevel

    assert VerificationLevel.TRAINING_PENALTY.value == "training_penalty"
    assert VerificationLevel.TRAINING_PENALTY is not VerificationLevel.MATHEMATICAL_GUARANTEE
