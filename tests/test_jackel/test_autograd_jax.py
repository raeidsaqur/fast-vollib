"""JAX ``custom_vjp`` differentiable IV — port of ``test_autograd.py``.

Mirrors the PyTorch test structure (3 models x 6 inputs, calls/puts,
heterogeneous batches, invalid-domain, low-vega) using JAX gradients
against the same implicit-function identities.
"""

from __future__ import annotations

import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
jax.config.update("jax_enable_x64", True)

from fast_vollib.jackel.differentiable_jax import (  # noqa: E402
    _bsm_price_j,
    _price_vega_d1d2_j,
    implied_volatility_autograd_jax,
)

# ---------------------------------------------------------------------------
# Well-conditioned chain (matches the PyTorch test fixture).
# ---------------------------------------------------------------------------


def _well_conditioned_chain():
    is_call = jnp.array([True, False, True, False, True, False])
    s = jnp.array([100.0, 100.0, 95.0, 105.0, 100.0, 100.0], dtype=jnp.float64)
    k = jnp.array([100.0, 100.0, 105.0, 95.0, 85.0, 115.0], dtype=jnp.float64)
    t = jnp.array([0.25, 0.5, 0.75, 1.0, 1.5, 2.0], dtype=jnp.float64)
    r = jnp.array([0.01, 0.02, 0.015, 0.025, 0.018, 0.012], dtype=jnp.float64)
    q = jnp.array([0.00, 0.01, 0.02, 0.005, 0.015, 0.0], dtype=jnp.float64)
    sigma = jnp.array([0.20, 0.35, 0.28, 0.22, 0.31, 0.25], dtype=jnp.float64)
    return is_call, s, k, t, r, q, sigma


def _q_for_price(model, r, q):
    if model == "black":
        return r
    if model == "black_scholes":
        return jnp.zeros_like(r)
    return q


def _forward_price(model, is_call, s, k, t, r, q, sigma):
    qp = _q_for_price(model, r, q)
    return _bsm_price_j(is_call, s, k, t, r, sigma, qp)


def _model_price_and_vega(model, is_call, s, k, t, r, q, sigma):
    qp = _q_for_price(model, r, q)
    price, vega, _, _ = _price_vega_d1d2_j(is_call, s, k, t, r, sigma, qp)
    return price, vega, qp


def _expected_implicit_grad(model, is_call, s, k, t, r, q, sigma, param: str):
    """Return -(d price_model / d param) / vega under the parametrization the
    JAX backward uses."""

    def price_of(s_, k_, t_, r_, q_):
        qp = _q_for_price(model, r_, q_)
        price_model, _, _, _ = _price_vega_d1d2_j(is_call, s_, k_, t_, r_, sigma, qp)
        return jnp.sum(price_model)

    grads = jax.grad(price_of, argnums=(0, 1, 2, 3, 4))(s, k, t, r, q)
    by_name = dict(zip(("s", "k", "t", "r", "q"), grads))
    grad_price = by_name[param]
    _, vega, _ = _model_price_and_vega(model, is_call, s, k, t, r, q, sigma)
    return -grad_price / vega


# ---------------------------------------------------------------------------
# Price gradient: d sigma / d price = 1 / vega
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model", ["black", "black_scholes", "black_scholes_merton"])
def test_autograd_jax_price_gradient_matches_inverse_vega(model: str) -> None:
    is_call, s, k, t, r, q, true_sigma = _well_conditioned_chain()
    price_model, vega, _ = _model_price_and_vega(model, is_call, s, k, t, r, q, true_sigma)
    price = jax.lax.stop_gradient(price_model)

    iv = implied_volatility_autograd_jax(price, s, k, t, r, is_call, q=q, model=model)
    assert jnp.allclose(iv, true_sigma, rtol=5e-9, atol=5e-10)

    def loss(p):
        return jnp.sum(implied_volatility_autograd_jax(p, s, k, t, r, is_call, q=q, model=model))

    grad_price = jax.grad(loss)(price)
    assert jnp.allclose(grad_price, 1.0 / vega, rtol=1e-8, atol=1e-10)


# ---------------------------------------------------------------------------
# Implicit-function identities for each structural input.
# Parametrized by (model, parameter).  Calls-and-puts mixed in the chain.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model", ["black", "black_scholes", "black_scholes_merton"])
@pytest.mark.parametrize("param", ["s", "k", "t", "r"])
def test_autograd_jax_implicit_input_gradient(model: str, param: str) -> None:
    is_call, s, k, t, r, q, true_sigma = _well_conditioned_chain()
    fixed_price = jax.lax.stop_gradient(_forward_price(model, is_call, s, k, t, r, q, true_sigma))

    arg_index = {"s": 0, "k": 1, "t": 2, "r": 3}[param]

    def iv_sum(*args):
        s_, k_, t_, r_ = args
        return jnp.sum(
            implied_volatility_autograd_jax(fixed_price, s_, k_, t_, r_, is_call, q=q, model=model)
        )

    grad = jax.grad(iv_sum, argnums=arg_index)(s, k, t, r)
    expected = _expected_implicit_grad(model, is_call, s, k, t, r, q, true_sigma, param=param)
    assert jnp.allclose(grad, expected, rtol=5e-8, atol=5e-10), (
        f"{model}/{param}: got={grad}\nexpected={expected}\n"
        f"max_abs_err={jnp.max(jnp.abs(grad - expected)).item():.3e}"
    )


def test_autograd_jax_implicit_dividend_gradient_bsm_only() -> None:
    is_call, s, k, t, r, q, true_sigma = _well_conditioned_chain()
    fixed_price = jax.lax.stop_gradient(
        _forward_price("black_scholes_merton", is_call, s, k, t, r, q, true_sigma)
    )

    def iv_sum(q_):
        return jnp.sum(
            implied_volatility_autograd_jax(
                fixed_price, s, k, t, r, is_call, q=q_, model="black_scholes_merton"
            )
        )

    grad_q = jax.grad(iv_sum)(q)
    expected = _expected_implicit_grad(
        "black_scholes_merton", is_call, s, k, t, r, q, true_sigma, param="q"
    )
    assert jnp.allclose(grad_q, expected, rtol=5e-8, atol=5e-10)


# ---------------------------------------------------------------------------
# Homogeneous flag batches.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("flag_value", [True, False], ids=["calls_only", "puts_only"])
def test_autograd_jax_homogeneous_flag_batch(flag_value: bool) -> None:
    n = 5
    is_call = jnp.full((n,), flag_value, dtype=bool)
    s = jnp.linspace(95.0, 105.0, n, dtype=jnp.float64)
    k = jnp.array([90.0, 95.0, 100.0, 105.0, 110.0], dtype=jnp.float64)
    t = jnp.array([0.5, 0.75, 1.0, 1.25, 1.5], dtype=jnp.float64)
    r = jnp.full((n,), 0.02, dtype=jnp.float64)
    q = jnp.zeros_like(r)
    true_sigma = jnp.array([0.22, 0.26, 0.30, 0.24, 0.28], dtype=jnp.float64)

    price = jax.lax.stop_gradient(_bsm_price_j(is_call, s, k, t, r, true_sigma, q))

    def price_loss(p):
        return jnp.sum(
            implied_volatility_autograd_jax(p, s, k, t, r, is_call, q=q, model="black_scholes")
        )

    iv = implied_volatility_autograd_jax(price, s, k, t, r, is_call, q=q, model="black_scholes")
    assert jnp.allclose(iv, true_sigma, rtol=5e-9, atol=5e-10)
    grad_price = jax.grad(price_loss)(price)
    _, vega, _, _ = _price_vega_d1d2_j(is_call, s, k, t, r, true_sigma, q)
    assert jnp.allclose(grad_price, 1.0 / vega, rtol=1e-8, atol=1e-10)

    def s_loss(s_):
        return jnp.sum(
            implied_volatility_autograd_jax(price, s_, k, t, r, is_call, q=q, model="black_scholes")
        )

    grad_s = jax.grad(s_loss)(s)
    expected_s = _expected_implicit_grad(
        "black_scholes", is_call, s, k, t, r, q, true_sigma, param="s"
    )
    assert jnp.allclose(grad_s, expected_s, rtol=5e-8, atol=5e-10)


# ---------------------------------------------------------------------------
# Heterogeneous maturity-and-spot batch.
# ---------------------------------------------------------------------------


def test_autograd_jax_heterogeneous_maturity_and_spot_batch() -> None:
    is_call = jnp.array([True, False, True, False, True, False, True])
    s = jnp.array([80.0, 100.0, 120.0, 90.0, 110.0, 95.0, 105.0], dtype=jnp.float64)
    k = jnp.array([85.0, 100.0, 115.0, 92.0, 108.0, 100.0, 100.0], dtype=jnp.float64)
    t = jnp.array([0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0], dtype=jnp.float64)
    r = jnp.array([0.01, 0.02, 0.03, 0.015, 0.025, 0.02, 0.018], dtype=jnp.float64)
    q = jnp.zeros_like(r)
    true_sigma = jnp.array([0.40, 0.30, 0.22, 0.28, 0.26, 0.24, 0.32], dtype=jnp.float64)

    fixed_price = jax.lax.stop_gradient(_bsm_price_j(is_call, s, k, t, r, true_sigma, q))

    iv = implied_volatility_autograd_jax(
        fixed_price, s, k, t, r, is_call, q=q, model="black_scholes"
    )
    assert jnp.allclose(iv, true_sigma, rtol=5e-9, atol=5e-10)

    def t_loss(t_):
        return jnp.sum(
            implied_volatility_autograd_jax(
                fixed_price, s, k, t_, r, is_call, q=q, model="black_scholes"
            )
        )

    def r_loss(r_):
        return jnp.sum(
            implied_volatility_autograd_jax(
                fixed_price, s, k, t, r_, is_call, q=q, model="black_scholes"
            )
        )

    grad_t = jax.grad(t_loss)(t)
    grad_r = jax.grad(r_loss)(r)
    expected_t = _expected_implicit_grad(
        "black_scholes", is_call, s, k, t, r, q, true_sigma, param="t"
    )
    expected_r = _expected_implicit_grad(
        "black_scholes", is_call, s, k, t, r, q, true_sigma, param="r"
    )
    assert jnp.allclose(grad_t, expected_t, rtol=5e-8, atol=5e-10)
    assert jnp.allclose(grad_r, expected_r, rtol=5e-8, atol=5e-10)


# ---------------------------------------------------------------------------
# Backward-compatibility — small black_scholes case retained.
# ---------------------------------------------------------------------------


def test_autograd_jax_simple_price_gradient_black_scholes() -> None:
    is_call = jnp.array([True, True, False])
    s = jnp.array([100.0, 105.0, 95.0], dtype=jnp.float64)
    k = jnp.array([100.0, 100.0, 100.0], dtype=jnp.float64)
    t = jnp.array([0.5, 1.0, 1.5], dtype=jnp.float64)
    r = jnp.array([0.01, 0.02, 0.015], dtype=jnp.float64)
    q = jnp.zeros_like(r)
    true_sigma = jnp.array([0.2, 0.35, 0.28], dtype=jnp.float64)

    price = jax.lax.stop_gradient(_bsm_price_j(is_call, s, k, t, r, true_sigma, q))
    iv = implied_volatility_autograd_jax(price, s, k, t, r, is_call, model="black_scholes")
    assert jnp.allclose(iv, true_sigma, rtol=5e-9, atol=5e-10)

    def loss(p):
        return jnp.sum(
            implied_volatility_autograd_jax(p, s, k, t, r, is_call, model="black_scholes")
        )

    grad_price = jax.grad(loss)(price)
    _, vega, _, _ = _price_vega_d1d2_j(is_call, s, k, t, r, true_sigma, q)
    assert jnp.allclose(grad_price, 1.0 / vega, rtol=5e-8, atol=5e-10)


def test_autograd_jax_simple_implicit_spot_gradient_black_scholes() -> None:
    is_call = jnp.array([True, False])
    s = jnp.array([100.0, 95.0], dtype=jnp.float64)
    k = jnp.array([102.0, 100.0], dtype=jnp.float64)
    t = jnp.array([0.75, 1.25], dtype=jnp.float64)
    r = jnp.array([0.01, 0.015], dtype=jnp.float64)
    q = jnp.zeros_like(r)
    true_sigma = jnp.array([0.24, 0.31], dtype=jnp.float64)
    fixed_price = jax.lax.stop_gradient(_bsm_price_j(is_call, s, k, t, r, true_sigma, q))

    def loss(s_):
        return jnp.sum(
            implied_volatility_autograd_jax(
                fixed_price, s_, k, t, r, is_call, model="black_scholes"
            )
        )

    grad_s = jax.grad(loss)(s)

    def price_of_s(s_):
        p, _, _, _ = _price_vega_d1d2_j(is_call, s_, k, t, r, true_sigma, q)
        return jnp.sum(p)

    price_spot_grad = jax.grad(price_of_s)(s)
    _, vega, _, _ = _price_vega_d1d2_j(is_call, s, k, t, r, true_sigma, q)
    expected = -price_spot_grad / vega
    assert jnp.allclose(grad_s, expected, rtol=5e-8, atol=5e-10)


# ---------------------------------------------------------------------------
# JIT compilation: regression guard for the ``nondiff_argnums`` tracer trap.
# The ``is_call`` flag must ride through ``custom_vjp`` as a regular
# positional float arg, not via ``nondiff_argnums``, otherwise tracing
# breaks under ``jax.jit``.
# ---------------------------------------------------------------------------


def test_autograd_jax_jit_grad_runs_under_jit() -> None:
    is_call, s, k, t, r, q, true_sigma = _well_conditioned_chain()
    # black_scholes uses q=0 internally; build vega the same way.
    q_zero = jnp.zeros_like(q)
    fixed_price = jax.lax.stop_gradient(_bsm_price_j(is_call, s, k, t, r, true_sigma, q_zero))

    def loss(p):
        return jnp.sum(
            implied_volatility_autograd_jax(p, s, k, t, r, is_call, model="black_scholes")
        )

    jit_grad = jax.jit(jax.grad(loss))
    grad_price = jit_grad(fixed_price)
    _, vega, _, _ = _price_vega_d1d2_j(is_call, s, k, t, r, true_sigma, q_zero)
    assert jnp.allclose(grad_price, 1.0 / vega, rtol=1e-8, atol=1e-10)


# ---------------------------------------------------------------------------
# Invalid-domain and low-vega contract.
# ---------------------------------------------------------------------------


def test_autograd_jax_below_intrinsic_returns_nan_iv() -> None:
    s = jnp.array([150.0, 100.0], dtype=jnp.float64)
    k = jnp.array([100.0, 150.0], dtype=jnp.float64)
    t = jnp.array([1.0, 1.0], dtype=jnp.float64)
    r = jnp.array([0.02, 0.02], dtype=jnp.float64)
    is_call = jnp.array([True, False])
    price = jnp.array([45.0, 45.0], dtype=jnp.float64)

    iv = implied_volatility_autograd_jax(price, s, k, t, r, is_call, model="black_scholes")
    assert jnp.all(jnp.isnan(iv))

    def loss(p):
        return jnp.sum(
            implied_volatility_autograd_jax(p, s, k, t, r, is_call, model="black_scholes")
        )

    grad_price = jax.grad(loss)(price)
    assert jnp.all(jnp.isnan(grad_price))


def test_autograd_jax_zero_and_negative_price_returns_nan() -> None:
    s = jnp.array([100.0, 100.0], dtype=jnp.float64)
    k = jnp.array([100.0, 100.0], dtype=jnp.float64)
    t = jnp.array([1.0, 1.0], dtype=jnp.float64)
    r = jnp.array([0.02, 0.02], dtype=jnp.float64)
    is_call = jnp.array([True, True])
    price = jnp.array([0.0, -1.0], dtype=jnp.float64)

    iv = implied_volatility_autograd_jax(price, s, k, t, r, is_call, model="black_scholes")
    assert jnp.all(jnp.isnan(iv))

    def loss(p):
        return jnp.sum(
            implied_volatility_autograd_jax(p, s, k, t, r, is_call, model="black_scholes")
        )

    grad_price = jax.grad(loss)(price)
    assert jnp.all(jnp.isnan(grad_price))


def test_autograd_jax_zero_maturity_returns_nan() -> None:
    s = jnp.array([100.0], dtype=jnp.float64)
    k = jnp.array([100.0], dtype=jnp.float64)
    t = jnp.array([0.0], dtype=jnp.float64)
    r = jnp.array([0.02], dtype=jnp.float64)
    is_call = jnp.array([True])
    price = jnp.array([1.0], dtype=jnp.float64)

    iv = implied_volatility_autograd_jax(price, s, k, t, r, is_call, model="black_scholes")
    assert jnp.all(jnp.isnan(iv))

    def loss(p):
        return jnp.sum(
            implied_volatility_autograd_jax(p, s, k, t, r, is_call, model="black_scholes")
        )

    grad_price = jax.grad(loss)(price)
    assert jnp.all(jnp.isnan(grad_price))


def test_autograd_jax_low_vega_gradient_masked_to_nan() -> None:
    """Deep-OTM short-maturity drives vega below the 1e-14 threshold."""
    is_call = jnp.array([True])
    s = jnp.array([100.0], dtype=jnp.float64)
    k = jnp.array([500.0], dtype=jnp.float64)
    t = jnp.array([0.01], dtype=jnp.float64)
    r = jnp.array([0.02], dtype=jnp.float64)
    q = jnp.zeros_like(r)
    true_sigma = jnp.array([0.10], dtype=jnp.float64)
    fixed_price = _bsm_price_j(is_call, s, k, t, r, true_sigma, q)
    _, vega, _, _ = _price_vega_d1d2_j(is_call, s, k, t, r, true_sigma, q)
    assert jnp.abs(vega).item() <= 1e-14, f"expected vega<=1e-14, got {vega}"

    def loss(p):
        return jnp.sum(
            implied_volatility_autograd_jax(p, s, k, t, r, is_call, model="black_scholes")
        )

    grad_price = jax.grad(loss)(fixed_price)
    assert jnp.all(jnp.isnan(grad_price))


def test_autograd_jax_mixed_valid_and_invalid_batch_does_not_contaminate_valid() -> None:
    """A single invalid row must not corrupt gradients for valid rows.

    Two upstream scenarios:
    1. ``jnp.nansum(iv)`` zero-outs NaN rows before grad — invalid rows must
       see clean zero gradient (not NaN, which would 0*NaN-poison via chain).
    2. ``jnp.sum(iv)`` asks for nonzero gradient uniformly — invalid rows
       see NaN (implicit identity is undefined there).
    """
    is_call = jnp.array([True, True, False, True])
    s = jnp.array([100.0, 150.0, 100.0, 100.0], dtype=jnp.float64)
    k = jnp.array([100.0, 100.0, 100.0, 100.0], dtype=jnp.float64)
    t = jnp.array([1.0, 1.0, 1.0, 0.0], dtype=jnp.float64)
    r = jnp.array([0.02, 0.02, 0.02, 0.02], dtype=jnp.float64)
    q = jnp.zeros_like(r)
    true_sigma = jnp.array([0.25, 0.30, 0.28, 0.20], dtype=jnp.float64)
    valid = jnp.array([True, False, True, False])

    prices = _bsm_price_j(is_call, s, k, jnp.clip(t, min=1e-6), r, true_sigma, q)
    prices = prices.at[1].set(5.0)  # below intrinsic
    prices = prices.at[3].set(1.0)  # zero T

    iv = implied_volatility_autograd_jax(prices, s, k, t, r, is_call, model="black_scholes")
    assert jnp.allclose(iv[valid], true_sigma[valid], rtol=5e-9, atol=5e-10)
    assert jnp.all(jnp.isnan(iv[~valid]))

    # Scenario 1: jnp.nansum
    def nansum_loss(p):
        return jnp.nansum(
            implied_volatility_autograd_jax(p, s, k, t, r, is_call, model="black_scholes")
        )

    grad_nansum = jax.grad(nansum_loss)(prices)
    _, vega, _, _ = _price_vega_d1d2_j(is_call, s, k, jnp.clip(t, min=1e-6), r, true_sigma, q)
    expected_valid = (1.0 / vega)[valid]
    got_valid = grad_nansum[valid]
    assert jnp.all(jnp.isfinite(got_valid))
    assert jnp.allclose(got_valid, expected_valid, rtol=1e-8, atol=1e-10)
    assert jnp.all(grad_nansum[~valid] == 0.0)

    # Scenario 2: plain sum -> NaN at invalid rows
    def sum_loss(p):
        return jnp.sum(
            implied_volatility_autograd_jax(p, s, k, t, r, is_call, model="black_scholes")
        )

    grad_sum = jax.grad(sum_loss)(prices)
    got_valid_sum = grad_sum[valid]
    assert jnp.all(jnp.isfinite(got_valid_sum))
    assert jnp.allclose(got_valid_sum, expected_valid, rtol=1e-8, atol=1e-10)
    assert jnp.all(jnp.isnan(grad_sum[~valid]))
