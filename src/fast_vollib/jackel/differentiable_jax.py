"""JAX ``custom_vjp`` differentiable Jaeckel implied volatility.

Mirrors the PyTorch ``implied_volatility_autograd`` API and contract.  The
forward pass invokes the existing JAX Jaeckel solver
(:func:`jackel_iv_black_jax`).  The backward pass does NOT differentiate
through the branch-heavy solver — instead it applies the implicit-function
theorem to the discounted Black-Scholes price equation:

    dsigma / dprice = 1 / vega
    dsigma / dtheta = - (d price_model / d theta) / vega

where ``theta`` is one of ``s, k, t, r, q``.  The price-model gradient is
obtained via :func:`jax.vjp` against a small JAX expression for the
discounted BSM price; the inner Jaeckel solver remains opaque to autodiff.

The contract — invalid-domain NaN-masking on the forward, and
upstream-aware NaN-masking on the backward at low-vega rows — matches
``differentiable.py`` so a JAX training loop with ``jnp.nanmean`` losses
behaves identically to the PyTorch version.
"""

from __future__ import annotations

from functools import partial
import math

import jax
import jax.numpy as jnp
import jax.scipy.special as jss
import numpy as np

from ..types import ModelLiteral

# Make sure x64 is enabled before any JAX op (JAX defaults to float32).
jax.config.update("jax_enable_x64", True)

_BELOW_INTRINSIC_SLACK = 1e-12
_LOW_VEGA_THRESHOLD = 1e-14

_SQRT2 = math.sqrt(2.0)
_SQRT2PI = math.sqrt(2.0 * math.pi)


# ---------------------------------------------------------------------------
# Core math helpers (JAX-native, autodiff-friendly).
#
# These mirror ``fast_vollib.backends.torch_backend._d1_d2`` and
# ``_price_vega_d1d2_t`` so the implicit-gradient identity computed in the
# backward matches the one the PyTorch path uses.  JAX sees through the
# arithmetic via ``jax.vjp`` — there is no need to write any closed-form
# derivative.
# ---------------------------------------------------------------------------


def _normal_cdf_j(x):
    return 0.5 * jss.erfc(-x / _SQRT2)


def _normal_pdf_j(x):
    return jnp.exp(-0.5 * x * x) / _SQRT2PI


def _d1_d2_j(s, k, t, r, sigma, q):
    sqrt_t = jnp.sqrt(jnp.clip(t, min=1e-32))
    vol_term = jnp.clip(sigma * sqrt_t, min=1e-32)
    d1 = (
        jnp.log(jnp.clip(s, min=1e-32) / jnp.clip(k, min=1e-32)) + (r - q + 0.5 * sigma**2) * t
    ) / vol_term
    d2 = d1 - sigma * sqrt_t
    return d1, d2


def _price_vega_d1d2_j(is_call, s, k, t, r, sigma, q):
    """JAX analogue of ``_price_vega_d1d2_t`` from the torch backend.

    Returns ``(price, raw_vega, d1, d2)`` for the discounted BSM price.  The
    ``q`` argument plays the same role as in the PyTorch path:

    * model="black"        → caller passes ``q = r``  (Black-76)
    * model="black_scholes" → caller passes ``q = 0``
    * model="black_scholes_merton" → caller passes the supplied ``q``
    """
    d1, d2 = _d1_d2_j(s, k, t, r, sigma, q)
    discounted_spot = s * jnp.exp(-q * t)
    discounted_strike = k * jnp.exp(-r * t)
    sqrt_t = jnp.sqrt(jnp.clip(t, min=1e-32))
    call = discounted_spot * _normal_cdf_j(d1) - discounted_strike * _normal_cdf_j(d2)
    put = discounted_strike * _normal_cdf_j(-d2) - discounted_spot * _normal_cdf_j(-d1)
    price = jnp.where(is_call, call, put)
    vega = discounted_spot * _normal_pdf_j(d1) * sqrt_t
    return price, vega, d1, d2


def _bsm_price_j(is_call, s, k, t, r, sigma, q):
    """Discounted BSM price on JAX (used in the docs example)."""
    price, _, _, _ = _price_vega_d1d2_j(is_call, s, k, t, r, sigma, q)
    return price


# ---------------------------------------------------------------------------
# Per-model parametrization helpers.
# ---------------------------------------------------------------------------


def _q_for_price(model: str, r, q):
    """Return the ``q`` argument the discounted price uses, matching
    PyTorch's ``differentiable._JackelImpliedVolatilityFunction.forward``."""
    if model == "black":
        return r
    if model == "black_scholes":
        return jnp.zeros_like(r)
    if model == "black_scholes_merton":
        return q
    raise ValueError(f"Unsupported model: {model!r}")


def _forward_for_solver(model: str, s, k, t, r, q):
    """Solver-side forward (F)."""
    if model == "black":
        return s
    if model == "black_scholes":
        return s * jnp.exp(r * t)
    if model == "black_scholes_merton":
        return s * jnp.exp((r - q) * t)
    raise ValueError(f"Unsupported model: {model!r}")


# ---------------------------------------------------------------------------
# Solver core (NaN-mask invalid-domain, return both clean and pre-clean).
# ---------------------------------------------------------------------------


def _solve_iv(price, s, k, t, r, q, is_call_float, model: str):
    """Run the JAX Jaeckel solver and apply the invalid-domain mask.

    ``is_call_float`` is a 0/1-valued float64 tensor.  We use a float dtype
    rather than bool so the call flag can ride through ``custom_vjp`` as a
    regular positional argument (JAX forbids tracer-valued bool arrays in
    ``nondiff_argnums``).  Inside the solver we recover the bool with a
    ``> 0.5`` comparison.

    Returns ``(sigma_out, sigma_for_backward, invalid_mask)``.  Replicates
    the PyTorch ``forward`` exactly: invalid-domain rows return NaN in
    ``sigma_out`` while ``sigma_for_backward`` falls back to ``0.1`` so the
    backward's price helper does not encounter NaN.
    """
    from .jax_backend import _jackel_iv_normalized_j, _jit_jackel_iv_normalized

    is_call = is_call_float > 0.5
    qm = _q_for_price(model, r, q)

    # Build the discounted/undiscounted price the solver expects.
    forward = _forward_for_solver(model, s, k, t, r, q)
    undiscounted_price = price * jnp.exp(r * t)

    # Reproduce the inner solver pipeline at array level (mirrors
    # ``jackel_iv_black_jax`` but inlined so we can use heterogeneous
    # F/T tensors).
    tiny = 2.2250738585072014e-308
    sqrt_T = jnp.sqrt(jnp.clip(t, min=0.0))
    q_arr = jnp.where(is_call, 1.0, -1.0)
    sqrt_FK = jnp.sqrt(forward * k)
    x = jnp.log(jnp.clip(forward, min=tiny) / jnp.clip(k, min=tiny))
    intrinsic = jnp.abs(jnp.maximum(q_arr * (forward - k), 0.0))
    itm = (q_arr * x) > 0.0
    price_red = jnp.where(
        itm, jnp.abs(jnp.maximum(undiscounted_price - intrinsic, 0.0)), undiscounted_price
    )
    x_red = jnp.where(x > 0.0, -x, x)
    beta = price_red / jnp.where(sqrt_FK > 0.0, sqrt_FK, tiny)

    if _jit_jackel_iv_normalized is not None:
        sigma_hat = _jit_jackel_iv_normalized(beta, x_red)
    else:
        sigma_hat = _jackel_iv_normalized_j(beta, x_red)

    sqrt_T_safe = jnp.where(sqrt_T > 0.0, sqrt_T, 1.0)
    sigma_solver = jnp.where(sqrt_T > 0.0, sigma_hat / sqrt_T_safe, 0.0)

    # Invalid-domain mask matches the PyTorch contract exactly.
    disc_spot = s * jnp.exp(-qm * t)
    disc_strike = k * jnp.exp(-r * t)
    call_intrinsic = jnp.maximum(disc_spot - disc_strike, 0.0)
    put_intrinsic = jnp.maximum(disc_strike - disc_spot, 0.0)
    intrinsic_disc = jnp.where(is_call, call_intrinsic, put_intrinsic)
    invalid = (
        (price < intrinsic_disc - _BELOW_INTRINSIC_SLACK)
        | (price <= 0.0)
        | (t <= 0.0)
        | (s <= 0.0)
        | (k <= 0.0)
    )
    nan = jnp.full_like(sigma_solver, jnp.nan)
    sigma_out = jnp.where(invalid, nan, sigma_solver)
    sigma_backward = jnp.where(invalid, jnp.full_like(sigma_solver, 0.1), sigma_solver)
    return sigma_out, sigma_backward, invalid


# ---------------------------------------------------------------------------
# custom_vjp implementation.
# ---------------------------------------------------------------------------
#
# We split the public, broadcast-aware wrapper from a tightly-typed inner
# function whose input arity matches the autodiff contract.  Non-differentiable
# arguments (``model`` string, ``is_call`` bool array) are bundled via
# ``nondiff_argnums``.  JAX's ``custom_vjp`` then sees five differentiable
# leaves: ``price, s, k, t, r, q``.
#
# Forward order: (price, s, k, t, r, q, is_call_float, model)
# Diff args are 0..6 (price, s, k, t, r, q, is_call_float).  Non-diff: 7 (model).
#
# The flag has no derivative — we return a zero cotangent for it — but it
# must ride through as a regular positional argument because under
# ``jax.jit`` the bool tensor becomes a tracer, and JAX forbids tracer
# values in ``nondiff_argnums`` (those are reserved for hashable Python
# values such as the model string).  We pass it as float64 0/1 and recover
# the bool with ``> 0.5`` inside ``_solve_iv``.


@partial(jax.custom_vjp, nondiff_argnums=(7,))
def _jackel_iv_jax_inner(price, s, k, t, r, q, is_call_float, model: str):
    """Inner ``custom_vjp`` entry point."""
    sigma_out, _, _ = _solve_iv(price, s, k, t, r, q, is_call_float, model)
    return sigma_out


def _jackel_iv_jax_fwd(price, s, k, t, r, q, is_call_float, model: str):
    sigma_out, sigma_backward, invalid = _solve_iv(price, s, k, t, r, q, is_call_float, model)
    residuals = (price, s, k, t, r, q, is_call_float, sigma_backward, invalid)
    return sigma_out, residuals


def _jackel_iv_jax_bwd(model: str, residuals, cotangent):
    """Implicit-function theorem backward.

    ``cotangent`` has the same shape as ``sigma_out``.  We compute
    ``d price_model / d theta`` via :func:`jax.vjp` against the discounted
    BSM price so analytic derivatives never need to be hand-derived.
    """
    price, s, k, t, r, q, is_call_float, sigma_bw, invalid = residuals
    is_call = is_call_float > 0.5
    qm = _q_for_price(model, r, q)

    # Vega for the implicit identity.  Use the saved sigma (NaN-clean) to
    # avoid the price helper hitting NaN at invalid rows.
    _, vega, _, _ = _price_vega_d1d2_j(is_call, s, k, t, r, sigma_bw, qm)

    well_cond = (~invalid) & (jnp.abs(vega) > _LOW_VEGA_THRESHOLD)
    zero_upstream = cotangent == 0.0
    safe_vega = jnp.where(well_cond, vega, jnp.ones_like(vega))
    raw_seed = -cotangent / safe_vega
    nan_filler = jnp.full_like(cotangent, jnp.nan)
    zero_filler = jnp.zeros_like(cotangent)

    implicit_seed = jnp.where(
        well_cond,
        raw_seed,
        jnp.where(zero_upstream, zero_filler, nan_filler),
    )

    # ``jax.vjp`` against the structural inputs.
    def _price_of(s_, k_, t_, r_, q_):
        if model == "black":
            qp = r_
        elif model == "black_scholes":
            qp = jnp.zeros_like(r_)
        else:
            qp = q_
        price_model, _, _, _ = _price_vega_d1d2_j(is_call, s_, k_, t_, r_, sigma_bw, qp)
        return price_model

    _, vjp_fn = jax.vjp(_price_of, s, k, t, r, q)
    cot_s, cot_k, cot_t, cot_r, cot_q = vjp_fn(implicit_seed)

    # ``d sigma / d price = 1 / vega`` (with the same upstream-aware mask).
    raw_grad_price = cotangent / safe_vega
    grad_price = jnp.where(
        well_cond,
        raw_grad_price,
        jnp.where(zero_upstream, zero_filler, nan_filler),
    )

    # For non-BSM models, ``q`` does not flow to the price helper, so JAX
    # returns a zero cotangent for it; we zero it explicitly for symmetry
    # with the PyTorch backward (which returns ``None`` there).
    if model != "black_scholes_merton":
        cot_q = jnp.zeros_like(q)

    # Flag has no gradient.
    cot_flag = jnp.zeros_like(is_call_float)

    return (grad_price, cot_s, cot_k, cot_t, cot_r, cot_q, cot_flag)


_jackel_iv_jax_inner.defvjp(_jackel_iv_jax_fwd, _jackel_iv_jax_bwd)


# ---------------------------------------------------------------------------
# Public wrapper — mirrors implied_volatility_autograd in differentiable.py
# ---------------------------------------------------------------------------


def _flag_to_bool_jax(flag, n):
    """Mirror :func:`differentiable._flag_to_bool_tensor` for JAX inputs."""
    if isinstance(flag, bool):
        return jnp.full((n,), flag, dtype=bool)
    if isinstance(flag, np.bool_):
        return jnp.full((n,), bool(flag), dtype=bool)
    arr = np.asarray(flag)
    if arr.dtype == bool:
        return jnp.asarray(arr, dtype=bool)
    if arr.dtype.kind in ("U", "S", "O"):
        flat = arr.astype(str)
        bool_arr = np.asarray([c[:1].lower() == "c" for c in flat.flatten()]).reshape(arr.shape)
        return jnp.asarray(bool_arr, dtype=bool)
    return jnp.asarray(arr > 0, dtype=bool)


def implied_volatility_autograd_jax(
    price,
    S,
    K,
    t,
    r,
    flag,
    q=None,
    *,
    model: ModelLiteral = "black_scholes",
):
    """Differentiable Jaeckel implied volatility for JAX.

    Mirror of :func:`fast_vollib.jackel.differentiable.implied_volatility_autograd`
    using ``jax.custom_vjp``.  The forward evaluates the Jaeckel
    ``Let's Be Rational`` solver.  The backward applies the implicit-function
    theorem to the discounted Black-Scholes price equation:
    ``d sigma / d price = 1 / vega`` and
    ``d sigma / d theta = -(d price_model / d theta) / vega`` for
    ``theta in {S, K, t, r, q}``.

    Invalid-domain contract:
        Entries where the price is below the discounted intrinsic value,
        non-positive, or where ``t``, ``S``, or ``K`` is non-positive
        return ``NaN`` in the forward and propagate ``NaN`` gradients for
        all differentiable inputs.

    Low-vega contract:
        The forward returns the solver's best estimate at low-vega inputs
        without NaN-masking (so that ``jnp.nanmean`` reductions downstream
        do not see ``NaN + 2*(NaN-c)`` chain-rule poisoning).  The backward
        instead returns ``NaN`` at rows where ``|vega| <= 1e-14`` *when
        upstream asks for a gradient there* and ``0`` when upstream is
        exactly zero (e.g. a NaN-aware reduction over ``iv``).

        Callers who need to filter low-vega points out of a training loss
        should compute vega separately (see :func:`_price_vega_d1d2_j`)
        and replace ``price`` at those rows with a detached BSM sentinel
        *before* calling this function — exactly as in the PyTorch case.
    """
    price_a = jnp.asarray(price, dtype=jnp.float64)
    S_a = jnp.asarray(S, dtype=jnp.float64)
    K_a = jnp.asarray(K, dtype=jnp.float64)
    t_a = jnp.asarray(t, dtype=jnp.float64)
    r_a = jnp.asarray(r, dtype=jnp.float64)
    q_a = jnp.zeros_like(r_a) if q is None else jnp.asarray(q, dtype=jnp.float64)

    is_call_bool = _flag_to_bool_jax(flag, price_a.size)
    # Cast to float64 0/1 so the flag can ride through ``custom_vjp`` as a
    # regular positional arg under ``jax.jit``.  Inside the inner solver we
    # recover the bool with ``> 0.5``.
    is_call_a = is_call_bool.astype(jnp.float64)

    price_a, S_a, K_a, t_a, r_a, q_a, is_call_a = jnp.broadcast_arrays(
        price_a, S_a, K_a, t_a, r_a, q_a, is_call_a
    )

    return _jackel_iv_jax_inner(price_a, S_a, K_a, t_a, r_a, q_a, is_call_a, model)


__all__ = [
    "implied_volatility_autograd_jax",
    "_price_vega_d1d2_j",
    "_bsm_price_j",
]
