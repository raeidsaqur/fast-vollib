from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from fast_vollib.backends.torch_backend import _bsm_price_t, _price_vega_d1d2_t
from fast_vollib.jackel.differentiable import implied_volatility_autograd

# ---------------------------------------------------------------------------
# Well-conditioned test chains.  Low-vega/invalid-domain tests live below.
# ---------------------------------------------------------------------------


def _well_conditioned_chain(dtype=torch.float64):
    """Return a well-conditioned option chain where vega is bounded away from zero.

    Moneyness is bounded to K/S in [0.8, 1.2], maturity in [0.25, 2.0], and
    sigma in [0.15, 0.5].  This keeps the implicit-gradient identity numerically
    sharp and separates correctness from low-vega amplification.
    """
    is_call = torch.tensor([True, False, True, False, True, False], dtype=torch.bool)
    s = torch.tensor([100.0, 100.0, 95.0, 105.0, 100.0, 100.0], dtype=dtype)
    k = torch.tensor([100.0, 100.0, 105.0, 95.0, 85.0, 115.0], dtype=dtype)
    t = torch.tensor([0.25, 0.5, 0.75, 1.0, 1.5, 2.0], dtype=dtype)
    r = torch.tensor([0.01, 0.02, 0.015, 0.025, 0.018, 0.012], dtype=dtype)
    q = torch.tensor([0.00, 0.01, 0.02, 0.005, 0.015, 0.0], dtype=dtype)
    sigma = torch.tensor([0.20, 0.35, 0.28, 0.22, 0.31, 0.25], dtype=dtype)
    return is_call, s, k, t, r, q, sigma


def _forward_price(model: str, is_call, s, k, t, r, q, sigma):
    """Discounted BSM price matching the parametrization used by the backward.

    model="black"        → q=r (no dividend; discounting via q=r reproduces
                            the Black-76 forward-to-spot discounted price).
    model="black_scholes" → q=0.
    model="black_scholes_merton" → q as supplied.
    """
    if model == "black":
        q_for_price = r
    elif model == "black_scholes":
        q_for_price = torch.zeros_like(r)
    else:
        q_for_price = q
    return _bsm_price_t(is_call, s, k, t, r, sigma, q_for_price)


def _model_price_and_vega(model: str, is_call, s, k, t, r, q, sigma):
    if model == "black":
        q_for_price = r
    elif model == "black_scholes":
        q_for_price = torch.zeros_like(r)
    else:
        q_for_price = q
    price, vega, _, _ = _price_vega_d1d2_t(is_call, s, k, t, r, sigma, q_for_price)
    return price, vega, q_for_price


def _expected_implicit_grad(model, is_call, s, k, t, r, q, sigma, param: str):
    """Return -(d price_model / d param) / vega under the same parametrization
    that the backward uses.  ``param`` selects which leaf is differentiated.
    """
    s_leaf = s.detach().requires_grad_(True)
    k_leaf = k.detach().requires_grad_(True)
    t_leaf = t.detach().requires_grad_(True)
    r_leaf = r.detach().requires_grad_(True)
    q_leaf = q.detach().requires_grad_(model == "black_scholes_merton")

    if model == "black":
        q_for_price = r_leaf
    elif model == "black_scholes":
        q_for_price = torch.zeros_like(r_leaf)
    else:
        q_for_price = q_leaf

    price_model, vega, _, _ = _price_vega_d1d2_t(
        is_call, s_leaf, k_leaf, t_leaf, r_leaf, sigma, q_for_price
    )

    leaf = {"s": s_leaf, "k": k_leaf, "t": t_leaf, "r": r_leaf, "q": q_leaf}[param]
    grad_price = torch.autograd.grad(
        price_model, leaf, torch.ones_like(price_model), allow_unused=True
    )[0]
    if grad_price is None:
        return torch.zeros_like(vega)
    return -grad_price / vega


# ---------------------------------------------------------------------------
# Price gradient:  d sigma / d price  =  1 / vega
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model", ["black", "black_scholes", "black_scholes_merton"])
def test_autograd_price_gradient_matches_inverse_vega(model: str) -> None:
    is_call, s, k, t, r, q, true_sigma = _well_conditioned_chain()
    price_model, vega, _ = _model_price_and_vega(model, is_call, s, k, t, r, q, true_sigma)
    price = price_model.detach().clone().requires_grad_(True)

    iv = implied_volatility_autograd(price, s, k, t, r, is_call, q=q, model=model)
    assert torch.allclose(iv, true_sigma, rtol=5e-9, atol=5e-10)

    iv.sum().backward()
    assert torch.allclose(price.grad, 1.0 / vega, rtol=1e-8, atol=1e-10)


# ---------------------------------------------------------------------------
# Implicit-function identities for each structural input.
# Parametrized by (model, parameter).  Calls-and-puts are mixed in the chain.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model", ["black", "black_scholes", "black_scholes_merton"])
@pytest.mark.parametrize("param", ["s", "k", "t", "r"])
def test_autograd_implicit_input_gradient(model: str, param: str) -> None:
    is_call, s, k, t, r, q, true_sigma = _well_conditioned_chain()
    fixed_price = _forward_price(model, is_call, s, k, t, r, q, true_sigma).detach()

    s_in = s.detach().requires_grad_(param == "s")
    k_in = k.detach().requires_grad_(param == "k")
    t_in = t.detach().requires_grad_(param == "t")
    r_in = r.detach().requires_grad_(param == "r")
    q_in = q.detach()

    iv = implied_volatility_autograd(
        fixed_price, s_in, k_in, t_in, r_in, is_call, q=q_in, model=model
    )
    iv.sum().backward()

    expected = _expected_implicit_grad(model, is_call, s, k, t, r, q, true_sigma, param=param)
    got = {"s": s_in.grad, "k": k_in.grad, "t": t_in.grad, "r": r_in.grad}[param]
    assert got is not None, f"{model}/{param}: grad was None"
    assert torch.allclose(got, expected, rtol=5e-8, atol=5e-10), (
        f"{model}/{param}: got={got}\nexpected={expected}\n"
        f"max_abs_err={(got - expected).abs().max().item():.3e}"
    )


def test_autograd_implicit_dividend_gradient_bsm_only() -> None:
    """q gradient is only defined for the black_scholes_merton model."""
    is_call, s, k, t, r, q, true_sigma = _well_conditioned_chain()
    fixed_price = _forward_price(
        "black_scholes_merton", is_call, s, k, t, r, q, true_sigma
    ).detach()

    q_in = q.detach().requires_grad_(True)
    iv = implied_volatility_autograd(
        fixed_price, s, k, t, r, is_call, q=q_in, model="black_scholes_merton"
    )
    iv.sum().backward()

    expected = _expected_implicit_grad(
        "black_scholes_merton", is_call, s, k, t, r, q, true_sigma, param="q"
    )
    assert q_in.grad is not None
    assert torch.allclose(q_in.grad, expected, rtol=5e-8, atol=5e-10)


# ---------------------------------------------------------------------------
# Separate call-only and put-only homogeneous batches.
# The mixed-flag chain above already exercises both branches, but a homogeneous
# batch guards against the torch.where dispatch silently collapsing one branch.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("flag_value", [True, False], ids=["calls_only", "puts_only"])
def test_autograd_homogeneous_flag_batch(flag_value: bool) -> None:
    dtype = torch.float64
    n = 5
    is_call = torch.full((n,), flag_value, dtype=torch.bool)
    s = torch.linspace(95.0, 105.0, n, dtype=dtype)
    k = torch.tensor([90.0, 95.0, 100.0, 105.0, 110.0], dtype=dtype)
    t = torch.tensor([0.5, 0.75, 1.0, 1.25, 1.5], dtype=dtype)
    r = torch.full((n,), 0.02, dtype=dtype)
    q = torch.zeros_like(r)
    true_sigma = torch.tensor([0.22, 0.26, 0.30, 0.24, 0.28], dtype=dtype)

    price = _bsm_price_t(is_call, s, k, t, r, true_sigma, q).detach().requires_grad_(True)
    s_in = s.detach().requires_grad_(True)

    iv = implied_volatility_autograd(price, s_in, k, t, r, is_call, q=q, model="black_scholes")
    assert torch.allclose(iv, true_sigma, rtol=5e-9, atol=5e-10)
    iv.sum().backward()

    _, vega, _, _ = _price_vega_d1d2_t(is_call, s, k, t, r, true_sigma, q)
    assert torch.allclose(price.grad, 1.0 / vega, rtol=1e-8, atol=1e-10)

    expected_s = _expected_implicit_grad(
        "black_scholes", is_call, s, k, t, r, q, true_sigma, param="s"
    )
    assert torch.allclose(s_in.grad, expected_s, rtol=5e-8, atol=5e-10)


# ---------------------------------------------------------------------------
# Heterogeneous maturity-and-forward batch.  Verifies the tensor F/T path.
# ---------------------------------------------------------------------------


def test_autograd_heterogeneous_maturity_and_spot_batch() -> None:
    dtype = torch.float64
    is_call = torch.tensor([True, False, True, False, True, False, True], dtype=torch.bool)
    s = torch.tensor([80.0, 100.0, 120.0, 90.0, 110.0, 95.0, 105.0], dtype=dtype)
    k = torch.tensor([85.0, 100.0, 115.0, 92.0, 108.0, 100.0, 100.0], dtype=dtype)
    t = torch.tensor([0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0], dtype=dtype)
    r = torch.tensor([0.01, 0.02, 0.03, 0.015, 0.025, 0.02, 0.018], dtype=dtype)
    q = torch.zeros_like(r)
    true_sigma = torch.tensor([0.40, 0.30, 0.22, 0.28, 0.26, 0.24, 0.32], dtype=dtype)

    fixed_price = _bsm_price_t(is_call, s, k, t, r, true_sigma, q).detach()
    t_in = t.detach().requires_grad_(True)
    r_in = r.detach().requires_grad_(True)

    iv = implied_volatility_autograd(
        fixed_price, s, k, t_in, r_in, is_call, q=q, model="black_scholes"
    )
    assert torch.allclose(iv, true_sigma, rtol=5e-9, atol=5e-10)

    iv.sum().backward()
    expected_t = _expected_implicit_grad(
        "black_scholes", is_call, s, k, t, r, q, true_sigma, param="t"
    )
    expected_r = _expected_implicit_grad(
        "black_scholes", is_call, s, k, t, r, q, true_sigma, param="r"
    )
    assert torch.allclose(t_in.grad, expected_t, rtol=5e-8, atol=5e-10)
    assert torch.allclose(r_in.grad, expected_r, rtol=5e-8, atol=5e-10)


# ---------------------------------------------------------------------------
# Backward-compatibility tests: retain the original black_scholes cases
# so existing behaviour remains pinned.
# ---------------------------------------------------------------------------


def test_jackel_autograd_price_gradient_matches_inverse_vega() -> None:
    dtype = torch.float64
    is_call = torch.tensor([True, True, False], dtype=torch.bool)
    s = torch.tensor([100.0, 105.0, 95.0], dtype=dtype)
    k = torch.tensor([100.0, 100.0, 100.0], dtype=dtype)
    t = torch.tensor([0.5, 1.0, 1.5], dtype=dtype)
    r = torch.tensor([0.01, 0.02, 0.015], dtype=dtype)
    q = torch.zeros_like(r)
    true_sigma = torch.tensor([0.2, 0.35, 0.28], dtype=dtype)

    price = _bsm_price_t(is_call, s, k, t, r, true_sigma, q).detach().requires_grad_(True)
    iv = implied_volatility_autograd(price, s, k, t, r, is_call, model="black_scholes")
    assert torch.allclose(iv, true_sigma, rtol=5e-9, atol=5e-10)

    iv.sum().backward()
    _, vega, _, _ = _price_vega_d1d2_t(is_call, s, k, t, r, true_sigma, q)
    assert torch.allclose(price.grad, 1.0 / vega, rtol=5e-8, atol=5e-10)


def test_jackel_autograd_implicit_spot_gradient_matches_black_scholes() -> None:
    dtype = torch.float64
    is_call = torch.tensor([True, False], dtype=torch.bool)
    s = torch.tensor([100.0, 95.0], dtype=dtype, requires_grad=True)
    k = torch.tensor([102.0, 100.0], dtype=dtype)
    t = torch.tensor([0.75, 1.25], dtype=dtype)
    r = torch.tensor([0.01, 0.015], dtype=dtype)
    q = torch.zeros_like(r)
    true_sigma = torch.tensor([0.24, 0.31], dtype=dtype)
    fixed_price = _bsm_price_t(is_call, s.detach(), k, t, r, true_sigma, q).detach()

    iv = implied_volatility_autograd(fixed_price, s, k, t, r, is_call, model="black_scholes")
    iv.sum().backward()

    s_ref = s.detach().requires_grad_(True)
    model_price, vega, _, _ = _price_vega_d1d2_t(is_call, s_ref, k, t, r, true_sigma, q)
    price_spot_grad = torch.autograd.grad(model_price, s_ref, torch.ones_like(model_price))[0]
    expected = -price_spot_grad / vega
    assert torch.allclose(s.grad, expected, rtol=5e-8, atol=5e-10)


# ---------------------------------------------------------------------------
# Invalid-domain and low-vega contract.
#
# The differentiable path should NaN-mask (a) prices below the discounted
# intrinsic, (b) non-positive price, s, k, or t, and (c) gradients where
# |vega| <= 1e-14.  Valid points interleaved with invalid ones must still
# produce correct sigma and 1/vega gradients.
# ---------------------------------------------------------------------------


def _is_nan(x: "torch.Tensor") -> "torch.Tensor":
    return x.isnan()


def test_invalid_domain_below_intrinsic_returns_nan_iv() -> None:
    dtype = torch.float64
    # Deep-ITM call whose quoted price is below the discounted intrinsic.
    s = torch.tensor([150.0, 100.0], dtype=dtype)
    k = torch.tensor([100.0, 150.0], dtype=dtype)
    t = torch.tensor([1.0, 1.0], dtype=dtype)
    r = torch.tensor([0.02, 0.02], dtype=dtype)
    is_call = torch.tensor([True, False], dtype=torch.bool)
    # Below-intrinsic prices for both sides.
    price = torch.tensor([45.0, 45.0], dtype=dtype, requires_grad=True)

    iv = implied_volatility_autograd(price, s, k, t, r, is_call, model="black_scholes")
    assert torch.all(_is_nan(iv)), f"below-intrinsic iv should be NaN, got {iv}"
    iv.sum().backward()
    assert torch.all(_is_nan(price.grad))


def test_invalid_domain_zero_and_negative_price_returns_nan() -> None:
    dtype = torch.float64
    s = torch.tensor([100.0, 100.0], dtype=dtype)
    k = torch.tensor([100.0, 100.0], dtype=dtype)
    t = torch.tensor([1.0, 1.0], dtype=dtype)
    r = torch.tensor([0.02, 0.02], dtype=dtype)
    is_call = torch.tensor([True, True], dtype=torch.bool)
    price = torch.tensor([0.0, -1.0], dtype=dtype, requires_grad=True)

    iv = implied_volatility_autograd(price, s, k, t, r, is_call, model="black_scholes")
    assert torch.all(_is_nan(iv))
    iv.sum().backward()
    assert torch.all(_is_nan(price.grad))


def test_invalid_domain_zero_maturity_returns_nan() -> None:
    dtype = torch.float64
    s = torch.tensor([100.0], dtype=dtype)
    k = torch.tensor([100.0], dtype=dtype)
    t = torch.tensor([0.0], dtype=dtype)
    r = torch.tensor([0.02], dtype=dtype)
    is_call = torch.tensor([True], dtype=torch.bool)
    price = torch.tensor([1.0], dtype=dtype, requires_grad=True)

    iv = implied_volatility_autograd(price, s, k, t, r, is_call, model="black_scholes")
    assert torch.all(_is_nan(iv))
    iv.sum().backward()
    assert torch.all(_is_nan(price.grad))


def test_low_vega_gradient_masked_to_nan() -> None:
    """Deep-OTM short-maturity chain drives vega below the 1e-14 threshold.

    The implicit-gradient formula ``1/vega`` would otherwise explode; the
    backward must mask those entries with NaN so downstream training can
    filter them.
    """
    dtype = torch.float64
    is_call = torch.tensor([True], dtype=torch.bool)
    s = torch.tensor([100.0], dtype=dtype)
    k = torch.tensor([500.0], dtype=dtype)
    t = torch.tensor([0.01], dtype=dtype)
    r = torch.tensor([0.02], dtype=dtype)
    q = torch.zeros_like(r)
    true_sigma = torch.tensor([0.10], dtype=dtype)
    fixed_price = _bsm_price_t(is_call, s, k, t, r, true_sigma, q).detach()
    # sanity: this parameterization drives vega to machine zero
    _, vega, _, _ = _price_vega_d1d2_t(is_call, s, k, t, r, true_sigma, q)
    assert vega.abs().item() <= 1e-14, f"expected vega<=1e-14, got {vega}"

    price = fixed_price.detach().clone().requires_grad_(True)
    iv = implied_volatility_autograd(price, s, k, t, r, is_call, model="black_scholes")
    iv.sum().backward()
    assert torch.all(_is_nan(price.grad))


def test_mixed_valid_and_invalid_batch_does_not_contaminate_valid_entries() -> None:
    """A single invalid row must not corrupt gradients for valid rows.

    Two upstream scenarios are checked:
    1. ``iv.nansum()`` zero-outs NaN rows before calling backward.  The
       expected backward contract at those rows is a *clean zero* (so the
       reduction flows cleanly), not NaN — a NaN would poison valid rows
       via the 0 * NaN = NaN chain rule.
    2. ``iv.sum()`` asks for a nonzero gradient uniformly.  At invalid rows
       the implicit identity is undefined, so the returned gradient is NaN.
    """
    dtype = torch.float64
    is_call = torch.tensor([True, True, False, True], dtype=torch.bool)
    s = torch.tensor([100.0, 150.0, 100.0, 100.0], dtype=dtype)
    k = torch.tensor([100.0, 100.0, 100.0, 100.0], dtype=dtype)
    t = torch.tensor([1.0, 1.0, 1.0, 0.0], dtype=dtype)  # last row: zero T
    r = torch.tensor([0.02, 0.02, 0.02, 0.02], dtype=dtype)
    q = torch.zeros_like(r)
    true_sigma = torch.tensor([0.25, 0.30, 0.28, 0.20], dtype=dtype)
    valid = torch.tensor([True, False, True, False])

    def build_price():
        prices = _bsm_price_t(is_call, s, k, t.clamp(min=1e-6), r, true_sigma, q).detach()
        prices[1] = 5.0
        prices[3] = 1.0
        return prices.detach().requires_grad_(True)

    # Scenario 1: nansum upstream → clean zero at invalid rows.
    price = build_price()
    iv = implied_volatility_autograd(price, s, k, t, r, is_call, model="black_scholes")
    assert torch.allclose(iv[valid], true_sigma[valid], rtol=5e-9, atol=5e-10)
    assert torch.all(_is_nan(iv[~valid]))
    iv.nansum().backward()
    _, vega, _, _ = _price_vega_d1d2_t(is_call, s, k, t.clamp(min=1e-6), r, true_sigma, q)
    expected_valid = (1.0 / vega)[valid]
    got_valid = price.grad[valid]
    assert torch.all(torch.isfinite(got_valid))
    assert torch.allclose(got_valid, expected_valid, rtol=1e-8, atol=1e-10)
    assert torch.all(price.grad[~valid] == 0.0)

    # Scenario 2: plain sum upstream → NaN at invalid rows (ill-conditioned).
    price = build_price()
    iv = implied_volatility_autograd(price, s, k, t, r, is_call, model="black_scholes")
    # sum propagates NaN from invalid rows into the scalar loss; backward from
    # a NaN scalar still runs, but asserting on the scalar value is not the
    # point of this scenario.  We backprop a vector of ones directly.
    iv.backward(gradient=torch.ones_like(iv))
    got_valid = price.grad[valid]
    assert torch.all(torch.isfinite(got_valid))
    assert torch.allclose(got_valid, expected_valid, rtol=1e-8, atol=1e-10)
    assert torch.all(_is_nan(price.grad[~valid]))
