from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from fast_vollib.backends.torch_backend import _bsm_price_t, _price_vega_d1d2_t
from fast_vollib.jackel.differentiable import implied_volatility_autograd


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
