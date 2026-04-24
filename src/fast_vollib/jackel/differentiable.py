from __future__ import annotations

import numpy as np
import torch

from ..backends.torch_backend import _price_vega_d1d2_t
from ..types import ModelLiteral
from ..utils.broadcast import preprocess_flags
from .torch_backend import jackel_iv_black_torch


def _as_tensor_like(value, ref: "torch.Tensor") -> "torch.Tensor":
    import torch

    if isinstance(value, torch.Tensor):
        return value.to(device=ref.device, dtype=ref.dtype)
    return torch.as_tensor(value, dtype=ref.dtype, device=ref.device)


def _flag_to_bool_tensor(flag, ref: "torch.Tensor") -> "torch.Tensor":
    import torch

    if isinstance(flag, torch.Tensor):
        if flag.dtype == torch.bool:
            return flag.to(device=ref.device)
        return flag.to(device=ref.device) > 0
    if isinstance(flag, (bool, np.bool_)):
        return torch.full_like(ref, bool(flag), dtype=torch.bool)
    flags = preprocess_flags(flag)
    return torch.as_tensor(flags == "c", dtype=torch.bool, device=ref.device)


class _JackelImpliedVolatilityFunction(torch.autograd.Function):  # type: ignore[name-defined]
    @staticmethod
    def forward(ctx, price, s, k, t, r, q, is_call, model: str):
        import torch

        with torch.no_grad():
            if model == "black":
                forward = s
            elif model == "black_scholes":
                forward = s * torch.exp(r * t)
            elif model == "black_scholes_merton":
                forward = s * torch.exp((r - q) * t)
            else:
                raise ValueError(f"Unsupported model: {model!r}")

            undiscounted_price = price * torch.exp(r * t)
            sigma = jackel_iv_black_torch(undiscounted_price, forward, k, t, is_call)

        ctx.model = model
        ctx.save_for_backward(s, k, t, r, q, is_call, sigma)
        return sigma

    @staticmethod
    def backward(ctx, grad_output):
        import torch

        s, k, t, r, q, is_call, sigma = ctx.saved_tensors
        model = ctx.model

        with torch.enable_grad():
            s_leaf = s.detach().requires_grad_(True)
            k_leaf = k.detach().requires_grad_(True)
            t_leaf = t.detach().requires_grad_(True)
            r_leaf = r.detach().requires_grad_(True)
            q_leaf = q.detach().requires_grad_(model == "black_scholes_merton")
            sigma_leaf = sigma.detach().requires_grad_(True)

            if model == "black":
                q_for_price = r_leaf
                grad_inputs = [s_leaf, k_leaf, t_leaf, r_leaf]
            elif model == "black_scholes":
                q_for_price = torch.zeros_like(r_leaf)
                grad_inputs = [s_leaf, k_leaf, t_leaf, r_leaf]
            else:
                q_for_price = q_leaf
                grad_inputs = [s_leaf, k_leaf, t_leaf, r_leaf, q_leaf]

            model_price, vega, _, _ = _price_vega_d1d2_t(
                is_call, s_leaf, k_leaf, t_leaf, r_leaf, sigma_leaf, q_for_price
            )
            safe_vega = torch.where(
                vega.abs() > 1e-14,
                vega,
                torch.full_like(vega, float("nan")),
            )
            implicit_seed = -grad_output / safe_vega
            grads = torch.autograd.grad(
                model_price,
                grad_inputs,
                grad_outputs=implicit_seed,
                allow_unused=True,
            )

        grad_price = grad_output / safe_vega if ctx.needs_input_grad[0] else None
        grad_s = grads[0] if ctx.needs_input_grad[1] else None
        grad_k = grads[1] if ctx.needs_input_grad[2] else None
        grad_t = grads[2] if ctx.needs_input_grad[3] else None
        grad_r = grads[3] if ctx.needs_input_grad[4] else None
        grad_q = None
        if model == "black_scholes_merton" and ctx.needs_input_grad[5]:
            grad_q = grads[4]

        return grad_price, grad_s, grad_k, grad_t, grad_r, grad_q, None, None


def implied_volatility_autograd(
    price,
    S,
    K,
    t,
    r,
    flag,
    q=None,
    *,
    model: ModelLiteral = "black_scholes",
) -> "torch.Tensor":
    """Differentiable Jäckel implied volatility for PyTorch.

    The forward pass evaluates the Jäckel ``Let's Be Rational`` solver.  The
    backward pass does not differentiate through the branch-heavy solver; it
    applies the implicit function theorem to the discounted Black-Scholes price:
    ``d sigma / d price = 1 / vega`` and
    ``d sigma / d theta = -(d price_model / d theta) / vega``.
    """
    import torch

    if not isinstance(price, torch.Tensor):
        price_t = torch.as_tensor(price, dtype=torch.float64)
    else:
        price_t = price

    s_t = _as_tensor_like(S, price_t)
    k_t = _as_tensor_like(K, price_t)
    t_t = _as_tensor_like(t, price_t)
    r_t = _as_tensor_like(r, price_t)
    q_t = torch.zeros_like(r_t) if q is None else _as_tensor_like(q, price_t)
    is_call_t = _flag_to_bool_tensor(flag, price_t)

    price_t, s_t, k_t, t_t, r_t, q_t, is_call_t = torch.broadcast_tensors(
        price_t, s_t, k_t, t_t, r_t, q_t, is_call_t
    )
    return _JackelImpliedVolatilityFunction.apply(
        price_t, s_t, k_t, t_t, r_t, q_t, is_call_t, model
    )


__all__ = ["implied_volatility_autograd"]
