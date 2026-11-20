from __future__ import annotations

import numpy as np
import torch

from ..backends.torch_backend import _price_vega_d1d2_t
from ..types import ModelLiteral, OnErrorLiteral
from ..utils.broadcast import preprocess_flags
from ..utils.validation import ensure_on_error
from . import _policy
from .torch_backend import jackel_iv_black_torch

_INV_SQRT2 = 1.0 / np.sqrt(2.0)


def _ndtr(x: torch.Tensor) -> torch.Tensor:
    """Normal CDF via erfc. ``torch.special.ndtr`` loses relative accuracy in the lower tail
    (about 1e-10 at -6, zero by -10), which made the repricing check reject correct volatilities
    of low-volatility rows."""
    return 0.5 * torch.special.erfc(-x * _INV_SQRT2)


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


def _jackel_otm_torch(tv, fo, ko, t):
    """Jäckel on the out-of-the-money call (fo <= ko) at time value tv (native torch)."""
    return jackel_iv_black_torch(tv, fo, ko, t, True)


class _JackelImpliedVolatilityFunction(torch.autograd.Function):  # type: ignore[name-defined]
    """Carries the policy-checked forward solution; backward applies the implicit-function rule."""

    @staticmethod
    def forward(ctx, price, s, k, t, r, q, is_call, model: str, sigma_out, sigma_backward, invalid):
        ctx.model = model
        ctx.invalid = invalid
        ctx.save_for_backward(s, k, t, r, q, is_call, sigma_backward)
        return sigma_out.clone()

    @staticmethod
    def backward(ctx, grad_output):
        import torch

        s, k, t, r, q, is_call, sigma = ctx.saved_tensors
        model = ctx.model
        invalid = ctx.invalid

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
            # Well-conditioned rows: sigma exists (not invalid-domain) and
            # vega is above the numerical-inversion threshold.  At other rows
            # the implicit identity 1/vega is undefined; return NaN when the
            # caller asks for a gradient and 0 when the upstream gradient is
            # already zero (e.g. ``torch.nansum(iv).backward()``).  This keeps
            # the NaN signal for ill-conditioning while preventing the
            # 0 * NaN = NaN chain rule from poisoning valid rows when the
            # caller already zeroed NaN rows.
            well_cond = (~invalid) & (vega.abs() > 1e-14)
            zero_upstream = grad_output == 0.0
            safe_vega = torch.where(well_cond, vega, torch.ones_like(vega))
            raw_seed = -grad_output / safe_vega
            nan_filler = torch.full_like(grad_output, float("nan"))
            zero_filler = torch.zeros_like(grad_output)
            implicit_seed = torch.where(
                well_cond,
                raw_seed,
                torch.where(zero_upstream, zero_filler, nan_filler),
            )
            grads = torch.autograd.grad(
                model_price,
                grad_inputs,
                grad_outputs=implicit_seed,
                allow_unused=True,
            )

        if ctx.needs_input_grad[0]:
            raw_grad_price = grad_output / safe_vega
            grad_price = torch.where(
                well_cond,
                raw_grad_price,
                torch.where(zero_upstream, zero_filler, nan_filler),
            )
        else:
            grad_price = None
        grad_s = grads[0] if ctx.needs_input_grad[1] else None
        grad_k = grads[1] if ctx.needs_input_grad[2] else None
        grad_t = grads[2] if ctx.needs_input_grad[3] else None
        grad_r = grads[3] if ctx.needs_input_grad[4] else None
        grad_q = None
        if model == "black_scholes_merton" and ctx.needs_input_grad[5]:
            grad_q = grads[4]

        return grad_price, grad_s, grad_k, grad_t, grad_r, grad_q, None, None, None, None, None


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
    on_error: OnErrorLiteral = "warn",
) -> "torch.Tensor":
    """Differentiable Jäckel implied volatility for PyTorch.

    The forward pass evaluates the Jäckel ``Let's Be Rational`` solver.  The
    backward pass does not differentiate through the branch-heavy solver; it
    applies the implicit function theorem to the discounted Black-Scholes price:
    ``d sigma / d price = 1 / vega`` and
    ``d sigma / d theta = -(d price_model / d theta) / vega``.

    Row policy (shared with every implied-volatility route, see ``jackel._policy``):
        Quotes exactly at intrinsic and ``t == 0`` return ``NaN`` here (the functional routes
        return 0.0 there; no gradient exists), without a signal. Zero prices, prices below
        intrinsic or at/above the upper bound, invalid inputs, normalised prices below the
        smallest normal double, and solutions that do not reproduce the quote return ``NaN``
        with one aggregated ``on_error`` signal per call (``"warn"``, ``"raise"`` or
        ``"ignore"``). All of these rows propagate ``NaN`` gradients (0 where the upstream
        gradient is exactly zero).

    Low-vega contract:
        The forward pass returns the solver's best estimate at low-vega
        inputs without NaN-masking (so that ``torch.nanmean`` reductions
        downstream do not see a NaN + 2*(NaN-c) chain-rule poison).  The
        backward instead NaN-masks the gradient at rows where
        ``|vega| <= 1e-14`` when upstream asks for a gradient there; when
        upstream is exactly zero (e.g. a NaN-aware reduction over
        ``iv``) the returned gradient is ``0`` to keep the backward
        numerically clean.  Callers who need to filter low-vega points out
        of a training loss should compute vega directly (e.g. via
        ``fast_vollib.backends.torch_backend._price_vega_d1d2_t``) and
        mask ``price`` with a detached replacement *before* calling this
        function.
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
    ensure_on_error(on_error)
    if model not in ("black", "black_scholes", "black_scholes_merton"):
        raise ValueError(f"Unsupported model: {model!r}")
    with torch.no_grad():
        q_model = q_t if model == "black_scholes_merton" else torch.zeros_like(r_t)
        sigma_out, invalid, code = _policy.solve_native(
            torch,
            _ndtr,
            _jackel_otm_torch,
            model,
            price_t.detach(),
            s_t.detach(),
            k_t.detach(),
            t_t.detach(),
            r_t.detach(),
            q_model.detach(),
            is_call_t,
            zero_vol_value=float("nan"),  # no gradient exists there; NaN as in v0.2.4
        )
        sigma_backward = torch.where(invalid, torch.full_like(sigma_out, 0.1), sigma_out)
    _policy.report_codes(code.cpu(), on_error)
    return _JackelImpliedVolatilityFunction.apply(
        price_t, s_t, k_t, t_t, r_t, q_t, is_call_t, model, sigma_out, sigma_backward, invalid
    )


__all__ = ["implied_volatility_autograd"]
