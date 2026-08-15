# Differentiable Jäckel IV — Training-Loop Example

`fast_vollib.jackel` ships a differentiable wrapper around Jäckel's solver
that you can drop into a PyTorch or JAX training loop.  The forward pass
runs the full machine-precision solver; the backward pass applies the
**implicit function theorem** to the discounted Black-Scholes price
equation, giving exact gradients without back-propagating through the
branch-heavy Householder iterations:

$$
\frac{\partial \sigma}{\partial \mathrm{price}} = \frac{1}{\nu}, \qquad
\frac{\partial \sigma}{\partial \theta} = -\frac{1}{\nu}\,\frac{\partial \mathrm{price}_{\mathrm{model}}}{\partial \theta},
\quad \theta \in \{S, K, t, r, q\},
$$

where $\nu = \partial \mathrm{price}_{\mathrm{model}}/\partial \sigma$ is
vega.  Two contracts the wrapper enforces:

- **Invalid domain → NaN forward and gradient.** Below-intrinsic, zero or
  negative price, zero maturity, or zero spot/strike all produce `NaN`
  in the forward and propagate `NaN` gradients.
- **Low-vega upstream-aware gradient.** When $|\nu| \le 10^{-14}$ the
  identity $1/\nu$ is numerically singular, so the backward returns
  `NaN` if the upstream cotangent is non-zero and `0` if it is exactly
  zero (e.g. a `nansum` upstream).  The latter prevents the
  `0 * NaN = NaN` chain-rule poison that would otherwise contaminate
  valid rows.

---

## PyTorch — IV-loss training

The simplest pattern: train an MLP to predict $\sigma$ from
$(\log m, \log t)$ and supervise with implied-volatility labels.

```python
import torch
from fast_vollib.backends.torch_backend import _bsm_price_t
from fast_vollib.jackel import implied_volatility_autograd


def make_chain(n: int, dtype=torch.float64):
    log_m = torch.empty(n, dtype=dtype).uniform_(-0.5, 0.5)
    s = torch.full((n,), 100.0, dtype=dtype)
    k = s * torch.exp(log_m)
    t = torch.empty(n, dtype=dtype).uniform_(0.1, 1.5)
    r = torch.full((n,), 0.02, dtype=dtype)
    q = torch.zeros_like(r)
    is_call = torch.rand(n) > 0.5
    sigma_star = 0.20 + 0.10 * torch.sqrt((log_m + 0.05) ** 2 + 0.04)
    price_star = _bsm_price_t(is_call, s, k, t, r, sigma_star, q).detach()
    return is_call, s, k, t, r, q, sigma_star, price_star


is_call, s, k, t, r, q, sigma_star, price_star = make_chain(2048)

mlp = torch.nn.Sequential(
    torch.nn.Linear(2, 64), torch.nn.SiLU(),
    torch.nn.Linear(64, 64), torch.nn.SiLU(),
    torch.nn.Linear(64, 1), torch.nn.Softplus(),
).double()

opt = torch.optim.AdamW(mlp.parameters(), lr=3e-3)
log_m = torch.log(k / s)
features = torch.stack([log_m, torch.log(t)], dim=-1)

for step in range(500):
    opt.zero_grad(set_to_none=True)
    sigma_hat = 0.02 + mlp(features).squeeze(-1)
    iv_label = implied_volatility_autograd(
        price_star, s, k, t, r, is_call, q=q, model="black_scholes"
    )
    loss = torch.mean((sigma_hat - iv_label) ** 2)
    loss.backward()
    opt.step()
```

The labels `iv_label` are produced by the Jäckel solver in the forward
pass; the training signal flows through `sigma_hat`'s parameters as
usual — the autograd wrapper only matters when **prices** are themselves
differentiable (see hybrid loss below).

---

## PyTorch — hybrid `price-loss + IV-roundtrip` (low-vega filter required!)

A common pattern is to combine a price-space loss with an IV-roundtrip
regulariser:

$$
\mathcal{L} = \underbrace{\Vert \mathrm{price}(\sigma_\theta) - \mathrm{price}^\star \Vert^2}_{\text{price-loss}}
  + \lambda \underbrace{\Vert \mathrm{IV}(\mathrm{price}(\sigma_\theta)) - \sigma^\star \Vert^2}_{\text{IV consistency}}
$$

The IV consistency term needs the **caller-side low-vega filter**:
without it, low-vega rows produce `NaN` gradients that the chain rule
turns into `NaN` weight updates.

```python
from fast_vollib.backends.torch_backend import _price_vega_d1d2_t

# Produce sentinel prices that round-trip cleanly to sigma_star.
with torch.no_grad():
    _, vega, _, _ = _price_vega_d1d2_t(is_call, s, k, t, r, sigma_hat, q)
    low_vega = vega.abs() <= 1e-6
    sentinel = _bsm_price_t(is_call, s, k, t, r, sigma_star, q)

price_hat = _bsm_price_t(is_call, s, k, t, r, sigma_hat, q)
price_clean = torch.where(low_vega, sentinel, price_hat)

iv_roundtrip = implied_volatility_autograd(
    price_clean, s, k, t, r, is_call, q=q, model="black_scholes"
)
loss = (
    torch.mean((price_hat - price_star) ** 2)
    + 0.1 * torch.mean((iv_roundtrip - sigma_star) ** 2)
)
loss.backward()
```

The `torch.where(low_vega, sentinel, price_hat)` line is the educational
point — it replaces low-vega entries with a price whose IV is exactly
$\sigma^\star$, so the squared-error term contributes nothing at those
rows, and no NaN gradient is produced.  The `_bsm_price_t` call is
detached automatically because it's inside a `torch.no_grad()` block.

---

## JAX — same contract, `custom_vjp` backend

The JAX entry point mirrors the PyTorch API exactly:

```python
import jax
import jax.numpy as jnp
from fast_vollib.jackel.differentiable_jax import (
    implied_volatility_autograd_jax,
    _price_vega_d1d2_j,
    _bsm_price_j,
)


def loss_fn(sigma_theta_params, features, ...):
    sigma_hat = mlp_apply(sigma_theta_params, features)
    price_hat = _bsm_price_j(is_call, s, k, t, r, sigma_hat, q)

    # Caller-side low-vega filter — same idea as the PyTorch version.
    _, vega, _, _ = _price_vega_d1d2_j(is_call, s, k, t, r, sigma_hat, q)
    low_vega = jnp.abs(vega) <= 1e-6
    sentinel = jax.lax.stop_gradient(
        _bsm_price_j(is_call, s, k, t, r, sigma_star, q)
    )
    price_clean = jnp.where(low_vega, sentinel, price_hat)

    iv_roundtrip = implied_volatility_autograd_jax(
        price_clean, s, k, t, r, is_call, q=q, model="black_scholes"
    )
    return (
        jnp.mean((price_hat - price_star) ** 2)
        + 0.1 * jnp.mean((iv_roundtrip - sigma_star) ** 2)
    )


grad_loss = jax.jit(jax.grad(loss_fn))
```

The JAX backward is wired through `jax.custom_vjp`.  The flag tensor is
internally cast to `float64` 0/1 so the function survives `jax.jit` —
JAX forbids tracer-valued non-differentiable arguments, so the bool
must ride through as a regular positional argument with a zero
cotangent.  This is invisible to callers.

---

## Validation

The implicit identity is exercised by 26 PyTorch tests
(`tests/test_jackel/test_autograd.py`) and 27 JAX tests
(`tests/test_jackel/test_autograd_jax.py`) covering:

- 3 models × 6 inputs (`price, s, k, t, r, q`) on a well-conditioned chain
- Heterogeneous F/T batches and homogeneous-flag batches
- Invalid-domain (below-intrinsic, $\le 0$ price, $t = 0$) returns NaN
- Low-vega gradient masking with both `nansum` and plain `sum` upstream
- A JIT regression guard for the `custom_vjp`/`is_call` tracer trap

The implicit identities are checked at `rtol = 5 \times 10^{-8}` and
`atol = 5 \times 10^{-10}`, which is much sharper than a finite-difference
sanity check.  Finite differences alone can hide sign errors that the
analytic identity catches — this is documented in
[`autoresearch-iv/neurips_plan.md`](https://github.com/raeidsaqur/autoresearch-iv/blob/main/neurips_plan.md).
