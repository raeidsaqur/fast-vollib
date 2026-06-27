# Surface Arbitrage-Evaluation Harness

`fast_vollib.surface` is a **generator-agnostic, backend-pluggable, differentiable**
evaluator for implied-volatility surfaces. It takes an *arbitrary* generated
surface on an *arbitrary* `(log-moneyness × maturity)` mesh and returns
calibrated, comparable arbitrage diagnostics — the packaged layer that
generative-surface work (VolGAN, deep-smoothing, VAE families) otherwise
re-derives inline, unnormalized, and tied to one parametrization.

It is **evaluation**, not construction: it *scores* a surface (and can repair
training via a soft penalty), rather than fitting an SSVI/eSSVI model.

## Why

- **Normalized, cross-model scores.** Every metric is dimensionless, so it
  compares across generators, meshes, and underlyings — unlike the raw scalar
  penalties reported in the literature.
- **One math path, two modes.** The same checks run (a) on CPU as an offline
  report and (b) **differentiably on-GPU** as an autograd penalty inside a
  training loop, because they route through the numpy/torch/jax backends.
- **Artifact-vs-arbitrage separation.** Violations whose stencil touches an
  interpolated node are bucketed separately from genuine model arbitrage.

## Quick start

```python
import numpy as np
from fast_vollib.surface import IVSurface, validate_surface

k = np.linspace(-0.4, 0.4, 21)            # forward log-moneyness, k = log(K/F)
T = np.array([0.1, 0.25, 0.5, 1.0])       # year-fractions to expiry
iv = np.full((k.size, T.size), 0.2)       # a flat, arbitrage-free surface

surf = IVSurface.from_logmoneyness(k, T, iv)
report = validate_surface(surf)
report.passed            # True
report.metrics           # {'ndm': 0.0, 'bfly_frac': 0.0, 'cal_frac': 0.0, ...}
report.sas               # 0.0  (Static-Arbitrage Score; 0 = clean)
```

### Constructors

| Constructor | Input |
|---|---|
| `IVSurface.from_logmoneyness(k, T, iv, forward=1.0, r=0.0, q=0.0)` | a shared forward-log-moneyness axis |
| `IVSurface.from_strikes(K, T, iv, spot=..., r=0.0, q=0.0)` | a shared strike vector + spot |
| `IVSurface.from_total_variance(k, T, w, ...)` | a total-variance grid `w = σ²T` |
| `IVSurface.from_call_prices(K, T, C, spot=..., ...)` | a call-price grid (inverted to IV) |

`SurfaceSequence([surf0, surf1, ...])` stacks frames of a surface evolving over
calendar time (the animation axis for the Part II UI).

## The conditions (design §4)

The harness runs two complementary families and reports them separately:

**Price-space discrete checks (primary, robust at the wings)** — convert the IV
grid to call prices and evaluate the model-free no-arbitrage inequalities
(Davis–Hobson 2007; Cousot 2007): call **convexity** (⇒ non-negative
risk-neutral density), the **slope/monotonicity** bound `−1 ≤ ∂c̃/∂K ≤ 0`, the
**price box**, and **calendar** monotonicity.

**Total-variance / IV-space checks (secondary, interpretable)** in `(k, T)` with
`w = σ²T` (Gatheral–Jacquier 2014; Roper 2010): **calendar** `∂_T w ≥ 0` and
**Durrleman's** `g(k) ≥ 0` (butterfly-free ⟺ valid density).

!!! note "Calendar coordinate"
    On a shared forward-log-moneyness grid (`IVSurface.shared_k is True`),
    calendar arbitrage is checked as `∂_T w ≥ 0`. On a fixed-strike grid under a
    term-varying forward, it is checked as undiscounted-call monotonicity at
    fixed strike — the coordinate-correct form there. `report.context["calendar_form"]`
    records which was used.

## Normalized metrics (design §5)

| Metric | Meaning | Range |
|---|---|---|
| `ndm` | integrated negative risk-neutral-density mass (max over slices) | `[0, 1]` |
| `bfly_frac` | fraction of interior strikes with Durrleman `g < −tol` | `[0, 1]` |
| `cal_depth_max` | max relative total-variance crossing depth | `[0, ∞)` |
| `cal_frac` | fraction of `(k, adjacent-T)` pairs that cross | `[0, 1]` |
| `vert_frac` | fraction of adjacent strike pairs breaking the slope bound | `[0, 1]` |
| `bound_frac` | fraction of nodes outside the price box | `[0, 1]` |
| `sas` | Static-Arbitrage Score — documented convex combination | `[0, 1]`, 0 = clean |

!!! warning "Always read the components"
    `sas` is reported **only alongside its components** — a single scalar hides
    *which* condition failed. The default weights are a modeling choice
    (`DEFAULT_SAS_WEIGHTS`, overridable via `weights=`); the principled
    normalization/weighting is itself an open research question.

`report.violations` is a list of localized `ArbitrageViolation`s
(`type`, `severity` ∈ {minor, moderate, severe}, normalized `value`, `location`,
and `origin` ∈ {native, interpolation_induced}). `report.trust_mask` is the
per-node round-trip `σ→C→σ'` LBR fixed-point mask (machine-tight where the quote
is well-posed).

`validate_surface(..., return_as="dict" | "json")` mirrors the rest of the
library's return conventions.

## Differentiable penalty (design §7)

The same checks become a single differentiable scalar suitable as a soft
no-arbitrage term in a generator's training loss — gradients flow back to `iv`:

```python
import torch
from fast_vollib.surface import arbitrage_penalty

iv = torch.tensor(generated_iv, requires_grad=True)   # (Nk, Nt)
loss = recon_loss + lam * arbitrage_penalty(iv, k, T, forward=1.0, r=0.0)
loss.backward()                                        # ∂penalty/∂iv flows
```

`arbitrage_penalty` stays in the input tensor's namespace (no host round-trip),
so it is autograd-traceable on torch and jax and matches the numpy report path
to machine precision. This is the reusable replacement for the inline penalty
functions the literature re-derives.

## Diagnostic figures (design §8)

`fast_vollib.diagnostics` provides the six publication-quality figures
(total-variance slices, Durrleman `g`, risk-neutral density with negative mass,
violation heatmap, calendar map, round-trip trust map). Matplotlib is gated
behind the `[viz]` extra and is **not** a core dependency:

```bash
pip install "fast-vollib[viz]"
```

```python
from fast_vollib.diagnostics import plot_durrleman_g, plot_density
fig = plot_durrleman_g(surf)         # matplotlib Figure; caller owns save/show
```

## References

Roper (2010); Gatheral–Jacquier (2014); Davis–Hobson (2007); Cousot (2007);
Fengler (2009); Breeden–Litzenberger (1978); Jäckel (2016, *Let's Be Rational*).
