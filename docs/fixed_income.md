# Fixed Income

A bond is not an option, and fast-vollib does not pretend otherwise. It gets a
third branch of the contract tree, its own dispatcher, and its own valuation
routine — because the thing that makes an option tractable, *one payment whose
size depends on the state at one horizon*, is exactly what a bond is not.

```python
import fast_vollib.rates as rates   # imports no torch, jax, numba, or triton
```

## Quick start

```python
from fast_vollib.instruments import ZeroCouponBond, cashflows
from fast_vollib.pricing import present_value
from fast_vollib.rates import CIRDiscountCurve, FlatDiscountCurve

bond = ZeroCouponBond(maturity=2.0, face_value=1000.0, currency="USD")

cashflows(bond)
# (Cashflow(payment_time=2.0, amount=1000.0),)

present_value(bond, discount_curve=FlatDiscountCurve(rate=0.03))
# 941.7645335842487

curve = CIRDiscountCurve(kappa=0.3, theta=0.04, volatility=0.1, initial_rate=0.04)
present_value(bond, discount_curve=curve)
# 923.4360545743322
```

A coupon bond against an observed curve:

```python
import numpy as np
from fast_vollib.instruments import FixedRateBond
from fast_vollib.rates import InterpolatedDiscountCurve

note = FixedRateBond(
    payment_times=(0.5, 1.0, 1.5, 2.0),
    accrual_fractions=(0.5, 0.5, 0.5, 0.5),
    coupon_rate=0.04,
    face_value=100.0,
)
observed = InterpolatedDiscountCurve(
    maturities=np.array([1.0, 2.0, 5.0]),
    discount_factors=np.array([0.97, 0.94, 0.85]),
)
present_value(note, discount_curve=observed)
# 101.6995359437322
```

## Why a bond is not routed through `payoff()`

`payoff(instrument, terminal_state)` maps the underlier's value at maturity to
one cashflow. That signature is what lets an engine simulate a contract, and it
cannot express "one unit at two years and one unit at five". Forcing a bond
through it would mean either inventing a terminal state the contract does not
have, or adding payments made at different dates as though they were worth the
same — and the second is not an approximation, it is an error.

So fixed-income securities expose `cashflows()` instead, and every entry point
that could otherwise return a plausible number refuses by name:

```python
from fast_vollib.instruments import payoff, price_instrument

payoff(bond, 100.0)
# UnsupportedInstrumentError: ZeroCouponBond has dated cashflows rather than a
# payoff: ... value it with fast_vollib.pricing.present_value(...).
```

`price_instrument`, `greeks_instrument`, `implied_volatility_instrument`,
`MonteCarloEngine.price`, and `EuropeanOptionBatch` all do the same.
`MonteCarloEngine.supports(ZeroCouponBond)` returns `False`, and
`capabilities(ZeroCouponBond)` reports `payoff=False`, `price=frozenset()`,
`cashflows=True`, `present_value=True`.

## Contracts

| Type | Terms | Cashflows |
|---|---|---|
| `ZeroCouponBond` | `maturity`, `face_value`, `currency`, `instrument_id` | one payment of `face_value` at `maturity` |
| `FixedRateBond` | `payment_times`, `accrual_fractions`, `coupon_rate`, `face_value`, `currency`, `instrument_id` | `face_value * coupon_rate * accrual_fractions[i]` at each `payment_times[i]`, with `face_value` added to the last |

`FixedRateBond` carries **no `frequency`**. A frequency cannot describe a stub,
a long first period, or an irregular redemption, so a contract holding one would
be unable to represent bonds that exist. The canonical terms are the schedule
itself; resolving dates, calendars, and day counts *into* that schedule belongs
to a conventions layer above this one, which is not implemented.

`accrual_fractions` is supplied rather than derived for the same reason.
The gap between two payment dates is not the accrual factor under 30/360,
ACT/365, or ACT/ACT, and the three disagree — deriving it would impose a
day-count convention the contract does not carry.

`face_value` rather than `Derivative.notional`: a notional is a contract
multiplier whose sign denotes a short position, and that meaning does not belong
on a primary security whose principal is redeemed.

A `maturity` of zero is a payment due now, not a degenerate case — it is the
last payment of a matured bond, and it is worth its face value under any curve.

Everything a calendar would decide is deliberately absent. `payment_time` is a
year fraction, never a date. Day counts, business-day rules, ex-coupon periods,
accrued interest, clean versus dirty price, and yield belong to a conventions
layer above this one and are not implemented.

## Curves

`DiscountCurve` is a structural protocol with one method. Any object providing
`discount_factor(maturity)` works, with no registration and no subclassing:

```python
class OurFirmCurve:
    def discount_factor(self, maturity):
        return ...

present_value(bond, discount_curve=OurFirmCurve())
```

| Curve | `P(0, t)` |
|---|---|
| `FlatDiscountCurve(rate)` | `exp(-rate * t)` |
| `CIRDiscountCurve(kappa, theta, volatility, initial_rate)` | the Cox–Ingersoll–Ross affine term structure |
| `InterpolatedDiscountCurve(maturities, discount_factors, extrapolation)` | log-linear in `P` between observed pillars |

All three are continuously compounded, matching `MonteCarloEngine._discount`
and `SurfaceMarket.discount_at`; the agreement is asserted by test rather than
assumed. All three return exactly `1.0` at a zero maturity — exactly, not to
within a tolerance.

`FlatDiscountCurve` accepts a negative rate, and then `P > 1`. That is a real
market condition rather than an input error.

The curve is always an argument. A `currency` on the contract is descriptive and
never selects one.

### Observed curves

`InterpolatedDiscountCurve` is a **container, not a bootstrapper**. It holds
factors somebody else produced — a stripped curve, a vendor file, a
calibration's output — and answers between them by one stated rule. It does not
build a curve from deposits and swaps.

Log-linear in `P`, which is linear in `r(T)·T`, evaluated as a convex
combination so a pillar's own log-factor comes back exactly rather than as a
rounding of itself.

**The origin is not a pillar.** `P(0,0) = 1` is a fact about discount factors
rather than an observation, so it anchors the short end: the region below the
first pillar is interpolation against it, and the zero rate there is flat at the
first pillar's — the usual market convention. That is why `discount_factor(0.0)`
is exactly `1.0` and `discount_factor(1e-9)` is a number rather than an error.
A pillar *at* zero is refused; it would be a second opinion about a fact.

`extrapolation` governs the far end only, and is never inferred:

| `extrapolation` | Past the last pillar |
|---|---|
| `"error"` (default) | raises `RateValidationError` |
| `"flat"` | holds the last pillar's **zero rate**, so `P` keeps decaying |

Holding the last *factor* instead would make `P` constant past the end, which
is a forward rate of zero — a strong statement about the market, made by
accident.

Factors are **not** required to be decreasing. A negative-rate curve
legitimately has `P > 1`, and refusing it would make the class unable to hold
the front end of several real government curves.

The whole curve is array-API native, so a bond's present value differentiates
back to the pillar factors it was read from — which is what a bucketed
sensitivity is.

#### Against `SurfaceMarket`

`fast_vollib.surface.SurfaceMarket` also interpolates a discount term structure,
**differently**: linearly in the zero rate `r(T)`, then `exp(-r(T)·T)`. This
curve is linear in `r(T)·T`. The two **agree at pillars and differ between
them**, by more the further apart the pillars and the more curved the rate.
Neither is more correct; they are different conventions, and this one is
canonical for fixed income. Both facts are asserted by test. This branch does
not change `SurfaceMarket`; reconciling the two is a separate change with its
own bitwise gates.

## The CIR term structure

Under the risk-neutral measure,

$$ dr_t = \kappa(\theta - r_t)\,dt + \sigma\sqrt{r_t}\,dW_t^{\mathbb{Q}} $$

and the bond price is affine in the state, $P(0,\tau) = A(\tau)e^{-B(\tau)r_0}$,
given in closed form by equation 23 of Cox, Ingersoll and Ross (1985).

`kappa` and `theta` are the **risk-neutral** values, with the market price of
rate risk already absorbed — which is what a bond price needs and what a
calibration to bond data recovers. Bakshi, Cao and Chen (1997) write the drift
as $\theta_R - \kappa_R R$, so their parameters map on as `kappa = kappa_R` and
`theta = theta_R / kappa_R`.

The Feller ratio $2\kappa\theta/\sigma^2$ is **reported and never enforced**:

```python
curve.feller_ratio      # 2.4
curve.satisfies_feller  # True
```

Calibrations violate it routinely, the bond formula is well defined either way,
and refusing those parameters would make the library unable to represent the
curves it exists to fit. CIR cannot produce a negative short rate; that is a
property of the model, stated rather than hidden.

### Stable evaluation of the published formula

`fast_vollib.rates.cir` implements an algebraic rearrangement, and the reason is
worth knowing before anyone "simplifies" it back.

Equation 23 as published contains $e^{\gamma\tau}$, which **overflows** past
$\gamma\tau \approx 709$ — a thirty-year bond reaches that at a vol-of-rate
of roughly 17 or more, depending on mean reversion. Factoring $e^{-\gamma\tau}$
through fixes the overflow and is where
most implementations stop. But it leaves $\gamma - \kappa$ to be formed by
subtraction, and as $\sigma \to 0$ those two agree to more and more digits: by
$\sigma = 10^{-8}$ the subtraction has **no significant digits left** and the
price is wrong in the third decimal place. A vanishing vol-of-rate is not
exotic — it is what a deterministic-rate limit, a calibration starting point,
and a finite-difference sensitivity all walk through.

The fix is to never form the difference. Since $\gamma^2 - \kappa^2 = 2\sigma^2$,

$$ \delta := \gamma - \kappa = \frac{2\sigma^2}{\gamma + \kappa}, $$

a quotient of positive quantities that avoids subtractive cancellation; the
affine base then reduces to $1/(1+x)$ with $x \to 0$, which is what `log1p` is
for, and the $\sigma^2$ in the exponent cancels analytically instead of
numerically.

Measured against `mpmath` at fifty digits over a table regenerated by
`scripts/generate_cir_reference_fixtures.py`, the shipped form holds the
exponent to under five units in the last place across the tested parameter
grid — $\sigma$ from $10^{-12}$ to 5, maturities to 500 years,
$\gamma\tau$ past 3500.

There is **no** small-$\sigma$ threshold and no asymptotic branch. The only
special case is $\sigma$ exactly zero, where the rate is deterministic and the
limit is written out directly.

## Gradients

`fast_vollib.rates` is the library's first analytic pricer written once against
the array API rather than once per backend, so a present value differentiates
back to the curve's own parameters:

```python
import torch
from fast_vollib.rates import FlatDiscountCurve

rate = torch.tensor(0.03, dtype=torch.float64, requires_grad=True)
value = present_value(
    ZeroCouponBond(maturity=4.0, face_value=1000.0),
    discount_curve=FlatDiscountCurve(rate=rate),
    return_native=True,
)
value.backward()
float(rate.grad)   # -3547.6817..., the dollar duration -T * face * exp(-rT)
```

This is a genuine exception to `fast_vollib.pricing`'s host-only rule, which
applies to the Fourier modules: a present value is a weighted sum of discount
factors, with no quadrature and no complex arithmetic, so it can be evaluated in
whatever namespace the curve hands back. `return_native=False` (the default)
returns a host `float`.

## Simulating the same model

The curve above is a closed form, and where a closed form exists this library
uses it. What a closed form cannot carry is a *path*, which is what a payoff
depending on the realized rate needs — there, the discount factor and the
payoff are functions of the same path and

$$ E\left[e^{-\int_0^T r_u du} X_T\right] \neq P(0,T)\, E[X_T] $$

whenever the two are correlated. `fast_vollib.processes.CIRShortRate` samples
the same dynamics with the same risk-neutral parameters, and
`fast_vollib.simulation.PathwiseShortRateDiscounting` integrates the sampled
rate; see the [simulation guide](simulation.md#the-cir-short-rate). The two
agree: the simulated bond price reproduces `CIRDiscountCurve` within Monte
Carlo error, and a process hands back its own curve with

```python
from fast_vollib.processes import CIRShortRate

process = CIRShortRate(kappa=0.3, theta=0.04, volatility=0.1)
process.discount_curve(initial_rate=0.04)
# CIRDiscountCurve(kappa=0.3, theta=0.04, volatility=0.1, initial_rate=0.04)
```

The convenience runs one way only. A curve is a valuation input that several
processes could have produced, so it does not hand back one.

## Not implemented

Named because a library that stays quiet about its edges invites the assumption
that it has none. Each of these needs its own design:

Bootstrapping a curve from deposits, futures, and swaps; credit, default, and
recovery; calendars, day counts, and business-day rules;
accrued interest, clean/dirty price, settlement lags, and yield-to-maturity;
floating, inflation-linked, callable, and convertible products; multi-factor or
negative-rate short-rate models.

## References

- Cox, J. C., Ingersoll, J. E., Ross, S. A. (1985). A Theory of the Term
  Structure of Interest Rates. *Econometrica* 53(2), 385–407, equation 23.
- Bakshi, G., Cao, C., Chen, Z. (1997). Empirical Performance of Alternative
  Option Pricing Models. *Journal of Finance* 52(5), 2003–2049.
