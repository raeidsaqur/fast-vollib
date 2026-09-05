"""``InterpolatedDiscountCurve``: the rule, the edges, and the neighbour it differs from.

Most of this file is about two decisions that a reader would otherwise have to
take on trust.

*The origin is not a pillar.*  ``P(0,0) = 1`` is a fact about discount factors,
not an observation, so it anchors the short end and the region below the first
pillar is interpolation against it.  The alternative -- special-casing zero and
refusing everything just above it -- is checked here to be *not* what happens,
because a curve that answers at ``0.0`` and raises at ``1e-9`` would be a rule
nobody could rely on.

*It disagrees with* :class:`~fast_vollib.surface.SurfaceMarket` *off-pillar, on
purpose.*  That class interpolates the zero rate linearly in ``T``; this one is
linear in ``r(T) T``.  Both are standard, they are not the same, and the
disagreement is asserted rather than left for someone to discover in a price
difference.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from fast_vollib.rates import (
    EXTRAPOLATIONS,
    DiscountCurve,
    InterpolatedDiscountCurve,
    RateValidationError,
)

PILLARS = np.array([1.0, 2.0, 5.0, 10.0])
FACTORS = np.array([0.97, 0.94, 0.85, 0.72])


def curve(**overrides):
    kwargs = {"maturities": PILLARS, "discount_factors": FACTORS}
    kwargs.update(overrides)
    return InterpolatedDiscountCurve(**kwargs)


# --- the protocol and the shape of the object ----------------------------------


def test_it_is_a_discount_curve() -> None:
    assert isinstance(curve(), DiscountCurve)


def test_it_reports_its_pillar_count_and_holds_no_valuation_state() -> None:
    assert curve().n_pillars == 4
    forbidden = {"process", "paths", "scenario", "rng", "device", "engine", "instrument"}
    assert forbidden.isdisjoint(set(curve().__dataclass_fields__))


def test_it_is_frozen() -> None:
    with pytest.raises(Exception):
        curve().extrapolation = "flat"  # type: ignore[misc]


def test_the_extrapolation_vocabulary_is_the_surface_layer_s_own() -> None:
    """One spelling of one decision. Two would eventually mean two things."""
    from fast_vollib.surface.market import EXTRAPOLATIONS as SURFACE_EXTRAPOLATIONS

    assert set(EXTRAPOLATIONS) == set(SURFACE_EXTRAPOLATIONS)


# --- the values ----------------------------------------------------------------


def test_a_zero_maturity_discounts_to_exactly_one() -> None:
    """Exactly, and it falls out of the anchor rather than being special-cased:
    ``log P(0)`` interpolates to ``0.0`` and ``exp(0.0)`` is ``1.0``."""
    assert float(curve().discount_factor(0.0)) == 1.0
    assert float(curve(extrapolation="flat").discount_factor(0.0)) == 1.0


@pytest.mark.parametrize("index", range(len(PILLARS)))
def test_the_curve_returns_its_own_pillars(index) -> None:
    """A container gives its data back. Exact in log space; the factor is that
    value round-tripped through ``exp(log(P))``, so at most one ulp away."""
    got = float(curve().discount_factor(float(PILLARS[index])))
    expected = float(FACTORS[index])
    assert abs(got - expected) <= math.ulp(expected)


def test_the_last_pillar_is_returned_as_exactly_as_the_others() -> None:
    """The convex form earns its keep here.

    With ``l0 + w * (l1 - l0)`` the final pillar is reached at ``w == 1`` and
    comes back as ``l0 + (l1 - l0)``, which is not ``l1`` in floating point.
    ``(1 - w) * l0 + w * l1`` is.
    """
    last = float(PILLARS[-1])
    assert abs(float(curve().discount_factor(last)) - float(FACTORS[-1])) <= math.ulp(
        float(FACTORS[-1])
    )


def test_the_interpolation_is_log_linear_in_the_discount_factor() -> None:
    """Computed by hand at a midpoint, which is what makes the rule checkable."""
    got = float(curve().discount_factor(3.5))
    weight = (3.5 - 2.0) / (5.0 - 2.0)
    expected = math.exp((1.0 - weight) * math.log(0.94) + weight * math.log(0.85))
    assert got == pytest.approx(expected, rel=1e-15)


def test_log_linear_is_linear_in_the_zero_rate_times_maturity() -> None:
    """The same rule stated the other way, which is how a rates desk says it."""
    left, right = 2.0, 5.0
    # ``-log P(T) == r(T) * T`` by definition, so linear interpolation of
    # ``log P`` *is* linear interpolation of ``r(T) T``.
    for maturity in (2.5, 3.0, 4.25):
        weight = (maturity - left) / (right - left)
        expected = (1.0 - weight) * (-math.log(0.94)) + weight * (-math.log(0.85))
        got = float(curve().zero_rate(maturity)) * maturity
        assert got == pytest.approx(expected, rel=1e-14)


def test_below_the_first_pillar_the_zero_rate_is_flat() -> None:
    """The consequence of anchoring at the origin, and the market convention."""
    first_rate = -math.log(0.97) / 1.0
    for maturity in (0.01, 0.25, 0.5, 1.0):
        assert float(curve().zero_rate(maturity)) == pytest.approx(first_rate, rel=1e-14)


def test_the_zero_rate_at_zero_maturity_is_the_first_pillar_s_rate() -> None:
    """``-log P / T`` is 0/0 there; the limit is not a convenience, it is the
    value the flat short end already has."""
    assert float(curve().zero_rate(0.0)) == pytest.approx(-math.log(0.97) / 1.0, rel=1e-15)


def test_a_query_just_above_zero_answers_rather_than_raising() -> None:
    """The short end is interpolation, not extrapolation, so ``error`` does not
    fire here. A curve that answered at 0.0 and raised at 1e-9 would be no rule."""
    assert 0.0 < float(curve().discount_factor(1e-9)) <= 1.0


def test_a_curve_with_one_pillar_is_flat_in_the_zero_rate() -> None:
    """The degenerate but legitimate case: one observation and the origin."""
    single = InterpolatedDiscountCurve(
        maturities=np.array([2.0]), discount_factors=np.array([0.9]), extrapolation="flat"
    )
    rate = -math.log(0.9) / 2.0
    for maturity in (0.5, 2.0, 7.0):
        assert float(single.discount_factor(maturity)) == pytest.approx(
            math.exp(-rate * maturity), rel=1e-14
        )


def test_factors_above_one_are_accepted_because_negative_rates_exist() -> None:
    """No monotonicity constraint: refusing this would make the class unable to
    hold the front end of several real government curves."""
    negative = InterpolatedDiscountCurve(
        maturities=np.array([1.0, 2.0]), discount_factors=np.array([1.004, 1.006])
    )
    assert float(negative.discount_factor(1.5)) > 1.0
    assert float(negative.zero_rate(1.5)) < 0.0


# --- extrapolation -------------------------------------------------------------


def test_past_the_last_pillar_the_default_refuses() -> None:
    with pytest.raises(RateValidationError, match="past the curve's last pillar"):
        curve().discount_factor(12.0)


def test_the_refusal_names_the_alternative() -> None:
    with pytest.raises(RateValidationError, match="extrapolation='flat'"):
        curve().discount_factor(12.0)


def test_the_last_pillar_itself_is_not_past_the_last_pillar() -> None:
    assert float(curve().discount_factor(10.0)) == pytest.approx(0.72, rel=1e-15)


def test_flat_extrapolation_holds_the_zero_rate_and_not_the_factor() -> None:
    """The distinction is the whole content of the word.

    Holding the last *factor* would make ``P`` constant past the end, which is
    a forward rate of zero -- a strong statement about the market, made by
    accident. Holding the last *rate* keeps ``P`` decaying at the rate the
    curve last observed.
    """
    flat = curve(extrapolation="flat")
    last_rate = -math.log(0.72) / 10.0
    for maturity in (10.5, 20.0, 40.0):
        assert float(flat.discount_factor(maturity)) == pytest.approx(
            math.exp(-last_rate * maturity), rel=1e-14
        )
        assert float(flat.discount_factor(maturity)) < 0.72


def test_flat_extrapolation_is_continuous_at_the_last_pillar() -> None:
    flat = curve(extrapolation="flat")
    inside = float(flat.discount_factor(10.0 - 1e-9))
    outside = float(flat.discount_factor(10.0 + 1e-9))
    assert abs(inside - outside) < 1e-9


def test_an_unknown_extrapolation_is_refused_at_construction() -> None:
    with pytest.raises(RateValidationError, match="extrapolation must be one of"):
        curve(extrapolation="linear")


# --- against SurfaceMarket -----------------------------------------------------


def market_and_curve():
    """A market and the curve built from the factors it quotes at its pillars."""
    from fast_vollib.surface import SurfaceMarket

    rates = -np.log(FACTORS) / PILLARS
    market = SurfaceMarket(T=PILLARS, forward=100.0, rate=rates)
    observed = InterpolatedDiscountCurve(
        maturities=PILLARS,
        discount_factors=np.array([float(market.discount_at(t)) for t in PILLARS]),
    )
    return market, observed


@pytest.mark.parametrize("index", range(len(PILLARS)))
def test_the_two_agree_at_every_pillar(index) -> None:
    """The two interpolation routes agree at pillars to 1e-15."""
    market, observed = market_and_curve()
    maturity = float(PILLARS[index])
    assert float(observed.discount_factor(maturity)) == pytest.approx(
        float(market.discount_at(maturity)), rel=1e-15, abs=0.0
    )


def test_the_two_disagree_between_pillars_and_that_is_the_documented_difference() -> None:
    """Not a bug in either, and not a tolerance to be widened.

    ``SurfaceMarket`` interpolates ``r(T)`` linearly in ``T`` and discounts by
    ``exp(-r(T) T)``; this curve is linear in ``r(T) T``. On a curve with any
    shape at all the two differ, and the difference grows with the pillar gap.
    """
    market, observed = market_and_curve()
    difference = abs(float(observed.discount_factor(3.5)) - float(market.discount_at(3.5)))
    assert difference > 1e-6, difference
    # Still small in absolute terms -- this is a convention difference, not a
    # disagreement about the market.
    assert difference < 1e-2, difference


def test_this_branch_does_not_change_surface_market() -> None:
    """The neighbour is untouched; reconciling them is a separate change."""
    from fast_vollib.surface import SurfaceMarket

    market = SurfaceMarket(T=PILLARS, forward=100.0, rate=0.03)
    assert float(market.discount_at(3.5)) == float(np.exp(-0.03 * 3.5))


# --- refusals ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"maturities": np.array([])}, "must not be empty"),
        ({"discount_factors": np.array([0.9, 0.8])}, "One factor per pillar"),
        ({"maturities": np.array([1.0, 0.5, 5.0, 10.0])}, "strictly increasing"),
        ({"maturities": np.array([1.0, 1.0, 5.0, 10.0])}, "strictly increasing"),
        ({"maturities": np.array([0.0, 2.0, 5.0, 10.0])}, "strictly positive"),
        ({"maturities": np.array([-1.0, 2.0, 5.0, 10.0])}, "strictly positive"),
        ({"maturities": np.array([1.0, np.nan, 5.0, 10.0])}, "finite"),
        ({"discount_factors": np.array([0.97, 0.0, 0.85, 0.72])}, "strictly positive"),
        ({"discount_factors": np.array([0.97, -0.1, 0.85, 0.72])}, "strictly positive"),
        ({"discount_factors": np.array([0.97, np.inf, 0.85, 0.72])}, "finite"),
        ({"maturities": np.zeros((2, 2))}, "one-dimensional"),
        ({"maturities": "abc"}, "not a string"),
    ],
)
def test_an_invalid_curve_is_refused_at_construction(kwargs, message) -> None:
    """Before it can price anything, so a bad curve cannot produce a number."""
    with pytest.raises(RateValidationError, match=message):
        curve(**kwargs)


def test_a_zero_pillar_is_refused_because_the_origin_is_not_an_observation() -> None:
    """``P(0,0) = 1`` is supplied; a pillar at zero would be a second opinion."""
    with pytest.raises(RateValidationError, match="anchors the short end"):
        curve(maturities=np.array([0.0, 2.0, 5.0, 10.0]))


def test_a_negative_maturity_is_refused() -> None:
    with pytest.raises(RateValidationError, match="maturity must be non-negative"):
        curve().discount_factor(-1.0)


def test_a_vector_of_maturities_is_refused_rather_than_broadcast() -> None:
    """One factor per call, matching the other two curves."""
    with pytest.raises(RateValidationError, match="must be a scalar"):
        curve().discount_factor(np.array([1.0, 2.0]))


# --- backends ------------------------------------------------------------------


def test_a_torch_curve_differentiates_in_its_own_pillar_factors() -> None:
    """A bucketed sensitivity of a price to the curve it was read from -- the
    reason the kernel is array-API native rather than NumPy with a cast."""
    torch = pytest.importorskip("torch")
    factors = torch.tensor(FACTORS, dtype=torch.float64, requires_grad=True)
    observed = InterpolatedDiscountCurve(
        maturities=torch.tensor(PILLARS, dtype=torch.float64), discount_factors=factors
    )
    value = observed.discount_factor(torch.tensor(3.5, dtype=torch.float64))
    value.backward()
    assert factors.grad is not None
    assert bool(torch.all(torch.isfinite(factors.grad))), factors.grad
    # Only the bracketing pillars move the answer.
    assert float(factors.grad[0]) == 0.0 and float(factors.grad[3]) == 0.0
    assert float(factors.grad[1]) > 0.0 and float(factors.grad[2]) > 0.0


def test_the_error_mode_does_not_poison_an_in_range_gradient() -> None:
    """``where`` multiplies the branch it did not take by zero, and zero times
    NaN is NaN. The out-of-range value is therefore a constant, not something
    built from the pillars."""
    torch = pytest.importorskip("torch")
    factors = torch.tensor(FACTORS, dtype=torch.float64, requires_grad=True)
    observed = InterpolatedDiscountCurve(
        maturities=torch.tensor(PILLARS, dtype=torch.float64),
        discount_factors=factors,
        extrapolation="error",
    )
    observed.discount_factor(torch.tensor(3.5, dtype=torch.float64)).backward()
    assert bool(torch.all(torch.isfinite(factors.grad))), factors.grad


def test_a_jax_curve_traces_and_differentiates() -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    def value(factors):
        return InterpolatedDiscountCurve(
            maturities=jnp.asarray(PILLARS), discount_factors=factors
        ).discount_factor(jnp.asarray(3.5))

    assert float(jax.jit(value)(jnp.asarray(FACTORS))) == pytest.approx(
        float(curve().discount_factor(3.5)), rel=1e-12
    )
    gradient = jax.grad(value)(jnp.asarray(FACTORS))
    assert bool(jnp.all(jnp.isfinite(gradient))), gradient


def test_a_curve_built_from_traced_pillars_is_constructible() -> None:
    """The value checks are skipped under a trace and the shape checks are not,
    which is the same split every other validator in this package makes."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    def value(pillars, factors):
        return InterpolatedDiscountCurve(
            maturities=pillars, discount_factors=factors, extrapolation="flat"
        ).discount_factor(jnp.asarray(3.5))

    got = float(jax.jit(value)(jnp.asarray(PILLARS), jnp.asarray(FACTORS)))
    assert got == pytest.approx(float(curve().discount_factor(3.5)), rel=1e-12)


def test_a_traced_query_past_the_last_pillar_is_not_a_plausible_number() -> None:
    """``error`` cannot raise on a value it cannot read, so the answer is NaN
    rather than a silent continuation of the last segment."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    def value(maturity):
        return InterpolatedDiscountCurve(
            maturities=jnp.asarray(PILLARS), discount_factors=jnp.asarray(FACTORS)
        ).discount_factor(maturity)

    assert math.isnan(float(jax.jit(value)(jnp.asarray(25.0))))
    assert not math.isnan(float(jax.jit(value)(jnp.asarray(3.5))))


def test_a_shape_mismatch_is_still_caught_under_a_trace() -> None:
    """Rank and length are static, so the guard that matters most still runs."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    def value(pillars):
        return InterpolatedDiscountCurve(
            maturities=pillars, discount_factors=jnp.asarray(FACTORS[:2])
        ).discount_factor(jnp.asarray(1.5))

    with pytest.raises(RateValidationError, match="One factor per pillar"):
        jax.jit(value)(jnp.asarray(PILLARS))


# --- present value -------------------------------------------------------------


def test_a_bond_prices_off_an_observed_curve() -> None:
    """The whole point of the class: a schedule read against quoted factors."""
    from fast_vollib.instruments import FixedRateBond
    from fast_vollib.pricing import present_value

    observed = curve(extrapolation="flat")
    bond = FixedRateBond(
        payment_times=(1.0, 2.0, 5.0, 10.0),
        accrual_fractions=(1.0, 1.0, 3.0, 5.0),
        coupon_rate=0.03,
        face_value=100.0,
    )
    priced = present_value(bond, discount_curve=observed)
    expected = sum(
        flow.amount * float(observed.discount_factor(flow.payment_time)) for flow in bond.cashflows
    )
    assert priced == pytest.approx(expected, rel=1e-14)


def test_pricing_a_bond_that_outlives_the_curve_refuses_rather_than_guesses() -> None:
    from fast_vollib.instruments import ZeroCouponBond
    from fast_vollib.pricing import present_value

    with pytest.raises(RateValidationError, match="past the curve's last pillar"):
        present_value(ZeroCouponBond(maturity=30.0), discount_curve=curve())
