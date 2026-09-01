"""A penalized tensor-product spline reproduces what it can and declines the rest."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest

from fast_vollib.surface import (
    DefiniteIVSurface,
    SurfaceCalibrationError,
    SurfaceCalibrator,
    SurfaceGridSpec,
    SurfaceObservations,
    SurfacePoints,
    SurfaceValidationError,
    materialize_surface,
)
from fast_vollib.surface.fitting.splines import (
    SplineIVSurface,
    SplineSmileCalibrator,
    SplineSurfaceCalibrator,
)

MONEYNESS = np.linspace(-0.4, 0.4, 21)
MATURITIES = np.array([0.1, 0.25, 0.5, 1.0, 2.0])

# The unpenalised calibrator: every exactness claim below is a claim about the
# solve, so the stabilising default penalty is switched off rather than assumed
# negligible.  DEFAULT_SMOOTHING biases a recovery test at the 1e-6 level.
EXACT = SplineSurfaceCalibrator(smoothing_k=0.0, smoothing_t=0.0)


def polynomial_w(k: Any, T: Any) -> np.ndarray:
    """A total variance of degree 2 in k and 1 in T, hence in every cubic span."""
    return T * (0.04 + 0.02 * k + 0.30 * k**2)


def svi_w(k: Any, T: Any) -> np.ndarray:
    """Gatheral-Jacquier raw SVI total variance, scaled linearly in T."""
    a, b, rho, m, sigma = 0.04, 0.40, -0.40, 0.0, 0.10
    shape = a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma**2))
    return shape * (T / MATURITIES[-1])


def observations_from(w_fn, k: Any = MONEYNESS, T: Any = MATURITIES) -> SurfaceObservations:
    """Quotes on the full ``k x T`` mesh implied by a total-variance function."""
    mesh_k, mesh_t = np.meshgrid(np.asarray(k), np.asarray(T), indexing="ij")
    w = w_fn(mesh_k, mesh_t)
    return SurfaceObservations(k=mesh_k.ravel(), T=mesh_t.ravel(), iv=np.sqrt(w / mesh_t).ravel())


# --- the protocols -----------------------------------------------------------


def test_the_spline_calibrators_and_their_surface_satisfy_the_protocols() -> None:
    surface = EXACT.fit(observations_from(polynomial_w))
    assert isinstance(surface, DefiniteIVSurface)
    assert isinstance(SplineSurfaceCalibrator(), SurfaceCalibrator)
    assert isinstance(SplineSmileCalibrator(), SurfaceCalibrator)


# --- exact recovery ----------------------------------------------------------


def test_a_total_variance_in_the_spline_span_is_recovered_to_solver_accuracy() -> None:
    # w = T (0.04 + 0.02 k + 0.30 k^2) is a bivariate polynomial of degree
    # (2, 1), and every polynomial of degree <= p lies in a degree-p B-spline
    # space whatever the knots are.  So the approximation error is exactly zero
    # and the only residual is the conditioning of the normal equations:
    # measured 5.2e-15 relative on this mesh, asserted at the float64
    # algebraic-identity band.
    surface = EXACT.fit(observations_from(polynomial_w))
    mesh_k, mesh_t = np.meshgrid(np.linspace(-0.38, 0.38, 37), np.linspace(0.12, 1.9, 13))
    query = SurfacePoints(k=mesh_k.ravel(), T=mesh_t.ravel())
    np.testing.assert_allclose(
        surface.total_variance(query.k, query.T),
        polynomial_w(query.k, query.T),
        rtol=1e-12,
        atol=1e-12,
    )
    prediction = surface.evaluate(query)
    assert bool(prediction.valid.all())
    np.testing.assert_allclose(
        prediction.iv,
        np.sqrt(polynomial_w(query.k, query.T) / query.T),
        rtol=1e-12,
        atol=1e-12,
    )


def test_the_fit_uses_a_different_basis_size_in_each_direction() -> None:
    # Guards the exactness test above: with n_basis_k == n_basis_t a transposed
    # Kronecker product in the penalty, or a transposed coefficient reshape,
    # would still produce a square system and could pass unnoticed.
    surface = EXACT.fit(observations_from(polynomial_w))
    assert surface.n_basis_k != surface.n_basis_t
    assert surface.coefficients.shape == (surface.n_basis_k, surface.n_basis_t)


def test_refining_the_knots_converges_at_the_cubic_approximation_order() -> None:
    # SVI total variance is not in any spline span, so the error is the cubic
    # approximation error, O(h^4) in the knot spacing: doubling the interior
    # knot count should divide the sup error by about 16.  The claim asserted
    # is 8 -- half the asymptotic rate -- because the coarsest meshes here are
    # not yet asymptotic.  Measured ratios: 12.1 then 32.1.
    fine_k = np.linspace(-0.5, 0.5, 201)
    quotes = observations_from(svi_w, k=fine_k, T=np.array([0.25, 1.0, 2.0]))
    probe_k = np.linspace(-0.49, 0.49, 401)
    probe_t = np.full(probe_k.size, 1.0)
    errors = []
    for interior in (8, 16, 32):
        fitted = SplineSurfaceCalibrator(
            n_interior_knots_k=interior, smoothing_k=0.0, smoothing_t=0.0
        ).fit(quotes)
        residual = fitted.total_variance(probe_k, probe_t) - svi_w(probe_k, probe_t)
        errors.append(float(np.max(np.abs(residual))))
    ratios = [coarse / fine for coarse, fine in zip(errors, errors[1:])]
    assert all(ratio >= 8.0 for ratio in ratios), ratios


# --- the penalty -------------------------------------------------------------


def test_raising_the_penalty_monotonically_lowers_the_roughness() -> None:
    # One multiplier scales both directions, which is the case in which the
    # penalty term of a ridge-type objective is provably non-increasing; two
    # independently varied weights are not covered by that result.
    quotes = observations_from(svi_w)
    roughness = [
        SplineSurfaceCalibrator(smoothing_k=scale, smoothing_t=scale).fit(quotes).roughness()
        for scale in (1e-6, 1e-4, 1e-2, 1.0)
    ]
    assert all(a > b for a, b in zip(roughness, roughness[1:])), roughness


def test_the_penalty_leaves_a_bilinear_coefficient_sheet_alone() -> None:
    # Second differences annihilate 1, a, b and a*b, so an infinite penalty
    # shrinks a surface towards a plane in the coefficients, not towards zero.
    sheet = np.array([[1.0 + 2.0 * a + 3.0 * b + 4.0 * a * b for b in range(4)] for a in range(5)])
    surface = SplineIVSurface(
        knots_k=np.concatenate([np.full(4, -0.4), [0.0], np.full(4, 0.4)]),
        knots_t=np.concatenate([np.full(4, 0.1), np.full(4, 2.0)]),
        degree_k=3,
        degree_t=3,
        coefficients=sheet,
    )
    assert surface.roughness() == pytest.approx(0.0, abs=1e-24)


# --- reproducibility ---------------------------------------------------------


def test_shuffling_the_observations_leaves_the_fit_bitwise_identical() -> None:
    # Bitwise by construction, not to a tolerance: rows are sorted into a
    # canonical (T, k, weight, w) order before the normal equations are
    # accumulated, so the summation order is a function of the data alone.
    quotes = observations_from(svi_w)
    shuffled = quotes.subset(np.random.default_rng(20260831).permutation(quotes.n))
    first = SplineSurfaceCalibrator().fit(quotes)
    second = SplineSurfaceCalibrator().fit(shuffled)
    np.testing.assert_array_equal(first.knots_k, second.knots_k)
    np.testing.assert_array_equal(first.knots_t, second.knots_t)
    np.testing.assert_array_equal(first.coefficients, second.coefficients)


def test_the_calibrator_holds_no_state_between_fits() -> None:
    calibrator = SplineSurfaceCalibrator()
    quotes = observations_from(polynomial_w)
    other = observations_from(svi_w)
    first = calibrator.fit(quotes)
    calibrator.fit(other)
    third = calibrator.fit(quotes)
    np.testing.assert_array_equal(first.coefficients, third.coefficients)


def test_evaluation_answers_every_query_row_in_query_order() -> None:
    surface = EXACT.fit(observations_from(polynomial_w))
    query = SurfacePoints(k=np.linspace(-0.3, 0.3, 11), T=np.linspace(0.2, 1.8, 11))
    order = np.random.default_rng(7).permutation(query.n)
    straight = surface.evaluate(query)
    scrambled = surface.evaluate(query.subset(order))
    assert straight.points.equals(query)
    np.testing.assert_array_equal(scrambled.iv, straight.iv[order])


# --- knots and degree --------------------------------------------------------


@pytest.mark.parametrize("n_maturities, expected_degree", [(1, 0), (2, 1), (3, 2), (4, 3), (5, 3)])
def test_the_degree_in_maturity_is_reduced_to_what_the_expiries_support(
    n_maturities: int, expected_degree: int
) -> None:
    quotes = observations_from(polynomial_w, T=MATURITIES[:n_maturities])
    surface = EXACT.fit(quotes)
    assert surface.degree_t == expected_degree
    assert surface.n_basis_t == n_maturities


def test_two_expiries_give_a_term_structure_that_is_linear_in_total_variance() -> None:
    # The reduction is not cosmetic: degree 1 in T means w interpolates
    # linearly between the two observed pillars, which is exactly what two
    # numbers can say about a term structure.
    quotes = observations_from(polynomial_w, T=np.array([0.5, 1.5]))
    surface = EXACT.fit(quotes)
    near, far = surface.total_variance(0.1, [0.5, 1.5])
    np.testing.assert_allclose(
        surface.total_variance(0.1, 1.0)[0], 0.5 * (near + far), rtol=1e-12, atol=1e-12
    )


@pytest.mark.parametrize("n_strikes", [5, 9, 21], ids=["sparse", "medium", "dense"])
def test_an_automatic_knot_count_never_outnumbers_the_coordinates_supporting_it(
    n_strikes: int,
) -> None:
    quotes = observations_from(polynomial_w, k=np.linspace(-0.3, 0.3, n_strikes))
    surface = EXACT.fit(quotes)
    assert surface.n_basis_k <= n_strikes
    assert surface.n_basis_t <= MATURITIES.size


# --- the domain --------------------------------------------------------------


def test_a_point_outside_the_knot_span_is_invalid_rather_than_extrapolated() -> None:
    surface = EXACT.fit(observations_from(polynomial_w))
    assert surface.domain_k == (float(MONEYNESS[0]), float(MONEYNESS[-1]))
    assert surface.domain_t == (float(MATURITIES[0]), float(MATURITIES[-1]))
    query = SurfacePoints(
        k=[0.0, -0.41, 0.41, 0.0, 0.0],
        T=[1.0, 1.0, 1.0, 0.09, 2.01],
    )
    prediction = surface.evaluate(query)
    assert prediction.valid.tolist() == [True, False, False, False, False]
    assert bool(np.all(np.isnan(prediction.iv[1:])))


def test_a_negative_fitted_total_variance_is_flagged_invalid_not_returned() -> None:
    # A coefficient block that dips below zero is what an unconstrained least
    # squares fit to noisy deep wings produces.  sqrt(max(w, 0) / T) is 0 there,
    # and 0 is a number; the validity flag is what says it is not an answer.
    surface = SplineIVSurface(
        knots_k=[-0.2, -0.2, 0.2, 0.2],
        knots_t=[0.5, 0.5, 2.0, 2.0],
        degree_k=1,
        degree_t=1,
        coefficients=[[-0.04, -0.08], [0.04, 0.08]],
    )
    prediction = surface.evaluate(SurfacePoints(k=[-0.2, 0.2], T=[1.0, 1.0]))
    assert prediction.valid.tolist() == [False, True]
    assert float(prediction.iv[0]) == 0.0


def test_evaluation_refuses_anything_that_is_not_a_point_set() -> None:
    surface = EXACT.fit(observations_from(polynomial_w))
    with pytest.raises(SurfaceValidationError, match="points must be a SurfacePoints"):
        surface.evaluate((np.array([0.0]), np.array([1.0])))  # type: ignore[arg-type]


# --- the arbitrage harness ---------------------------------------------------


def test_a_flat_volatility_data_set_is_fitted_flat_and_validates() -> None:
    # A constant sigma gives w = sigma^2 T, linear in T and constant in k, so
    # it lies in the span exactly; the residual is the solve, asserted at the
    # float64 algebraic-identity band.
    quotes = observations_from(lambda k, T: 0.04 * T + 0.0 * k)
    surface = EXACT.fit(quotes)
    grid = SurfaceGridSpec(k=np.linspace(-0.35, 0.35, 21), T=[0.2, 0.5, 1.0, 1.8])
    mesh = materialize_surface(surface, grid)
    np.testing.assert_allclose(mesh.iv, 0.2, rtol=1e-12, atol=1e-12)
    assert mesh.validate().passed


# --- serialization -----------------------------------------------------------


def test_the_parameters_mapping_round_trips_through_json() -> None:
    surface = EXACT.fit(observations_from(polynomial_w))
    parameters = surface.parameters()
    assert sorted(parameters) == [
        "coefficients",
        "degree_k",
        "degree_t",
        "knots_k",
        "knots_t",
    ]
    encoded = json.dumps(parameters, allow_nan=False)
    rebuilt = SplineIVSurface(**json.loads(encoded))
    query = SurfacePoints(k=[-0.2, 0.0, 0.3], T=[0.3, 1.0, 1.7])
    np.testing.assert_array_equal(rebuilt.coefficients, surface.coefficients)
    np.testing.assert_array_equal(rebuilt.evaluate(query).iv, surface.evaluate(query).iv)


# --- weights -----------------------------------------------------------------


def test_a_zero_weight_observation_does_not_move_the_fit() -> None:
    quotes = observations_from(polynomial_w)
    poisoned = SurfaceObservations(
        k=np.append(quotes.k, 0.0),
        T=np.append(quotes.T, 1.0),
        iv=np.append(quotes.iv, 5.0),
        weight=np.append(np.ones(quotes.n), 0.0),
    )
    weighted = EXACT.fit(poisoned)
    np.testing.assert_array_equal(weighted.coefficients, EXACT.fit(quotes).coefficients)


def test_declining_to_use_weights_lets_the_ignored_quote_back_in() -> None:
    quotes = observations_from(polynomial_w)
    poisoned = SurfaceObservations(
        k=np.append(quotes.k, 0.0),
        T=np.append(quotes.T, 1.0),
        iv=np.append(quotes.iv, 5.0),
        weight=np.append(np.ones(quotes.n), 0.0),
    )
    unweighted = SplineSurfaceCalibrator(smoothing_k=0.0, smoothing_t=0.0, use_weights=False).fit(
        poisoned
    )
    assert not np.allclose(unweighted.coefficients, EXACT.fit(quotes).coefficients)


# --- refusals ----------------------------------------------------------------


def test_an_unpenalised_fit_with_fewer_observations_than_basis_functions_is_refused() -> None:
    sparse = SurfaceObservations(
        k=[-0.2, -0.1, 0.0, 0.1, 0.2], T=[1.0] * 5, iv=[0.22, 0.21, 0.20, 0.21, 0.23]
    )
    with pytest.raises(
        SurfaceCalibrationError, match="at least as many usable observations as basis functions"
    ):
        SplineSurfaceCalibrator(n_interior_knots_k=8, smoothing_k=0.0, smoothing_t=0.0).fit(sparse)


def test_a_penalty_in_one_direction_does_not_rescue_a_deficiency_in_the_other() -> None:
    # Three distinct strikes cannot identify an eleven-function basis in k, and
    # a maturity penalty has nothing to say about that: the stacked system stays
    # column-rank deficient, which is where a singular normal-equation system
    # shows up when the normal equations are not formed.
    repeated = SurfaceObservations(
        k=np.tile(np.repeat([-0.2, 0.0, 0.2], 2), 20),
        T=np.tile([0.5, 1.0], 60),
        iv=np.full(120, 0.2),
    )
    with pytest.raises(SurfaceCalibrationError, match="must have full column rank"):
        SplineSurfaceCalibrator(n_interior_knots_k=8, smoothing_k=0.0, smoothing_t=1e-3).fit(
            repeated
        )


def test_a_fit_with_no_usable_observation_is_refused() -> None:
    empty = SurfaceObservations(k=[0.0, 0.1], T=[1.0, 1.0], iv=[np.nan, np.nan])
    with pytest.raises(SurfaceCalibrationError, match="at least one usable observation"):
        SplineSurfaceCalibrator().fit(empty)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"knots_k": [-0.2, -0.2, 0.2]}, "knots_k must hold at least"),
        ({"knots_k": [0.2, 0.2, -0.2, -0.2]}, "knots_k must be non-decreasing"),
        ({"knots_t": [-1.0, -1.0, 2.0, 2.0]}, "knots_t must be strictly positive"),
        ({"degree_k": -1}, "degree_k must be non-negative"),
        ({"degree_k": 1.5}, "degree_k must be an integer"),
        ({"coefficients": [[0.04, 0.08, 0.1], [0.04, 0.08, 0.1]]}, "coefficients must have shape"),
    ],
)
def test_a_malformed_spline_surface_is_refused_at_construction(
    kwargs: dict[str, Any], message: str
) -> None:
    fields: dict[str, Any] = {
        "knots_k": [-0.2, -0.2, 0.2, 0.2],
        "knots_t": [0.5, 0.5, 2.0, 2.0],
        "degree_k": 1,
        "degree_t": 1,
        "coefficients": [[0.04, 0.08], [0.04, 0.08]],
    }
    fields.update(kwargs)
    with pytest.raises(SurfaceValidationError, match=message):
        SplineIVSurface(**fields)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"smoothing_k": -1e-6}, "smoothing_k must be non-negative"),
        ({"smoothing_t": np.inf}, "smoothing_t must be finite"),
        ({"n_interior_knots_k": -1}, "n_interior_knots_k must be non-negative"),
        ({"degree_t": True}, "degree_t must be an integer"),
    ],
)
def test_a_malformed_calibrator_configuration_is_refused(
    kwargs: dict[str, Any], message: str
) -> None:
    with pytest.raises(SurfaceValidationError, match=message):
        SplineSurfaceCalibrator(**kwargs)


# --- the smile calibrator ----------------------------------------------------


def test_a_smile_speaks_only_for_the_expiry_it_was_fitted_to() -> None:
    expiry = 0.5
    strikes = np.linspace(-0.25, 0.25, 11)
    w = 0.02 + 0.01 * strikes + 0.05 * strikes**2
    quotes = SurfaceObservations(k=strikes, T=np.full(11, expiry), iv=np.sqrt(w / expiry))
    smile = SplineSmileCalibrator(smoothing=0.0).fit(quotes)
    assert (smile.degree_t, smile.n_basis_t) == (0, 1)
    assert smile.domain_t == (expiry, expiry)
    prediction = smile.evaluate(SurfacePoints(k=[0.0, 0.0], T=[expiry, 1.0]))
    assert prediction.valid.tolist() == [True, False]
    # w is quadratic in k and so lies in the cubic span whatever the knots are,
    # which makes the only residual the solve: float64 algebraic-identity band.
    np.testing.assert_allclose(
        float(prediction.iv[0]), np.sqrt(0.02 / expiry), rtol=1e-12, atol=1e-12
    )


def test_a_smile_refuses_to_pool_several_expiries() -> None:
    with pytest.raises(SurfaceCalibrationError, match="must be given exactly one maturity"):
        SplineSmileCalibrator().fit(observations_from(polynomial_w))


# --- what an adversarial review found the tests could not see -----------------


def test_the_fit_matches_an_independently_weighted_least_squares() -> None:
    """Non-degenerate weights, pinned against a solve this module did not perform.

    Every earlier weight test used weights in {0, 1}, and a zero-weight row is
    already removed by the row filter before the design is scaled -- so the
    scaling itself was unconstrained, and mutating it to ignore the weights, or
    to invert them, left the whole suite green.  Weights move the answer: on a
    noisy surface with weights spread over two orders of magnitude the
    coefficients shift by about 6e-3.

    The reference below builds the same penalized normal equations from scratch
    and solves them directly.  The two routes differ -- this module stacks and
    calls lstsq -- so 1e-9 is the honest band for a well-conditioned B-spline
    design, not machine precision.
    """
    generator = np.random.default_rng(20260831)
    k = np.tile(np.linspace(-0.35, 0.35, 15), 3)
    T = np.repeat([0.25, 0.5, 1.0], 15)
    truth = T * (0.04 + 0.02 * k + 0.30 * k**2)
    iv = np.sqrt(truth / T) + 0.004 * generator.standard_normal(k.size)
    weight = generator.uniform(0.05, 20.0, size=k.size)
    observations = SurfaceObservations(k=k, T=T, iv=iv, weight=weight)

    calibrator = SplineSurfaceCalibrator(smoothing_k=3e-3, smoothing_t=7e-4)
    fitted = calibrator.fit(observations)

    # Rebuild the same problem independently: basis rows from scipy, the two
    # Kronecker second-difference penalties, and one dense solve.
    from scipy.interpolate import BSpline

    order = np.lexsort((iv * iv * T, weight, k, T))
    ks, Ts, ws, ys = k[order], T[order], weight[order], (iv * iv * T)[order]
    basis_k = BSpline.design_matrix(ks, fitted.knots_k, fitted.degree_k).toarray()
    basis_t = BSpline.design_matrix(Ts, fitted.knots_t, fitted.degree_t).toarray()
    design = (basis_k[:, :, None] * basis_t[:, None, :]).reshape(ks.size, -1)
    n_k, n_t = fitted.n_basis_k, fitted.n_basis_t

    def second_difference(size: int) -> np.ndarray:
        if size < 3:
            return np.zeros((0, size))
        rows = np.zeros((size - 2, size))
        for row in range(size - 2):
            rows[row, row : row + 3] = (1.0, -2.0, 1.0)
        return rows

    penalty = 3e-3 * np.kron(second_difference(n_k), np.eye(n_t)).T @ np.kron(
        second_difference(n_k), np.eye(n_t)
    ) + 7e-4 * np.kron(np.eye(n_k), second_difference(n_t)).T @ np.kron(
        np.eye(n_k), second_difference(n_t)
    )
    weighted = design * ws[:, None]
    expected = np.linalg.solve(design.T @ weighted + penalty, weighted.T @ ys)
    np.testing.assert_allclose(
        np.asarray(fitted.coefficients).reshape(-1), expected, rtol=1e-9, atol=1e-9
    )


def test_ignoring_or_inverting_the_weights_would_change_the_answer() -> None:
    # The companion to the test above: it establishes that the weighted fit is
    # genuinely different from the unweighted one, so the comparison there is
    # not one an unweighted implementation could also pass.
    generator = np.random.default_rng(7)
    k = np.tile(np.linspace(-0.35, 0.35, 15), 3)
    T = np.repeat([0.25, 0.5, 1.0], 15)
    iv = np.sqrt((0.04 + 0.30 * k**2)) + 0.004 * generator.standard_normal(k.size)
    weight = generator.uniform(0.05, 20.0, size=k.size)
    weighted = SplineSurfaceCalibrator().fit(SurfaceObservations(k=k, T=T, iv=iv, weight=weight))
    inverted = SplineSurfaceCalibrator().fit(
        SurfaceObservations(k=k, T=T, iv=iv, weight=1.0 / weight)
    )
    ignored = SplineSurfaceCalibrator(use_weights=False).fit(
        SurfaceObservations(k=k, T=T, iv=iv, weight=weight)
    )
    a = np.asarray(weighted.coefficients)
    assert float(np.max(np.abs(a - np.asarray(inverted.coefficients)))) > 1e-4
    assert float(np.max(np.abs(a - np.asarray(ignored.coefficients)))) > 1e-4


def test_the_moneyness_smoothing_weight_is_the_one_that_smooths_moneyness() -> None:
    """Raising ``smoothing_k`` alone must smooth the *moneyness* direction.

    ``roughness()`` sums both directions, and it is not monotone in
    ``smoothing_k``: as the moneyness penalty bites, the fit is free to buy
    smoothness in ``k`` by paying a little roughness in ``T``, and past a point
    the total turns back up.  Measured here: 1.26e-2, 1.15e-2, 8.8e-4, 9.6e-4
    over four decades of weight.  So the claim is asserted on the quantity the
    weight actually multiplies -- the squared second differences of the
    coefficient block down its moneyness axis -- computed here rather than read
    off the surface, so a mutation that hard-wired ``smoothing_k`` to zero
    cannot pass.
    """
    generator = np.random.default_rng(11)
    k = np.tile(np.linspace(-0.35, 0.35, 15), 3)
    T = np.repeat([0.25, 0.5, 1.0], 15)
    iv = np.sqrt(0.04 + 0.30 * k**2) + 0.01 * generator.standard_normal(k.size)
    observations = SurfaceObservations(k=k, T=T, iv=iv)

    def moneyness_roughness(value: float) -> float:
        fitted = SplineSurfaceCalibrator(
            n_interior_knots_k=6, smoothing_k=value, smoothing_t=1e-6
        ).fit(observations)
        block = np.asarray(fitted.coefficients)
        return float(np.sum(np.diff(block, n=2, axis=0) ** 2))

    measured = [moneyness_roughness(value) for value in (1e-8, 1e-4, 1e-1, 10.0)]
    assert measured == sorted(measured, reverse=True), measured
    assert measured[0] > 100.0 * measured[-1]


def test_a_maturity_penalty_needs_three_maturities_to_exist_at_all() -> None:
    """A second-difference operator on two pillars has no rows.

    Worth pinning because a test that set ``smoothing_t`` on a two-expiry fixture
    and concluded something about the maturity penalty would be describing an
    empty matrix.  With three expiries the operator is real and the penalty bites.
    """
    two = SplineSurfaceCalibrator(smoothing_t=1e3).fit(
        SurfaceObservations(
            k=np.tile(np.linspace(-0.3, 0.3, 9), 2),
            T=np.repeat([0.5, 1.0], 9),
            iv=np.full(18, 0.2),
        )
    )
    assert two.n_basis_t == 2

    k = np.tile(np.linspace(-0.3, 0.3, 9), 3)
    T = np.repeat([0.25, 0.5, 1.0], 9)
    generator = np.random.default_rng(3)
    iv = 0.2 + 0.02 * generator.standard_normal(27)
    observations = SurfaceObservations(k=k, T=T, iv=iv)
    light = SplineSurfaceCalibrator(smoothing_t=1e-8).fit(observations)
    heavy = SplineSurfaceCalibrator(smoothing_t=1e2).fit(observations)
    assert light.n_basis_t >= 3
    assert heavy.roughness() < light.roughness()
