"""Generic regularization: exact operators, stable solves, honest coupling."""

from __future__ import annotations

import json

import numpy as np
from numpy.testing import assert_array_equal
import pytest

from fast_vollib.surface import SurfaceValidationError
from fast_vollib.surface.fitting.regularized import (
    MAX_DIFFERENCE_ORDER,
    LeastSquaresDiagnostics,
    TikhonovPenalty,
    couple_parameter_sequence,
    difference_matrix,
    solve_penalized_least_squares,
)

EPS = float(np.finfo(np.float64).eps)

# Predeclared tolerances.
#
# ALGEBRAIC -- one identity reached by two float64 code paths: the stacked
# solve here against a direct numpy.linalg.lstsq there, a permuted row order,
# or a column coupled inside a block against the same column coupled alone.
# The paths differ only in the order of the accumulated inner products, so they
# agree to a few ulps; the measured gaps below are 0 to 1e-14 on unit-scale
# data. 1e-12 is the float64 algebraic-identity tolerance used across this
# suite, and is not widened anywhere in this file.
ALGEBRAIC = 1e-12

# MEAN_LIMIT -- how close the order-1 coupling of a five-step path gets to the
# path's mean at weight 1e12. Two terms, both predeclared:
#   analytic bias   ||p - mean|| / (weight * mu_min) with mu_min = 4 sin^2(pi/2n)
#                   = 0.382 the smallest non-zero eigenvalue of D^T D, giving
#                   0.06 / (1e12 * 0.382) = 1.6e-13; and
#   solver floor    cond(stacked) * eps = sqrt(1e12 * 3.6) * 2.2e-16 = 4.2e-10.
# Their sum is 4.2e-10, so 1e-9 is the smallest round tolerance above the
# bound. The measured deviation is 2.2e-11.
MEAN_LIMIT = 1e-9


def _vandermonde_fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A genuinely ill-conditioned design with a known exact solution.

    A degree-11 monomial basis on ``[0, 1]``, sampled at 40 points: condition
    number 1.18e8, which is ordinary for a polynomial basis over a wide range
    and fatal once squared.
    """
    x = np.linspace(0.0, 1.0, 40)
    design = np.vander(x, 12, increasing=True)
    truth = np.arange(1.0, 13.0)
    return design, truth, design @ truth


def _well_conditioned() -> tuple[np.ndarray, np.ndarray]:
    """A well-conditioned random design (condition number 1.6) and a target."""
    rng = np.random.default_rng(20260831)
    return rng.standard_normal((30, 4)), rng.standard_normal(30)


# --- the difference operator --------------------------------------------------


def test_the_first_and_second_difference_rows_are_the_exact_binomial_stencils() -> None:
    # Exact by construction: the stencil is built in int64 and cast once.
    assert_array_equal(
        difference_matrix(4, 1),
        [[-1.0, 1.0, 0.0, 0.0], [0.0, -1.0, 1.0, 0.0], [0.0, 0.0, -1.0, 1.0]],
    )
    assert_array_equal(
        difference_matrix(4, 2),
        [[1.0, -2.0, 1.0, 0.0], [0.0, 1.0, -2.0, 1.0]],
    )
    assert_array_equal(difference_matrix(6, 3)[0], [-1.0, 3.0, -3.0, 1.0, 0.0, 0.0])


def test_the_operator_of_order_zero_is_the_identity() -> None:
    assert_array_equal(difference_matrix(3, 0), np.eye(3))


@pytest.mark.parametrize("n, order", [(2, 1), (5, 1), (5, 2), (9, 4), (57, 56)], ids=str)
def test_the_operator_has_one_row_per_window_of_order_plus_one(n: int, order: int) -> None:
    assert difference_matrix(n, order).shape == (n - order, n)


@pytest.mark.parametrize(
    "slope, intercept", [(0.0, 1.0), (2.5, -3.0), (-0.125, 7.0)], ids=["flat", "up", "down"]
)
def test_a_second_difference_annihilates_every_affine_sequence_exactly(
    slope: float, intercept: float
) -> None:
    # Exact, not approximate: every entry is a dyadic rational, so the
    # cancellation 1 - 2 + 1 is performed on exactly representable numbers.
    index = np.arange(9.0)
    assert_array_equal(difference_matrix(9, 2) @ (intercept + slope * index), np.zeros(7))


def test_a_third_difference_annihilates_a_quadratic_sequence_exactly() -> None:
    index = np.arange(8.0)
    quadratic = 2.0 - 0.5 * index + 0.25 * index * index
    assert_array_equal(difference_matrix(8, 3) @ quadratic, np.zeros(5))


def test_an_order_that_would_leave_no_rows_is_refused() -> None:
    with pytest.raises(SurfaceValidationError, match="order must be smaller than n"):
        difference_matrix(3, 3)


def test_an_order_beyond_float64_exactness_is_refused() -> None:
    assert MAX_DIFFERENCE_ORDER == 56  # binom(56, 28) < 2**53 < binom(57, 28)
    with pytest.raises(SurfaceValidationError, match="order must be at most 56"):
        difference_matrix(200, 57)


@pytest.mark.parametrize("bad", [2.0, "4", True], ids=["float", "str", "bool"])
def test_a_dimension_that_is_not_an_integer_is_refused(bad: object) -> None:
    with pytest.raises(SurfaceValidationError, match="n must be an integer"):
        difference_matrix(bad, 1)  # type: ignore[arg-type]


# --- the penalty object -------------------------------------------------------


def test_a_penalty_owns_a_read_only_copy_of_its_operator() -> None:
    operator = np.eye(2)
    penalty = TikhonovPenalty(operator=operator, weight=1.0)
    operator[0, 0] = 99.0
    assert penalty.operator[0, 0] == 1.0
    assert penalty.operator.flags.writeable is False


def test_the_ridge_and_difference_helpers_build_the_documented_operators() -> None:
    assert_array_equal(TikhonovPenalty.ridge(3, 1.0).operator, np.eye(3))
    assert_array_equal(
        TikhonovPenalty.difference(5, 2.0, order=2).operator, difference_matrix(5, 2)
    )
    assert TikhonovPenalty.difference(5, 2.0, order=1).n_rows == 4


def test_the_penalty_value_is_the_weight_times_the_squared_operator_norm() -> None:
    penalty = TikhonovPenalty.difference(4, 3.0, order=1)
    coefficients = np.array([0.0, 1.0, 3.0, 6.0])
    # differences 1, 2, 3 -> squared norm 14 -> 3 * 14 = 42, exact in float64.
    assert penalty.value(coefficients) == 42.0


def test_a_second_difference_penalty_is_exactly_zero_on_an_affine_vector() -> None:
    assert TikhonovPenalty.difference(6, 7.5, order=2).value(np.arange(6.0) * 0.5 + 2.0) == 0.0


def test_the_zero_weight_and_the_weighted_operator_agree_with_the_convention() -> None:
    penalty = TikhonovPenalty.difference(4, 9.0, order=1)
    # lambda multiplies the SQUARED norm, so the stacked block carries sqrt(lambda).
    np.testing.assert_allclose(
        penalty.weighted_operator(), 3.0 * difference_matrix(4, 1), rtol=ALGEBRAIC, atol=ALGEBRAIC
    )
    assert TikhonovPenalty.difference(4, 0.0, order=1).weighted_operator().max() == 0.0


@pytest.mark.parametrize(
    "weight, message",
    [
        (-1e-12, "weight must be non-negative"),
        (float("nan"), "weight must be finite"),
        (float("inf"), "weight must be finite"),
    ],
    ids=["negative", "nan", "infinite"],
)
def test_an_inadmissible_penalty_weight_is_refused(weight: float, message: str) -> None:
    with pytest.raises(SurfaceValidationError, match=message):
        TikhonovPenalty.ridge(3, weight)


def test_a_penalty_evaluated_on_the_wrong_coefficients_is_refused() -> None:
    with pytest.raises(SurfaceValidationError, match="one entry per operator column"):
        TikhonovPenalty.ridge(3, 1.0).value([1.0, 2.0])


# --- the penalized solve ------------------------------------------------------


def test_an_unpenalized_solve_reproduces_numpy_lstsq() -> None:
    design, target = _well_conditioned()
    coefficients, diagnostics = solve_penalized_least_squares(design, target)
    reference = np.linalg.lstsq(design, target, rcond=None)[0]
    np.testing.assert_allclose(coefficients, reference, rtol=ALGEBRAIC, atol=ALGEBRAIC)
    assert diagnostics.penalty_norm == 0.0
    assert diagnostics.effective_rank == 4


def test_a_zero_weight_penalty_reproduces_the_unpenalized_solution() -> None:
    design, target = _well_conditioned()
    plain, _ = solve_penalized_least_squares(design, target)
    inert, diagnostics = solve_penalized_least_squares(
        design, target, penalties=[TikhonovPenalty.difference(4, 0.0, order=2)]
    )
    # A zero-weight penalty is kept in the stack, so this is the same problem
    # solved through a taller matrix -- a different code path, hence ALGEBRAIC
    # rather than assert_array_equal (the measured difference is in fact 0).
    np.testing.assert_allclose(inert, plain, rtol=ALGEBRAIC, atol=ALGEBRAIC)
    assert diagnostics.penalty_norm == 0.0


def test_a_row_weight_of_four_is_that_row_entered_four_times() -> None:
    design, target = _well_conditioned()
    weights = np.concatenate([[4.0], np.ones(design.shape[0] - 1)])
    weighted, _ = solve_penalized_least_squares(design, target, weights=weights)
    repeated, _ = solve_penalized_least_squares(
        np.vstack([design, np.repeat(design[:1], 3, axis=0)]),
        np.concatenate([target, np.repeat(target[:1], 3)]),
    )
    np.testing.assert_allclose(weighted, repeated, rtol=ALGEBRAIC, atol=ALGEBRAIC)


def test_increasing_the_ridge_weight_monotonically_shrinks_the_coefficients() -> None:
    design, target = _well_conditioned()
    norms = [
        float(
            np.linalg.norm(
                solve_penalized_least_squares(
                    design, target, penalties=[TikhonovPenalty.ridge(4, weight)]
                )[0]
            )
        )
        for weight in (0.0, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
    ]
    assert all(later < earlier for earlier, later in zip(norms, norms[1:]))


def test_the_solution_does_not_depend_on_the_order_of_the_rows() -> None:
    # Stated on a well-conditioned design on purpose: row order legitimately
    # perturbs an SVD of an ill-conditioned one beyond ALGEBRAIC.
    design, target = _well_conditioned()
    order = np.random.default_rng(7).permutation(design.shape[0])
    plain, _ = solve_penalized_least_squares(design, target)
    permuted, _ = solve_penalized_least_squares(design[order], target[order])
    np.testing.assert_allclose(permuted, plain, rtol=ALGEBRAIC, atol=ALGEBRAIC)


def test_an_ill_conditioned_design_is_solved_to_its_conditioning_and_reports_it() -> None:
    design, truth, target = _vandermonde_fixture()
    coefficients, diagnostics = solve_penalized_least_squares(design, target)
    # The forward error of a backward-stable least-squares solve is bounded by
    # cond(A) * eps * ||c||; here 1.18e8 * 2.2e-16 * 25.5 = 6.7e-7. The bound is
    # asserted rather than a fitted constant, and the measured error is 9.8e-9.
    bound = diagnostics.condition_number * EPS * float(np.linalg.norm(truth))
    assert diagnostics.condition_number > 1e7
    assert float(np.max(np.abs(coefficients - truth))) < bound


def test_the_stacked_solve_beats_the_normal_equations_on_the_same_design() -> None:
    design, truth, target = _vandermonde_fixture()
    coefficients, _ = solve_penalized_least_squares(design, target)
    # This is the measurement behind the module's design decision, asserted so
    # the decision cannot be quietly reverted: forming A^T A squares 1.18e8 into
    # 1.4e16, past what float64 carries. Measured: 9.8e-9 against 2.6e-1.
    normal = np.linalg.solve(design.T @ design, design.T @ target)
    stacked_error = float(np.max(np.abs(coefficients - truth)))
    normal_error = float(np.max(np.abs(normal - truth)))
    assert normal_error > 1e4 * stacked_error


def test_a_penalty_lowers_the_reported_condition_number_and_raises_the_residual() -> None:
    design, _truth, target = _vandermonde_fixture()
    _, plain = solve_penalized_least_squares(design, target)
    _, ridged = solve_penalized_least_squares(
        design, target, penalties=[TikhonovPenalty.ridge(12, 1e-6)]
    )
    assert ridged.condition_number < plain.condition_number / 1e3
    assert ridged.residual_norm > plain.residual_norm
    assert ridged.penalty_norm > 0.0


def test_a_rank_deficient_system_is_reported_and_not_hidden() -> None:
    design = np.ones((3, 2))  # two identical columns: one direction is free
    coefficients, diagnostics = solve_penalized_least_squares(design, np.ones(3))
    assert diagnostics.effective_rank == 1
    assert diagnostics.rank_deficient is True
    assert diagnostics.condition_number > 1e12
    # The minimum-norm member of the family, not an arbitrary one.
    np.testing.assert_allclose(coefficients, [0.5, 0.5], rtol=ALGEBRAIC, atol=ALGEBRAIC)


def test_a_penalty_restores_the_rank_a_degenerate_design_lost() -> None:
    design = np.ones((3, 2))
    _, diagnostics = solve_penalized_least_squares(
        design, np.ones(3), penalties=[TikhonovPenalty.ridge(2, 1e-6)]
    )
    assert diagnostics.rank_deficient is False


def test_the_diagnostics_survive_a_strict_json_round_trip() -> None:
    record = LeastSquaresDiagnostics(
        residual_norm=1.0,
        penalty_norm=2.0,
        effective_rank=1,
        condition_number=float("inf"),
        n_parameters=3,
    )
    restored = json.loads(json.dumps(record.to_dict(), allow_nan=False))
    assert restored["condition_number"] is None  # Infinity is not valid JSON
    assert restored["rank_deficient"] is True


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"target": [1.0, 2.0]}, "one entry per design row"),
        ({"design": [[1.0, 2.0], [np.nan, 1.0], [0.0, 1.0]]}, "design must be finite"),
        ({"target": [1.0, np.inf, 3.0]}, "target must be finite"),
        ({"weights": [1.0, -1.0, 1.0]}, "weights must be non-negative"),
        ({"weights": [1.0, 1.0]}, "one entry per design row"),
        ({"penalties": [TikhonovPenalty.ridge(3, 1.0)]}, "must act on 2 coefficients"),
        ({"penalties": [np.eye(2)]}, "must be a TikhonovPenalty"),
        ({"design": [1.0, 2.0, 3.0]}, "design must be two-dimensional"),
    ],
    ids=[
        "short-target",
        "nan-design",
        "inf-target",
        "negative-weight",
        "short-weights",
        "wrong-width-penalty",
        "bare-matrix-penalty",
        "one-dimensional-design",
    ],
)
def test_a_malformed_penalized_problem_is_refused(kwargs: dict, message: str) -> None:
    call = {"design": [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], "target": [1.0, 2.0, 3.0]}
    call.update(kwargs)
    design = call.pop("design")
    target = call.pop("target")
    with pytest.raises(SurfaceValidationError, match=message):
        solve_penalized_least_squares(design, target, **call)


# --- coupling a parameter path ------------------------------------------------


def test_coupling_at_zero_weight_returns_the_fitted_path_unchanged() -> None:
    path = np.array([0.20, 0.30, 0.20, 0.30, 0.20])
    # The identity claim, through the general solver rather than a shortcut
    # branch: at weight 0 the stacked system is the identity, so ALGEBRAIC is
    # the tolerance for a second code path (measured difference: bitwise 0).
    np.testing.assert_allclose(
        couple_parameter_sequence(path, weight=0.0), path, rtol=ALGEBRAIC, atol=ALGEBRAIC
    )


def test_coupling_tends_to_the_sequence_mean_as_the_weight_grows() -> None:
    path = np.array([0.20, 0.30, 0.20, 0.30, 0.20])
    coupled = couple_parameter_sequence(path, weight=1e12)
    np.testing.assert_allclose(coupled, np.full(5, path.mean()), rtol=0.0, atol=MEAN_LIMIT)


def test_coupling_tends_to_the_weighted_mean_when_the_steps_are_weighted() -> None:
    path = np.array([0.20, 0.30, 0.20, 0.30, 0.20])
    step_weights = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    coupled = couple_parameter_sequence(path, weight=1e12, data_weights=step_weights)
    weighted_mean = float((step_weights * path).sum() / step_weights.sum())
    np.testing.assert_allclose(coupled, np.full(5, weighted_mean), rtol=0.0, atol=MEAN_LIMIT)


def test_the_coupled_path_is_smoother_than_the_fitted_one() -> None:
    path = np.array([0.20, 0.30, 0.20, 0.30, 0.20])
    roughness = float(np.abs(np.diff(couple_parameter_sequence(path, weight=1.0))).sum())
    assert roughness < float(np.abs(np.diff(path)).sum())


def test_an_affine_path_survives_second_order_coupling_unchanged() -> None:
    # A second-difference penalty is exactly zero on an affine path, so the
    # minimizer is the path itself; the residual 5e-15 is the solve, not bias.
    path = 1.0 + 0.5 * np.arange(7.0)
    for weight in (3.7, 1e6):
        coupled = couple_parameter_sequence(path, weight=weight, order=2)
        np.testing.assert_allclose(coupled, path, rtol=ALGEBRAIC, atol=ALGEBRAIC)


def test_the_columns_of_a_parameter_block_do_not_interact() -> None:
    path = np.array([0.20, 0.30, 0.20, 0.30, 0.20])
    block = np.column_stack([path, 100.0 * path])
    coupled = couple_parameter_sequence(block, weight=7.0)
    alone = couple_parameter_sequence(path, weight=7.0)
    np.testing.assert_allclose(coupled[:, 0], alone, rtol=ALGEBRAIC, atol=ALGEBRAIC)
    np.testing.assert_allclose(coupled[:, 1], 100.0 * alone, rtol=ALGEBRAIC, atol=ALGEBRAIC)


def test_a_step_with_zero_weight_is_reconstructed_from_its_neighbours() -> None:
    corrupted = [1.0, 2.0, 999.0, 4.0, 5.0]
    filled = couple_parameter_sequence(
        corrupted, weight=1.0, data_weights=[1.0, 1.0, 0.0, 1.0, 1.0]
    )
    # By symmetry the untrusted step lands exactly on the ramp it interrupts.
    assert abs(float(filled[2]) - 3.0) < ALGEBRAIC
    assert float(filled.max()) < 5.0


def test_reversing_a_path_reverses_its_coupling() -> None:
    # The stencil of an even-order difference is symmetric, so D^T D is
    # persymmetric and the coupling commutes with reversing the index.
    path = np.array([0.10, 0.40, 0.20, 0.35, 0.15, 0.50])
    forward = couple_parameter_sequence(path, weight=2.5, order=2)
    backward = couple_parameter_sequence(path[::-1], weight=2.5, order=2)[::-1]
    np.testing.assert_allclose(backward, forward, rtol=ALGEBRAIC, atol=ALGEBRAIC)


def test_the_coupled_path_keeps_the_shape_it_was_given() -> None:
    path = np.array([0.20, 0.30, 0.20, 0.30, 0.20])
    assert couple_parameter_sequence(path, weight=1.0).shape == (5,)
    assert couple_parameter_sequence(path[:, None], weight=1.0).shape == (5, 1)
    assert couple_parameter_sequence(np.column_stack([path] * 3), weight=1.0).shape == (5, 3)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"weight": -1e-9}, "weight must be non-negative"),
        ({"weight": float("inf")}, "weight must be finite"),
        ({"order": 0}, "order must be at least 1"),
        ({"parameters": [[1.0, 2.0], [3.0]]}, "rectangular array of real numbers"),
        ({"parameters": [1.0, 2.0], "order": 2}, "steps; got 2"),
        ({"parameters": [1.0, np.nan, 3.0]}, "parameters must be finite"),
        ({"data_weights": [1.0, 1.0]}, "one weight per step"),
        ({"data_weights": [1.0, -1.0, 1.0]}, "data_weights must be non-negative"),
        ({"data_weights": [0.0, 0.0, 0.0]}, "strictly positive entries"),
        ({"data_weights": [1.0, 1.0, 0.0], "weight": 0.0}, "strictly positive entries"),
    ],
    ids=[
        "negative-weight",
        "infinite-weight",
        "order-zero",
        "ragged-path",
        "too-short-for-order",
        "nan-parameter",
        "short-data-weights",
        "negative-data-weight",
        "no-anchored-step",
        "uncoupled-unanchored-step",
    ],
)
def test_a_malformed_coupling_request_is_refused(kwargs: dict, message: str) -> None:
    call = {"parameters": [1.0, 2.0, 3.0], "weight": 1.0}
    call.update(kwargs)
    parameters = call.pop("parameters")
    with pytest.raises(SurfaceValidationError, match=message):
        couple_parameter_sequence(parameters, **call)


# --- the conditioning diagnostic tells the truth about a null space -----------


def test_a_rank_deficient_system_reports_infinite_conditioning() -> None:
    """A singular operator's condition number is infinite, and lstsq hides that.

    ``numpy.linalg.lstsq`` returns only ``min(rows, columns)`` singular values,
    so on a wide system the zero ones are never in the list and a ratio over
    what is there describes a range direction rather than the operator.  The
    over-complete basis below -- nine columns fitted to six points, the module's
    own headline motivation -- has three exactly null directions, and reporting a
    comfortable four-digit conditioning for it would be worse than reporting
    nothing.
    """
    x = np.linspace(-0.4, 0.4, 6)
    design = np.column_stack([x**power for power in range(9)])
    _, diagnostics = solve_penalized_least_squares(design, 0.2 + 0.1 * x**2)
    assert diagnostics.effective_rank == 6
    assert diagnostics.n_parameters == 9
    assert diagnostics.rank_deficient is True
    assert diagnostics.condition_number == float("inf")
    assert diagnostics.to_dict()["condition_number"] is None


def test_an_inert_penalty_does_not_change_the_reported_conditioning() -> None:
    # Same objective, same minimiser, same infinite solution family. A zero-weight
    # penalty is documented as inert, so a diagnostic that changed across it would
    # be describing the stacking rather than the problem.
    design = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    target = [1.0, 2.0]
    plain = solve_penalized_least_squares(design, target)[1]
    padded = solve_penalized_least_squares(
        design, target, penalties=[TikhonovPenalty.ridge(3, 0.0)]
    )[1]
    assert plain.condition_number == padded.condition_number == float("inf")
    assert plain.effective_rank == padded.effective_rank


def test_a_full_rank_system_reports_the_singular_value_ratio_it_claims() -> None:
    # Pinned against an independent SVD of the same stacked matrix, so inflating
    # the reported number by any factor fails here rather than merely loosening
    # some other test's bound.
    x = np.linspace(-0.4, 0.4, 6)
    design = np.column_stack([np.ones_like(x), x, x**2])
    _, diagnostics = solve_penalized_least_squares(design, 0.2 + 0.1 * x**2)
    singular = np.linalg.svd(design, compute_uv=False)
    assert diagnostics.rank_deficient is False
    np.testing.assert_allclose(
        diagnostics.condition_number, singular[0] / singular[-1], rtol=1e-12, atol=0.0
    )


def test_a_penalty_lowers_the_reported_conditioning_by_the_amount_it_should() -> None:
    # Again pinned to an independent SVD of the stacked operator, not to a ratio
    # between two reported numbers that a common factor would cancel out of.
    x = np.linspace(-0.4, 0.4, 8)
    design = np.column_stack([x**power for power in range(6)])
    weight = 1e-3
    penalty = TikhonovPenalty.ridge(6, weight)
    _, ridged = solve_penalized_least_squares(design, 0.2 + 0.1 * x**2, penalties=[penalty])
    stacked = np.vstack([design, np.sqrt(weight) * np.eye(6)])
    singular = np.linalg.svd(stacked, compute_uv=False)
    np.testing.assert_allclose(
        ridged.condition_number, singular[0] / singular[-1], rtol=1e-12, atol=0.0
    )
    plain = solve_penalized_least_squares(design, 0.2 + 0.1 * x**2)[1]
    assert ridged.condition_number < plain.condition_number
