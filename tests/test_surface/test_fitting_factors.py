"""A factor basis recovers a known subspace, and its signs do not depend on LAPACK."""

from __future__ import annotations

import inspect
import json

import numpy as np
import pytest

from fast_vollib.surface import (
    DefiniteIVSurface,
    GridIVSurface,
    IVSurface,
    SurfaceCalibrationError,
    SurfaceCalibrator,
    SurfaceGridSpec,
    SurfaceObservations,
    SurfacePoints,
    SurfaceValidationError,
)
from fast_vollib.surface.fitting.factors import (
    FactorIVSurface,
    FactorSurfaceCalibrator,
    SurfaceFactorBasis,
    _component_signs,
    fit_factor_basis,
)

# --- predeclared tolerances --------------------------------------------------
#
# EXACT_ATOL covers every claim that is exact in exact arithmetic: an SVD
# reconstruction of a matrix that lies in the retained subspace, a
# projection/reconstruction round trip, and a grid read at a node. The backward
# error of a LAPACK float64 SVD is O(eps * ||X||), and this ensemble has
# ||X|| ~ 1e-1, so the observed errors are ~3e-17. Four to five orders of margin
# is deliberate: nothing here was widened to make a test pass.
EXACT_ATOL = 1e-12

# Loadings compared between two algebraically identical fits -- the same
# centred matrix negated, or with its rows permuted. The row space is
# identical, but LAPACK reduces the rows in a different order, so the singular
# vectors agree to the componentwise accuracy of the decomposition rather than
# bit for bit; the observed deviation on unit-norm rows is ~4e-15.
SIGN_STABILITY_ATOL = 1e-12

# The relative band inside which two entries of a component count as tied for
# the pivot. Restated here rather than imported so that these tests state the
# convention independently of the code that implements it.
PIVOT_TIE_RTOL = 1e-9

# --- a synthetic ensemble of known rank --------------------------------------

GRID_K = np.linspace(-0.30, 0.30, 7)
GRID_T = np.array([0.25, 0.50, 1.00])
MEAN_W = 0.04 + 0.02 * np.ones((7, 1)) * GRID_T[None, :]

#: Three known node-space directions: a term-proportional level, a skew that is
#: antisymmetric in moneyness, and a curvature. The skew is what makes the tie
#: rule load-bearing -- its extreme entries are equal in magnitude.
MODES = (
    0.010 * np.ones((7, 1)) * GRID_T[None, :],
    0.005 * GRID_K[:, None] * np.ones((1, 3)),
    0.004 * GRID_K[:, None] ** 2 * np.ones((1, 3)),
)

#: Coefficients whose columns sum to zero, so the ensemble mean is exactly
#: MEAN_W, and whose columns are independent, so the rank is the column count.
RANK_TWO = np.array([[-2.0, 1.0], [-1.0, -1.0], [0.0, 0.0], [1.0, -1.0], [2.0, 1.0]])
RANK_THREE = np.array(
    [
        [-2.0, 1.0, 0.5],
        [-1.0, -1.0, -1.0],
        [0.0, 0.0, 1.0],
        [1.0, -1.0, -1.0],
        [2.0, 1.0, 0.5],
    ]
)


def total_variance(coefficients: np.ndarray) -> np.ndarray:
    """The ensemble's total-variance nodes, shape ``(n_surfaces, 7, 3)``."""
    coefficients = np.asarray(coefficients, dtype=np.float64)
    return np.stack(
        [
            MEAN_W + sum(row[j] * MODES[j] for j in range(coefficients.shape[1]))
            for row in coefficients
        ]
    )


def ensemble(coefficients: np.ndarray) -> list[IVSurface]:
    """The same ensemble as implied-volatility surfaces on the reference grid."""
    return [
        IVSurface.from_total_variance(GRID_K, GRID_T, values)
        for values in total_variance(coefficients)
    ]


def pivot_of(component: np.ndarray) -> int:
    """The lowest node index whose magnitude ties the component's largest."""
    tied = np.abs(component) >= (1.0 - PIVOT_TIE_RTOL) * np.max(np.abs(component))
    return int(np.argmax(tied))


def complete_observations(surface: IVSurface, basis: SurfaceFactorBasis) -> SurfaceObservations:
    """``surface`` observed at every node of ``basis``, in the grid's own order."""
    return SurfaceObservations.from_points(
        basis.grid.to_points(), iv=np.asarray(surface.iv).reshape(-1)
    )


# --- the basis recovers a known subspace -------------------------------------


def test_a_rank_two_ensemble_is_reconstructed_to_machine_precision() -> None:
    basis = fit_factor_basis(ensemble(RANK_TWO))
    assert basis.n_factors == 2
    for values in total_variance(RANK_TWO):
        rebuilt = basis.reconstruct(basis.project(values))
        np.testing.assert_allclose(rebuilt, values, rtol=0.0, atol=EXACT_ATOL)


def test_the_recovered_loadings_span_the_subspace_the_ensemble_was_built_from() -> None:
    # The basis is unique only up to a rotation inside a degenerate block, so
    # the loadings are never compared elementwise to the modes. The testable
    # claim is that the two spans coincide, which is a residual on both sides.
    basis = fit_factor_basis(ensemble(RANK_TWO))
    loadings = basis.loadings
    for mode in MODES[:2]:
        vector = mode.reshape(-1)
        residual = vector - loadings.T @ (loadings @ vector)
        assert np.max(np.abs(residual)) <= EXACT_ATOL * np.linalg.norm(vector)
    span, _ = np.linalg.qr(np.column_stack([mode.reshape(-1) for mode in MODES[:2]]))
    for component in loadings:
        residual = component - span @ (span.T @ component)
        assert np.max(np.abs(residual)) <= EXACT_ATOL


def test_the_kept_components_are_the_ensembles_numerical_rank() -> None:
    # The default keeps s_j > max(m, n) * eps * max|X|, the resolution the
    # centring itself leaves. The cut is nowhere near a boundary here: the
    # second singular value sits thirteen orders above that floor and the third
    # an order below it.
    matrix = total_variance(RANK_TWO).reshape(5, -1)
    spectrum = np.linalg.svd(matrix - matrix.mean(axis=0), compute_uv=False)
    resolution = max(matrix.shape) * float(np.finfo(np.float64).eps) * np.max(np.abs(matrix))
    assert spectrum[1] / resolution > 1e10
    assert spectrum[2] / resolution < 0.1
    assert fit_factor_basis(ensemble(RANK_TWO)).n_factors == 2


def test_the_mean_is_the_node_wise_mean_of_the_ensemble() -> None:
    basis = fit_factor_basis(ensemble(RANK_TWO))
    np.testing.assert_allclose(basis.mean.reshape(7, 3), MEAN_W, rtol=0.0, atol=EXACT_ATOL)


# --- the sign convention ------------------------------------------------------


def test_every_component_is_positive_at_its_pivot_entry() -> None:
    for coefficients in (RANK_TWO, RANK_THREE):
        basis = fit_factor_basis(ensemble(coefficients))
        for component in basis.loadings:
            assert component[pivot_of(component)] > 0.0


def test_the_sign_convention_breaks_a_tie_at_the_lowest_flat_node_index() -> None:
    # Row 0 ties at nodes 0 and 1 and is left alone because node 0 is positive;
    # row 1 ties at the same pair and is flipped because node 0 is negative.
    loadings = np.array([[1.0, -1.0, 0.0], [-2.0, 2.0, 0.5]])
    np.testing.assert_array_equal(_component_signs(loadings), [1.0, -1.0])


def test_a_tie_broken_only_by_rounding_is_still_a_tie() -> None:
    # This is the case a strict argmax gets wrong: the two candidates are equal
    # in exact arithmetic and carry opposite signs, so the last bit of the
    # decomposition would otherwise decide the sign of the whole component.
    np.testing.assert_array_equal(_component_signs(np.array([[-1.0, 1.0 + 2.0e-16]])), [-1.0])
    np.testing.assert_array_equal(_component_signs(np.array([[-1.0, 1.0 - 2.0e-16]])), [-1.0])


def test_refitting_the_same_ensemble_returns_bitwise_identical_loadings() -> None:
    members = ensemble(RANK_TWO)
    first = fit_factor_basis(members)
    second = fit_factor_basis(members)
    np.testing.assert_array_equal(first.loadings, second.loadings)
    np.testing.assert_array_equal(first.singular_values, second.singular_values)


def test_negating_the_ensemble_coefficients_leaves_the_loadings_unchanged() -> None:
    # Negating every coefficient negates the centred matrix, which negates each
    # singular vector LAPACK may return and changes nothing about the subspace.
    # The convention is what makes the returned basis the same one.
    basis = fit_factor_basis(ensemble(RANK_TWO))
    negated = fit_factor_basis(ensemble(-RANK_TWO))
    np.testing.assert_allclose(negated.loadings, basis.loadings, rtol=0.0, atol=SIGN_STABILITY_ATOL)
    for component in negated.loadings:
        assert component[pivot_of(component)] > 0.0


def test_permuting_the_ensemble_leaves_the_loadings_and_the_scores_unchanged() -> None:
    order = [3, 0, 4, 1, 2]
    members = ensemble(RANK_TWO)
    basis = fit_factor_basis(members)
    permuted = fit_factor_basis([members[index] for index in order])
    np.testing.assert_allclose(
        permuted.loadings, basis.loadings, rtol=0.0, atol=SIGN_STABILITY_ATOL
    )
    for values in total_variance(RANK_TWO):
        np.testing.assert_allclose(
            permuted.project(values), basis.project(values), rtol=0.0, atol=SIGN_STABILITY_ATOL
        )


# --- explained variance -------------------------------------------------------


def test_explained_variance_ratios_are_non_increasing_and_sum_to_at_most_one() -> None:
    for coefficients in (RANK_TWO, RANK_THREE):
        ratio = fit_factor_basis(ensemble(coefficients)).explained_variance_ratio
        assert bool(np.all(np.diff(ratio) <= 0.0))
        assert bool(np.all(ratio >= 0.0))
        # Summing at most a few dozen float64 terms cannot drift past this.
        assert float(ratio.sum()) <= 1.0 + EXACT_ATOL


def test_a_truncated_basis_reports_the_share_it_kept_rather_than_renormalizing() -> None:
    basis = fit_factor_basis(ensemble(RANK_THREE))
    truncated = basis.truncate(1)
    assert truncated.n_factors == 1
    assert float(truncated.explained_variance_ratio.sum()) < 1.0
    np.testing.assert_array_equal(
        truncated.explained_variance_ratio, basis.explained_variance_ratio[:1]
    )


def test_truncating_to_fewer_factors_increases_the_reconstruction_error() -> None:
    basis = fit_factor_basis(ensemble(RANK_THREE))
    assert basis.n_factors == 3
    values = total_variance(RANK_THREE)
    errors = [
        max(
            float(np.sqrt(np.mean((part.reconstruct(part.project(one)) - one) ** 2)))
            for one in values
        )
        for part in (basis.truncate(1), basis.truncate(2), basis)
    ]
    assert errors[0] > errors[1] > errors[2]
    assert errors[2] <= EXACT_ATOL


# --- round trips --------------------------------------------------------------


def test_a_surface_rebuilt_from_all_its_factors_equals_the_original() -> None:
    basis = fit_factor_basis(ensemble(RANK_THREE))
    for values in total_variance(RANK_THREE):
        np.testing.assert_allclose(
            basis.reconstruct(basis.project(values)), values, rtol=0.0, atol=EXACT_ATOL
        )


def test_the_fitted_surface_reproduces_its_own_grid_nodes() -> None:
    # The read squares the node volatility and takes the root back, a two-ulp
    # round trip, and interpolation at a node is exact by construction.
    members = ensemble(RANK_TWO)
    basis = fit_factor_basis(members)
    surface = FactorIVSurface(basis=basis, scores=basis.project(total_variance(RANK_TWO)[0]))
    prediction = surface.evaluate(basis.grid.to_points())
    np.testing.assert_allclose(
        prediction.iv, np.asarray(members[0].iv).reshape(-1), rtol=EXACT_ATOL, atol=0.0
    )


def test_projecting_a_training_surface_recovers_the_scores_that_rebuild_it() -> None:
    members = ensemble(RANK_TWO)
    basis = fit_factor_basis(members)
    calibrator = FactorSurfaceCalibrator(basis=basis)
    for member, values in zip(members, total_variance(RANK_TWO)):
        fitted = calibrator.fit(complete_observations(member, basis))
        np.testing.assert_allclose(fitted.scores, basis.project(values), rtol=0.0, atol=EXACT_ATOL)


def test_a_surface_observed_off_the_nodes_projects_to_the_scores_that_generated_it() -> None:
    # The design matrix is the model's own linear map -- each column is a
    # loading read through the same interpolator the surface evaluates through
    # -- so an exactly-representable surface is recovered from scattered
    # interior quotes, not merely approximated by them.
    basis = fit_factor_basis(ensemble(RANK_TWO))
    scores = basis.project(total_variance(RANK_TWO)[0])
    generator = np.random.default_rng(20260831)
    points = SurfacePoints(k=generator.uniform(-0.25, 0.25, 12), T=generator.uniform(0.3, 0.9, 12))
    quotes = FactorIVSurface(basis=basis, scores=scores).evaluate(points)
    fitted = FactorSurfaceCalibrator(basis=basis).fit(
        SurfaceObservations.from_points(points, iv=quotes.iv)
    )
    np.testing.assert_allclose(fitted.scores, scores, rtol=0.0, atol=EXACT_ATOL)


def test_an_ensemble_of_materialized_observations_gives_the_same_basis_as_one_of_surfaces() -> None:
    members = ensemble(RANK_TWO)
    basis = fit_factor_basis(members)
    from_rows = fit_factor_basis(
        [complete_observations(member, basis) for member in members], grid=basis.grid
    )
    np.testing.assert_array_equal(from_rows.loadings, basis.loadings)
    np.testing.assert_array_equal(from_rows.mean, basis.mean)


def test_a_basis_fitted_in_implied_volatility_factors_sigma_rather_than_total_variance() -> None:
    members = ensemble(RANK_TWO)
    basis = fit_factor_basis(members, value_space="implied_volatility")
    assert basis.value_space == "implied_volatility"
    # sigma -> sigma^2 T is not linear, so an ensemble of rank two in total
    # variance is of full centred rank here; that is the point of the choice.
    assert basis.n_factors == len(members) - 1
    observed = np.stack([np.asarray(member.iv).reshape(-1) for member in members])
    np.testing.assert_allclose(basis.mean, observed.mean(axis=0), rtol=0.0, atol=EXACT_ATOL)
    for member, values in zip(members, observed):
        np.testing.assert_allclose(
            basis.reconstruct(basis.project(values)).reshape(-1),
            values,
            rtol=0.0,
            atol=EXACT_ATOL,
        )
    surface = FactorIVSurface(basis=basis, scores=basis.project(observed[0]))
    assert surface.policy == "implied_volatility"


# --- evaluation declines rather than extrapolates ------------------------------


def test_a_point_outside_the_reference_grid_is_declined_rather_than_extrapolated() -> None:
    basis = fit_factor_basis(ensemble(RANK_TWO))
    surface = FactorIVSurface(basis=basis, scores=np.zeros(basis.n_factors))
    prediction = surface.evaluate(SurfacePoints(k=[0.0, 3.0, 0.0], T=[0.5, 0.5, 9.0]))
    assert prediction.valid.tolist() == [True, False, False]
    assert prediction.invalid_count == 2


def test_evaluation_answers_every_query_row_in_query_order() -> None:
    basis = fit_factor_basis(ensemble(RANK_TWO))
    surface = FactorIVSurface(basis=basis, scores=basis.project(total_variance(RANK_TWO)[0]))
    k = np.array([-0.2, 0.05, 0.1])
    T = np.array([0.4, 0.6, 0.9])
    straight = surface.evaluate(SurfacePoints(k=k, T=T))
    order = np.array([2, 0, 1])
    shuffled = surface.evaluate(SurfacePoints(k=k[order], T=T[order]))
    assert straight.n == 3
    np.testing.assert_array_equal(shuffled.iv, straight.iv[order])


def test_the_surface_reads_through_the_one_grid_interpolator_this_package_has() -> None:
    # Bit-for-bit equality with the shared adapter is the check that there is no
    # second interpolator hiding in this module.
    basis = fit_factor_basis(ensemble(RANK_TWO))
    surface = FactorIVSurface(basis=basis, scores=basis.project(total_variance(RANK_TWO)[0]))
    points = SurfacePoints(k=[-0.11, 0.07], T=[0.4, 0.8])
    reader = GridIVSurface(surface.to_grid(), policy="total_variance", extrapolation="invalid")
    np.testing.assert_array_equal(surface.evaluate(points).iv, reader.evaluate(points).iv)


def test_the_read_policy_defaults_to_the_space_the_basis_was_fitted_in() -> None:
    basis = fit_factor_basis(ensemble(RANK_TWO))
    assert FactorIVSurface(basis=basis, scores=[0.0, 0.0]).policy == "total_variance"
    assert FactorIVSurface(basis=basis, scores=[0.0, 0.0], policy="nearest").policy == "nearest"
    with pytest.raises(SurfaceValidationError, match="policy must be one of"):
        FactorIVSurface(basis=basis, scores=[0.0, 0.0], policy="cubic")


# --- missing data is refused, never imputed ------------------------------------


def test_a_missing_node_is_refused_rather_than_imputed() -> None:
    values = total_variance(RANK_TWO)
    values[2, 3, 1] = np.nan
    members = [IVSurface.from_total_variance(GRID_K, GRID_T, one) for one in values]
    with pytest.raises(SurfaceValidationError, match="Every node must be present"):
        fit_factor_basis(members)


def test_the_message_for_a_missing_node_names_the_surface_and_the_remedy() -> None:
    values = total_variance(RANK_TWO)
    values[1] = np.nan
    members = [IVSurface.from_total_variance(GRID_K, GRID_T, one) for one in values]
    with pytest.raises(SurfaceValidationError, match="surface 1 has 21 missing node"):
        fit_factor_basis(members)
    with pytest.raises(SurfaceValidationError, match="Materialize a complete grid first"):
        fit_factor_basis(members)


def test_projecting_a_grid_with_a_hole_is_refused() -> None:
    basis = fit_factor_basis(ensemble(RANK_TWO))
    values = total_variance(RANK_TWO)[0]
    values[0, 0] = np.nan
    with pytest.raises(SurfaceValidationError, match="Every node must be present"):
        basis.project(values)


def test_a_missing_quote_costs_its_row_and_not_the_fit() -> None:
    # The observed surface lies exactly in the basis, so every retained row has
    # zero residual and dropping one leaves the same least-squares solution.
    members = ensemble(RANK_TWO)
    basis = fit_factor_basis(members)
    iv = np.asarray(members[0].iv).reshape(-1).copy()
    iv[4] = np.nan
    fitted = FactorSurfaceCalibrator(basis=basis).fit(
        SurfaceObservations.from_points(basis.grid.to_points(), iv=iv)
    )
    np.testing.assert_allclose(
        fitted.scores, basis.project(total_variance(RANK_TWO)[0]), rtol=0.0, atol=EXACT_ATOL
    )


def test_a_zero_weight_row_cannot_reach_the_fit() -> None:
    members = ensemble(RANK_TWO)
    basis = fit_factor_basis(members)
    iv = np.asarray(members[0].iv).reshape(-1).copy()
    weight = np.ones(iv.size)
    iv[7], weight[7] = 5.0, 0.0
    fitted = FactorSurfaceCalibrator(basis=basis).fit(
        SurfaceObservations.from_points(basis.grid.to_points(), iv=iv, weight=weight)
    )
    np.testing.assert_allclose(
        fitted.scores, basis.project(total_variance(RANK_TWO)[0]), rtol=0.0, atol=EXACT_ATOL
    )


# --- a fit the calibrator cannot stand behind ----------------------------------


def test_fewer_usable_observations_than_factors_is_a_calibration_error() -> None:
    basis = fit_factor_basis(ensemble(RANK_TWO))
    single = SurfaceObservations(k=[0.0], T=[0.5], iv=[0.3])
    with pytest.raises(SurfaceCalibrationError, match="must use at least 2 observation"):
        FactorSurfaceCalibrator(basis=basis).fit(single)


def test_a_rank_deficient_projection_is_a_calibration_error() -> None:
    # Two rows, but both at the same coordinate: the design matrix has rank one
    # and any split of the score between the two components fits equally well.
    basis = fit_factor_basis(ensemble(RANK_TWO))
    repeated = SurfaceObservations(k=[0.0, 0.0], T=[0.5, 0.5], iv=[0.3, 0.3])
    with pytest.raises(SurfaceCalibrationError, match="must resolve all 2 factor"):
        FactorSurfaceCalibrator(basis=basis).fit(repeated)


def test_observations_entirely_outside_the_reference_grid_are_a_calibration_error() -> None:
    basis = fit_factor_basis(ensemble(RANK_TWO))
    elsewhere = SurfaceObservations(k=[4.0, 4.1, 4.2], T=[0.5, 0.5, 0.5], iv=[0.3, 0.3, 0.3])
    with pytest.raises(SurfaceCalibrationError, match="must use at least 2 observation"):
        FactorSurfaceCalibrator(basis=basis).fit(elsewhere)


def test_an_ensemble_that_does_not_vary_has_no_direction_to_factor() -> None:
    # The three surfaces are bit-for-bit identical, and their centred matrix is
    # still not exactly zero: the mean of three equal floats is not exactly
    # either of them. A guard against zero alone would hand back that ~1e-18 of
    # rounding error as a component with a mean, a loading, and a sign.
    members = ensemble(np.zeros((3, 2)))
    values = np.stack([np.asarray(member.iv).reshape(-1) for member in members])
    np.testing.assert_array_equal(values[0], values[2])
    assert 0.0 < np.max(np.abs(values - values.mean(axis=0))) < 1e-15
    with pytest.raises(SurfaceCalibrationError, match="An ensemble must vary by more than"):
        fit_factor_basis(members)


def test_asking_for_more_factors_than_the_ensemble_resolves_is_refused() -> None:
    with pytest.raises(SurfaceCalibrationError, match="must be at most the 2 component"):
        fit_factor_basis(ensemble(RANK_TWO), n_factors=3)
    with pytest.raises(SurfaceValidationError, match="n_factors must be at least 1"):
        fit_factor_basis(ensemble(RANK_TWO), n_factors=0)


# --- validation of the value objects -------------------------------------------


def test_an_unknown_value_space_is_refused() -> None:
    with pytest.raises(SurfaceValidationError, match="value_space must be one of"):
        fit_factor_basis(ensemble(RANK_TWO), value_space="log_variance")


def test_an_ensemble_needs_at_least_two_surfaces() -> None:
    with pytest.raises(SurfaceValidationError, match="at least 2 surfaces"):
        fit_factor_basis(ensemble(RANK_TWO)[:1])


def test_surfaces_on_different_meshes_cannot_share_a_basis() -> None:
    members = ensemble(RANK_TWO)
    members[1] = IVSurface.from_total_variance(GRID_K, GRID_T * 2.0, total_variance(RANK_TWO)[1])
    with pytest.raises(SurfaceValidationError, match="must be on the same reference grid"):
        fit_factor_basis(members)


def test_observations_without_a_grid_cannot_declare_a_mesh() -> None:
    basis = fit_factor_basis(ensemble(RANK_TWO))
    rows = [complete_observations(member, basis) for member in ensemble(RANK_TWO)]
    with pytest.raises(SurfaceValidationError, match="grid must be given"):
        fit_factor_basis(rows)


def test_loadings_that_are_not_orthonormal_are_refused() -> None:
    grid = SurfaceGridSpec(k=[-0.1, 0.0, 0.1], T=[1.0])
    with pytest.raises(SurfaceValidationError, match="must have orthonormal rows"):
        SurfaceFactorBasis(
            grid=grid,
            mean=[0.04, 0.04, 0.04],
            loadings=[[1.0, 1.0, 1.0]],
            singular_values=[0.01],
            explained_variance_ratio=[1.0],
        )


def test_a_spectrum_that_is_not_a_set_of_ratios_is_refused() -> None:
    grid = SurfaceGridSpec(k=[-0.1, 0.0, 0.1], T=[1.0])
    loadings = np.array([[-1.0, 0.0, 1.0], [1.0, 0.0, 1.0]]) / np.sqrt(2.0)
    with pytest.raises(SurfaceValidationError, match="must sum to at most 1"):
        SurfaceFactorBasis(
            grid=grid,
            mean=[0.04, 0.04, 0.04],
            loadings=loadings,
            singular_values=[0.02, 0.01],
            explained_variance_ratio=[0.8, 0.6],
        )
    with pytest.raises(SurfaceValidationError, match="singular_values must be non-increasing"):
        SurfaceFactorBasis(
            grid=grid,
            mean=[0.04, 0.04, 0.04],
            loadings=loadings,
            singular_values=[0.01, 0.02],
            explained_variance_ratio=[0.6, 0.3],
        )


def test_the_basis_arrays_are_owned_read_only_copies() -> None:
    mean = np.array([0.04, 0.04, 0.04])
    basis = SurfaceFactorBasis(
        grid=SurfaceGridSpec(k=[-0.1, 0.0, 0.1], T=[1.0]),
        mean=mean,
        loadings=np.array([[-1.0, 0.0, 1.0]]) / np.sqrt(2.0),
        singular_values=[0.01],
        explained_variance_ratio=[1.0],
    )
    mean[0] = 99.0
    assert basis.mean[0] == 0.04
    assert basis.mean.flags.writeable is False
    assert basis.loadings.flags.writeable is False
    assert basis.singular_values.flags.writeable is False


def test_a_score_vector_must_match_the_basis_it_is_expressed_in() -> None:
    basis = fit_factor_basis(ensemble(RANK_TWO))
    with pytest.raises(SurfaceValidationError, match="one entry per component"):
        FactorIVSurface(basis=basis, scores=[0.0])
    with pytest.raises(SurfaceValidationError, match="scores must be finite"):
        FactorIVSurface(basis=basis, scores=[0.0, np.nan])


# --- the protocols -------------------------------------------------------------


def test_the_factor_calibrator_and_its_surface_satisfy_their_protocols() -> None:
    basis = fit_factor_basis(ensemble(RANK_TWO))
    assert isinstance(FactorSurfaceCalibrator(basis=basis), SurfaceCalibrator)
    assert isinstance(FactorIVSurface(basis=basis, scores=[0.0, 0.0]), DefiniteIVSurface)


def test_a_calibrator_holds_no_state_between_fits() -> None:
    members = ensemble(RANK_TWO)
    basis = fit_factor_basis(members)
    calibrator = FactorSurfaceCalibrator(basis=basis)
    first = calibrator.fit(complete_observations(members[0], basis))
    second = calibrator.fit(complete_observations(members[3], basis))
    third = calibrator.fit(complete_observations(members[0], basis))
    np.testing.assert_array_equal(third.scores, first.scores)
    assert not np.array_equal(second.scores, first.scores)


def test_no_signature_names_a_dataset_a_split_or_a_path() -> None:
    assert list(inspect.signature(fit_factor_basis).parameters) == [
        "surfaces",
        "grid",
        "n_factors",
        "value_space",
    ]
    assert list(inspect.signature(FactorSurfaceCalibrator.fit).parameters) == [
        "self",
        "observations",
        "rng",
    ]
    assert list(inspect.signature(FactorIVSurface.evaluate).parameters) == [
        "self",
        "points",
        "market",
    ]


def test_a_fitted_surface_reports_json_safe_parameters() -> None:
    basis = fit_factor_basis(ensemble(RANK_TWO))
    parameters = FactorIVSurface(basis=basis, scores=[0.25, -0.5]).parameters()
    assert json.loads(json.dumps(parameters, allow_nan=False)) == parameters
    assert parameters["value_space"] == "total_variance"


# --- what an adversarial review found the tests could not see -----------------


def test_a_clamped_extrapolation_is_refused_for_fitting() -> None:
    """Clamping reads an off-mesh quote at a node's maturity; the target keeps its own.

    Under ``total_variance`` the design column is then built at ``T_clamped``
    while the target is ``iv^2 T_observed``, so a two-year quote is fitted as if
    it were a one-year one.  Measured before the refusal was added: scores
    sign-flipped and doubled, and a surface generated by this module failed to
    round-trip through its own calibrator with a 42% implied-volatility error.
    Clamping stays available on :class:`FactorIVSurface`, where the maturity
    being read is the one that was asked for.
    """
    from fast_vollib.surface.fitting.factors import FactorPCARecipe

    basis = fit_factor_basis(ensemble(RANK_TWO), n_factors=2)
    with pytest.raises(SurfaceValidationError, match="not available for fitting"):
        FactorSurfaceCalibrator(basis=basis, extrapolation="clamp")
    with pytest.raises(SurfaceValidationError, match="not available for fitting"):
        FactorPCARecipe(extrapolation="clamp")


def test_the_documented_sign_convention_is_the_banded_one() -> None:
    """The headline rule has to be the rule, including the band.

    On this fixture the plain ``argmax`` pivot of component 1 is *negative*: two
    entries tie in magnitude and the last bit decides which one ``argmax``
    returns.  A reader who took the unqualified sentence at face value would
    assert the wrong thing, which is why the module docstring states the banded
    form from its first mention.
    """
    basis = fit_factor_basis(ensemble(RANK_TWO), n_factors=2)
    for row in basis.loadings:
        largest = float(np.max(np.abs(row)))
        tied = np.flatnonzero(np.abs(row) >= largest * (1.0 - 1e-9))
        assert float(row[tied[0]]) > 0.0


def test_the_recipe_trains_a_basis_and_produces_a_calibrator() -> None:
    from fast_vollib.surface.fitting.factors import FactorPCARecipe

    recipe = FactorPCARecipe(n_factors=2)
    basis = recipe.train(ensemble(RANK_TWO))
    assert basis.n_factors == 2
    calibrator = recipe.calibrator(basis)
    assert isinstance(calibrator, FactorSurfaceCalibrator)
    # The recipe's own defaults must not quietly differ from the calibrator's:
    # policy=None means "read in the space the basis was fitted in", and a
    # concrete string here would decouple them without anybody asking.
    assert calibrator.policy is None
    assert FactorPCARecipe().policy is None


def test_the_recipe_holds_no_state_between_trainings() -> None:
    from fast_vollib.surface.fitting.factors import FactorPCARecipe

    recipe = FactorPCARecipe(n_factors=1)
    first = recipe.train(ensemble(RANK_TWO))
    recipe.train(ensemble(RANK_TWO)[:3])
    third = recipe.train(ensemble(RANK_TWO))
    np.testing.assert_array_equal(first.loadings, third.loadings)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"n_factors": 0}, "n_factors must be at least 1"),
        ({"value_space": "price"}, "value_space must be one of"),
        ({"policy": "cubic"}, "policy must be one of"),
        ({"extrapolation": "wrap"}, "extrapolation must be one of"),
    ],
)
def test_invalid_recipe_configuration_is_refused(kwargs, message) -> None:
    from fast_vollib.surface.fitting.factors import FactorPCARecipe

    with pytest.raises(SurfaceValidationError, match=message):
        FactorPCARecipe(**kwargs)
