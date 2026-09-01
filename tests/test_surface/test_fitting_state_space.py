"""Explicit-state Kalman recursions: no hidden state, Joseph covariances, honoured horizons."""

from __future__ import annotations

import numpy as np
import pytest

from fast_vollib.surface import (
    ForecastHorizon,
    SurfaceCalibrationError,
    SurfaceForecaster,
    SurfaceObservations,
    SurfacePoints,
    SurfaceValidationError,
)
from fast_vollib.surface.fitting import (
    FlatIVSurface,
    FlatVolatilityCalibrator,
    PersistenceForecaster,
)
from fast_vollib.surface.fitting.state_space import (
    FilteredPath,
    GaussianState,
    LinearGaussianModel,
    StateSpaceForecaster,
    flat_level_parameters,
    flat_surface_from_parameters,
    kalman_filter,
    kalman_predict,
    kalman_smooth,
    kalman_update,
)

#: A well-conditioned scalar local level: F = 0.97, q = 3e-3, r = 5e-2.
SCALAR = LinearGaussianModel(
    transition_matrix=[[0.97]],
    transition_covariance=[[3e-3]],
    observation_matrix=[[1.0]],
    observation_covariance=[[5e-2]],
)

#: The steady-state test's variances; q / r = 4 makes the iteration contract fast.
STEADY_Q, STEADY_R = 4e-4, 1e-4

#: Query points every forecast comparison is evaluated on.
POINTS = SurfacePoints(k=[-0.2, 0.0, 0.2], T=[0.5, 1.0, 2.0])


def history(levels: tuple[float, ...]) -> list[SurfaceObservations]:
    """One observation set per level, each a flat three-quote smile."""
    return [
        SurfaceObservations(k=[-0.1, 0.0, 0.1], T=[1.0, 1.0, 1.0], iv=[level] * 3)
        for level in levels
    ]


def correlated_model() -> LinearGaussianModel:
    """A three-dimensional model with full-rank, non-diagonal Q and R."""
    rng = np.random.default_rng(7)
    a = rng.standard_normal((3, 3))
    b = rng.standard_normal((3, 3))
    return LinearGaussianModel(
        transition_matrix=[[0.99, 0.02, 0.0], [0.0, 0.95, 0.01], [0.0, 0.0, 0.9]],
        transition_covariance=a @ a.T * 1e-4,
        observation_matrix=[[1.0, 0.0, 0.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0]],
        observation_covariance=b @ b.T * 1e-3,
    )


def simulate_random_walk(q: float, r: float, n: int, seed: int) -> np.ndarray:
    """A scalar random-walk-plus-noise path, shape ``(n, 1)``."""
    rng = np.random.default_rng(seed)
    state = 0.2
    out = np.empty(n)
    for step in range(n):
        state += np.sqrt(q) * rng.standard_normal()
        out[step] = state + np.sqrt(r) * rng.standard_normal()
    return out.reshape(-1, 1)


# --- the containers validate what they hold ----------------------------------


def test_a_gaussian_state_owns_read_only_copies_of_its_arrays() -> None:
    mean = np.array([0.2, 0.3])
    covariance = np.array([[1e-4, 0.0], [0.0, 1e-6]])
    state = GaussianState(mean=mean, covariance=covariance)
    mean[0] = 99.0
    covariance[0, 0] = 99.0
    assert state.mean[0] == 0.2
    assert state.covariance[0, 0] == 1e-4
    assert state.mean.flags.writeable is False
    assert state.covariance.flags.writeable is False


def test_an_asymmetric_covariance_is_refused() -> None:
    with pytest.raises(SurfaceValidationError, match="covariance must be symmetric"):
        GaussianState(mean=[0.0, 0.0], covariance=[[1.0, 0.5], [0.4, 1.0]])


def test_a_covariance_with_a_negative_eigenvalue_is_refused() -> None:
    with pytest.raises(SurfaceValidationError, match="must be positive semi-definite"):
        GaussianState(mean=[0.0, 0.0], covariance=[[1.0, 2.0], [2.0, 1.0]])


def test_asymmetry_within_the_declared_round_off_gate_is_accepted() -> None:
    # The gate is COVARIANCE_TOLERANCE_FACTOR * n * eps * max|P| = 16 * 2 * eps * 1.
    # A perturbation of eps is inside it; one of 1e-12 is not.
    perturbation = float(np.finfo(np.float64).eps)
    GaussianState(mean=[0.0, 0.0], covariance=[[1.0, 0.5 + perturbation], [0.5, 1.0]])
    with pytest.raises(SurfaceValidationError, match="covariance must be symmetric"):
        GaussianState(mean=[0.0, 0.0], covariance=[[1.0, 0.5 + 1e-12], [0.5, 1.0]])


def test_a_non_finite_state_mean_is_refused() -> None:
    with pytest.raises(SurfaceValidationError, match="mean must be finite everywhere"):
        GaussianState(mean=[np.nan], covariance=[[1.0]])


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"transition_matrix": [[1.0, 0.0]]}, "transition_matrix must be a square matrix"),
        ({"observation_matrix": [1.0]}, "observation_matrix must be two-dimensional"),
        ({"observation_matrix": [[1.0, 0.0]]}, "observation_matrix must have 1 column"),
        ({"transition_covariance": [[-1.0]]}, "must be positive semi-definite"),
        ({"drift": [0.0, 0.0]}, "drift must have one entry per state component"),
    ],
)
def test_the_model_validates_its_shapes(kwargs, message) -> None:
    call = {
        "transition_matrix": [[1.0]],
        "transition_covariance": [[1e-4]],
        "observation_matrix": [[1.0]],
        "observation_covariance": [[1e-4]],
    }
    call.update(kwargs)
    with pytest.raises(SurfaceValidationError, match=message):
        LinearGaussianModel(**call)


def test_an_absent_drift_is_stored_as_an_explicit_zero_vector() -> None:
    model = LinearGaussianModel.random_walk(3)
    np.testing.assert_array_equal(model.drift, np.zeros(3))
    assert model.drift.flags.writeable is False


# --- the recursions are functions on values ----------------------------------


def test_kalman_predict_returns_a_new_state_and_advances_nothing() -> None:
    state = GaussianState.isotropic([0.2], 0.04)
    first = kalman_predict(state, SCALAR)
    second = kalman_predict(state, SCALAR)
    # Exact-by-construction: the same inputs run the same operations.
    np.testing.assert_array_equal(first.mean, second.mean)
    np.testing.assert_array_equal(first.covariance, second.covariance)
    np.testing.assert_array_equal(state.mean, [0.2])
    np.testing.assert_array_equal(state.covariance, [[0.04]])


def test_kalman_update_returns_a_new_state_and_leaves_its_argument_alone() -> None:
    state = GaussianState.isotropic([0.0], 1.0)
    update = kalman_update(state, SCALAR, [1.0])
    np.testing.assert_array_equal(state.mean, [0.0])
    assert update.state.mean[0] > 0.9  # r << P, so the measurement wins
    assert update.state is not state


def test_the_one_dimensional_filter_matches_the_closed_form_scalar_recursion() -> None:
    """The vector code reproduces x = f x + K v, P = (1 - K)^2 P^- + K^2 r term by term."""
    # Algebraic identity in float64 evaluated two ways (a 1x1 Cholesky solve against a
    # scalar division), so the gap is a few units in the last place: rtol = atol = 1e-12.
    observations = simulate_random_walk(3e-3, 5e-2, 50, seed=11)
    initial = GaussianState.isotropic([0.1], 0.7)
    path = kalman_filter(observations, SCALAR, initial)

    f = float(SCALAR.transition_matrix[0, 0])
    q = float(SCALAR.transition_covariance[0, 0])
    r = float(SCALAR.observation_covariance[0, 0])
    x, p = float(initial.mean[0]), float(initial.covariance[0, 0])
    total = 0.0
    for step, row in enumerate(observations):
        x_predicted = f * x
        p_predicted = f * f * p + q
        s = p_predicted + r
        gain = p_predicted / s
        innovation = float(row[0]) - x_predicted
        x = x_predicted + gain * innovation
        p = (1.0 - gain) ** 2 * p_predicted + gain * gain * r
        total += -0.5 * (np.log(2.0 * np.pi) + np.log(s) + innovation * innovation / s)
        np.testing.assert_allclose(path.filtered[step].mean[0], x, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(path.filtered[step].covariance[0, 0], p, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(path.log_likelihood, total, rtol=1e-12, atol=1e-12)


def test_the_paths_log_likelihood_is_the_sum_of_its_step_contributions() -> None:
    # Same additions in the same order, so the identity is exact rather than approximate.
    observations = simulate_random_walk(3e-3, 5e-2, 20, seed=3)
    initial = GaussianState.isotropic([0.1], 0.7)
    path = kalman_filter(observations, SCALAR, initial)
    total = 0.0
    state = initial
    for row in observations:
        update = kalman_update(kalman_predict(state, SCALAR), SCALAR, row)
        total += update.log_likelihood
        state = update.state
    assert total == path.log_likelihood


def test_the_joseph_form_keeps_the_covariance_symmetric_and_positive_for_500_steps() -> None:
    """500 Joseph updates leave a bitwise-symmetric covariance with no negative eigenvalue."""
    # No tolerance is spent here. (P + P.T) / 2 is bitwise symmetric because float
    # addition commutes, and the Joseph form is a sum of two positive semi-definite
    # terms, so a negative eigenvalue would be a defect and not round-off. The smallest
    # eigenvalue observed over the run stays near 1e-6, eleven orders of magnitude above
    # the gate GaussianState would have allowed (16 * 3 * eps * max|P| is about 1e-17).
    model = correlated_model()
    rng = np.random.default_rng(23)
    state = GaussianState.isotropic(np.zeros(3), 1.0)
    smallest = np.inf
    for _ in range(500):
        update = kalman_update(kalman_predict(state, model), model, rng.standard_normal(3) * 0.05)
        covariance = update.state.covariance
        np.testing.assert_array_equal(covariance, covariance.T)
        smallest = min(smallest, float(np.linalg.eigvalsh(covariance)[0]))
        state = update.state
    assert smallest >= 0.0
    assert smallest > 1e-9


def test_the_scalar_gain_converges_to_the_riccati_fixed_point() -> None:
    """The random-walk-plus-noise gain converges to K = (sqrt(q^2 + 4qr) - q) / (2r).

    With F = H = 1 the predicted variance obeys p <- q + p r / (p + r).  Its fixed
    point solves p^2 - q p - q r = 0, so p* = (q + sqrt(q^2 + 4 q r)) / 2, and the
    gain K* = p* / (p* + r) equals (sqrt(q^2 + 4 q r) - q) / (2 r); the two forms are
    the same number, which the first assertion checks rather than assumes.
    """
    q, r = STEADY_Q, STEADY_R
    discriminant = np.sqrt(q * q + 4.0 * q * r)
    gain_star = (discriminant - q) / (2.0 * r)
    variance_star = (q + discriminant) / 2.0
    # Two algebraic forms of one root, in float64: rtol = 1e-12.
    np.testing.assert_allclose(
        variance_star / (variance_star + r), gain_star, rtol=1e-12, atol=1e-12
    )

    model = LinearGaussianModel.random_walk(1, transition_variance=q, observation_variance=r)
    rng = np.random.default_rng(5)
    state = GaussianState.isotropic([0.0], 1.0)  # 2000x the fixed point, so it must travel
    gains = []
    for _ in range(500):
        update = kalman_update(
            kalman_predict(state, model), model, [0.2 + 0.01 * rng.standard_normal()]
        )
        gains.append(float(update.gain[0, 0]))
        state = update.state

    # The iteration contracts by (1 - K*)^2 = 0.029 per step, so it reaches the float64
    # floor in about ten steps and 500 is far past that; what is left is the round-off of
    # one square root and two divisions -- measured at exactly 0.0 here. 1e-13 is that
    # floor with room to spare, not a tolerance fitted to make the assertion pass.
    assert abs(gains[0] - gain_star) > 0.1
    assert abs(gains[-1] - gain_star) <= 1e-13
    np.testing.assert_allclose(state.covariance[0, 0] + q, variance_star, rtol=1e-12, atol=1e-12)


def test_the_log_likelihood_is_maximised_at_the_true_parameters_on_a_coarse_grid() -> None:
    """On a 5x5 factor-of-two grid the filtered log-likelihood peaks at the true (q, r).

    The grid is q_true * {0.25, 0.5, 1, 2, 4} crossed with r_true * the same factors,
    with q_true = 4e-4 and r_true = 1e-4, over a 1200-step simulated path.  The claim
    is statistical, so it is stated as a margin: the relative standard error of a
    variance estimated from n = 1200 observations is about sqrt(2 / n) = 4%, and the
    grid spacing is a factor of two, so the maximiser sits roughly seventeen standard
    errors inside the correct cell.  The seed is fixed; the same cell wins for all of
    the first twelve seeds at this length, and n was chosen as the length at which that
    is true -- at 800 steps one seed in twelve picks the neighbouring r.
    """
    q_true, r_true = 4e-4, 1e-4
    factors = (0.25, 0.5, 1.0, 2.0, 4.0)
    observations = simulate_random_walk(q_true, r_true, 1200, seed=20260831)
    initial = GaussianState.isotropic([0.2], 1.0)
    scores = {
        (fq, fr): kalman_filter(
            observations,
            LinearGaussianModel.random_walk(
                1, transition_variance=q_true * fq, observation_variance=r_true * fr
            ),
            initial,
        ).log_likelihood
        for fq in factors
        for fr in factors
    }
    assert max(scores, key=scores.__getitem__) == (1.0, 1.0)


def test_an_empty_observation_path_is_refused() -> None:
    with pytest.raises(SurfaceValidationError, match="must contain at least one step"):
        kalman_filter(np.zeros((0, 1)), SCALAR, GaussianState.isotropic([0.0], 1.0))


def test_a_one_dimensional_observation_array_is_refused_rather_than_guessed() -> None:
    with pytest.raises(SurfaceValidationError, match="observations must be two-dimensional"):
        kalman_filter(np.zeros(4), SCALAR, GaussianState.isotropic([0.0], 1.0))


# --- missing observations are skipped, not imputed ---------------------------


def test_a_missing_observation_leaves_the_mean_unchanged_and_widens_the_covariance() -> None:
    """A fully masked step contributes no innovation, no likelihood, and no information."""
    model = LinearGaussianModel.random_walk(2, transition_variance=1e-3, observation_variance=1e-2)
    state = GaussianState.isotropic([0.2, 0.3], 0.05)
    update = kalman_update(kalman_predict(state, model), model, [np.nan, np.nan])
    # Exact by construction: an identity transition with a zero drift is a copy, and a
    # masked update returns the predicted state itself.
    np.testing.assert_array_equal(update.state.mean, state.mean)
    assert update.state.trace > state.trace
    # tr(P + Q) = tr(P) + tr(Q) is an identity; the two sides sum the same numbers in
    # different orders, so rel = 1e-12 covers the last-bit difference and nothing more.
    assert update.state.trace == pytest.approx(state.trace + 2e-3, rel=1e-12)
    assert update.n_observed == 0
    assert update.log_likelihood == 0.0


def test_an_explicit_mask_and_the_nan_convention_agree() -> None:
    model = LinearGaussianModel.random_walk(2, transition_variance=1e-3, observation_variance=1e-2)
    state = GaussianState.isotropic([0.2, 0.3], 0.05)
    by_nan = kalman_update(state, model, [0.25, np.nan])
    by_mask = kalman_update(state, model, [0.25, 12.0], observation_mask=[True, False])
    np.testing.assert_array_equal(by_nan.state.mean, by_mask.state.mean)
    assert by_nan.n_observed == 1
    assert by_nan.gain.shape == (2, 1)


def test_a_partially_masked_step_moves_only_the_rows_it_observed() -> None:
    model = LinearGaussianModel.random_walk(2, transition_variance=0.0, observation_variance=1e-2)
    state = GaussianState.isotropic([0.2, 0.3], 0.05)
    update = kalman_update(state, model, [0.25, np.nan])
    assert update.state.mean[0] > 0.2  # pulled toward the observed row
    # The second component is uncorrelated with the first under an isotropic prior, so an
    # exact copy is the right answer, not an approximation.
    np.testing.assert_array_equal(update.state.mean[1], state.mean[1])


def test_a_non_finite_value_on_a_kept_row_is_refused() -> None:
    with pytest.raises(SurfaceValidationError, match="must be finite on every row"):
        kalman_update(
            GaussianState.isotropic([0.0], 1.0), SCALAR, [np.nan], observation_mask=[True]
        )
    # An infinity is not a missing quote, so the NaN convention must not absorb it.
    with pytest.raises(SurfaceValidationError, match="must be finite on every row"):
        kalman_update(GaussianState.isotropic([0.0], 1.0), SCALAR, [np.inf])


# --- exact arithmetic where the model is exact -------------------------------


def test_a_zero_noise_model_forecasts_an_exactly_linear_state_path_exactly() -> None:
    """Two noiseless observations of a line pin the local linear trend, bitwise.

    The constants are dyadic (a = 1/4, b = 1/8, P0 = I/2), and every quantity the
    recursion forms -- the innovation covariances 1 and 1/4, their Cholesky factors 1
    and 1/2, the gains (1, 1/2) and (1, 1) -- is a power of two, so the whole filter is
    exact in binary floating point.  This is the one place the house rule for an
    exact-by-construction claim applies: rtol = atol = 0.
    """
    a, b = 0.25, 0.125
    model = LinearGaussianModel(
        transition_matrix=[[1.0, 1.0], [0.0, 1.0]],
        transition_covariance=np.zeros((2, 2)),
        observation_matrix=[[1.0, 0.0]],
        observation_covariance=np.zeros((1, 1)),
    )
    initial = GaussianState(mean=[0.0, 0.0], covariance=[[0.5, 0.0], [0.0, 0.5]])
    path = kalman_filter(np.array([[a], [a + b]]), model, initial)
    np.testing.assert_array_equal(path.terminal.mean, [a + b, b])
    np.testing.assert_array_equal(path.terminal.covariance, np.zeros((2, 2)))
    state = path.terminal
    for step in range(1, 6):
        state = kalman_predict(state, model)
        np.testing.assert_array_equal(state.mean, [a + b * (1 + step), b])


def test_a_singular_innovation_covariance_is_reported_rather_than_regularised() -> None:
    model = LinearGaussianModel.random_walk(1, transition_variance=0.0, observation_variance=0.0)
    state = GaussianState.isotropic([0.2], 1.0)
    first = kalman_update(kalman_predict(state, model), model, [0.2])
    with pytest.raises(SurfaceCalibrationError, match="must be positive definite"):
        kalman_update(kalman_predict(first.state, model), model, [0.3])


# --- the smoother ------------------------------------------------------------


def test_the_smoothers_terminal_state_is_the_filters_terminal_state() -> None:
    """Conditioning on everything at the last step is what the filter already did."""
    model = correlated_model()
    rng = np.random.default_rng(31)
    path = kalman_filter(
        rng.standard_normal((40, 3)) * 0.05, model, GaussianState.isotropic(np.zeros(3), 1.0)
    )
    smoothed = kalman_smooth(path, model)
    assert smoothed[-1] is path.terminal
    np.testing.assert_array_equal(smoothed[-1].mean, path.terminal.mean)
    np.testing.assert_array_equal(smoothed[-1].covariance, path.terminal.covariance)
    assert len(smoothed) == path.n_steps


def test_the_smoother_never_leaves_a_step_less_certain_than_the_filter_did() -> None:
    model = correlated_model()
    rng = np.random.default_rng(37)
    path = kalman_filter(
        rng.standard_normal((40, 3)) * 0.05, model, GaussianState.isotropic(np.zeros(3), 1.0)
    )
    smoothed = kalman_smooth(path, model)
    # The RTS covariance is P^f + J (P^s - P^p) J', and P^s <= P^p in the Loewner order,
    # so the trace can only fall. It is an inequality, so no tolerance is involved --
    # except at the terminal step, where the two are the same object.
    for index in range(path.n_steps):
        assert smoothed[index].trace <= path.filtered[index].trace


def test_a_smoother_needs_the_path_it_is_given() -> None:
    with pytest.raises(SurfaceValidationError, match="path must be a FilteredPath"):
        kalman_smooth([GaussianState.isotropic([0.0], 1.0)], SCALAR)


def test_a_filtered_path_must_carry_one_state_per_step() -> None:
    state = GaussianState.isotropic([0.0], 1.0)
    with pytest.raises(SurfaceValidationError, match="one state per step"):
        FilteredPath(predicted=[state, state], filtered=[state], log_likelihood=0.0)


# --- the forecaster ----------------------------------------------------------


def test_the_forecaster_satisfies_the_forecaster_protocol() -> None:
    assert isinstance(StateSpaceForecaster(), SurfaceForecaster)


def test_an_empty_history_is_refused() -> None:
    with pytest.raises(SurfaceValidationError, match="history is empty"):
        StateSpaceForecaster().forecast([], ForecastHorizon(steps=1))


def test_the_forecaster_reproduces_persistence_when_the_observation_is_taken_as_exact() -> None:
    """Zero observation noise plus an identity transition is the persistence forecast.

    With R = 0 the gain is the identity, so each update adopts its observation exactly
    and the terminal state is the last fitted level; an identity transition then carries
    it to any horizon.  The transition variance must be strictly *positive* for this:
    with both variances zero the model says the parameter never moves and was seen
    without error, so it is pinned by the *first* observation, which is a different
    model rather than a persistence forecast.
    """
    # K = I only up to the round-off of one Cholesky solve on a matrix of condition
    # number 1, i.e. a few eps; rtol = atol = 1e-12 is far above that floor and far
    # below the 0.01 spacing between the levels in the history.
    path = history((0.20, 0.21, 0.19, 0.24))
    state_space = StateSpaceForecaster(transition_variance=1e-4, observation_variance=0.0)
    persistence = PersistenceForecaster(calibrator=FlatVolatilityCalibrator())
    for steps in (1, 3, 10):
        horizon = ForecastHorizon(steps=steps)
        np.testing.assert_allclose(
            state_space.forecast(path, horizon).evaluate(POINTS).iv,
            persistence.forecast(path, horizon).evaluate(POINTS).iv,
            rtol=1e-12,
            atol=1e-12,
        )


def test_a_fully_deterministic_model_reproduces_persistence_on_a_one_step_history() -> None:
    """With q = r = 0 and one observation the innovation is exactly zero, so nothing moves."""
    path = history((0.22,))
    forecaster = StateSpaceForecaster(transition_variance=0.0, observation_variance=0.0)
    persistence = PersistenceForecaster(calibrator=FlatVolatilityCalibrator())
    horizon = ForecastHorizon(steps=4)
    # Exact by construction: the initial mean is the first fitted level, so v = y - x = 0
    # and the update adds K @ 0; the identity transition then copies it h times.
    np.testing.assert_array_equal(
        forecaster.forecast(path, horizon).evaluate(POINTS).iv,
        persistence.forecast(path, horizon).evaluate(POINTS).iv,
    )


def test_a_fully_deterministic_model_refuses_a_history_it_cannot_filter() -> None:
    forecaster = StateSpaceForecaster(transition_variance=0.0, observation_variance=0.0)
    with pytest.raises(SurfaceCalibrationError, match="must be positive definite"):
        forecaster.forecast(history((0.20, 0.24)), ForecastHorizon(steps=1))


def test_a_horizon_of_h_applies_the_transition_exactly_h_times() -> None:
    """Forecasting h steps is h transitions, not one closed form that happens to agree."""
    model = LinearGaussianModel(
        transition_matrix=[[0.5]],
        transition_covariance=[[1e-4]],
        observation_matrix=[[1.0]],
        observation_covariance=[[1e-6]],
    )
    forecaster = StateSpaceForecaster(model=model)
    path = history((0.20, 0.21, 0.19, 0.24))
    parameters = forecaster.parameter_path(path)
    terminal = kalman_filter(
        parameters, model, GaussianState.isotropic(parameters[0], 1.0)
    ).terminal
    for steps in (1, 2, 5):
        expected = terminal
        for _ in range(steps):
            expected = kalman_predict(expected, model)
        forecast = forecaster.forecast_state(path, ForecastHorizon(steps=steps))
        # The same operations in the same order: bitwise equality, no tolerance.
        np.testing.assert_array_equal(forecast.mean, expected.mean)
        np.testing.assert_array_equal(forecast.covariance, expected.covariance)
        # Halving is exact in binary floating point, so the analytic check is exact too.
        np.testing.assert_array_equal(forecast.mean, terminal.mean * 0.5**steps)


def test_the_calendar_spacing_of_a_step_is_accepted_and_unused() -> None:
    forecaster = StateSpaceForecaster()
    path = history((0.20, 0.24))
    bare = forecaster.forecast(path, ForecastHorizon(steps=2))
    dated = forecaster.forecast(path, ForecastHorizon(steps=2, step_years=1 / 252))
    assert bare.level == dated.level


def test_the_point_forecast_carries_no_invented_interval() -> None:
    """The predictive covariance is over parameters, and is not passed off as a band."""
    forecaster = StateSpaceForecaster()
    path = history((0.20, 0.24))
    prediction = forecaster.forecast(path, ForecastHorizon(steps=1)).evaluate(POINTS)
    assert prediction.has_uncertainty is False
    assert forecaster.forecast_state(path, ForecastHorizon(steps=1)).trace > 0.0


def test_the_forecaster_holds_no_state_between_forecasts() -> None:
    """Two forecasts from one history agree bitwise, whatever ran between them."""
    forecaster = StateSpaceForecaster()
    first = history((0.20, 0.21, 0.24))
    second = history((0.31, 0.29))
    horizon = ForecastHorizon(steps=2)
    before = forecaster.forecast(first, horizon).level
    forecaster.forecast(second, horizon)
    forecaster.forecast(second, ForecastHorizon(steps=7))
    after = forecaster.forecast(first, horizon).level
    assert before == after


def test_two_forecasters_configured_alike_produce_the_same_forecast() -> None:
    path = history((0.20, 0.21, 0.24))
    horizon = ForecastHorizon(steps=3)
    assert (
        StateSpaceForecaster().forecast(path, horizon).level
        == StateSpaceForecaster().forecast(path, horizon).level
    )


def test_a_model_that_does_not_observe_every_state_component_requires_an_initial_state() -> None:
    trend = LinearGaussianModel(
        transition_matrix=[[1.0, 1.0], [0.0, 1.0]],
        transition_covariance=np.diag([1e-4, 1e-6]),
        observation_matrix=[[1.0, 0.0]],
        observation_covariance=[[1e-6]],
    )
    path = history((0.20, 0.22, 0.24))
    with pytest.raises(SurfaceValidationError, match="initial_state is required"):
        StateSpaceForecaster(model=trend).forecast(path, ForecastHorizon(steps=1))
    with_state = StateSpaceForecaster(
        model=trend, initial_state=GaussianState(mean=[0.2, 0.0], covariance=np.diag([1.0, 0.01]))
    )
    forecast = with_state.forecast(path, ForecastHorizon(steps=1))
    # The level has been rising by 0.02 a step, so a trend model must forecast above the
    # last fitted level; the exact value depends on the noise ratio and is not asserted.
    assert forecast.level > 0.24


def test_a_model_whose_observation_does_not_match_the_parameters_is_refused() -> None:
    forecaster = StateSpaceForecaster(model=LinearGaussianModel.random_walk(2))
    with pytest.raises(SurfaceValidationError, match="must observe one row per extracted"):
        forecaster.forecast(history((0.2, 0.24)), ForecastHorizon(steps=1))


def test_the_horizon_must_be_a_forecast_horizon() -> None:
    with pytest.raises(SurfaceValidationError, match="horizon must be a ForecastHorizon"):
        StateSpaceForecaster().forecast(history((0.2,)), 3)


def test_the_default_extractor_and_reconstructor_round_trip_a_flat_level() -> None:
    surface = FlatIVSurface(level=0.23)
    np.testing.assert_array_equal(flat_level_parameters(surface), [0.23])
    assert flat_surface_from_parameters(flat_level_parameters(surface)).level == 0.23


def test_an_extractor_that_does_not_match_the_calibrator_is_refused() -> None:
    class Nameless:
        def evaluate(self, points, *, market=None):  # pragma: no cover - never called
            raise AssertionError("not evaluated")

    with pytest.raises(SurfaceValidationError, match="must expose a scalar 'level'"):
        flat_level_parameters(Nameless())


def test_an_extractor_that_changes_width_between_steps_is_refused() -> None:
    widths = iter([1, 2])

    def extract(surface) -> np.ndarray:
        return np.full(next(widths), float(surface.level))

    forecaster = StateSpaceForecaster(extract_parameters=extract)
    with pytest.raises(SurfaceCalibrationError, match="same number of parameters"):
        forecaster.forecast(history((0.2, 0.24)), ForecastHorizon(steps=1))


def test_an_initial_variance_of_zero_is_refused() -> None:
    with pytest.raises(SurfaceValidationError, match="initial_variance must be finite and"):
        StateSpaceForecaster(initial_variance=0.0)


# --- an independent oracle: the joint Gaussian, conditioned by hand ------------


def _joint_gaussian_oracle(model, initial, observations):
    """Filtered, smoothed and log-likelihood from first principles.

    Builds the full joint law of ``(x_1..x_S, y_1..y_S)`` from the stacked linear
    map and conditions it by hand.  It shares no code path with the recursions --
    no predict, no update, no backward gain -- so agreement is evidence about the
    recursions rather than about a helper both sides call.

    Returns ``(filtered_means, smoothed_means, log_likelihood)``.
    """
    from scipy.stats import multivariate_normal

    F = np.asarray(model.transition_matrix, dtype=np.float64)
    Q = np.asarray(model.transition_covariance, dtype=np.float64)
    H = np.asarray(model.observation_matrix, dtype=np.float64)
    R = np.asarray(model.observation_covariance, dtype=np.float64)
    b = np.asarray(model.drift, dtype=np.float64)
    steps = len(observations)
    n = F.shape[0]

    # One transition happens before the first observation, matching kalman_filter.
    means = []
    covariances = []
    mean = F @ np.asarray(initial.mean, dtype=np.float64) + b
    covariance = F @ np.asarray(initial.covariance, dtype=np.float64) @ F.T + Q
    for _ in range(steps):
        means.append(mean)
        covariances.append(covariance)
        mean = F @ mean + b
        covariance = F @ covariance @ F.T + Q

    cross = np.zeros((steps * n, steps * n))
    for i in range(steps):
        for j in range(steps):
            if i <= j:
                block = covariances[i] @ np.linalg.matrix_power(F.T, j - i)
            else:
                block = np.linalg.matrix_power(F, i - j) @ covariances[j]
            cross[i * n : (i + 1) * n, j * n : (j + 1) * n] = block

    stacked_H = np.kron(np.eye(steps), H)
    mu_x = np.concatenate(means)
    mu_y = stacked_H @ mu_x
    cov_xy = cross @ stacked_H.T
    cov_yy = stacked_H @ cross @ stacked_H.T + np.kron(np.eye(steps), R)

    y = np.asarray(observations, dtype=np.float64).reshape(-1)
    smoothed = (mu_x + cov_xy @ np.linalg.solve(cov_yy, y - mu_y)).reshape(steps, n)

    filtered = []
    for t in range(1, steps + 1):
        rows = t * H.shape[0]
        gain = cov_xy[t * n - n : t * n, :rows]
        filtered.append(
            means[t - 1] + gain @ np.linalg.solve(cov_yy[:rows, :rows], y[:rows] - mu_y[:rows])
        )
    log_likelihood = float(multivariate_normal(mean=mu_y, cov=cov_yy).logpdf(y))
    return np.array(filtered), smoothed, log_likelihood


#: Tolerance against the oracle.  Both sides are float64 solves of the same
#: well-conditioned 8x8 system by different routes -- one recursive, one a single
#: dense solve -- so the difference is accumulated rounding, not method error.
ORACLE_TOL = 1e-9


def _oracle_model():
    from fast_vollib.surface.fitting.state_space import GaussianState, LinearGaussianModel

    return (
        LinearGaussianModel(
            transition_matrix=np.array([[0.9, 0.1], [0.0, 0.8]]),
            transition_covariance=np.array([[0.02, 0.005], [0.005, 0.01]]),
            observation_matrix=np.array([[1.0, 0.0], [0.5, 1.0]]),
            observation_covariance=np.array([[0.03, 0.0], [0.0, 0.05]]),
            drift=np.array([0.03, -0.02]),
        ),
        GaussianState(mean=np.array([0.2, -0.1]), covariance=np.diag([0.5, 0.4])),
    )


def test_the_filtered_means_match_the_joint_gaussian_oracle() -> None:
    from fast_vollib.surface.fitting.state_space import kalman_filter

    model, initial = _oracle_model()
    observations = np.array([[0.21, 0.02], [0.18, -0.03], [0.25, 0.06], [0.19, 0.01]])
    path = kalman_filter(observations, model, initial)
    expected, _, _ = _joint_gaussian_oracle(model, initial, observations)
    np.testing.assert_allclose(
        np.array([state.mean for state in path.filtered]),
        expected,
        rtol=ORACLE_TOL,
        atol=ORACLE_TOL,
    )


def test_the_smoothed_means_match_the_joint_gaussian_oracle() -> None:
    """The smoother is pinned to a value, not only to a shape and an inequality.

    An identity smoother -- one that returned the filtered path unchanged --
    satisfies "the terminal state is the filter's" and "the trace never rises",
    so neither of those can detect a wrong backward gain.  This one can: it
    compares every non-terminal mean against the joint law conditioned on the
    whole observation sequence.
    """
    from fast_vollib.surface.fitting.state_space import kalman_filter, kalman_smooth

    model, initial = _oracle_model()
    observations = np.array([[0.21, 0.02], [0.18, -0.03], [0.25, 0.06], [0.19, 0.01]])
    path = kalman_filter(observations, model, initial)
    smoothed = kalman_smooth(path, model)
    _, expected, _ = _joint_gaussian_oracle(model, initial, observations)
    actual = np.array([state.mean for state in smoothed])
    np.testing.assert_allclose(actual, expected, rtol=ORACLE_TOL, atol=ORACLE_TOL)
    # And it is genuinely different from the filtered path, so the comparison
    # above is not one an identity smoother could pass.
    filtered = np.array([state.mean for state in path.filtered])
    assert float(np.max(np.abs(actual[:-1] - filtered[:-1]))) > 1e-3


def test_the_path_log_likelihood_matches_the_joint_gaussian_oracle() -> None:
    from fast_vollib.surface.fitting.state_space import kalman_filter

    model, initial = _oracle_model()
    observations = np.array([[0.21, 0.02], [0.18, -0.03], [0.25, 0.06], [0.19, 0.01]])
    path = kalman_filter(observations, model, initial)
    _, _, expected = _joint_gaussian_oracle(model, initial, observations)
    assert abs(path.log_likelihood - expected) < ORACLE_TOL


def test_the_drift_moves_the_filtered_path_and_the_likelihood() -> None:
    """The drift term is exercised numerically, not only stored.

    Dropping ``+ b`` from the prediction leaves a filter that reports a
    likelihood for a model it is not fitting, and every other test in this file
    uses a zero drift, so nothing else would notice.
    """
    from fast_vollib.surface.fitting.state_space import LinearGaussianModel, kalman_filter

    observations = np.array([[0.21, 0.02], [0.18, -0.03], [0.25, 0.06], [0.19, 0.01]])
    with_drift, initial = _oracle_model()
    without_drift = LinearGaussianModel(
        transition_matrix=with_drift.transition_matrix,
        transition_covariance=with_drift.transition_covariance,
        observation_matrix=with_drift.observation_matrix,
        observation_covariance=with_drift.observation_covariance,
    )
    driven = kalman_filter(observations, with_drift, initial)
    undriven = kalman_filter(observations, without_drift, initial)
    assert float(np.max(np.abs(driven.filtered[0].mean - undriven.filtered[0].mean))) > 1e-3
    assert abs(driven.log_likelihood - undriven.log_likelihood) > 1e-3
    # A zero-drift model must reproduce the no-drift one exactly, so the term is
    # additive rather than, say, multiplicative.
    zero = LinearGaussianModel(
        transition_matrix=with_drift.transition_matrix,
        transition_covariance=with_drift.transition_covariance,
        observation_matrix=with_drift.observation_matrix,
        observation_covariance=with_drift.observation_covariance,
        drift=np.zeros(2),
    )
    np.testing.assert_array_equal(
        kalman_filter(observations, zero, initial).filtered[-1].mean,
        undriven.filtered[-1].mean,
    )


def test_a_predicted_state_uses_the_drift_the_model_carries() -> None:
    from fast_vollib.surface.fitting.state_space import kalman_predict

    model, initial = _oracle_model()
    predicted = kalman_predict(initial, model)
    expected = np.asarray(model.transition_matrix) @ np.asarray(initial.mean) + np.asarray(
        model.drift
    )
    np.testing.assert_allclose(predicted.mean, expected, rtol=1e-14, atol=1e-15)
