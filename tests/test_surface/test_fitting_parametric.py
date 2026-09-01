"""SVI, SVI-JW and SSVI: exact conversions, exact recovery, and honest arbitrage claims."""

from __future__ import annotations

import numpy as np
import pytest

from fast_vollib.surface import (
    SurfaceGridSpec,
    SurfaceObservations,
    SurfacePoints,
    materialize_surface,
)
from fast_vollib.surface.density import durrleman_g as mesh_durrleman_g
from fast_vollib.surface.errors import SurfaceCalibrationError, SurfaceValidationError
from fast_vollib.surface.fitting.parametric import (
    SAFE_ETA,
    HestonLikePhi,
    PowerLawPhi,
    SSVICalibrator,
    SSVISurface,
    SVICalibrator,
    SVIJumpWings,
    SVIParameters,
    SVISmile,
    SVISurface,
    svi_total_variance,
)
from fast_vollib.surface.protocols import DefiniteIVSurface, SurfaceCalibrator

# A well-behaved slice: the wings are not so steep that Durrleman's g dips, and
# the curvature is wide enough that a 21-point mesh resolves it.
REFERENCE = SVIParameters(a=0.04, b=0.40, rho=-0.40, m=0.0, sigma=0.10)

# Four slices chosen so that the *data* is arbitrage-free: each has a strictly
# positive Durrleman g, and their total variances are strictly ordered in
# maturity over the moneyness range the tests use. Both properties are asserted
# below rather than assumed -- a reference set that was quietly arbitrageable
# would turn every downstream arbitrage assertion into a test of the fixture.
TERM_STRUCTURE = {
    0.25: SVIParameters(a=0.010, b=0.12, rho=-0.50, m=0.02, sigma=0.08),
    0.50: SVIParameters(a=0.020, b=0.20, rho=-0.45, m=0.01, sigma=0.09),
    1.00: SVIParameters(a=0.040, b=0.40, rho=-0.40, m=0.00, sigma=0.10),
    2.00: SVIParameters(a=0.080, b=0.35, rho=-0.25, m=0.00, sigma=0.30),
}

#: Tolerance for an algebraic identity evaluated in float64.  Both sides are
#: O(0.1) sums of a handful of square roots, so the accumulated rounding is a
#: few ulps; 1e-12 is six orders of magnitude of headroom over that and still
#: catches any real algebra error.
IDENTITY_TOL = 1e-12

#: Tolerance for parameters recovered from noiseless data by a least-squares
#: solver run to xtol = ftol = gtol = 1e-14.  The solver's own convergence
#: criterion is the binding constraint, not the arithmetic.
RECOVERY_TOL = 1e-8


def _observations(parameters: dict[float, SVIParameters], n_k: int = 21, span: float = 0.4):
    k, T, iv = [], [], []
    for maturity, slice_parameters in parameters.items():
        axis = np.linspace(-span, span, n_k)
        k.append(axis)
        T.append(np.full(n_k, maturity))
        iv.append(np.sqrt(slice_parameters.total_variance(axis) / maturity))
    return SurfaceObservations(k=np.concatenate(k), T=np.concatenate(T), iv=np.concatenate(iv))


# --- the raw parameterization -------------------------------------------------


def test_the_atm_total_variance_is_what_the_formula_says() -> None:
    # w(0) = a + b(-rho m + sqrt(m^2 + sigma^2)); with m = 0 that is a + b sigma.
    np.testing.assert_allclose(
        REFERENCE.total_variance(0.0), 0.04 + 0.4 * 0.1, rtol=IDENTITY_TOL, atol=IDENTITY_TOL
    )


def test_the_wings_are_asymptotically_linear_with_the_stated_slopes() -> None:
    # w(k) -> a - b m rho + b(1 + rho)k as k -> +inf, and slope b(1 - rho) going left.
    far = 1e6
    right = (REFERENCE.total_variance(far + 1.0) - REFERENCE.total_variance(far)) / 1.0
    left = (REFERENCE.total_variance(-far) - REFERENCE.total_variance(-far - 1.0)) / 1.0
    np.testing.assert_allclose(right, 0.4 * (1 - 0.4), rtol=1e-6, atol=0.0)
    np.testing.assert_allclose(left, -0.4 * (1 + 0.4), rtol=1e-6, atol=0.0)


def test_the_minimum_total_variance_is_attained_where_the_formula_says() -> None:
    axis = np.linspace(REFERENCE.minimum_at - 0.5, REFERENCE.minimum_at + 0.5, 100001)
    numerical = float(np.min(REFERENCE.total_variance(axis)))
    np.testing.assert_allclose(numerical, REFERENCE.minimum_total_variance, rtol=1e-9, atol=1e-12)


def test_the_closed_form_derivatives_match_central_differences() -> None:
    # A central difference is second-order accurate, so with h = 1e-5 the error
    # is O(h^2) = 1e-10 on w' and O(h^2) on w'' with an extra 1/h^2 amplification
    # of rounding; 1e-6 is the honest band for the second derivative.
    k = np.linspace(-0.4, 0.4, 17)
    h = 1e-5
    w, w_prime, w_second = REFERENCE.derivatives(k)
    up = REFERENCE.total_variance(k + h)
    down = REFERENCE.total_variance(k - h)
    np.testing.assert_allclose(w, REFERENCE.total_variance(k), rtol=IDENTITY_TOL, atol=0.0)
    np.testing.assert_allclose(w_prime, (up - down) / (2 * h), rtol=1e-8, atol=1e-9)
    np.testing.assert_allclose(w_second, (up - 2 * w + down) / h**2, rtol=1e-6, atol=1e-6)


def test_the_analytic_durrleman_g_is_the_limit_of_the_mesh_estimator() -> None:
    """The mesh estimator converges to the closed form at second order.

    The mesh fits a local parabola through three nodes, so its error is O(h^2).
    Asserting a fixed band would require guessing the constant; halving the
    spacing and requiring the error to fall by a factor near four tests the
    thing that actually matters -- that the two computations are of the same
    quantity -- and cannot be satisfied by a coincidence of scale.
    """
    from fast_vollib._array_api import numpy_namespace

    def max_error(n_nodes: int) -> float:
        k = np.linspace(-0.4, 0.4, n_nodes)
        w = REFERENCE.total_variance(k)
        mesh = mesh_durrleman_g(k[:, None], w[:, None], numpy_namespace())[:, 0]
        return float(np.max(np.abs(mesh - REFERENCE.durrleman_g(k[1:-1]))))

    coarse = max_error(201)
    fine = max_error(401)
    assert coarse < 1e-2  # the two are already close on a coarse mesh
    assert 3.0 < coarse / fine < 5.0  # second order: halving h quarters the error


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"b": -0.1}, "b must be non-negative"),
        ({"rho": 1.0}, "rho must lie strictly inside"),
        ({"rho": -1.5}, "rho must lie strictly inside"),
        ({"sigma": 0.0}, "sigma must be strictly positive"),
        ({"a": -1.0}, "minimum total variance"),
        ({"a": np.inf}, "a must be finite"),
    ],
)
def test_invalid_raw_parameters_are_refused(kwargs, message) -> None:
    base = {"a": 0.04, "b": 0.4, "rho": -0.4, "m": 0.0, "sigma": 0.1}
    with pytest.raises(SurfaceValidationError, match=message):
        SVIParameters(**{**base, **kwargs})


def test_the_namespace_generic_kernel_agrees_with_the_dataclass() -> None:
    k = np.linspace(-0.5, 0.5, 11)
    np.testing.assert_allclose(
        svi_total_variance(k, 0.04, 0.4, -0.4, 0.0, 0.1),
        REFERENCE.total_variance(k),
        rtol=0,
        atol=0,
    )


def test_the_kernel_is_differentiable_on_the_torch_backend() -> None:
    torch = pytest.importorskip("torch", reason="torch not installed")
    k = torch.linspace(-0.4, 0.4, 9, dtype=torch.float64)
    b = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)
    total = svi_total_variance(k, 0.04, b, -0.4, 0.0, 0.1).sum()
    total.backward()
    # d/db sum_i w(k_i) = sum_i (rho (k_i - m) + sqrt((k_i - m)^2 + sigma^2)).
    expected = float(np.sum(-0.4 * k.numpy() + np.sqrt(k.numpy() ** 2 + 0.01)))
    assert abs(float(b.grad) - expected) < 1e-12


# --- jump wings ---------------------------------------------------------------


@pytest.mark.parametrize("maturity", [0.25, 1.0, 3.0], ids=["short", "one-year", "long"])
def test_the_jump_wings_round_trip_is_the_identity(maturity) -> None:
    for parameters in TERM_STRUCTURE.values():
        restored = parameters.to_jump_wings(maturity).to_raw(maturity)
        for name in parameters.parameters():
            assert abs(getattr(restored, name) - getattr(parameters, name)) < IDENTITY_TOL


def test_each_jump_wing_parameter_means_what_it_claims() -> None:
    # A mutually inverse pair of wrong formulas would round-trip perfectly, so
    # every jump-wings quantity is checked against its own operational meaning.
    maturity = 1.5
    jw = REFERENCE.to_jump_wings(maturity)
    w_atm = float(REFERENCE.total_variance(0.0))
    _, w_prime_atm, _ = REFERENCE.derivatives(np.array(0.0))

    np.testing.assert_allclose(jw.v, w_atm / maturity, rtol=IDENTITY_TOL, atol=0.0)
    np.testing.assert_allclose(
        jw.psi, float(w_prime_atm) / (2 * np.sqrt(w_atm)), rtol=IDENTITY_TOL, atol=0.0
    )
    np.testing.assert_allclose(
        jw.v_tilde, REFERENCE.minimum_total_variance / maturity, rtol=IDENTITY_TOL, atol=0.0
    )
    far = 1e7
    left_slope = -(REFERENCE.total_variance(-far) - REFERENCE.total_variance(-far - 1.0))
    right_slope = REFERENCE.total_variance(far + 1.0) - REFERENCE.total_variance(far)
    np.testing.assert_allclose(jw.p, left_slope / np.sqrt(w_atm), rtol=1e-6, atol=0.0)
    np.testing.assert_allclose(jw.c, right_slope / np.sqrt(w_atm), rtol=1e-6, atol=0.0)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"v": 0.0}, "v must be strictly positive"),
        ({"v_tilde": -0.1}, "v_tilde must be non-negative"),
        ({"p": -0.1}, "Wing slopes must be non-negative"),
    ],
)
def test_invalid_jump_wings_are_refused(kwargs, message) -> None:
    base = {"v": 0.08, "psi": -0.1, "p": 1.9, "c": 0.8, "v_tilde": 0.076}
    with pytest.raises(SurfaceValidationError, match=message):
        SVIJumpWings(**{**base, **kwargs})


def test_wingless_jump_wings_cannot_be_inverted() -> None:
    with pytest.raises(SurfaceValidationError, match="A smile with no wings"):
        SVIJumpWings(v=0.08, psi=0.0, p=0.0, c=0.0, v_tilde=0.08).to_raw(1.0)


# --- the smile and the surface as definite surfaces ---------------------------


def test_a_smile_answers_only_at_its_own_maturity() -> None:
    smile = SVISmile(parameters=REFERENCE, maturity=1.0)
    prediction = smile.evaluate(SurfacePoints(k=[0.0, 0.0, 0.0], T=[1.0, 2.0, 0.5]))
    assert prediction.valid.tolist() == [True, False, False]
    np.testing.assert_allclose(prediction.iv[0], np.sqrt(0.08), rtol=IDENTITY_TOL, atol=0.0)


def test_a_surface_interpolates_total_variance_between_its_slices() -> None:
    surface = SVISurface(
        maturities=(1.0, 2.0),
        slices=(TERM_STRUCTURE[1.0], TERM_STRUCTURE[2.0]),
    )
    midpoint = surface.total_variance(np.array([0.1]), np.array([1.5]))
    expected = 0.5 * (
        TERM_STRUCTURE[1.0].total_variance(0.1) + TERM_STRUCTURE[2.0].total_variance(0.1)
    )
    np.testing.assert_allclose(midpoint, expected, rtol=IDENTITY_TOL, atol=0.0)


def test_a_surface_declines_maturities_outside_its_fitted_term() -> None:
    surface = SVISurface(maturities=(1.0, 2.0), slices=(TERM_STRUCTURE[1.0], TERM_STRUCTURE[2.0]))
    prediction = surface.evaluate(SurfacePoints(k=[0.0, 0.0], T=[0.5, 3.0]))
    assert prediction.valid.tolist() == [False, False]


def test_the_no_interpolation_policy_answers_only_at_the_pillars() -> None:
    surface = SVISurface(
        maturities=(1.0, 2.0),
        slices=(TERM_STRUCTURE[1.0], TERM_STRUCTURE[2.0]),
        maturity_interpolation="none",
    )
    prediction = surface.evaluate(SurfacePoints(k=[0.0, 0.0], T=[1.0, 1.5]))
    assert prediction.valid.tolist() == [True, False]


def test_a_surface_refuses_unordered_maturities() -> None:
    with pytest.raises(SurfaceValidationError, match="strictly increasing"):
        SVISurface(maturities=(2.0, 1.0), slices=(REFERENCE, REFERENCE))


# --- calibration --------------------------------------------------------------


def test_the_reference_term_structure_is_itself_arbitrage_free() -> None:
    # Guards every downstream arbitrage assertion: if this fixture were
    # arbitrageable, a fit that reproduced it faithfully would look like a bug.
    axis = np.linspace(-1.0, 1.0, 2001)
    for parameters in TERM_STRUCTURE.values():
        assert float(np.min(parameters.durrleman_g(axis))) > 0.0
    inner = np.linspace(-0.45, 0.45, 401)
    total_variance = np.stack(
        [parameters.total_variance(inner) for parameters in TERM_STRUCTURE.values()]
    )
    assert bool(np.all(np.diff(total_variance, axis=0) > 0.0))


def test_svi_recovers_the_parameters_it_was_generated_from() -> None:
    fitted = SVICalibrator().fit(_observations(TERM_STRUCTURE))
    assert fitted.maturities == tuple(sorted(TERM_STRUCTURE))
    for maturity, parameters in zip(fitted.maturities, fitted.slices):
        truth = TERM_STRUCTURE[maturity]
        for name in truth.parameters():
            assert abs(getattr(parameters, name) - getattr(truth, name)) < RECOVERY_TOL


def test_the_fitted_surface_passes_the_mesh_arbitrage_harness() -> None:
    fitted = SVICalibrator().fit(_observations(TERM_STRUCTURE))
    grid = SurfaceGridSpec(k=np.linspace(-0.35, 0.35, 25), T=list(TERM_STRUCTURE))
    assert materialize_surface(fitted, grid).validate().passed is True


def test_calibration_is_deterministic_and_independent_of_row_order() -> None:
    generator = np.random.default_rng(20260831)
    clean = _observations(TERM_STRUCTURE)
    noisy = SurfaceObservations(
        k=clean.k, T=clean.T, iv=clean.iv + 0.002 * generator.standard_normal(clean.n)
    )
    first = SVICalibrator().fit(noisy)
    again = SVICalibrator().fit(noisy)
    shuffled = SVICalibrator().fit(noisy.subset(generator.permutation(noisy.n)))
    for a, b, c in zip(first.slices, again.slices, shuffled.slices):
        assert a.parameters() == b.parameters()
        for name in a.parameters():
            assert abs(getattr(a, name) - getattr(c, name)) < RECOVERY_TOL


def test_noise_degrades_the_fit_without_breaking_it() -> None:
    generator = np.random.default_rng(7)
    clean = _observations(TERM_STRUCTURE)
    noise = 0.005
    noisy = SurfaceObservations(
        k=clean.k, T=clean.T, iv=clean.iv + noise * generator.standard_normal(clean.n)
    )
    fitted = SVICalibrator().fit(noisy)
    # The residual standard deviation should sit near the injected noise; a fit
    # far below it is interpolating the noise, and far above it has not converged.
    for record in fitted.diagnostics:
        rms = np.sqrt(2 * record.cost / record.n_observations) / record.maturity
        assert 0.1 * noise < rms < 10 * noise


def test_a_maturity_with_too_few_quotes_is_refused() -> None:
    observations = SurfaceObservations(
        k=[-0.1, 0.0, 0.1, 0.2], T=[1.0] * 4, iv=[0.22, 0.20, 0.21, 0.23]
    )
    with pytest.raises(SurfaceCalibrationError, match="cannot be identified"):
        SVICalibrator().fit(observations)


def test_an_all_missing_observation_set_is_refused() -> None:
    observations = SurfaceObservations(
        k=np.linspace(-0.2, 0.2, 6), T=np.full(6, 1.0), iv=np.full(6, np.nan)
    )
    with pytest.raises(SurfaceCalibrationError, match="no slice to fit"):
        SVICalibrator().fit(observations)


def test_the_butterfly_penalty_improves_the_worst_durrleman_g() -> None:
    # A steep, narrow smile whose unconstrained fit dips below zero.
    arbitrageable = SVIParameters(a=0.002, b=0.9, rho=-0.85, m=0.0, sigma=0.02)
    observations = _observations({1.0: arbitrageable}, n_k=41, span=0.3)
    plain = SVICalibrator(butterfly_penalty=0.0).fit(observations)
    penalized = SVICalibrator(butterfly_penalty=1e4).fit(observations)
    assert plain.diagnostics[0].min_durrleman_g < 0.0
    assert penalized.diagnostics[0].min_durrleman_g > plain.diagnostics[0].min_durrleman_g


def test_the_fit_record_reports_the_optimizer_status_it_terminated_on() -> None:
    fitted = SVICalibrator().fit(_observations(TERM_STRUCTURE))
    for record in fitted.diagnostics:
        assert record.status > 0, record.message
        assert record.n_function_evaluations > 0
        assert set(record.to_dict()) == {
            "maturity",
            "parameters",
            "cost",
            "optimality",
            "status",
            "message",
            "n_observations",
            "n_function_evaluations",
            "min_durrleman_g",
        }


def test_the_calibrator_holds_no_state_between_fits() -> None:
    calibrator = SVICalibrator()
    first = calibrator.fit(_observations({1.0: TERM_STRUCTURE[1.0]}))
    calibrator.fit(_observations({2.0: TERM_STRUCTURE[2.0]}))
    third = calibrator.fit(_observations({1.0: TERM_STRUCTURE[1.0]}))
    assert first.parameters() == third.parameters()


# --- SSVI ---------------------------------------------------------------------


def test_an_ssvi_slice_is_exactly_a_raw_svi_slice() -> None:
    # a = theta(1 - rho^2)/2, b = theta phi / 2, m = -rho/phi, sigma = sqrt(1 - rho^2)/phi.
    surface = SSVISurface(
        maturities=(0.5, 1.0),
        theta=(0.04, 0.08),
        rho=-0.3,
        phi=PowerLawPhi(eta=1.0, gamma=0.5),
    )
    k = np.linspace(-0.6, 0.6, 25)
    for maturity in surface.maturities:
        direct = surface.total_variance(k, np.full(k.size, maturity))
        via_raw = surface.slice_parameters(maturity).total_variance(k)
        np.testing.assert_allclose(direct, via_raw, rtol=IDENTITY_TOL, atol=IDENTITY_TOL)


def test_theta_is_the_atm_total_variance_exactly() -> None:
    surface = SSVISurface(maturities=(1.0,), theta=(0.08,), rho=-0.3, phi=PowerLawPhi(eta=1.0))
    np.testing.assert_allclose(
        surface.total_variance(np.array([0.0]), np.array([1.0])),
        [0.08],
        rtol=IDENTITY_TOL,
        atol=0.0,
    )


def test_a_falling_theta_term_structure_is_refused() -> None:
    with pytest.raises(SurfaceValidationError, match="calendar-spread arbitrage"):
        SSVISurface(maturities=(1.0, 2.0), theta=(0.08, 0.04), rho=-0.3, phi=PowerLawPhi(eta=1.0))


def test_ssvi_recovers_the_parameters_it_was_generated_from() -> None:
    truth = SSVISurface(
        maturities=(0.25, 0.5, 1.0, 2.0),
        theta=(0.02, 0.035, 0.07, 0.14),
        rho=-0.45,
        phi=PowerLawPhi(eta=1.2, gamma=0.4),
    )
    k, T, iv = [], [], []
    for maturity in truth.maturities:
        axis = np.linspace(-0.5, 0.5, 25)
        k.append(axis)
        T.append(np.full(25, maturity))
        iv.append(np.sqrt(truth.total_variance(axis, np.full(25, maturity)) / maturity))
    observations = SurfaceObservations(
        k=np.concatenate(k), T=np.concatenate(T), iv=np.concatenate(iv)
    )
    fitted = SSVICalibrator().fit(observations)
    np.testing.assert_allclose(fitted.theta, truth.theta, rtol=RECOVERY_TOL, atol=1e-12)
    assert abs(fitted.rho - truth.rho) < RECOVERY_TOL
    assert abs(fitted.phi.eta - 1.2) < RECOVERY_TOL
    assert abs(fitted.phi.gamma - 0.4) < RECOVERY_TOL


def test_the_ssvi_fit_is_arbitrage_free_on_the_mesh() -> None:
    observations = _observations(TERM_STRUCTURE)
    fitted = SSVICalibrator().fit(observations)
    grid = SurfaceGridSpec(k=np.linspace(-0.35, 0.35, 25), T=list(TERM_STRUCTURE))
    assert materialize_surface(fitted, grid).validate().passed is True


def test_the_sufficient_condition_is_conservative_and_says_so() -> None:
    # A surface can fail the sufficient inequalities and still have a strictly
    # positive Durrleman g everywhere. The report separates the two claims, and
    # this test is what makes the separation more than a docstring.
    surface = SSVISurface(
        maturities=(1.0,), theta=(0.10,), rho=-0.35, phi=PowerLawPhi(eta=SAFE_ETA, gamma=0.5)
    )
    condition = surface.sufficient_butterfly_condition(1.0)
    assert condition["theta_phi_term"] < 4.0
    assert condition["satisfied"] is False
    assert float(np.min(surface.slice_parameters(1.0).durrleman_g(np.linspace(-1, 1, 401)))) > 0.0


def test_the_enforced_eta_bound_keeps_the_first_condition_satisfied() -> None:
    fitted = SSVICalibrator(enforce_no_butterfly=True).fit(_observations(TERM_STRUCTURE))
    assert fitted.phi.eta <= SAFE_ETA + 1e-9
    for maturity in fitted.maturities:
        assert fitted.sufficient_butterfly_condition(maturity)["theta_phi_term"] < 4.0


def test_the_heston_like_smoothing_function_is_positive_and_decreasing() -> None:
    phi = HestonLikePhi(lam=2.0)
    theta = np.linspace(0.01, 1.0, 50)
    values = phi(theta)
    assert bool(np.all(values > 0.0))
    assert bool(np.all(np.diff(values) < 0.0))


def test_ssvi_can_be_fitted_with_the_heston_like_family() -> None:
    fitted = SSVICalibrator(phi_family="heston_like").fit(_observations(TERM_STRUCTURE))
    assert isinstance(fitted.phi, HestonLikePhi)
    assert fitted.diagnostics["status"] > 0


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"eta": 0.0}, "eta must be finite and strictly positive"),
        ({"gamma": 1.0}, "gamma must lie strictly inside"),
    ],
)
def test_invalid_power_law_parameters_are_refused(kwargs, message) -> None:
    with pytest.raises(SurfaceValidationError, match=message):
        PowerLawPhi(**{"eta": 1.0, "gamma": 0.5, **kwargs})


def test_an_underdetermined_ssvi_fit_is_refused() -> None:
    observations = SurfaceObservations(k=[-0.1, 0.0, 0.1], T=[1.0] * 3, iv=[0.22, 0.20, 0.21])
    with pytest.raises(SurfaceCalibrationError, match="not identified"):
        SSVICalibrator().fit(observations)


# --- the protocols ------------------------------------------------------------


def test_every_parametric_type_satisfies_the_protocol_it_claims() -> None:
    surface = SSVISurface(maturities=(1.0,), theta=(0.08,), rho=-0.3, phi=PowerLawPhi(eta=1.0))
    assert isinstance(SVISmile(parameters=REFERENCE, maturity=1.0), DefiniteIVSurface)
    assert isinstance(SVISurface(maturities=(1.0,), slices=(REFERENCE,)), DefiniteIVSurface)
    assert isinstance(surface, DefiniteIVSurface)
    assert isinstance(SVICalibrator(), SurfaceCalibrator)
    assert isinstance(SSVICalibrator(), SurfaceCalibrator)


def test_the_parameter_mappings_are_json_safe_plain_data() -> None:
    import json

    surface = SSVISurface(maturities=(1.0,), theta=(0.08,), rho=-0.3, phi=PowerLawPhi(eta=1.0))
    for payload in (
        REFERENCE.parameters(),
        REFERENCE.to_jump_wings(1.0).parameters(),
        SVISurface(maturities=(1.0,), slices=(REFERENCE,)).parameters(),
        surface.parameters(),
    ):
        assert json.loads(json.dumps(payload, allow_nan=False)) == payload
