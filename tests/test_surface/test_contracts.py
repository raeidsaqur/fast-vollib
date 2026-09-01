"""The surface contracts: coordinates, ownership, identity, and honest absence."""

from __future__ import annotations

import json
import pickle

import numpy as np
import pytest

from fast_vollib.surface import (
    CoordinateConvention,
    DefiniteIVSurface,
    ForecastHorizon,
    GenerativeSurfaceModel,
    GridIVSurface,
    IVSurface,
    MissingMarketStateError,
    SurfaceCalibrator,
    SurfaceDistribution,
    SurfaceForecaster,
    SurfaceGridSpec,
    SurfaceMarket,
    SurfaceObservations,
    SurfacePoints,
    SurfacePrediction,
    SurfaceSamples,
    SurfaceTypeError,
    SurfaceValidationError,
    align_predictions,
    materialize_samples,
    materialize_surface,
    points_from_forward_delta,
    points_from_spot_moneyness,
    points_from_strikes,
)
from fast_vollib.surface.fitting import FlatIVSurface, FlatVolatilityCalibrator


def _points(n: int = 3) -> SurfacePoints:
    return SurfacePoints(
        k=np.linspace(-0.1, 0.1, n),
        T=np.full(n, 0.5),
        point_id=np.arange(n),
    )


# --- coordinates are canonical and validated ---------------------------------


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"k": [np.inf], "T": [1.0]}, "k must be finite"),
        ({"k": [0.0], "T": [np.nan]}, "T must be finite"),
        ({"k": [0.0], "T": [0.0]}, "T must be strictly positive"),
        ({"k": [0.0], "T": [-1.0]}, "T must be strictly positive"),
        ({"k": [0.0, 0.1], "T": [1.0, 2.0, 3.0]}, "same length"),
    ],
)
def test_point_coordinates_are_validated_unconditionally(kwargs, message) -> None:
    with pytest.raises(SurfaceValidationError, match=message):
        SurfacePoints(**kwargs)


def test_points_default_to_a_single_surface_labelled_zero() -> None:
    points = SurfacePoints(k=[0.0, 0.1], T=[1.0, 1.0])
    assert points.surface_ids() == [0]
    assert points.has_point_ids is False


def test_a_strike_adapter_records_the_market_it_used() -> None:
    points = points_from_strikes(
        [90.0, 100.0, 110.0],
        [1.0, 1.0, 1.0],
        forward=100.0,
        market_source="synthetic flat forward",
    )
    np.testing.assert_allclose(points.k, np.log([0.9, 1.0, 1.1]), rtol=1e-12, atol=1e-14)
    assert points.convention.source == "strike"
    assert points.convention.market_source == "synthetic flat forward"


def test_spot_moneyness_and_strike_adapters_agree_when_the_carry_is_undone() -> None:
    forward, spot = 102.0, 100.0
    strikes = np.array([90.0, 100.0, 110.0])
    by_strike = points_from_strikes(strikes, [1.0] * 3, forward=forward)
    by_moneyness = points_from_spot_moneyness(
        np.log(strikes / spot), [1.0] * 3, forward=forward, spot=spot
    )
    np.testing.assert_allclose(by_strike.k, by_moneyness.k, rtol=1e-12, atol=1e-14)


def test_the_delta_adapter_inverts_the_black_forward_delta_exactly() -> None:
    # k = w/2 - sqrt(w) N^{-1}(delta); at delta = 1/2 the inverse normal is 0.
    points = points_from_forward_delta([0.5, 0.25], [1.0, 1.0], [0.2, 0.2])
    np.testing.assert_allclose(points.k[0], 0.5 * 0.04, rtol=1e-12, atol=1e-14)
    assert points.k[1] > points.k[0]  # a lower call delta is a higher strike
    assert points.convention.source == "forward_delta"


def test_a_put_delta_maps_to_its_call_delta() -> None:
    call = points_from_forward_delta([0.25], [1.0], [0.2], is_call=True)
    put = points_from_forward_delta([-0.75], [1.0], [0.2], is_call=False)
    np.testing.assert_allclose(call.k, put.k, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("delta", [0.0, 1.0, 1.5], ids=["zero", "one", "above-one"])
def test_a_degenerate_delta_is_refused(delta) -> None:
    with pytest.raises(SurfaceValidationError, match="strictly inside"):
        points_from_forward_delta([delta], [1.0], [0.2])


# --- ownership ---------------------------------------------------------------


def test_point_arrays_are_owned_copies_and_read_only() -> None:
    k = np.array([-0.1, 0.0, 0.1])
    points = SurfacePoints(k=k, T=[1.0, 1.0, 1.0])
    k[0] = 99.0
    assert points.k[0] == -0.1
    assert points.k.flags.writeable is False


def test_prediction_and_sample_arrays_are_read_only() -> None:
    points = _points()
    prediction = SurfacePrediction(points=points, iv=[0.2, 0.2, 0.2])
    samples = SurfaceSamples(points=points, iv=np.full((2, 3), 0.2))
    assert prediction.iv.flags.writeable is False
    assert samples.iv.flags.writeable is False


def test_observations_carry_optional_weights_and_prices_as_owned_copies() -> None:
    weight = np.array([1.0, 2.0, 3.0])
    observations = SurfaceObservations(
        k=[-0.1, 0.0, 0.1],
        T=[1.0, 1.0, 1.0],
        iv=[0.2, 0.2, 0.2],
        weight=weight,
        price=[0.05, 0.08, 0.05],
    )
    weight[0] = 99.0
    assert observations.has_weights and observations.has_prices
    assert observations.weight[0] == 1.0
    assert observations.weight.flags.writeable is False


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"weight": [-1.0, 0.0, 1.0]}, "weight must be non-negative"),
        ({"weight": [np.nan, 0.0, 1.0]}, "a missing weight is 0"),
        ({"price": [-1.0, 0.0, 1.0]}, "price must be non-negative"),
        ({"price": [np.inf, 0.0, 1.0]}, "infinities are rejected"),
    ],
)
def test_weight_and_price_validation(kwargs, message) -> None:
    with pytest.raises(SurfaceValidationError, match=message):
        SurfaceObservations(k=[-0.1, 0.0, 0.1], T=[1.0] * 3, iv=[0.2] * 3, **kwargs)


# --- identity and alignment --------------------------------------------------


def test_duplicate_coordinates_stay_distinguishable_through_point_ids() -> None:
    observations = SurfaceObservations(
        k=[0.0, 0.0], T=[1.0, 1.0], iv=[0.2, 0.3], point_id=["lit", "otc"]
    )
    assert observations.n == 2
    assert observations.point_id.tolist() == ["lit", "otc"]


def test_alignment_never_falls_back_to_row_order() -> None:
    truth = SurfaceObservations(
        k=[-0.1, 0.0, 0.1], T=[1.0] * 3, iv=[0.2, 0.21, 0.22], point_id=["a", "b", "c"]
    )
    shuffled = SurfaceObservations(
        k=[0.1, -0.1, 0.0], T=[1.0] * 3, iv=[0.22, 0.2, 0.21], point_id=["c", "a", "b"]
    )
    aligned = align_predictions(truth, shuffled)
    np.testing.assert_allclose(aligned, truth.iv, rtol=0, atol=0)


def test_alignment_on_ids_requires_point_ids_on_both_sides() -> None:
    truth = SurfaceObservations(k=[0.0], T=[1.0], iv=[0.2])
    with pytest.raises(SurfaceValidationError, match="requires point_id on both sides"):
        align_predictions(truth, truth)


def test_matches_domain_ignores_the_surface_label_and_equals_does_not() -> None:
    left = SurfacePoints(k=[0.0], T=[1.0], surface_id="grid", point_id=[0])
    right = SurfacePoints(k=[0.0], T=[1.0], surface_id="draw-7", point_id=[0])
    assert left.matches_domain(right) is True
    assert left.equals(right) is False


def test_point_equality_is_exact_and_not_tolerant() -> None:
    left = SurfacePoints(k=[0.1], T=[1.0])
    right = SurfacePoints(k=[0.1 + 1e-15], T=[1.0])
    assert left.equals(right) is False


# --- missingness is explicit -------------------------------------------------


def test_an_invalid_prediction_is_counted_not_dropped() -> None:
    prediction = SurfacePrediction(points=_points(), iv=[0.2, -0.1, np.nan])
    assert prediction.valid.tolist() == [True, False, False]
    assert (prediction.valid_count, prediction.invalid_count) == (1, 2)
    assert prediction.coverage == pytest.approx(1 / 3, rel=1e-12)
    # the offending values survive: an evaluator classifies them, it is not
    # handed a sanitized copy.
    assert prediction.iv[1] == -0.1


def test_a_model_may_declare_a_point_unanswered_without_emitting_nan() -> None:
    prediction = SurfacePrediction(points=_points(), iv=[0.2, 0.2, 0.2], valid=[True, True, False])
    assert prediction.invalid_count == 1


def test_quantiles_require_their_levels_and_must_not_cross() -> None:
    points = _points(2)
    with pytest.raises(SurfaceValidationError, match="must be given together"):
        SurfacePrediction(points=points, iv=[0.2, 0.2], quantiles=[[0.1, 0.1]])
    with pytest.raises(SurfaceValidationError, match="non-decreasing"):
        SurfacePrediction(
            points=points,
            iv=[0.2, 0.2],
            quantiles=[[0.3, 0.3], [0.1, 0.1]],
            quantile_levels=[0.1, 0.9],
        )


def test_sample_summaries_use_only_valid_draws() -> None:
    points = _points(2)
    iv = np.array([[0.2, 0.2], [0.4, np.nan], [0.6, 0.4]])
    samples = SurfaceSamples(points=points, iv=iv, rng_policy="numpy.default_rng(7)")
    mean = samples.mean_prediction()
    np.testing.assert_allclose(mean.iv, [0.4, 0.3], rtol=1e-12, atol=1e-14)
    median = samples.median_prediction()
    np.testing.assert_allclose(median.iv, [0.4, 0.3], rtol=1e-12, atol=1e-14)


# --- market state is never inferred ------------------------------------------


def test_a_grid_without_a_market_refuses_to_invent_one() -> None:
    grid = SurfaceGridSpec(k=np.linspace(-0.2, 0.2, 5), T=[1.0])
    with pytest.raises(MissingMarketStateError, match="not a price"):
        grid.require_market("Price-space error")


def test_the_market_term_structure_interpolates_under_a_declared_policy() -> None:
    market = SurfaceMarket.from_spot(spot=100.0, T=[0.25, 1.0], rate=0.03, carry=0.01)
    # log-linear in the forward reproduces the pillar exactly.
    np.testing.assert_allclose(market.forward_at(1.0), 100.0 * np.exp(0.02), rtol=1e-12, atol=1e-12)
    exact = SurfaceMarket(T=[0.25, 1.0], forward=[100.0, 102.0], interpolation="exact")
    with pytest.raises(SurfaceValidationError, match="every maturity to be a pillar"):
        exact.forward_at(0.5)


def test_a_market_refuses_unordered_pillars_and_non_positive_forwards() -> None:
    with pytest.raises(SurfaceValidationError, match="strictly increasing"):
        SurfaceMarket(T=[1.0, 0.5], forward=[100.0, 100.0])
    with pytest.raises(SurfaceValidationError, match="strictly positive at every pillar"):
        SurfaceMarket(T=[1.0], forward=[0.0])


# --- grids and materialization ------------------------------------------------


def test_an_ivsurface_has_no_evaluate_method() -> None:
    # The grid is a materialized mesh, not a continuous surface. Off-grid
    # evaluation goes through GridIVSurface, which names its policy.
    surface = IVSurface.from_logmoneyness(
        np.array([-0.1, 0.0, 0.1]), np.array([1.0]), np.full((3, 1), 0.2)
    )
    assert not hasattr(surface, "evaluate")
    assert not isinstance(surface, DefiniteIVSurface)


def test_the_grid_adapter_satisfies_the_definite_surface_protocol() -> None:
    surface = IVSurface.from_logmoneyness(
        np.array([-0.1, 0.0, 0.1]), np.array([0.5, 1.0]), np.full((3, 2), 0.2)
    )
    assert isinstance(GridIVSurface(surface, policy="total_variance"), DefiniteIVSurface)


def test_an_unknown_interpolation_policy_is_refused() -> None:
    surface = IVSurface.from_logmoneyness(
        np.array([-0.1, 0.0, 0.1]), np.array([1.0]), np.full((3, 1), 0.2)
    )
    with pytest.raises(SurfaceValidationError, match="policy must be one of"):
        GridIVSurface(surface, policy="cubic")


def test_interpolating_total_variance_and_implied_volatility_differ_off_the_nodes() -> None:
    # A term structure that is flat in total variance is falling in implied
    # volatility, so the two policies disagree away from a node -- which is why
    # the policy is a required argument rather than a default.
    T = np.array([0.5, 1.0])
    w = np.array([[0.02, 0.02], [0.02, 0.02], [0.02, 0.02]])
    iv = np.sqrt(w / T[None, :])
    surface = IVSurface.from_logmoneyness(np.array([-0.1, 0.0, 0.1]), T, iv)
    query = SurfacePoints(k=[0.0], T=[0.75])
    in_w = GridIVSurface(surface, policy="total_variance").evaluate(query).iv[0]
    in_iv = GridIVSurface(surface, policy="implied_volatility").evaluate(query).iv[0]
    np.testing.assert_allclose(in_w, np.sqrt(0.02 / 0.75), rtol=1e-12, atol=1e-14)
    assert abs(in_w - in_iv) > 1e-4


def test_a_point_outside_the_mesh_is_unanswered_rather_than_extrapolated() -> None:
    surface = IVSurface.from_logmoneyness(
        np.array([-0.1, 0.0, 0.1]), np.array([0.5, 1.0]), np.full((3, 2), 0.2)
    )
    reader = GridIVSurface(surface, policy="total_variance")
    prediction = reader.evaluate(SurfacePoints(k=[5.0], T=[0.75]))
    assert prediction.valid.tolist() == [False]


def test_materialization_puts_model_output_on_the_declared_mesh() -> None:
    grid = SurfaceGridSpec(k=np.linspace(-0.2, 0.2, 5), T=[0.25, 1.0])
    materialized = materialize_surface(FlatIVSurface(level=0.2), grid)
    assert materialized.iv.shape == (5, 2)
    np.testing.assert_allclose(materialized.iv, 0.2, rtol=0, atol=0)


def test_materialization_detects_a_surface_that_reorders_its_answer() -> None:
    class Reversing:
        def evaluate(self, points, *, market=None):
            reversed_points = points.subset(np.arange(points.n)[::-1])
            return SurfacePrediction(points=reversed_points, iv=np.full(points.n, 0.2))

    grid = SurfaceGridSpec(k=np.linspace(-0.2, 0.2, 5), T=[1.0])
    with pytest.raises(SurfaceValidationError, match="different points than it was asked"):
        materialize_surface(Reversing(), grid)


def test_samples_materialize_onto_the_grid_they_were_drawn_on() -> None:
    grid = SurfaceGridSpec(k=np.linspace(-0.2, 0.2, 5), T=[0.5, 1.0])
    points = grid.to_points()
    samples = SurfaceSamples(points=points, iv=np.full((3, points.n), 0.2))
    surfaces = list(materialize_samples(samples, grid))
    assert len(surfaces) == 3
    assert all(surface.iv.shape == (5, 2) for surface in surfaces)


def test_samples_drawn_elsewhere_cannot_be_reshaped_onto_a_grid() -> None:
    grid = SurfaceGridSpec(k=np.linspace(-0.2, 0.2, 5), T=[0.5, 1.0])
    elsewhere = SurfacePoints(k=np.linspace(-0.2, 0.2, 10), T=np.full(10, 1.0))
    samples = SurfaceSamples(points=elsewhere, iv=np.full((2, 10), 0.2))
    with pytest.raises(SurfaceValidationError, match="other than this grid's nodes"):
        list(materialize_samples(samples, grid))


def test_a_grid_refuses_a_topology_that_disagrees_with_its_coordinates() -> None:
    with pytest.raises(SurfaceValidationError, match="requires a 1-D k"):
        SurfaceGridSpec(k=np.zeros((3, 2)), T=[0.5, 1.0])
    with pytest.raises(SurfaceValidationError, match="requires a 2-D k"):
        SurfaceGridSpec(k=[-0.1, 0.0, 0.1], T=[1.0], topology="fixed_strike")


def test_a_fixed_strike_grid_is_two_dimensional_in_moneyness() -> None:
    market = SurfaceMarket.from_spot(spot=100.0, T=[0.5, 1.0], rate=0.05)
    grid = SurfaceGridSpec.from_strikes([90.0, 100.0, 110.0], [0.5, 1.0], market=market)
    assert grid.topology == "fixed_strike"
    assert grid.k.shape == (3, 2)
    assert grid.shared_k is False


def test_grid_nodes_carry_flat_indices_so_a_prediction_can_be_put_back() -> None:
    grid = SurfaceGridSpec(k=np.linspace(-0.2, 0.2, 4), T=[0.5, 1.0])
    points = grid.to_points()
    assert points.point_id.tolist() == list(range(8))


# --- the protocols -----------------------------------------------------------


def test_the_baseline_calibrator_and_forecaster_satisfy_their_protocols() -> None:
    from fast_vollib.surface.fitting import PersistenceForecaster

    assert isinstance(FlatVolatilityCalibrator(), SurfaceCalibrator)
    assert isinstance(PersistenceForecaster(), SurfaceForecaster)
    assert isinstance(FlatIVSurface(level=0.2), DefiniteIVSurface)


def test_a_distribution_and_a_generative_model_are_recognized_structurally() -> None:
    class Distribution:
        def sample(self, points, *, n_samples, rng, market=None):
            generator = np.random.default_rng(rng)
            return SurfaceSamples(
                points=points,
                iv=0.2 + 0.01 * generator.standard_normal((n_samples, points.n)),
            )

    class Model:
        def distribution(self, context, *, horizon=None):
            return Distribution()

    assert isinstance(Distribution(), SurfaceDistribution)
    assert isinstance(Model(), GenerativeSurfaceModel)


def test_a_calibrator_holds_no_state_between_fits() -> None:
    calibrator = FlatVolatilityCalibrator()
    first = calibrator.fit(SurfaceObservations(k=[0.0], T=[1.0], iv=[0.2]))
    second = calibrator.fit(SurfaceObservations(k=[0.0], T=[1.0], iv=[0.3]))
    third = calibrator.fit(SurfaceObservations(k=[0.0], T=[1.0], iv=[0.2]))
    assert (first.level, second.level, third.level) == (0.2, 0.3, 0.2)


def test_batched_and_per_surface_fits_agree() -> None:
    from fast_vollib.surface.fitting import fit_each

    observations = SurfaceObservations(
        k=[0.0, 0.1, 0.0, 0.1],
        T=[1.0, 1.0, 1.0, 1.0],
        iv=[0.2, 0.3, 0.4, 0.5],
        surface_id=["a", "a", "b", "b"],
    )
    batched = fit_each(FlatVolatilityCalibrator(), observations)
    for label, subset in observations.surfaces():
        alone = FlatVolatilityCalibrator().fit(subset)
        assert batched[label].level == alone.level


def test_a_horizon_is_a_validated_value() -> None:
    assert ForecastHorizon(steps=5, step_years=1 / 252).years == pytest.approx(5 / 252)
    assert ForecastHorizon().years is None
    with pytest.raises(SurfaceValidationError, match="steps must be at least 1"):
        ForecastHorizon(steps=0)
    with pytest.raises(SurfaceValidationError, match="step_years must be strictly positive"):
        ForecastHorizon(steps=1, step_years=0.0)


# --- randomness is explicit ---------------------------------------------------


def test_no_surface_module_reads_a_module_global_random_stream() -> None:
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parents[2] / "src" / "fast_vollib" / "surface"
    offenders: dict[str, list[int]] = {}
    banned = {"seed", "rand", "randn", "random_sample", "choice", "shuffle", "permutation"}
    for path in sorted(root.rglob("*.py")):
        lines: list[int] = []
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=str(path))):
            if not isinstance(node, ast.Attribute) or node.attr not in banned:
                continue
            owner = node.value
            if isinstance(owner, ast.Attribute) and owner.attr == "random":
                lines.append(node.lineno)
        if lines:
            offenders[path.name] = lines
    assert offenders == {}


# --- serialization ------------------------------------------------------------


def test_every_value_object_serializes_to_json_safe_plain_data() -> None:
    market = SurfaceMarket.from_spot(spot=100.0, T=[0.5, 1.0], rate=0.02, source="synthetic")
    grid = SurfaceGridSpec(k=np.linspace(-0.2, 0.2, 3), T=[0.5, 1.0], market=market, name="demo")
    payloads = [
        CoordinateConvention().to_dict(),
        market.to_dict(),
        grid.to_dict(),
        ForecastHorizon(steps=2, step_years=1 / 252).to_dict(),
    ]
    for payload in payloads:
        assert json.loads(json.dumps(payload, allow_nan=False)) == payload


def test_serialization_is_deterministic_across_repeated_renders() -> None:
    grid = SurfaceGridSpec(k=np.linspace(-0.2, 0.2, 3), T=[0.5, 1.0], name="demo")
    first = json.dumps(grid.to_dict(), sort_keys=False)
    second = json.dumps(grid.to_dict(), sort_keys=False)
    assert first == second


def test_value_objects_survive_a_pickle_round_trip() -> None:
    points = _points()
    restored = pickle.loads(pickle.dumps(points))
    assert restored.equals(points)


# --- labels must survive JSON ------------------------------------------------


@pytest.mark.parametrize("labels", [[1.5, 2.5], [True, False]], ids=["float", "bool"])
def test_a_label_that_does_not_round_trip_through_json_is_refused(labels) -> None:
    with pytest.raises(SurfaceTypeError):
        SurfacePoints(k=[0.0, 0.1], T=[1.0, 1.0], surface_id=labels)
