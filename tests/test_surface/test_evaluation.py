"""Deterministic evaluation: coverage is part of the error, and a grid is not a proof."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import jsonschema
import numpy as np
import pytest

from fast_vollib.surface import (
    IVSurface,
    MissingMarketStateError,
    SurfaceGridSpec,
    SurfaceMarket,
    SurfaceObservations,
    SurfacePoints,
    SurfacePrediction,
    SurfaceValidationError,
    VerificationLevel,
    evaluate_prediction,
    materialize_surface,
)
from fast_vollib.surface.evaluation import (
    SCHEMA_VERSION,
    evaluation_json_schema,
)
from fast_vollib.surface.fitting import FlatIVSurface, SVIParameters

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = ROOT / "docs" / "schemas" / "fast-vollib-surface-evaluation-v1.schema.json"
GENERATOR = ROOT / "scripts" / "generate_surface_schemas.py"

REFERENCE = SVIParameters(a=0.04, b=0.40, rho=-0.40, m=0.0, sigma=0.10)


def _observations(n: int = 15) -> SurfaceObservations:
    k = np.linspace(-0.3, 0.3, n)
    T = np.full(n, 1.0)
    return SurfaceObservations(k=k, T=T, iv=np.sqrt(REFERENCE.total_variance(k) / T))


# --- coverage is part of the error --------------------------------------------


def test_a_perfect_prediction_scores_zero_error_and_full_coverage() -> None:
    observations = _observations()
    prediction = SurfacePrediction(points=observations.points, iv=observations.iv)
    evaluation = evaluate_prediction(prediction, observations)
    assert evaluation.iv_rmse == pytest.approx(0.0, abs=1e-15)
    assert evaluation.coverage == 1.0
    assert (evaluation.target_count, evaluation.valid_count, evaluation.invalid_count) == (
        15,
        15,
        0,
    )


def test_a_partial_prediction_is_not_flattered_by_its_own_gaps() -> None:
    """The failure this accounting exists to prevent, stated as a test.

    A model that answers three rows perfectly and declines twelve has an RMSE of
    zero over what it answered.  The three counts keep that visible, so it cannot
    be compared with a model that answered all fifteen without the difference
    being on the page.
    """
    observations = _observations()
    iv = observations.iv.copy()
    iv[3:] = np.nan
    prediction = SurfacePrediction(points=observations.points, iv=iv)
    evaluation = evaluate_prediction(prediction, observations)
    assert evaluation.iv_rmse == pytest.approx(0.0, abs=1e-15)
    assert (evaluation.valid_count, evaluation.invalid_count) == (3, 12)
    assert evaluation.coverage == pytest.approx(3 / 15, rel=1e-12)
    assert evaluation.invalid_rate == pytest.approx(12 / 15, rel=1e-12)


def test_a_negative_prediction_is_invalid_and_counted() -> None:
    observations = _observations(3)
    prediction = SurfacePrediction(points=observations.points, iv=[0.2, -0.1, 0.2])
    evaluation = evaluate_prediction(prediction, observations)
    assert evaluation.invalid_count == 1


def test_nothing_measured_reports_none_rather_than_zero() -> None:
    observations = SurfaceObservations(k=[0.0], T=[1.0], iv=[np.nan])
    prediction = SurfacePrediction(points=observations.points, iv=[0.2])
    evaluation = evaluate_prediction(prediction, observations)
    assert evaluation.iv_rmse is None
    assert evaluation.coverage is None
    assert evaluation.to_dict()["implied_volatility"]["rmse"] is None


# --- the error itself ----------------------------------------------------------


def test_the_error_metrics_are_what_they_say_they_are() -> None:
    observations = _observations(4)
    offsets = np.array([0.01, -0.02, 0.03, 0.0])
    prediction = SurfacePrediction(points=observations.points, iv=observations.iv + offsets)
    evaluation = evaluate_prediction(prediction, observations)
    np.testing.assert_allclose(
        evaluation.iv_rmse, np.sqrt(np.mean(offsets**2)), rtol=1e-12, atol=0.0
    )
    np.testing.assert_allclose(evaluation.iv_mae, np.mean(np.abs(offsets)), rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(evaluation.max_absolute_iv_error, 0.03, rtol=1e-12, atol=0.0)


def test_weights_change_the_weighted_error_and_not_the_plain_one() -> None:
    k = np.linspace(-0.2, 0.2, 4)
    T = np.full(4, 1.0)
    iv = np.sqrt(REFERENCE.total_variance(k) / T)
    weighted = SurfaceObservations(k=k, T=T, iv=iv, weight=[1.0, 1.0, 1.0, 100.0])
    plain = SurfaceObservations(k=k, T=T, iv=iv)
    offsets = np.array([0.01, 0.01, 0.01, 0.0])
    prediction = SurfacePrediction(points=plain.points, iv=iv + offsets)
    with_weights = evaluate_prediction(prediction, weighted)
    without = evaluate_prediction(prediction, plain)
    assert with_weights.iv_rmse == pytest.approx(without.iv_rmse, rel=1e-12)
    assert with_weights.weighted_iv_rmse < without.weighted_iv_rmse


def test_the_vega_weighted_error_needs_no_market() -> None:
    observations = _observations()
    prediction = SurfacePrediction(points=observations.points, iv=observations.iv + 0.01)
    evaluation = evaluate_prediction(prediction, observations)
    assert evaluation.vega_weighted_iv_rmse is not None
    assert evaluation.price_rmse is None


# --- the market is never inferred ----------------------------------------------


def test_price_error_is_absent_without_a_market_and_present_with_one() -> None:
    observations = _observations()
    prediction = SurfacePrediction(points=observations.points, iv=observations.iv + 0.005)
    without = evaluate_prediction(prediction, observations)
    assert without.price_rmse is None and without.market_source is None

    market = SurfaceMarket.flat(forward=100.0, rate=0.02, source="synthetic flat")
    with_market = evaluate_prediction(prediction, observations, market=market)
    assert with_market.price_rmse is not None
    assert with_market.market_source == "synthetic flat"


def test_the_price_error_scales_with_the_forward() -> None:
    # The forward-normalised call is dimensionless; a price is that scaled by the
    # discounted forward, so doubling the forward doubles the error exactly.
    observations = _observations()
    prediction = SurfacePrediction(points=observations.points, iv=observations.iv + 0.005)
    one = evaluate_prediction(prediction, observations, market=SurfaceMarket.flat(forward=100.0))
    two = evaluate_prediction(prediction, observations, market=SurfaceMarket.flat(forward=200.0))
    assert two.price_rmse == pytest.approx(2.0 * one.price_rmse, rel=1e-12)


def test_requiring_prices_without_a_market_raises() -> None:
    observations = _observations()
    prediction = SurfacePrediction(points=observations.points, iv=observations.iv)
    with pytest.raises(MissingMarketStateError, match="invented forward"):
        evaluate_prediction(prediction, observations, require_prices=True)


def test_the_spread_block_is_absent_rather_than_zero_without_quotes() -> None:
    observations = _observations()
    prediction = SurfacePrediction(points=observations.points, iv=observations.iv)
    assert evaluate_prediction(prediction, observations).inside_spread_fraction is None


def test_a_prediction_inside_the_quoted_spread_is_counted() -> None:
    k = np.array([-0.1, 0.0, 0.1])
    T = np.full(3, 1.0)
    iv = np.sqrt(REFERENCE.total_variance(k) / T)
    from fast_vollib._array_api import numpy_namespace
    from fast_vollib.surface.transforms import undiscounted_call

    fair = undiscounted_call(k, iv * iv * T, np.ones(3), numpy_namespace())
    observations = SurfaceObservations(k=k, T=T, iv=iv, bid=fair - 1e-3, ask=fair + 1e-3)
    prediction = SurfacePrediction(points=observations.points, iv=iv)
    assert evaluate_prediction(prediction, observations).inside_spread_fraction == 1.0


# --- splits --------------------------------------------------------------------


def test_the_split_counts_add_back_to_the_whole() -> None:
    observations = _observations(21)
    prediction = SurfacePrediction(points=observations.points, iv=observations.iv + 0.002)
    evaluation = evaluate_prediction(prediction, observations)
    assert sum(entry.valid_count for entry in evaluation.by_maturity) == evaluation.valid_count
    assert [entry.maturity for entry in evaluation.by_maturity] == [1.0]
    assert [entry.name for entry in evaluation.by_region] == ["liquid"]


def test_maturities_are_split_and_reported_ascending() -> None:
    k = np.tile(np.linspace(-0.2, 0.2, 5), 3)
    T = np.repeat([2.0, 0.5, 1.0], 5)
    iv = np.sqrt(REFERENCE.total_variance(k) / T)
    observations = SurfaceObservations(k=k, T=T, iv=iv)
    prediction = SurfacePrediction(points=observations.points, iv=iv)
    evaluation = evaluate_prediction(prediction, observations)
    assert [entry.maturity for entry in evaluation.by_maturity] == [0.5, 1.0, 2.0]


# --- verification levels --------------------------------------------------------


def test_a_grid_check_reports_the_empirical_level_and_nothing_stronger() -> None:
    observations = _observations()
    prediction = SurfacePrediction(points=observations.points, iv=observations.iv)
    grid = SurfaceGridSpec(k=np.linspace(-0.25, 0.25, 11), T=[1.0])
    materialized = materialize_surface(FlatIVSurface(level=0.2), grid)
    evaluation = evaluate_prediction(prediction, observations, grid=grid, materialized=materialized)
    assert evaluation.verification is VerificationLevel.EMPIRICAL_FINITE_GRID
    assert evaluation.arbitrage is not None
    assert evaluation.grid_shape == (11, 1)


def test_a_stronger_level_must_be_asserted_deliberately() -> None:
    # SSVI's non-decreasing theta really is a mathematical guarantee for the
    # calendar condition, and a caller who knows that can say so. It is never
    # inferred from a passing grid, which is the whole point of the vocabulary.
    observations = _observations()
    prediction = SurfacePrediction(points=observations.points, iv=observations.iv)
    grid = SurfaceGridSpec(k=np.linspace(-0.25, 0.25, 11), T=[1.0])
    materialized = materialize_surface(FlatIVSurface(level=0.2), grid)
    evaluation = evaluate_prediction(
        prediction,
        observations,
        grid=grid,
        materialized=materialized,
        verification=VerificationLevel.MATHEMATICAL_GUARANTEE,
    )
    assert evaluation.verification is VerificationLevel.MATHEMATICAL_GUARANTEE


def test_interpolated_nodes_are_reported_as_such() -> None:
    grid = SurfaceGridSpec(
        k=np.linspace(-0.25, 0.25, 5),
        T=[0.5, 1.0],
        native_mask=np.array(
            [[True, True], [False, False], [True, True], [False, False], [True, True]]
        ),
    )
    observations = _observations()
    prediction = SurfacePrediction(points=observations.points, iv=observations.iv)
    materialized = materialize_surface(FlatIVSurface(level=0.2), grid)
    evaluation = evaluate_prediction(prediction, observations, grid=grid, materialized=materialized)
    assert evaluation.native_node_fraction == pytest.approx(0.6, rel=1e-12)


def test_a_grid_without_a_materialized_surface_is_refused() -> None:
    observations = _observations()
    prediction = SurfacePrediction(points=observations.points, iv=observations.iv)
    grid = SurfaceGridSpec(k=np.linspace(-0.25, 0.25, 5), T=[1.0])
    with pytest.raises(SurfaceValidationError, match="without a materialized surface"):
        evaluate_prediction(prediction, observations, grid=grid)


# --- alignment ------------------------------------------------------------------


def test_a_prediction_on_other_points_is_refused() -> None:
    observations = _observations(5)
    elsewhere = SurfacePoints(k=np.linspace(-0.4, 0.4, 5), T=np.full(5, 1.0))
    prediction = SurfacePrediction(points=elsewhere, iv=np.full(5, 0.2))
    with pytest.raises(SurfaceValidationError, match="different points"):
        evaluate_prediction(prediction, observations)


# --- the wire contract -----------------------------------------------------------


def test_the_record_is_canonical_json_with_no_non_standard_floats() -> None:
    observations = _observations()
    prediction = SurfacePrediction(points=observations.points, iv=observations.iv)
    evaluation = evaluate_prediction(prediction, observations)
    text = evaluation.to_json(indent=2)
    assert text.endswith("\n")
    assert json.loads(text)["schema"] == SCHEMA_VERSION
    # allow_nan=False is what makes an unavailable number null rather than NaN.
    json.dumps(evaluation.to_dict(), allow_nan=False)


def test_rendering_is_deterministic() -> None:
    observations = _observations()
    prediction = SurfacePrediction(points=observations.points, iv=observations.iv + 0.001)
    evaluation = evaluate_prediction(prediction, observations)
    assert evaluation.to_json() == evaluation.to_json()


def test_the_committed_schema_is_byte_identical_to_the_generator() -> None:
    spec = importlib.util.spec_from_file_location("generate_surface_schemas", GENERATOR)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    for path, content in module.render().items():
        assert path.read_text(encoding="utf-8") == content, f"{path.name} is stale"


def test_the_committed_schema_is_a_valid_closed_draft_2020_12_schema() -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)

    def assert_closed(node):
        if isinstance(node, dict):
            if node.get("type") == "object" and "properties" in node:
                assert node.get("additionalProperties") is False
                assert sorted(node["required"]) == sorted(node["properties"])
            for value in node.values():
                assert_closed(value)
        elif isinstance(node, list):
            for item in node:
                assert_closed(item)

    assert_closed(schema)


def test_an_evaluation_validates_against_its_committed_schema() -> None:
    observations = _observations()
    prediction = SurfacePrediction(points=observations.points, iv=observations.iv + 0.001)
    grid = SurfaceGridSpec(k=np.linspace(-0.25, 0.25, 11), T=[1.0])
    materialized = materialize_surface(FlatIVSurface(level=0.2), grid)
    evaluation = evaluate_prediction(
        prediction,
        observations,
        market=SurfaceMarket.flat(forward=100.0, source="synthetic"),
        grid=grid,
        materialized=materialized,
    )
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(schema).validate(evaluation.to_dict())


def test_the_evaluation_schema_and_the_module_agree() -> None:
    assert json.loads(SCHEMA_PATH.read_text(encoding="utf-8")) == evaluation_json_schema()


def test_an_ivsurface_is_still_not_a_definite_surface() -> None:
    # Guards the boundary the evaluation depends on: a materialized mesh cannot
    # be handed to evaluate_prediction as if it were a model.
    surface = IVSurface.from_logmoneyness(
        np.array([-0.1, 0.0, 0.1]), np.array([1.0]), np.full((3, 1), 0.2)
    )
    assert not hasattr(surface, "evaluate")
