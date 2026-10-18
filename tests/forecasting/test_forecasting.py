"""Independent ownership, identity, time and existing-engine parity checks."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone
import os

import numpy as np
import pytest

from fast_vollib.forecasting import (
    CurveBlock,
    EventBlock,
    FixedGridDrawSet,
    ForecastDistribution,
    ForecastDrawSet,
    ForecastError,
    ForecastMeta,
    ForecastProjectionError,
    ForecastQuery,
    ForecastQueryError,
    ForecastResult,
    ForecastTarget,
    MarginalBlock,
    MarkedEvent,
    ProcessDistribution,
    Quantity,
    ReturnSpec,
    ScenarioView,
    SurfaceBlock,
    SurfaceDistributionAdapter,
    TrajectoryBlock,
    returns_to_spot,
)
from fast_vollib.instruments import InstrumentRef
from fast_vollib.processes import GBM
from fast_vollib.simulation import Scenario, simulate
from fast_vollib.surface import SurfacePoints, SurfaceSamples
from fast_vollib.surface.fitting import FlatIVSurface
from fast_vollib.surface.generative import GaussianFieldSurfaceDistribution

NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)
REF = InstrumentRef(identifier="ACME", currency="USD")


def query(**kw):
    return ForecastQuery(
        **(
            dict(
                origin=NOW,
                times=(0.0, 0.25, 1.0),
                convention="act_365",
                targets=("asset",),
                kind="path",
            )
            | kw
        )
    )


def meta(**kw):
    return ForecastMeta(
        **(
            dict(
                measure="physical",
                weighting_scheme="equal",
                sampling_dependence="unknown",
                coupling="joint",
                initial_state="point",
                generator="synthetic",
                draw_set_id="draw1",
                conditioning_cutoff=NOW,
                observations_digest="synthetic:observations",
            )
            | kw
        )
    )


def target(**kw):
    return ForecastTarget(
        **(
            dict(
                target_id="asset",
                entity=REF,
                quantity=Quantity.SPOT,
                units="USD",
                resolution="fixed_grid",
            )
            | kw
        )
    )


def result(**kw):
    b = TrajectoryBlock(
        target_id="asset",
        times=(0.0, 0.25, 1.0),
        values=np.arange(18.0, dtype=float).reshape(2, 3, 3),
        channels=("spot", "variance", "other"),
    )
    return ForecastResult(
        **(
            dict(
                query=query(),
                targets=(target(),),
                blocks={"asset": b},
                sample_ids=["draw1:0", "draw1:1"],
                weights=None,
                meta=meta(),
            )
            | kw
        )
    )


def process_distribution(**kw):
    return ProcessDistribution(
        **(
            dict(
                process=GBM(drift=0.07, volatility=0.2),
                initial_state=100.0,
                underlier=REF,
                query=query(),
                measure="physical",
                backend_metadata={"backend": "numpy"},
                conditioning_cutoff=NOW,
                observations_digest="synthetic:observations",
            )
            | kw
        )
    )


@pytest.mark.parametrize(
    "times,kind",
    [
        ((-1.0,), "marginal"),
        ((0.0, 0.0), "path"),
        ((0.0, float("nan")), "path"),
        ((0.1, 1.0), "path"),
        ((0.0,), "path"),
        ((0.0, NOW), "path"),
        ((datetime(2026, 1, 1),), "marginal"),
        ((True, 1.0), "path"),
    ],
)
def test_query_refuses_invalid_time_grids(times, kind):
    with pytest.raises(ForecastError):
        query(times=times, kind=kind)


def test_query_time_normalization_and_target_validation():
    a = query(times=(NOW, NOW + timedelta(days=365)))
    assert a.times == (0.0, 1.0)
    assert query(times=(NOW, NOW + timedelta(days=360)), convention="act_360").times == (0.0, 1.0)
    assert query(times=(NOW, NOW + timedelta(days=365.25)), convention="act_365_25").times == (
        0.0,
        1.0,
    )
    with pytest.raises(ForecastError):
        query(targets=("asset", "asset"))
    with pytest.raises(ForecastError):
        query(convention="trading_252")
    with pytest.raises(ForecastError):
        target(quantity=Quantity.RETURN)
    with pytest.raises(ForecastError):
        target(
            return_spec=ReturnSpec(
                kind="log", cumulative=True, base_time=NOW, includes_distributions=False
            )
        )
    with pytest.raises(ForecastError):
        target(entity={"mutable": 1})


def test_process_parity_retention_and_no_randomness_in_queries():
    rng = np.random.default_rng(17)
    d = process_distribution()
    assert isinstance(d, ForecastDistribution)
    draw = d.draw(n_samples=5, rng=rng)
    assert isinstance(draw, ForecastDrawSet)
    state = repr(rng.bit_generator.state)
    a, b = draw.query(query()), draw.query(query())
    expected = simulate(
        REF, d.process, initial_state=100.0, time_grid=list(query().times), n_paths=5, rng=17
    )
    np.testing.assert_array_equal(a.blocks["asset"].values, expected.states)
    np.testing.assert_array_equal(a.blocks["asset"].values, b.blocks["asset"].values)
    np.testing.assert_array_equal(a.sample_ids, b.sample_ids)
    assert state == repr(rng.bit_generator.state)
    subset = draw.query(query(times=(0.0, 1.0)))
    np.testing.assert_array_equal(subset.blocks["asset"].values, expected.states[:, [0, 2], :])
    marginal = draw.query(query(times=(0.25,), kind="marginal"))
    np.testing.assert_array_equal(marginal.blocks["asset"].values, expected.states[:, 1, :])
    with pytest.raises(ForecastProjectionError):
        marginal.to_scenario_view(target_id="asset")
    view = a.to_scenario_view(target_id="asset")
    np.testing.assert_array_equal(view.scenario.states, expected.states)
    assert view.meta.measure == "physical" and view.target.entity == REF
    assert (
        draw.query(query()).meta.draw_set_id != d.draw(n_samples=5, rng=17).result.meta.draw_set_id
    )


@pytest.mark.parametrize(
    "changed",
    [
        dict(times=(0.0, 0.5, 1.0)),
        dict(times=(0.0, 2.0)),
        dict(origin=NOW + timedelta(seconds=1)),
        dict(convention="act_360"),
        dict(targets=("different",)),
    ],
)
def test_fixed_grid_refuses_refinement_and_identity_changes(changed):
    draws = process_distribution().draw(n_samples=2, rng=5)
    with pytest.raises(ForecastQueryError):
        draws.query(query(**changed))


def test_target_identity_does_not_collapse_equal_channel_names():
    a = result()
    b = replace(a.blocks["asset"], target_id="volume", values=a.blocks["asset"].values + 1000)
    q = query(targets=("asset", "volume"))
    both = ForecastResult(
        query=q,
        targets=(target(), target(target_id="volume", quantity=Quantity.VOLUME)),
        blocks={"asset": a.blocks["asset"], "volume": b},
        sample_ids=a.sample_ids,
        weights=None,
        meta=a.meta,
    )
    chosen = FixedGridDrawSet(both).query(query(targets=("volume", "asset"), times=(0.0, 1.0)))
    assert chosen.targets[0].quantity == Quantity.VOLUME
    assert chosen.blocks["volume"].values[0, 0, 0] == 1000
    assert chosen.blocks["asset"].values[0, 0, 0] == 0
    with pytest.raises(ForecastProjectionError):
        chosen.to_scenario_view(target_id="volume")
    # Volume failure does not invalidate the spot block.
    bad_volume = replace(b, values=np.full_like(b.values, np.nan), valid=np.zeros_like(b.valid))
    partial = replace(both, blocks={"asset": a.blocks["asset"], "volume": bad_volume})
    assert partial.to_scenario_view(target_id="asset").valid.all()


@pytest.mark.parametrize("weights", [[-1.0, 2.0], [0.0, 0.0], [np.inf, 1.0], [np.nan, 1.0], [1.0]])
def test_importance_weight_validation(weights):
    with pytest.raises(ForecastError):
        result(weights=weights, meta=meta(weighting_scheme="importance"))


def test_weights_sample_ids_and_owned_buffers():
    values = np.arange(18.0, dtype=float).reshape(2, 3, 3)
    b = TrajectoryBlock(
        target_id="asset", times=(0.0, 0.25, 1.0), values=values, channels=("spot", "x", "y")
    )
    weights = np.array([2.0, 3.0])
    ids = np.array(["draw1:0", "draw1:1"])
    mapping = {"asset": b}
    a = result(
        blocks=mapping, sample_ids=ids, weights=weights, meta=meta(weighting_scheme="importance")
    )
    values[:] = 500
    weights[:] = 200
    ids[:] = "mutated"
    mapping.clear()
    np.testing.assert_array_equal(a.weights, [0.4, 0.6])
    assert a.original_weight_mass == 5 and a.blocks["asset"].values[0, 0, 0] == 0
    for value in (a.weights, a.sample_ids, a.blocks["asset"].values, a.blocks["asset"].times):
        with pytest.raises(ValueError):
            value.setflags(write=True)
    chosen = FixedGridDrawSet(a).query(query(times=(0.0, 1.0)))
    np.testing.assert_array_equal(chosen.weights, a.weights)
    assert chosen.original_weight_mass == 5
    view = a.to_scenario_view(target_id="asset", channels=("spot",))
    with pytest.raises(ValueError):
        view.scenario.states.setflags(write=True)
    with pytest.raises(ForecastError):
        result(sample_ids=["draw1:0", "draw1:0"])
    with pytest.raises(ForecastError):
        result(sample_ids=["other:0", "other:1"])
    with pytest.raises(ForecastError):
        result(weights=[1.0, 1.0])
    with pytest.raises(ForecastError):
        result(meta=meta(weighting_scheme="importance"))
    with pytest.raises(ForecastError):
        meta(measure="stress_only")
    stress = result(meta=meta(measure="stress_only", weighting_scheme="none"))
    assert stress.weights is None
    assert not hasattr(stress, "probability_of_profit")


def test_missing_measure_marginal_and_direct_mutable_scenario_projection():
    with pytest.raises(ForecastProjectionError):
        result(meta=meta(measure=None)).to_scenario_view(target_id="asset")
    marginal = MarginalBlock(
        target_id="asset", time=1.0, values=np.array([[1.0], [2.0]]), channels=("spot",)
    )
    r = result(query=query(times=(1.0,), kind="marginal"), blocks={"asset": marginal})
    with pytest.raises(ForecastQueryError):
        FixedGridDrawSet(r).query(query(times=(0.0, 1.0)))
    with pytest.raises(ForecastProjectionError):
        r.to_scenario_view(target_id="asset")
    with pytest.raises(ForecastError):
        replace(r, query=query(times=(0.0, 1.0)))
    state = np.ones((2, 3, 1))
    scenario = Scenario.from_states(REF, time_grid=[0.0, 0.25, 1.0], states=state)
    view = ScenarioView(
        scenario=scenario,
        sample_ids=["draw1:0", "draw1:1"],
        weights=None,
        valid=np.ones((2, 3), dtype=bool),
        meta=meta(),
        origin=NOW,
        convention="act_365",
        target=target(),
    )
    scenario.states.setflags(write=True)
    scenario.states[:] = 999
    assert view.scenario.states[0, 0, 0] == 1
    with pytest.raises(ForecastError):
        replace(view, target=target(entity="wrong"))


def test_typed_surface_curve_events_and_shapes():
    points = SurfacePoints(k=[-0.1, 0.1], T=[1.0, 1.0])
    s = SurfaceBlock(target_id="surface", time=0.25, points=points, iv=[[0.2, 0.21], [0.22, 0.23]])
    c = CurveBlock(
        target_id="curve", time=0.25, tenors=[1.0, 2.0], values=np.array([[0.9, 0.8], [0.95, 0.85]])
    )
    for block, quantity, coords in (
        (s, Quantity.IV_SURFACE, ("k", "T")),
        (c, Quantity.DISCOUNT_CURVE, ("tenor",)),
    ):
        t = target(target_id=block.target_id, quantity=quantity, coordinates=coords)
        q = query(times=(0.25,), kind="marginal", targets=(t.target_id,))
        r = result(query=q, targets=(t,), blocks={t.target_id: block})
        assert FixedGridDrawSet(r).query(q).blocks[t.target_id] is block
        with pytest.raises(ForecastProjectionError):
            r.to_scenario_view(target_id=t.target_id)
    points.k.setflags(write=True)
    points.k[:] = 10
    assert s.points.k[0] == -0.1
    with pytest.raises(ValueError):
        s.points.k.setflags(write=True)
    payload = {"value": [1]}
    event = MarkedEvent(time=0.5, kind="event", payload=payload)
    e = EventBlock(target_id="asset", events_by_sample=((event,), ()))
    r = result(blocks={"asset": e}, targets=(target(quantity=Quantity.OTHER),))
    payload["value"][0] = 100
    assert e.events_by_sample[0][0].payload["value"] == (1,)
    selected = FixedGridDrawSet(r).query(query(times=(0.0, 0.25)))
    assert selected.blocks["asset"].events_by_sample == ((), ())
    with pytest.raises(ForecastError):
        replace(s, iv=[[0.2]])
    with pytest.raises(ForecastError):
        replace(s, surfaces=(object(),))
    with pytest.raises(ForecastError):
        replace(c, tenors=[2.0, 1.0])
    with pytest.raises(ForecastError):
        replace(result(), sample_ids=["draw1:0"])
    with pytest.raises(ForecastError):
        replace(result().blocks["asset"], valid=np.ones((2, 3)))


def test_surface_adapter_draws_once_single_date_and_identity():
    class Counting:
        calls = 0

        def sample(self, points, **kw):
            self.calls += 1
            return GaussianFieldSurfaceDistribution(
                base=FlatIVSurface(level=0.2), volatility=0.05
            ).sample(points, **kw)

    law = Counting()
    points = SurfacePoints(k=[-0.1, 0.1], T=[0.5, 1.0], point_id=[1, 2])
    date = NOW + timedelta(days=365)
    q = query(times=(date,), kind="marginal", targets=("iv",))
    d = SurfaceDistributionAdapter(
        surface_distribution=law,
        points=points,
        date=date,
        query=q,
        target=target(target_id="iv", quantity=Quantity.IV_SURFACE, coordinates=("k", "T")),
        meta=meta(),
    )
    draws = d.draw(n_samples=3, rng=2)
    a, b = draws.query(q), draws.query(q)
    assert law.calls == 1
    np.testing.assert_array_equal(a.blocks["iv"].iv, b.blocks["iv"].iv)
    with pytest.raises(ForecastError):
        replace(d, date=NOW)
    with pytest.raises(ForecastQueryError):
        draws.query(replace(q, times=(0.5,)))

    class Wrong:
        def sample(self, points, **kw):
            return SurfaceSamples(
                points=SurfacePoints(k=[0.0, 0.0], T=[0.5, 1.0]), iv=np.ones((kw["n_samples"], 2))
            )

    with pytest.raises(ForecastError):
        replace(d, surface_distribution=Wrong()).draw(n_samples=3, rng=2)


@pytest.mark.parametrize(
    "kind,cumulative", [("log", True), ("log", False), ("simple", True), ("simple", False)]
)
def test_explicit_financial_return_projection(kind, cumulative):
    spec = ReturnSpec(kind=kind, cumulative=cumulative, base_time=NOW, includes_distributions=False)
    r = np.array([0.0, 0.1, 0.2]) if cumulative else np.array([0.0, 0.1, 0.1])
    block = TrajectoryBlock(
        target_id="asset", times=(0.0, 0.25, 1.0), values=r[None, :, None], channels=("return",)
    )
    r0 = result(
        targets=(target(quantity=Quantity.RETURN, return_spec=spec),),
        blocks={"asset": block},
        sample_ids=["draw1:0"],
    )
    with pytest.raises(ForecastProjectionError):
        r0.to_scenario_view(target_id="asset")
    converted = returns_to_spot(r0, target_id="asset", initial_spot=100.0, underlier=REF)
    expected = 100 * np.exp(0.2) if kind == "log" else (120 if cumulative else 121)
    assert converted.to_scenario_view(target_id="asset").scenario.states[0, -1, 0] == pytest.approx(
        expected
    )
    with pytest.raises(ForecastProjectionError):
        returns_to_spot(
            r0, target_id="asset", initial_spot=100.0, underlier=InstrumentRef(identifier="OTHER")
        )
    total = replace(
        r0,
        targets=(replace(r0.targets[0], return_spec=replace(spec, includes_distributions=True)),),
    )
    with pytest.raises(ForecastProjectionError):
        returns_to_spot(total, target_id="asset", initial_spot=100.0, underlier=REF)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("device", ["cpu", "cuda"] if os.getenv("FV_TEST_CUDA") == "1" else ["cpu"])
def test_native_torch_simulation_parity_and_retained_ownership(dtype, device):
    torch = pytest.importorskip("torch")
    assert device != "cuda" or torch.cuda.is_available()
    value = torch.tensor(100.0, dtype=getattr(torch, dtype), device=device, requires_grad=True)
    p = process_distribution(
        initial_state=value, backend_metadata={"backend": "torch", "device": device}
    )
    draw = p.draw(n_samples=6, rng=17)
    expected = simulate(
        REF, p.process, initial_state=value, time_grid=list(query().times), n_paths=6, rng=17
    )
    np.testing.assert_array_equal(
        draw.result.blocks["asset"].values, expected.states.detach().cpu().numpy()
    )
    assert expected.states.device.type == device and draw.result.blocks[
        "asset"
    ].values.dtype == getattr(np, dtype)
    with torch.no_grad():
        value.fill_(20)
    assert draw.result.blocks["asset"].values[0, 0, 0] == 100
    np.testing.assert_array_equal(
        draw.result.to_scenario_view(target_id="asset").scenario.states,
        expected.states.detach().cpu().numpy(),
    )


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_native_jax_simulation_parity(dtype):
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    jax.config.update("jax_enable_x64", True)
    initial = jnp.asarray(100.0, dtype=getattr(jnp, dtype))
    p = process_distribution(
        initial_state=initial, backend_metadata={"backend": "jax", "device": "cpu"}
    )
    key = jax.random.key(17)
    draw = p.draw(n_samples=6, rng=key)
    expected = simulate(
        REF, p.process, initial_state=initial, time_grid=list(query().times), n_paths=6, rng=key
    )
    np.testing.assert_array_equal(draw.result.blocks["asset"].values, np.asarray(expected.states))
    assert draw.result.blocks["asset"].values.dtype == getattr(np, dtype)


def test_executable_forecast_example():
    from pathlib import Path
    import runpy

    runpy.run_path(str(Path(__file__).resolve().parents[2] / "examples/retained_forecasts.py"))
