from dataclasses import replace
from datetime import date, time
from decimal import Decimal as D
import json
import os

import numpy as np
import pytest
from scipy.special import ndtr

from fast_vollib.instruments import (
    EuropeanOption,
    SettlementType,
    VanillaMarketInputs,
    greeks_instrument,
    payoff,
    price_instrument,
)
from fast_vollib.pnl import (
    EuropeanProxyProfile,
    EuropeanProxySensitivityBand,
    FlatVol,
    HorizonMarket,
    LognormalTerminal,
    PerLegVol,
    PnLError,
    PnLHorizonError,
    ProportionalCarry,
    SurfaceSensitivityBand,
    SurfaceVol,
    SurfaceVolSamples,
    european_proxy_profile,
    european_proxy_samples,
    expiry_profile,
    pnl_at_horizon,
    pnl_at_horizon_samples,
    position_greeks,
    probability_of_profit,
    repricing_index,
)
from fast_vollib.pnl.serialization import loads
from fast_vollib.portfolio import (
    OptionLeg,
    Position,
    hypothetical_snapshot,
    resolve_expiry_instant,
    year_fraction,
)
from fast_vollib.surface import SurfaceMarket, SurfacePoints, SurfacePrediction, SurfaceSamples
from fast_vollib.surface.fitting import FlatIVSurface

from ._fixtures import ASSUMPTION, NOW, WORLD, contract, executed, fill
from .test_terminal import condor, snap


def euro(kind="call", strike="100", expiry=date(2026, 11, 20), american=False):
    c = contract(kind, strike, expiry=expiry)
    if not american:
        c = replace(
            c,
            exercise="european",
            settlement=SettlementType.CASH,
            deliverable="standard_cash_equivalent:100",
        )
    return resolve_expiry_instant(c, policy=(time(10), "UTC"))


def position(*, american=False, mixed=False):
    return Position.from_legs(
        (
            OptionLeg(
                contract=euro("put", "95", american=american),
                quantity=2,
                allocated_entry=D(2),
                allocated_fees=D(1),
                leg_id="put",
            ),
            OptionLeg(
                contract=euro(
                    "call",
                    "105",
                    expiry=date(2026, 12, 18) if mixed else date(2026, 11, 20),
                    american=american,
                ),
                quantity=-3,
                allocated_entry=D(3),
                allocated_fees=D(1),
                leg_id="call",
            ),
        )
    )


def snapshot(**kw):
    return hypothetical_snapshot(
        position(**kw), valuation_time=NOW, knowledge_cutoff=NOW, world_id=WORLD
    )


def maturity(s):
    return year_fraction(NOW, s.position.legs[0].contract.expiry_instant, convention="act_365")


def market(vol=None, rate=0.03, q=0.01, **kw):
    return HorizonMarket(
        rate=rate, dividend_yield=q, volatility=FlatVol(0.2) if vol is None else vol, **kw
    )


def evaluate(s=None, spots=(90.0, 100.0, 110.0), **kw):
    s = s or snapshot()
    options = dict(
        market=market(), model="black_scholes_merton", backend="numpy", convention="act_365"
    )
    options.update(kw)
    if "horizons" not in options:
        options["horizons"] = (0.0, maturity(s) / 2, maturity(s))
    return pnl_at_horizon(s, spots, **options)


def vol(
    volclass=SurfaceVol,
    *,
    spot_policy="sticky_forward_moneyness",
    time_policy="constant_tenor",
    surface=None,
    distribution=None,
    n_samples=3,
):
    forward = ProportionalCarry(
        origin_spot=100.0,
        base_forward=lambda h, t: 100.0 * np.exp(0.02 * np.asarray(t)),
        policy_id="flat_carry/1",
    )
    ref = SurfaceMarket.from_spot(spot=100.0, T=[1e-12, 1.0], rate=0.03, carry=0.01)
    args = dict(
        reference_market=ref,
        spot_movement=spot_policy,
        time_movement=time_policy,
        forward_mapping=forward,
        forward_mapping_id=forward.policy_id,
    )
    return volclass(
        **args,
        **(
            dict(surface=surface or FlatIVSurface(level=0.2))
            if volclass is SurfaceVol
            else dict(distribution=distribution, n_samples=n_samples, rng=17)
        ),
    )


@pytest.mark.parametrize("chunk", [1, 7, 29, 100])
def test_non_square_chunked_unit_batches_equal_scalar_prices(chunk, monkeypatch):
    import fast_vollib.instruments as instruments

    original = instruments.price_instrument
    calls = []

    def record(batch, *args, **kwargs):
        calls.append(len(batch))
        np.testing.assert_array_equal(batch.notional, np.ones(len(batch)))
        return original(batch, *args, **kwargs)

    monkeypatch.setattr(instruments, "price_instrument", record)
    s = snapshot()
    grid = np.array([0.0, 90.0, 100.0, 110.0, 120.0])
    p = evaluate(s, grid, chunk_rows=chunk)
    assert max(calls) <= chunk and p.leg_values.shape == (3, 5, 2)
    expected = np.empty_like(p.leg_values)
    for h, elapsed in enumerate(p.horizons):
        for j, spot in enumerate(grid):
            for i, leg in enumerate(s.position.legs):
                remaining = maturity(s) - elapsed
                unit = EuropeanOption(
                    underlier=leg.contract.underlier,
                    option_type=leg.contract.option_type,
                    strike=float(leg.contract.strike),
                    maturity=max(remaining, 0.0),
                    notional=1.0,
                )
                value = (
                    payoff(unit, np.array([spot]))[0]
                    if remaining < 1e-12
                    else price_instrument(
                        unit,
                        VanillaMarketInputs(
                            underlying=spot, rate=0.03, dividend_yield=0.01, volatility=0.2
                        ),
                        model="black_scholes_merton",
                        backend="numpy",
                    )[0]
                )
                expected[h, j, i] = float(leg.signed_size) * value
    np.testing.assert_allclose(p.leg_values, expected, rtol=1e-12, atol=1e-11)
    np.testing.assert_allclose(p.total, float(s.cumulative_cash) + expected.sum(-1))
    indices = repricing_index(np.arange(30), n_spots=5, n_legs=2)
    for a, b in zip(indices, np.unravel_index(np.arange(30), (3, 5, 2))):
        np.testing.assert_array_equal(a, b)
    restored = loads(p.to_json())
    np.testing.assert_array_equal(restored.total, p.total)
    assert restored.meta.model == "black_scholes_merton"
    with pytest.raises(ValueError):
        p.total.setflags(write=True)


def test_remaining_maturities_conventions_and_lifecycle_limits():
    s = snapshot(mixed=True)
    T = maturity(s)
    p = evaluate(s, horizons=(0.0, T))
    assert p.leg_values.shape == (2, 3, 2)
    assert p.leg_values[-1, 1, 0] == 0 and p.leg_values[-1, 1, 1] < 0
    with pytest.raises(PnLHorizonError):
        evaluate(s, horizons=(T + 1e-4,))
    unresolved = replace(
        s,
        position=Position.from_legs(
            (
                replace(
                    s.position.legs[0],
                    contract=replace(
                        s.position.legs[0].contract, expiry_instant=None, expiry_policy=None
                    ),
                ),
            )
        ),
    )
    with pytest.raises(PnLHorizonError):
        evaluate(unresolved, horizons=(0.0,))
    for convention in ("act_360", "act_365_25"):
        elapsed = year_fraction(
            NOW, s.position.legs[0].contract.expiry_instant, convention=convention
        )
        out = evaluate(s, horizons=(elapsed,), convention=convention)
        assert out.meta.time_policy == convention and out.leg_values[0, 1, 0] == 0
    same = snapshot()
    rounded = float(np.float32(maturity(same)))
    p = evaluate(same, np.array([90.0, 100.0, 110.0], dtype=np.float32), horizons=(rounded,))
    terminal = expiry_profile(
        same, [90.0, 100.0, 110.0], terminal_assumption="standard_intrinsic_equivalent"
    )
    np.testing.assert_array_equal(p.total, terminal.total)
    closed = executed([fill(1, "2", id="open", seconds=-2), fill(-1, "3", id="close", seconds=-1)])
    np.testing.assert_array_equal(
        evaluate(closed, horizons=(0.0, 2.0)).total, np.full((2, 3), 100.0)
    )


def test_flat_per_leg_and_flat_surface_parity_and_required_policies():
    s = snapshot()
    flat = evaluate(s)
    per = evaluate(s, market=market(PerLegVol({"put": 0.2, "call": 0.2})))
    surface = evaluate(s, market=market(vol()))
    np.testing.assert_allclose(flat.total, per.total, rtol=0, atol=1e-12)
    np.testing.assert_allclose(flat.total, surface.total, rtol=1e-12, atol=1e-11)
    for bad in (PerLegVol({"put": 0.2}), PerLegVol({"put": 0.2, "call": 0.2, "extra": 0.2})):
        with pytest.raises(PnLError):
            evaluate(s, market=market(bad))
    with pytest.raises(PnLError):
        evaluate(s, model="black_scholes")
    with pytest.raises(PnLError):
        evaluate(s, market=market(q=None))
    with pytest.raises(PnLError):
        evaluate(s, model="black")
    with pytest.raises(PnLError):
        evaluate(s, chunk_rows=0)
    with pytest.raises(PnLHorizonError):
        evaluate(s, horizons=(False, 0.01))


@pytest.mark.parametrize("spot_policy", ["sticky_strike", "sticky_forward_moneyness"])
@pytest.mark.parametrize("time_policy", ["fixed_expiry", "constant_tenor"])
def test_surface_movement_coordinates_are_explicit(spot_policy, time_policy):
    class Recording:
        seen = []

        def evaluate(self, points, *, market):
            self.seen.append((points, market))
            return SurfacePrediction(points=points, iv=np.full(points.n, 0.2))

    surface = Recording()
    v = vol(spot_policy=spot_policy, time_policy=time_policy, surface=surface)
    s = snapshot()
    p = evaluate(s, spots=[80.0, 120.0], horizons=(maturity(s) / 2,), market=market(v))
    points, reference = surface.seen[0]
    assert reference is v.reference_market
    tenor = np.repeat(maturity(s) if time_policy == "fixed_expiry" else maturity(s) / 2, 4)
    strikes = np.tile([95.0, 105.0], 2)
    denominator = (
        reference.forward_at(tenor)
        if spot_policy == "sticky_strike"
        else np.repeat([80.0, 120.0], 2) * np.exp(0.02 * maturity(s) / 2)
    )
    np.testing.assert_allclose(points.k, np.log(strikes / denominator), rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(points.T, tenor)
    assert p.meta.spot_movement == spot_policy and p.meta.time_movement == time_policy
    bad = replace(v, forward_mapping=lambda spot, h, t: np.asarray(spot) * 2)
    with pytest.raises(PnLError):
        evaluate(s, market=market(bad))


def test_black76_requires_forward_mapping_and_discount_convention():
    s = snapshot()

    def mapping(spot, h, t):
        return np.asarray(spot) * np.exp(0.02 * np.asarray(t))

    m = market(
        forward_mapping=mapping,
        forward_mapping_id="constant_carry/1",
        discount_convention="constant_continuous_rate",
    )
    np.testing.assert_allclose(
        evaluate(s, model="black", market=m).total, evaluate(s).total, rtol=1e-12, atol=1e-11
    )
    with pytest.raises(PnLError):
        evaluate(s, model="black", market=replace(m, discount_convention=None))


def test_surface_invalid_cells_and_zero_spot_stay_unavailable():
    class Invalid:
        def evaluate(self, points, *, market):
            return SurfacePrediction(points=points, iv=np.full(points.n, 0.2), valid=points.k < 0)

    v = vol(surface=Invalid())
    with pytest.raises(PnLError):
        evaluate(market=market(v))
    out = evaluate(market=market(v), diagnostic=True)
    assert not out.valid.all() and np.isnan(out.total[~out.valid]).all()
    with pytest.raises(PnLError):
        evaluate(spots=[0.0, 100.0], market=market(vol()))
    out = evaluate(spots=[0.0, 100.0], market=market(vol()), diagnostic=True)
    assert not out.valid[0, 0] and out.valid[-1, 0]


class SharedSurface:
    def __init__(self, z):
        self.z = z

    def evaluate(self, points, *, market):
        return SurfacePrediction(
            points=points, iv=0.2 + 0.01 * self.z * (1 + 0.1 * points.k + 0.1 * points.T)
        )


class JointLaw:
    def __init__(self):
        self.calls = []

    def sample(self, points, *, n_samples, rng, market):
        self.calls.append(points)
        z = np.linspace(-1.0, 1.0, n_samples)
        return SurfaceSamples(
            points=points,
            iv=np.stack([SharedSurface(x).evaluate(points, market=market).iv for x in z]),
        )


@pytest.mark.parametrize("chunk", [1, 7, 100])
def test_coherent_surface_union_before_sample_and_model_chunking(chunk, monkeypatch):
    import fast_vollib.instruments as instruments

    original = instruments.price_instrument
    calls = []

    def record(batch, *args, **kwargs):
        calls.append(len(batch))
        return original(batch, *args, **kwargs)

    monkeypatch.setattr(instruments, "price_instrument", record)
    law = JointLaw()
    v = vol(SurfaceVolSamples, distribution=law)
    s = snapshot()
    hs = (0.0, maturity(s) / 2, maturity(s))
    spots = [90.0, 100.0, 110.0]
    m = replace(market(), volatility=None)
    band = pnl_at_horizon_samples(
        s,
        spots,
        horizons=hs,
        volatility=v,
        market=m,
        model="black_scholes_merton",
        backend="numpy",
        convention="act_365",
        chunk_rows=chunk,
    )
    assert isinstance(band, SurfaceSensitivityBand)
    assert len(law.calls) == 1 and band.samples.shape == (3, 3, 3, 2) and max(calls) <= chunk
    for i, z in enumerate(np.linspace(-1.0, 1.0, 3)):
        expected = evaluate(s, spots, horizons=hs, market=market(vol(surface=SharedSurface(z))))
        np.testing.assert_allclose(band.samples[i], expected.leg_values, rtol=1e-12, atol=1e-11)
    assert not hasattr(band, "probability_of_profit") and not hasattr(band, "summary")
    assert "not joint spot/volatility P&L probabilities" in band.meta.caveat
    saved = loads(band.to_json())
    np.testing.assert_array_equal(saved.total, band.total)
    with pytest.raises(ValueError):
        band.samples.setflags(write=True)
    np.testing.assert_allclose(
        band.quantiles([0.0, 1.0]), np.stack([band.total.min(0), band.total.max(0)])
    )


def test_union_deduplication_refinement_limits_and_invalid_mass():
    s = snapshot()
    hs = (0.0, maturity(s) / 2)
    law = JointLaw()
    v = vol(
        SurfaceVolSamples, distribution=law, spot_policy="sticky_strike", time_policy="fixed_expiry"
    )
    args = dict(
        horizons=hs,
        volatility=v,
        market=replace(market(), volatility=None),
        model="black_scholes_merton",
        backend="numpy",
        convention="act_365",
    )
    pnl_at_horizon_samples(s, [90.0, 100.0, 110.0], **args)
    assert law.calls[0].n == 2
    with pytest.raises(PnLError):
        pnl_at_horizon_samples(s, [90.0, 100.0, 110.0], **args, union_point_limit=1)
    assert len(law.calls) == 1

    class Invalid:
        def sample(self, points, *, n_samples, rng, market):
            values = np.full((n_samples, points.n), 0.2)
            mask = np.ones_like(values, dtype=bool)
            mask[0, 0] = False
            return SurfaceSamples(points=points, iv=values, valid=mask)

    bad = replace(v, distribution=Invalid())
    with pytest.raises(PnLError):
        pnl_at_horizon_samples(s, [100.0], **(args | {"volatility": bad}))
    result = pnl_at_horizon_samples(s, [100.0], **(args | {"volatility": bad}), diagnostic=True)
    assert np.isnan(result.quantiles([0.5])).all()


def test_american_standard_refusals_and_proxy_type_survives_export():
    s = snapshot(american=True)
    args = dict(
        horizons=(0.0,),
        market=market(),
        model="black_scholes_merton",
        backend="numpy",
        convention="act_365",
    )
    with pytest.raises(PnLError):
        pnl_at_horizon(s, [100.0], **args)
    p = european_proxy_profile(s, [100.0], **args)
    assert (
        isinstance(p, EuropeanProxyProfile)
        and p.meta.basis == "european_proxy"
        and p.contracts[0].exercise == "american"
    )
    assert isinstance(loads(p.to_json()), EuropeanProxyProfile)
    with pytest.raises(PnLError):
        probability_of_profit(
            p,
            LognormalTerminal(
                spot=100.0,
                drift=0.03,
                sigma=0.2,
                T=maturity(s),
                measure="physical",
                convention="act_365",
            ),
        )
    with pytest.raises(PnLError):
        position_greeks(
            s,
            VanillaMarketInputs(underlying=100.0, rate=0.03, volatility=0.2, dividend_yield=0.01),
            model="black_scholes_merton",
            backend="numpy",
            convention="act_365",
        )
    with pytest.raises(PnLError):
        european_proxy_profile(snapshot(), [100.0], **args)
    v = vol(SurfaceVolSamples, distribution=JointLaw())
    b = european_proxy_samples(
        s, [100.0], **(args | {"market": replace(market(), volatility=None), "volatility": v})
    )
    assert isinstance(b, EuropeanProxySensitivityBand) and isinstance(
        loads(b.to_json()), EuropeanProxySensitivityBand
    )
    assert "European surrogate illustration" in b.meta.caveat


def test_position_greeks_scale_existing_units_and_annual_unit_choice():
    s = snapshot()
    m = VanillaMarketInputs(underlying=100.0, rate=0.03, volatility=0.2, dividend_yield=0.01)
    out = position_greeks(s, m, model="black_scholes_merton", backend="numpy", convention="act_365")
    annual = position_greeks(
        s,
        m,
        model="black_scholes_merton",
        backend="numpy",
        convention="act_365",
        report_units="annual_unit",
    )
    for i, leg in enumerate(s.position.legs):
        unit = EuropeanOption(
            underlier=leg.contract.underlier,
            option_type=leg.contract.option_type,
            strike=float(leg.contract.strike),
            maturity=maturity(s),
        )
        expected = greeks_instrument(unit, m, model="black_scholes_merton", backend="numpy")
        for name in expected:
            assert out.per_leg[name][i] == pytest.approx(expected[name][0] * float(leg.signed_size))
    for name, factor in (
        ("theta", 365.0),
        ("vega", 100.0),
        ("rho", 100.0),
        ("delta", 1.0),
        ("gamma", 1.0),
    ):
        assert annual.net[name] == pytest.approx(out.net[name] * factor)
    assert annual.units["theta"] == "per model year"
    restored = loads(out.to_json())
    assert restored.net["delta"] == out.net["delta"]
    with pytest.raises(ValueError):
        out.per_leg["delta"].setflags(write=True)


def test_lognormal_profit_probability_and_strict_degenerate_atoms():
    pos = condor("b", "4")
    legs = tuple(
        replace(leg, contract=resolve_expiry_instant(leg.contract, policy=(time(10), "UTC")))
        for leg in pos.legs
    )
    s = snap(Position.from_legs(legs))
    T = maturity(s)
    p = expiry_profile(
        s, [500.0, 600.0, 700.0], terminal_assumption=ASSUMPTION, convention="act_365"
    )
    law = LognormalTerminal(
        spot=600.0, drift=0.05, sigma=0.2, T=T, measure="physical", convention="act_365"
    )
    out = probability_of_profit(p, law)

    def z(x):
        return (np.log(x / 600) - (0.05 - 0.5 * 0.2**2) * T) / (0.2 * np.sqrt(T))

    assert out.probability == pytest.approx(ndtr(z(616.56)) - ndtr(z(583.44)), abs=1e-13)
    assert out.meta.measure == "physical" and out.meta.caveat == p.meta.caveat
    for spot, expected in ((583.44, 0.0), (600.0, 1.0), (616.56, 0.0)):
        assert (
            probability_of_profit(p, replace(law, spot=spot, drift=0.0, sigma=0.0)).probability
            == expected
        )
    at_expiry = replace(
        s,
        valuation_time=legs[0].contract.expiry_instant,
        knowledge_cutoff=legs[0].contract.expiry_instant,
    )
    expiry = expiry_profile(
        at_expiry, [600.0], terminal_assumption=ASSUMPTION, convention="act_365"
    )
    assert probability_of_profit(expiry, replace(law, T=0.0, spot=583.44)).probability == 0
    assert loads(out.to_json()).probability == out.probability
    with pytest.raises(PnLHorizonError):
        probability_of_profit(p, replace(law, T=T + 0.1))
    with pytest.raises(PnLHorizonError):
        probability_of_profit(p, replace(law, convention="act_360"))
    for invalid in (
        dict(spot=0),
        dict(T=-1),
        dict(sigma=-1),
        dict(drift=np.inf),
        dict(measure="stress_only"),
    ):
        with pytest.raises(PnLError):
            replace(law, **invalid)


@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_existing_native_backend_model_parity(backend):
    pytest.importorskip(backend)
    s = snapshot()
    expected = evaluate(s)
    out = evaluate(s, backend=backend, chunk_rows=7)
    np.testing.assert_allclose(out.total, expected.total, rtol=1e-10, atol=1e-8)


def test_surface_sampler_point_identity_and_model_overflow_are_refused(monkeypatch):
    s = snapshot()

    class Wrong:
        def sample(self, points, *, n_samples, rng, market):
            other = SurfacePoints(k=points.k, T=points.T, point_id=np.arange(points.n) + 100)
            return SurfaceSamples(points=other, iv=np.full((n_samples, points.n), 0.2))

    v = vol(SurfaceVolSamples, distribution=Wrong())
    with pytest.raises(PnLError):
        pnl_at_horizon_samples(
            s,
            [100.0],
            horizons=(0.0,),
            volatility=v,
            market=replace(market(), volatility=None),
            model="black_scholes_merton",
            backend="numpy",
            convention="act_365",
        )
    import fast_vollib.instruments as instruments

    monkeypatch.setattr(
        instruments, "price_instrument", lambda batch, *args, **kw: np.full(len(batch), np.inf)
    )
    out = evaluate(s, horizons=(0.0,), diagnostic=True)
    assert not out.valid.any() and np.isnan(out.leg_values).all()
    assert "positive_unbounded" not in out.to_json()


def test_greeks_annual_units_match_independent_price_differences():
    s = snapshot()
    m = VanillaMarketInputs(underlying=100.0, rate=0.03, volatility=0.2, dividend_yield=0.01)
    greeks = position_greeks(
        s,
        m,
        model="black_scholes_merton",
        backend="numpy",
        convention="act_365",
        report_units="annual_unit",
    )

    def price(spot=100.0, sigma=0.2, rate=0.03, dt=0.0):
        total = 0.0
        for leg in s.position.legs:
            unit = EuropeanOption(
                underlier=leg.contract.underlier,
                option_type=leg.contract.option_type,
                strike=float(leg.contract.strike),
                maturity=maturity(s) + dt,
            )
            total += (
                float(leg.signed_size)
                * price_instrument(
                    unit,
                    VanillaMarketInputs(
                        underlying=spot, rate=rate, volatility=sigma, dividend_yield=0.01
                    ),
                    model="black_scholes_merton",
                    backend="numpy",
                )[0]
            )
        return total

    h = 1e-5
    expected = {
        "delta": (price(spot=100 + h) - price(spot=100 - h)) / (2 * h),
        "vega": (price(sigma=0.2 + h) - price(sigma=0.2 - h)) / (2 * h),
        "rho": (price(rate=0.03 + h) - price(rate=0.03 - h)) / (2 * h),
        "theta": -(price(dt=h) - price(dt=-h)) / (2 * h),
    }
    expected["gamma"] = (price(spot=100.01) - 2 * price() + price(spot=99.99)) / (0.01**2)
    for name, value in expected.items():
        assert greeks.net[name] == pytest.approx(value, rel=3e-5, abs=2e-4)


def test_proxy_caveats_and_surface_bands_survive_plot_and_file_export(tmp_path):
    plt = pytest.importorskip("matplotlib.pyplot", reason="optional viz extra is not installed")

    from fast_vollib.pnl import export_pnl, export_surface_band, plot_pnl, plot_pnl_band

    s = snapshot(american=True)
    args = dict(
        horizons=(0.0,),
        market=market(),
        model="black_scholes_merton",
        backend="numpy",
        convention="act_365",
    )
    proxy = european_proxy_profile(s, [90.0, 100.0, 110.0], **args)
    fig = plot_pnl([proxy])
    assert "European surrogate illustration" in " ".join(
        " ".join(t.get_text().split()) for t in fig.texts
    )
    paths = export_pnl(proxy, tmp_path / "proxy", figure=fig)
    assert isinstance(loads(paths[0].read_text()), EuropeanProxyProfile)
    assert "european_proxy" in paths[1].read_text()
    v = vol(SurfaceVolSamples, distribution=JointLaw())
    band = european_proxy_samples(
        s,
        [90.0, 100.0, 110.0],
        **(args | {"volatility": v, "market": replace(market(), volatility=None)}),
    )
    figure = plot_pnl_band(band)
    text = " ".join(" ".join(t.get_text().split()) for t in figure.texts)
    assert (
        "not joint spot/volatility P&L probabilities" in text
        and "European surrogate illustration" in text
    )
    saved = export_surface_band(band, tmp_path / "sensitivity", figure=figure)
    assert isinstance(loads(saved[0].read_text()), EuropeanProxySensitivityBand)
    for result in (proxy, band):
        wire = json.loads(result.to_json())
        wire["result_type"] = "profile" if result is proxy else "surface_sensitivity"
        with pytest.raises(PnLError):
            loads(json.dumps(wire))
    plt.close("all")


def test_cuda_horizon_route_executes_existing_cuda_kernel(monkeypatch):
    if os.getenv("FV_TEST_CUDA") != "1":
        pytest.skip("explicit CUDA lane is disabled")
    torch = pytest.importorskip("torch")
    from fast_vollib.backends import torch_backend

    assert torch.cuda.is_available()
    called = []
    original = torch_backend._device

    def device():
        actual = original()
        called.append(actual)
        assert actual.type == "cuda"
        return actual

    monkeypatch.setattr(torch_backend, "_device", device)
    out = evaluate(snapshot(), horizons=(0.0,), backend="torch")
    assert called and out.valid.all()
    np.testing.assert_allclose(
        out.total, evaluate(snapshot(), horizons=(0.0,)).total, rtol=1e-10, atol=1e-8
    )


def test_executable_horizon_example():
    from pathlib import Path
    import runpy

    plt = pytest.importorskip("matplotlib.pyplot", reason="optional viz extra is not installed")

    data = runpy.run_path(str(Path(__file__).resolve().parents[2] / "examples/horizon_pnl.py"))
    assert data["band"].samples.shape == (8, 3, 41, 2)
    plt.close("all")


def test_model_and_band_codecs_refuse_inconsistent_availability_and_horizons():
    profile = evaluate()
    for changes in (
        dict(horizons=(None, *profile.horizons[1:])),
        dict(horizons=profile.horizons[::-1]),
        dict(valid=np.zeros_like(profile.valid)),
        dict(leg_valid=np.zeros_like(profile.leg_valid)),
    ):
        with pytest.raises(PnLError):
            replace(profile, **changes)
    band = pnl_at_horizon_samples(
        snapshot(),
        [100.0],
        horizons=(0.0,),
        volatility=vol(SurfaceVolSamples, distribution=JointLaw()),
        market=HorizonMarket(rate=0.03, dividend_yield=0.01, volatility=None),
        model="black_scholes_merton",
        backend="numpy",
        convention="act_365",
    )
    with pytest.raises(PnLError):
        replace(band, valid=np.zeros_like(band.valid))
    unknown = replace(band, cumulative_cash=None, total=np.full_like(band.total, np.nan))
    assert unknown.to_dict()["band"]["total"][0][0][0] == {"status": "missing"}
    assert np.isnan(loads(unknown.to_json()).total).all()
    assert "knowledge_cutoff" in band.to_frame()
    with pytest.raises(PnLError):
        replace(unknown, total=np.full_like(band.total, np.inf))
    malformed = json.loads(profile.to_json())
    malformed["profile"]["horizons"] = [None] * len(profile.horizons)
    with pytest.raises(PnLError):
        loads(json.dumps(malformed))


def test_terminal_probability_refuses_overflow_and_inexact_interval_relabelling():
    s = snapshot()
    p = expiry_profile(
        s, [100.0], terminal_assumption="standard_intrinsic_equivalent", convention="act_365"
    )
    law = LognormalTerminal(
        spot=100.0,
        drift=0.03,
        sigma=0.2,
        T=maturity(s),
        measure="risk_neutral",
        convention="act_365",
    )
    for changes in (dict(sigma=1e308), dict(sigma=0.0, drift=1e308)):
        with pytest.raises(PnLError, match="numeric domain"):
            probability_of_profit(p, replace(law, **changes))
    result = probability_of_profit(p, law)
    for changes in (
        dict(probability=True),
        dict(profitable_intervals=((0.0, None),)),
        dict(meta=replace(result.meta, time_policy="act_360")),
    ):
        with pytest.raises(PnLError):
            replace(result, **changes)
