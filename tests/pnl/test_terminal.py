from dataclasses import replace
from datetime import date, time, timedelta
from decimal import Decimal as D, localcontext
from fractions import Fraction as F
import os
from pathlib import Path
import runpy

import numpy as np
import pytest

from fast_vollib.instruments import OptionType, SettlementType
from fast_vollib.pnl import (
    PnLAssumptionError,
    PnLError,
    PnLHorizonError,
    expiry_profile,
    plot_pnl,
    prepare_terminal,
    terminal_pnl,
)
from fast_vollib.pnl.serialization import dumps, loads
from fast_vollib.portfolio import (
    ButterflyPrices,
    CalendarPrices,
    DiagonalPrices,
    IronButterflyPrices,
    IronCondorPrices,
    OptionLeg,
    PortfolioError,
    Position,
    StraddlePrices,
    StranglePrices,
    UnsupportedContractError,
    VerticalPrices,
    butterfly,
    calendar_spread,
    diagonal_spread,
    hypothetical_snapshot,
    iron_butterfly,
    iron_condor,
    resolve_expiry_instant,
    straddle,
    strangle,
    vertical_spread,
    year_fraction,
)

from ._fixtures import ASSUMPTION, NOW, WORLD, contract, executed, fill


def condor(which="a", fees="0"):
    strikes, prices = (
        (("570", "580", "620", "630"), ("1.20", "2.10", "2.30", "1.30"))
        if which == "a"
        else (("580", "585", "615", "620"), ("1.20", "2.00", "1.80", "1.00"))
    )
    return iron_condor(
        put_long=contract("put", strikes[0]),
        put_short=contract("put", strikes[1]),
        call_short=contract("call", strikes[2]),
        call_long=contract("call", strikes[3]),
        quantity=1,
        prices=IronCondorPrices(
            **dict(zip(("put_long", "put_short", "call_short", "call_long"), map(D, prices)))
        ),
        fees=D(fees),
    )


def snap(p):
    return hypothetical_snapshot(p, valuation_time=NOW, knowledge_cutoff=NOW, world_id=WORLD)


def oracle(snapshot, s, future="0"):
    with localcontext() as ctx:
        ctx.prec = 100
        out = snapshot.cumulative_cash - D(future)
        for leg in snapshot.position.legs:
            diff = (
                s - leg.contract.strike
                if leg.contract.option_type is OptionType.CALL
                else leg.contract.strike - s
            )
            out += leg.quantity * leg.contract.multiplier * max(diff, D(0))
        return out


def test_condor_and_butterfly_fixtures():
    a = expiry_profile(snap(condor()), [575.0, 600.0], terminal_assumption=ASSUMPTION)
    assert a.cumulative_cash == 190 and a.maximum_pnl == 190 and a.minimum_pnl == -810
    assert a.breakevens == (F("578.10"), F("621.90"))
    assert a.metrics.maximum_regions == ((F(580), F(620)),)
    assert a.total[0, 0] == -310
    b = expiry_profile(snap(condor("b", "4")), [0.0, 600.0, 1000.0], terminal_assumption=ASSUMPTION)
    assert b.cumulative_cash == 156 and b.minimum_pnl == -344
    assert b.breakevens == (F("583.44"), F("616.56"))
    fly = butterfly(
        lower=contract("call", "590"),
        middle=contract("call", "600"),
        upper=contract("call", "610"),
        quantity=1,
        prices=ButterflyPrices(lower=D(12), middle=D(6), upper=D(2)),
        fees=D(4),
    )
    c = expiry_profile(snap(fly), [0.0, 600.0, 1000.0], terminal_assumption=ASSUMPTION)
    assert c.cumulative_cash == -204 and c.minimum_pnl == -204 and c.maximum_pnl == 796
    assert c.metrics.maximum_regions == ((F(600), F(600)),)
    assert c.breakevens == (F("592.04"), F("607.96"))


def test_grid_independence_unequal_wings_and_oracle():
    p = condor()
    p = Position(legs=(*p.legs[:3], replace(p.legs[3], contract=contract("call", "640"))))
    s = snap(p)
    profiles = [
        expiry_profile(s, grid, terminal_assumption=ASSUMPTION)
        for grid in ([600.0], [0.0, 500.0, 575.0, 600.0, 621.9, 700.0, 1e7])
    ]
    assert profiles[0].metrics == profiles[1].metrics
    assert profiles[0].minimum_pnl == -1810
    for profile in profiles:
        truth = np.array([float(oracle(s, D(str(x)))) for x in profile.spot])
        np.testing.assert_allclose(profile.total[0], truth, rtol=0, atol=1e-8)


def test_unbounded_tail_zero_interval_and_nonterminating_root():
    short = Position(
        legs=(
            OptionLeg(
                contract=contract(),
                quantity=-1,
                allocated_entry=D(2),
                allocated_fees=D(0),
                leg_id="short",
            ),
        )
    )
    r = expiry_profile(snap(short), [99.0], terminal_assumption=ASSUMPTION)
    assert r.minimum_pnl == D("-Infinity") and r.maximum_pnl == 200 and r.breakevens == (F(102),)
    zero = replace(short.legs[0], quantity=1, allocated_entry=D(0))
    r = expiry_profile(snap(Position(legs=(zero,))), [101.0], terminal_assumption=ASSUMPTION)
    assert r.metrics.zero_intervals == ((F(0), F(100)),) and not r.breakevens
    exact = replace(zero, quantity=3, allocated_entry=D("0"), allocated_fees=D(1))
    r = expiry_profile(snap(Position(legs=(exact,))), [0.0], terminal_assumption=ASSUMPTION)
    assert r.breakevens == (F(30001, 300),)
    assert loads(dumps(r)).breakevens == r.breakevens
    closed = executed(
        [fill(1, "2"), fill(-1, "2", id="close", seconds=1)],
        valuation_time=NOW + timedelta(seconds=2),
    )
    z = expiry_profile(closed, [100.0], terminal_assumption=ASSUMPTION)
    assert z.metrics.zero_intervals == ((F(0), None),) and z.minimum_pnl == z.maximum_pnl == 0


def test_cost_ownership_and_report_ownership():
    s = snap(condor("b", "4"))
    grid = np.array([580.0, 600.0, 620.0])
    a = expiry_profile(s, grid, terminal_assumption=ASSUMPTION)
    b = expiry_profile(s, grid, terminal_assumption=ASSUMPTION, future_costs=D(4))
    np.testing.assert_array_equal(a.total - b.total, np.full((1, 3), 4.0))
    assert a.paid_fees == b.paid_fees == 4 and b.maximum_pnl == 152
    assert np.allclose(b.total, float(b.cumulative_cash) + b.leg_values.sum(-1) - 4)
    grid[:] = 1
    assert a.spot[0] == 580.0
    for value in (a.spot, a.total, a.leg_values, a.valid, a.conversion_bound):
        with pytest.raises(ValueError):
            value.setflags(write=True)
    f = a.to_frame()
    f.loc[0, "total"] = 999
    assert a.total[0, 0] != 999
    assert loads(dumps(a)).meta == a.meta
    raw = a.to_dict()
    raw["profile"]["unknown"] = 1
    from fast_vollib.pnl.serialization import profile_from_dict

    with pytest.raises(PnLError):
        profile_from_dict(raw)


def test_assumptions_and_unsupported_terms():
    s = snap(condor())
    with pytest.raises(PnLAssumptionError):
        prepare_terminal(s, terminal_assumption="standard_intrinsic_equivalent")
    with pytest.raises(PnLAssumptionError):
        prepare_terminal(s, terminal_assumption="guess")
    for terms in ({"deliverable": "adjusted shares and cash"}, {"multiplier": D(10)}):
        p = Position(
            legs=(
                replace(s.position.legs[0], contract=replace(s.position.legs[0].contract, **terms)),
            )
        )
        with pytest.raises(UnsupportedContractError):
            prepare_terminal(snap(p), terminal_assumption=ASSUMPTION)
    cash = replace(
        s.position.legs[0],
        contract=replace(
            s.position.legs[0].contract,
            exercise="european",
            settlement=SettlementType.CASH,
            deliverable="standard_cash_equivalent:100",
        ),
    )
    assert (
        prepare_terminal(
            snap(Position(legs=(cash,))), terminal_assumption="standard_intrinsic_equivalent"
        ).caveat
        is None
    )


def test_all_strategy_builders_and_mixed_expiry_refusal():
    p = contract("put", "90")
    c = contract("call", "110")
    st = strangle(put=p, call=c, quantity=2, prices=StranglePrices(put=D(1), call=D(1)), fees=D(1))
    assert sum(x.allocated_fees for x in st.legs) == 1
    straddle(
        put=contract("put"),
        call=contract(),
        quantity=1,
        prices=StraddlePrices(put=D(1), call=D(1)),
        fees=D(0),
    )
    iron_butterfly(
        put_long=p,
        put_short=contract("put"),
        call_short=contract(),
        call_long=c,
        quantity=1,
        prices=IronButterflyPrices(put_long=D(1), put_short=D(2), call_short=D(2), call_long=D(1)),
        fees=D(0),
    )
    vertical_spread(
        long=contract("call", "90"),
        short=c,
        quantity=1,
        prices=VerticalPrices(long=D(2), short=D(1)),
        fees=D(0),
    )
    later = replace(c, key="later", expiry_date=date(2026, 12, 18))
    calendar = calendar_spread(
        long=later, short=c, quantity=1, prices=CalendarPrices(long=D(2), short=D(1)), fees=D(0)
    )
    assert not calendar.is_single_expiry
    diagonal_spread(
        long=replace(later, strike=D(120)),
        short=c,
        quantity=1,
        prices=DiagonalPrices(long=D(2), short=D(1)),
        fees=D(0),
    )
    with pytest.raises(PnLHorizonError):
        expiry_profile(snap(calendar), [100.0], terminal_assumption=ASSUMPTION)
    with pytest.raises(PortfolioError):
        strangle(
            put=contract("put", "120"),
            call=c,
            quantity=1,
            prices=StranglePrices(put=D(1), call=D(1)),
            fees=D(0),
        )
    with pytest.raises(PortfolioError):
        vertical_spread(
            long=c, short=c, quantity=1, prices=VerticalPrices(long=D(1), short=D(1)), fees=D(0)
        )


def test_explicit_daycount_resolution_and_dst():
    c = contract()
    assert c.expiry_instant is None
    resolved = resolve_expiry_instant(c, policy=(time(16), "America/New_York"))
    assert (
        c.expiry_instant is None and resolved.expiry_instant.hour == 21 and resolved.expiry_policy
    )
    p = Position(
        legs=(
            OptionLeg(
                contract=resolved, quantity=1, allocated_entry=D(2), allocated_fees=D(0), leg_id="x"
            ),
        )
    )
    r = expiry_profile(snap(p), [100.0], terminal_assumption=ASSUMPTION, convention="act_365")
    assert r.horizons == (year_fraction(NOW, resolved.expiry_instant, convention="act_365"),)
    assert year_fraction(NOW, NOW + timedelta(hours=12), convention="act_360") == 0.5 / 360
    assert year_fraction(NOW.date(), NOW.date() + timedelta(days=365), convention="act_365") == 1
    for day, clock in ((date(2026, 3, 8), time(2, 30)), (date(2026, 11, 1), time(1, 30))):
        with pytest.raises(PortfolioError):
            resolve_expiry_instant(replace(c, expiry_date=day), policy=(clock, "America/New_York"))
    with pytest.raises(PnLHorizonError):
        prepare_terminal(
            replace(snap(p), valuation_time=resolved.expiry_instant + timedelta(seconds=1)),
            terminal_assumption=ASSUMPTION,
        )
    with pytest.raises(PortfolioError):
        year_fraction(NOW, NOW, convention="trading_days")


def test_scenario_uses_actual_maturity_and_existing_guards():
    from fast_vollib.simulation import Scenario

    c = resolve_expiry_instant(contract(), policy=(time(16), "America/New_York"))
    p = Position(
        legs=(
            OptionLeg(
                contract=c, quantity=1, allocated_entry=D(2), allocated_fees=D(0), leg_id="x"
            ),
        )
    )
    s = snap(p)
    prep = prepare_terminal(s, terminal_assumption=ASSUMPTION, convention="act_365")
    horizon = year_fraction(NOW, c.expiry_instant, convention="act_365")
    states = np.array([[[100.0], [110.0]], [[100.0], [90.0]]])
    sc = Scenario.from_states(c.underlier, time_grid=[0.0, horizon], states=states)
    np.testing.assert_allclose(terminal_pnl(prep, sc), terminal_pnl(prep, states[:, -1, 0]))
    from fast_vollib.simulation import ScenarioMismatchError

    with pytest.raises(ScenarioMismatchError):
        terminal_pnl(prep, Scenario.from_states("OTHER", time_grid=[0.0, horizon], states=states))
    with pytest.raises(ScenarioMismatchError):
        terminal_pnl(
            prep, Scenario.from_states(c.underlier, time_grid=[0.0, horizon / 2], states=states)
        )
    with pytest.raises(PnLHorizonError):
        terminal_pnl(prepare_terminal(s, terminal_assumption=ASSUMPTION), sc)
    unresolved = replace(p, legs=(replace(p.legs[0], contract=contract()),))
    with pytest.raises(PnLHorizonError):
        terminal_pnl(
            prepare_terminal(
                snap(unresolved), terminal_assumption=ASSUMPTION, convention="act_365"
            ),
            sc,
        )


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_numpy_noncontiguous_shape_domain_and_conversion(dtype):
    s = snap(condor())
    prep = prepare_terminal(s, terminal_assumption=ASSUMPTION, dtype=dtype)
    x = np.arange(540, 661, dtype=dtype).reshape(11, 11).T
    assert not x.flags.c_contiguous
    out = terminal_pnl(prep, x)
    assert out.shape == x.shape and out.dtype == x.dtype
    truth = np.array([float(oracle(s, D(str(v)))) for v in x.flat]).reshape(x.shape)
    np.testing.assert_allclose(out, truth, atol=1e-4 if dtype == "float32" else 1e-10, rtol=0)
    assert terminal_pnl(prep, np.array(575.0, dtype=dtype)).shape == ()
    for bad in (
        np.array([1], dtype=int),
        np.array([1.0], dtype="float64" if dtype == "float32" else "float32"),
        np.array([-1.0], dtype=dtype),
        np.array([np.inf], dtype=dtype),
    ):
        with pytest.raises(PnLError):
            terminal_pnl(prep, bad)
    # Nonrepresentable Decimal coefficients have explicit pointwise conversion bounds.
    leg = OptionLeg(
        contract=replace(contract(), strike=D("100.123456789123456789")),
        quantity=3,
        allocated_entry=D("1.23456789123456789"),
        allocated_fees=D(".123456789123456789"),
        leg_id="x",
    )
    q = prepare_terminal(snap(Position(legs=(leg,))), terminal_assumption=ASSUMPTION, dtype=dtype)
    x = np.array([0.0, 100.0, 101.0, 1000.0], dtype=dtype)
    exact_coeff = np.array([float(oracle(q.snapshot, D(float(v)))) for v in x])
    error = np.abs(terminal_pnl(q, x) - exact_coeff)
    bound = q.conversion_error_bound(x)
    assert np.all(error <= bound + 32 * np.finfo(dtype).eps * (abs(exact_coeff) + 1))
    assert any(e.absolute_error > 0 for e in q.conversion_errors)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("device", ["cpu", "cuda"] if os.getenv("FV_TEST_CUDA") == "1" else ["cpu"])
def test_torch_native_gradient_device(dtype, device):
    torch = pytest.importorskip("torch")
    assert device != "cuda" or torch.cuda.is_available()
    prep = prepare_terminal(snap(condor()), terminal_assumption=ASSUMPTION, dtype=dtype)
    x = (
        torch.tensor([[575.0, 600.0], [625.0, 640.0]], dtype=getattr(torch, dtype), device=device)
        .T.detach()
        .requires_grad_(True)
    )
    assert not x.is_contiguous()
    y = terminal_pnl(prep, x)
    assert y.dtype == x.dtype and y.device == x.device and y.shape == x.shape
    y.sum().backward()
    np.testing.assert_allclose(x.grad.cpu(), [[100.0, -100.0], [0.0, 0.0]])
    np.testing.assert_allclose(y.detach().cpu(), [[-310.0, -310.0], [190.0, -810.0]])
    closed = executed(
        [fill(1, "2"), fill(-1, "3", id="close", seconds=1)],
        valuation_time=NOW + timedelta(seconds=2),
    )
    x.grad = None
    terminal_pnl(
        prepare_terminal(closed, terminal_assumption=ASSUMPTION, dtype=dtype), x
    ).sum().backward()
    assert torch.equal(x.grad, torch.zeros_like(x))


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_jax_jit_gradient_and_invalid_dynamic_values(dtype):
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    jax.config.update("jax_enable_x64", True)
    prep = prepare_terminal(snap(condor()), terminal_assumption=ASSUMPTION, dtype=dtype)
    x = jnp.array([575.0, 600.0, 625.0, 640.0], dtype=getattr(jnp, dtype))
    fn = jax.jit(lambda z: terminal_pnl(prep, z))
    y = fn(x)
    assert y.dtype == x.dtype and y.shape == x.shape
    np.testing.assert_allclose(y, [-310.0, 190.0, -310.0, -810.0])
    np.testing.assert_allclose(jax.grad(lambda z: jnp.sum(fn(z)))(x), [100.0, 0.0, -100.0, 0.0])
    bad = fn(jnp.array([-1.0, jnp.nan, jnp.inf, 600.0], dtype=getattr(jnp, dtype)))
    assert np.isnan(np.asarray(bad)[:3]).all() and bad[3] == 190
    closed = executed(
        [fill(1, "2"), fill(-1, "3", id="close", seconds=1)],
        valuation_time=NOW + timedelta(seconds=2),
    )
    q = prepare_terminal(closed, terminal_assumption=ASSUMPTION, dtype=dtype)
    np.testing.assert_array_equal(jax.grad(lambda z: jnp.sum(terminal_pnl(q, z)))(x), np.zeros(4))


def test_plot_and_executable_example():
    matplotlib = pytest.importorskip("matplotlib", reason="optional viz extra is not installed")

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    r = expiry_profile(
        snap(condor()), np.linspace(540.0, 660.0, 481), terminal_assumption=ASSUMPTION
    )
    fig = plot_pnl([r], show_legs=True)
    assert r.meta.caveat in " ".join(t.get_text().replace("\n", " ") for t in fig.texts)
    plt.close(fig)
    result = runpy.run_path(
        str(Path(__file__).resolve().parents[2] / "examples" / "terminal_pnl.py")
    )
    assert result["profile"].maximum_pnl == 190
    plt.close(result["fig"])


def test_metadata_is_owned_and_unbounded_codec_is_explicit():
    p = Position(
        legs=(
            OptionLeg(
                contract=contract(),
                quantity=-1,
                allocated_entry=D(2),
                allocated_fees=D(0),
                leg_id="short",
            ),
        )
    )
    r = expiry_profile(snap(p), [100.0], terminal_assumption=ASSUMPTION)
    encoded = dumps(r)
    assert "negative_unbounded" in encoded and "missing" in encoded
    assert "NaN" not in encoded and "Infinity" not in encoded
    restored = loads(encoded)
    assert restored.minimum_pnl == D("-Infinity") and restored.meta.caveat == r.meta.caveat
    assert restored.contracts[0].exercise == "american"
    assert restored.contracts[0].settlement is SettlementType.PHYSICAL
    with pytest.raises(PnLError):
        replace(r.meta, caveat=["mutable"])
    with pytest.raises(PnLError):
        replace(r, conversion_errors=({"mutable": []},))
