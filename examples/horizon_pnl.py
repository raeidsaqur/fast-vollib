"""European model curves, joint surface sensitivity and a specified terminal law."""

from datetime import date, datetime, time, timezone
from decimal import Decimal as D

import numpy as np

from fast_vollib.instruments import InstrumentRef, OptionType, SettlementType, VanillaMarketInputs
from fast_vollib.observations import ObservationStamp
from fast_vollib.pnl import (
    FlatVol,
    HorizonMarket,
    LognormalTerminal,
    ProportionalCarry,
    SurfaceVolSamples,
    expiry_profile,
    plot_pnl,
    plot_pnl_band,
    pnl_at_horizon,
    pnl_at_horizon_samples,
    position_greeks,
    probability_of_profit,
)
from fast_vollib.portfolio import (
    ListedContract,
    OptionLeg,
    Position,
    hypothetical_snapshot,
    resolve_expiry_instant,
)
from fast_vollib.surface import SurfaceMarket
from fast_vollib.surface.fitting import FlatIVSurface
from fast_vollib.surface.generative import GaussianFieldSurfaceDistribution

origin = datetime(2026, 1, 1, tzinfo=timezone.utc)
world = "hypothetical:european-horizon-example"
stamp = ObservationStamp(
    event_time=origin, available_time=origin, recorded_at=origin, source="synthetic", world_id=world
)
ref = InstrumentRef(identifier="ACME", currency="USD")


def contract(kind, strike):
    value = ListedContract(
        key=f"synthetic:ACME:2027-01-01:{kind.value}:{strike}",
        underlier=ref,
        option_type=kind,
        strike=D(strike),
        expiry_date=date(2027, 1, 1),
        exercise="european",
        settlement=SettlementType.CASH,
        multiplier=D(100),
        deliverable="standard_cash_equivalent:100",
        currency="USD",
        stamp=stamp,
    )
    return resolve_expiry_instant(value, policy=(time(0), "UTC"))


position = Position.from_legs(
    (
        OptionLeg(
            contract=contract(OptionType.PUT, "90"),
            quantity=1,
            allocated_entry=D(1),
            allocated_fees=D(".5"),
            leg_id="put",
        ),
        OptionLeg(
            contract=contract(OptionType.CALL, "110"),
            quantity=-1,
            allocated_entry=D("1.5"),
            allocated_fees=D(".5"),
            leg_id="call",
        ),
    )
)
snapshot = hypothetical_snapshot(
    position, valuation_time=origin, knowledge_cutoff=origin, world_id=world
)
spot = np.linspace(60.0, 140.0, 41)
market = HorizonMarket(rate=0.03, dividend_yield=0.01, volatility=FlatVol(0.2))
profile = pnl_at_horizon(
    snapshot,
    spot,
    horizons=(0.0, 0.5, 1.0),
    market=market,
    model="black_scholes_merton",
    backend="numpy",
    convention="act_365",
    chunk_rows=128,
)
figure = plot_pnl([profile])
forward = ProportionalCarry(
    origin_spot=100.0,
    base_forward=lambda h, t: 100 * np.exp(0.02 * np.asarray(t)),
    policy_id="flat_carry/1",
)
reference = SurfaceMarket.from_spot(spot=100.0, T=[0.5, 1.0], rate=0.03, carry=0.01)
volatility = SurfaceVolSamples(
    distribution=GaussianFieldSurfaceDistribution(base=FlatIVSurface(level=0.2), volatility=0.05),
    reference_market=reference,
    spot_movement="sticky_forward_moneyness",
    time_movement="constant_tenor",
    forward_mapping=forward,
    forward_mapping_id=forward.policy_id,
    n_samples=8,
    rng=17,
)
band = pnl_at_horizon_samples(
    snapshot,
    spot,
    horizons=(0.0, 0.5, 1.0),
    volatility=volatility,
    market=HorizonMarket(rate=0.03, dividend_yield=0.01, volatility=None),
    model="black_scholes_merton",
    backend="numpy",
    convention="act_365",
    chunk_rows=128,
)
band_figure = plot_pnl_band(band, horizon_index=1)
terminal = expiry_profile(
    snapshot, spot, terminal_assumption="standard_intrinsic_equivalent", convention="act_365"
)
probability = probability_of_profit(
    terminal,
    LognormalTerminal(
        spot=100.0, drift=0.05, sigma=0.2, T=1.0, measure="physical", convention="act_365"
    ),
)
greeks = position_greeks(
    snapshot,
    VanillaMarketInputs(underlying=100.0, rate=0.03, volatility=0.2, dividend_yield=0.01),
    model="black_scholes_merton",
    backend="numpy",
    convention="act_365",
    report_units="annual_unit",
)
assert profile.leg_values.shape == (3, 41, 2) and band.samples.shape == (8, 3, 41, 2)
assert 0 <= probability.probability <= 1 and greeks.units["theta"] == "per model year"
