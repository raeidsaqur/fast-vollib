"""Offline listed-option terminal curve with explicit delivery assumptions."""

from datetime import date, datetime, timezone
from decimal import Decimal as D

import numpy as np

from fast_vollib.instruments import InstrumentRef, OptionType, SettlementType
from fast_vollib.observations import ObservationStamp
from fast_vollib.pnl import expiry_profile, plot_pnl
from fast_vollib.portfolio import (
    IronCondorPrices,
    ListedContract,
    hypothetical_snapshot,
    iron_condor,
)

as_of = datetime(2026, 10, 17, 20, 0, tzinfo=timezone.utc)
world = "hypothetical:condor-example"
stamp = ObservationStamp(
    event_time=as_of, available_time=as_of, recorded_at=as_of, source="synthetic", world_id=world
)


def contract(kind, strike):
    return ListedContract(
        key=f"synthetic:SPY:2026-11-20:{kind.value}:{strike}",
        underlier=InstrumentRef(identifier="SPY", currency="USD"),
        option_type=kind,
        strike=D(strike),
        expiry_date=date(2026, 11, 20),
        exercise="american",
        settlement=SettlementType.PHYSICAL,
        multiplier=D("100"),
        deliverable="100 SPY",
        currency="USD",
        stamp=stamp,
    )


pos = iron_condor(
    put_long=contract(OptionType.PUT, "570"),
    put_short=contract(OptionType.PUT, "580"),
    call_short=contract(OptionType.CALL, "620"),
    call_long=contract(OptionType.CALL, "630"),
    quantity=1,
    fees=D("0"),
    prices=IronCondorPrices(
        put_long=D("1.20"), put_short=D("2.10"), call_short=D("2.30"), call_long=D("1.30")
    ),
)
snap = hypothetical_snapshot(pos, valuation_time=as_of, knowledge_cutoff=as_of, world_id=world)
profile = expiry_profile(
    snap, np.linspace(540.0, 660.0, 481), terminal_assumption="hold_to_expiry_no_early_exercise"
)
fig = plot_pnl([profile])
assert profile.meta.origin == "hypothetical"
assert profile.meta.caveat  # visible American/physical-delivery assumption
