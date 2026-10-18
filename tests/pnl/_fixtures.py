from datetime import date, datetime, timezone
from decimal import Decimal as D

from fast_vollib.instruments import InstrumentRef, OptionType, SettlementType
from fast_vollib.observations import ObservationStamp, select_as_of
from fast_vollib.portfolio import Execution, FeeRecord, ListedContract, Trade, position_snapshot
from fast_vollib.portfolio.serialization import contract_record, execution_record

NOW = datetime(2026, 10, 17, 10, tzinfo=timezone.utc)
WORLD = "hypothetical:test"
ASSUMPTION = "hold_to_expiry_no_early_exercise"


def stamp(*, world=WORLD, event=NOW, available=None, **kwargs):
    return ObservationStamp(
        event_time=event,
        available_time=available or event,
        recorded_at=available or event,
        world_id=world,
        source="synthetic" if world != "actual" else "manual",
        **kwargs,
    )


def contract(kind="call", strike="100", *, world=WORLD, expiry=date(2026, 11, 20), key=None):
    return ListedContract(
        key=key or f"test:{kind}:{strike}:{expiry}",
        underlier=InstrumentRef(identifier="SPY", currency="USD"),
        option_type=OptionType(kind),
        strike=D(strike),
        expiry_date=expiry,
        exercise="american",
        settlement=SettlementType.PHYSICAL,
        multiplier=D(100),
        deliverable="100 SPY",
        currency="USD",
        stamp=stamp(world=world),
    )


def executed(fills, *, valuation_time=NOW, knowledge_cutoff=None, completeness="complete"):
    c = contract(world="actual")
    records = [contract_record(c, record_id="contract")]
    records += [execution_record(f, record_id=f.execution_id) for f in fills]
    obs = select_as_of(records, knowledge_cutoff=knowledge_cutoff or valuation_time)
    trade = Trade(
        label="explicit",
        currency="USD",
        executions=tuple(fills),
        contracts={c.key: c},
        completeness=completeness,
    )
    return position_snapshot(trade, observations=obs, valuation_time=valuation_time, policy="fifo")


def fill(n, p, *, id="fill", fee="0", seconds=0, sequence=None):
    from datetime import timedelta

    c = contract(world="actual")
    return Execution(
        contract_key=c.key,
        quantity=n,
        premium=D(p),
        execution_id=id,
        stamp=stamp(world="actual", event=NOW + timedelta(seconds=seconds)),
        source_sequence=sequence,
        fees=(FeeRecord(kind="commission", amount=D(fee), currency="USD"),),
    )
