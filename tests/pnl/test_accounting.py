from dataclasses import replace
from datetime import timedelta
from decimal import Decimal as D, getcontext
import json

import numpy as np
import pytest

from fast_vollib.observations import (
    ObservationConflictError,
    ObservationError,
    ObservationRecord,
    RevisionError,
    select_as_of,
)
from fast_vollib.observations.serialization import (
    dumps as obs_dumps,
    loads as obs_loads,
    snapshot_from_dict,
    snapshot_to_dict,
)
from fast_vollib.pnl import expiry_profile, prepare_terminal
from fast_vollib.portfolio import (
    FeeRecord,
    IncompleteBasisError,
    OptionLeg,
    PortfolioError,
    Position,
    Trade,
    hypothetical_snapshot,
    match_lots,
    position_snapshot,
)
from fast_vollib.portfolio.serialization import (
    contract_record,
    dumps,
    execution_record,
    loads,
    position_record,
)

from ._fixtures import ASSUMPTION, NOW, contract, executed, fill, stamp


def test_partial_close_and_closed_cash():
    fs = [fill(2, "2"), fill(-1, "3", id="close", seconds=1)]
    snap = executed(fs, valuation_time=NOW + timedelta(seconds=2))
    assert snap.cumulative_cash == D(-100)
    assert snap.accounting(D(250)) == (D(100), D(50), D(150))
    c = contract(world="actual")
    ledger = match_lots([fs[1], fs[0], fs[0]], contracts={c.key: c}, policy="fifo")
    assert ledger.cumulative_cash == snap.cumulative_cash
    closed = executed(
        [*fs, fill(-1, "2.5", id="last", seconds=2)], valuation_time=NOW + timedelta(seconds=3)
    )
    assert not closed.position.legs
    profile = expiry_profile(closed, np.array([0.0, 100.0, 1000.0]), terminal_assumption=ASSUMPTION)
    assert profile.target_date is None and profile.horizons == (None,)
    np.testing.assert_array_equal(profile.total, [[150.0, 150.0, 150.0]])
    assert profile.minimum_pnl == profile.maximum_pnl == 150


def test_crossing_zero_and_fee_conservation():
    snap = executed(
        [fill(2, "2", fee="3"), fill(-3, "3", fee="3", id="cross", seconds=1)],
        valuation_time=NOW + timedelta(seconds=2),
    )
    assert snap.cumulative_cash == D(494)
    assert snap.accounting(D(-300)) == (D(195), D(-1), D(194))
    assert snap.paid_fees == 6
    assert snap.lots[0].opening_fees == 1
    assert snap.position.legs[0].quantity == -1
    assert loads(dumps(snap)) == snap


def test_repeated_fractional_fee_splits_preserve_identity_and_context():
    before = getcontext().copy()
    snap = executed(
        [
            fill(3, "2", fee="1"),
            fill(-1, "3", fee="1", id="b", seconds=1),
            fill(-1, "4", fee="1", id="c", seconds=2),
        ],
        valuation_time=NOW + timedelta(seconds=3),
    )
    assert snap.paid_fees == 3
    realized, unrealized, total = snap.accounting(D(500))
    from decimal import localcontext

    from fast_vollib.portfolio._validate import LEDGER_CONTEXT

    with localcontext(LEDGER_CONTEXT):
        assert realized + unrealized == total == 597
    assert getcontext() == before or (getcontext().prec, getcontext().rounding) == (
        before.prec,
        before.rounding,
    )


def test_equal_timestamp_needs_explicit_order():
    fs = [fill(2, "2"), fill(-3, "3", id="b")]
    c = contract(world="actual")
    with pytest.raises(PortfolioError, match="equal execution"):
        match_lots(fs, contracts={c.key: c}, policy="fifo")
    a = match_lots(fs, contracts={c.key: c}, policy="fifo", tie_policy="execution_id")
    b = match_lots(reversed(fs), contracts={c.key: c}, policy="fifo", tie_policy="execution_id")
    assert a == b
    fs = [replace(fs[0], source_sequence=1), replace(fs[1], source_sequence=2)]
    assert match_lots(fs, contracts={c.key: c}, policy="fifo") == match_lots(
        reversed(fs), contracts={c.key: c}, policy="fifo"
    )


def test_three_cutoffs_and_later_correction():
    c = replace(
        contract(world="actual"), stamp=stamp(world="actual", event=NOW - timedelta(days=1))
    )
    initial = replace(
        fill(1, "2", fee="1"),
        stamp=stamp(
            world="actual", available=NOW + timedelta(seconds=2), source_record_id="fill-source"
        ),
    )
    correction = replace(
        initial,
        fees=(FeeRecord(kind="commission", amount=D(3), currency="USD"),),
        stamp=replace(
            initial.stamp,
            available_time=NOW + timedelta(hours=1),
            recorded_at=NOW + timedelta(hours=1),
            revision_of="original",
            version=1,
        ),
    )
    records = [
        execution_record(correction, record_id="corrected"),
        contract_record(c, record_id="contract"),
        execution_record(initial, record_id="original"),
    ]
    trade = Trade(label="one fill", currency="USD", contracts={c.key: c}, executions=(initial,))
    saved = []
    for seconds, cash in [(1, D(0)), (3, D(-201)), (3660, D(-203))]:
        obs = select_as_of(records + records, knowledge_cutoff=NOW + timedelta(seconds=seconds))
        snap = position_snapshot(
            trade, observations=obs, valuation_time=NOW + timedelta(seconds=3), policy="fifo"
        )
        assert snap.cumulative_cash == cash
        assert snap.knowledge_label == ("as_known" if seconds <= 3 else "latest_corrected")
        assert snapshot_from_dict(snapshot_to_dict(obs)).digest == obs.digest
        saved.append(snap)
    assert saved[1].cumulative_cash == -201
    assert saved[1].observations_digest != saved[2].observations_digest
    # Detached caller values cannot change selected record accounting.
    detached = replace(trade, executions=(replace(initial, premium=D(999)),))
    assert (
        position_snapshot(
            detached,
            observations=select_as_of(records, knowledge_cutoff=NOW + timedelta(seconds=3)),
            valuation_time=NOW + timedelta(seconds=3),
            policy="fifo",
        ).cumulative_cash
        == -201
    )


def test_known_future_execution_is_excluded_by_valuation_time():
    c = contract(world="actual")
    future = replace(
        fill(1, "2"), stamp=stamp(world="actual", event=NOW + timedelta(days=1), available=NOW)
    )
    obs = select_as_of(
        [contract_record(c, record_id="c"), execution_record(future, record_id="f")],
        knowledge_cutoff=NOW,
    )
    assert len(obs.selected_records) == 2  # generic selection keeps future notices
    trade = Trade(label="future", currency="USD", contracts={c.key: c}, executions=(future,))
    assert (
        position_snapshot(
            trade, observations=obs, valuation_time=NOW, policy="fifo"
        ).cumulative_cash
        == 0
    )


def record(*, id="r", st=None, payload=None):
    return ObservationRecord(
        record_id=id,
        kind="test",
        entities=("economic-id",),
        payload_schema="test/1",
        payload={"value": [D("1.25")]} if payload is None else payload,
        stamp=st or stamp(world="actual"),
    )


def test_payload_ownership_digest_and_codecs():
    tree = {"a": [{"b": D("2.5")}], "when": NOW}
    r = record(payload=tree)
    snapshot = select_as_of([r], knowledge_cutoff=NOW)
    original = snapshot.digest
    tree["a"][0]["b"] = D(999)
    tree["a"].append("later")
    assert r.payload["a"][0]["b"] == D("2.5") and snapshot.digest == original
    assert obs_loads(obs_dumps(r)) == r
    with pytest.raises(TypeError):
        r.payload["a"][0]["b"] = D(3)
    with pytest.raises(ObservationError):
        replace(r, payload_digest="bad")
    for bad in (object(), lambda: None, np.array([1]), {"$decimal": "spoof"}):
        with pytest.raises(ObservationError):
            record(payload=bad)
    for bad in (
        '{"schema":1,"schema":2}',
        obs_dumps(r).replace("observation.record/1", "observation.record/999"),
    ):
        with pytest.raises(ObservationError):
            obs_loads(bad)
    data = json.loads(obs_dumps(r))
    data["extra"] = 1
    with pytest.raises(ObservationError):
        obs_loads(json.dumps(data))


def test_dedup_conflicts_fallback_and_graph_refusals():
    a = record(st=stamp(world="actual", source_record_id="source"))
    assert len(select_as_of([a, a], knowledge_cutoff=NOW).selected_records) == 1
    with pytest.raises(ObservationConflictError):
        select_as_of(
            [a, replace(a, payload={"different": True}, payload_digest="")], knowledge_cutoff=NOW
        )
    x = record(id="x")
    y = record(id="y")
    assert len(select_as_of([x, y], knowledge_cutoff=NOW).selected_records) == 2
    b = record(id="b", st=replace(a.stamp, version=1, revision_of="r"))
    fork = record(id="fork", st=replace(a.stamp, version=2, revision_of="r"))
    for records in (
        [b],
        [a, b, fork],
        [a, replace(b, entities=("other",))],
        [replace(a, stamp=replace(a.stamp, revision_of="b")), b],
        [a, replace(b, stamp=replace(b.stamp, source="different"))],
    ):
        with pytest.raises(RevisionError):
            select_as_of(records, knowledge_cutoff=NOW)
    assert select_as_of([b, a], knowledge_cutoff=NOW).selected_records == (b,)


def test_timestamps_and_worlds_and_unknown_policies():
    with pytest.raises(ObservationError):
        stamp(world="actual", event=NOW.replace(tzinfo=None))
    with pytest.raises(ObservationError):
        replace(stamp(), world_id="actual")
    with pytest.raises(ObservationError):
        replace(stamp(), event_time=(NOW, NOW - timedelta(seconds=1)))
    # Archive availability does not assert the local ingestion time.
    assert replace(stamp(world="actual"), recorded_at=NOW - timedelta(days=1))
    with pytest.raises(ObservationError):
        select_as_of([], knowledge_cutoff=NOW, policy="guessed")
    p = Position(
        legs=(
            OptionLeg(
                contract=contract(),
                quantity=1,
                allocated_entry=D(2),
                allocated_fees=D(0),
                leg_id="a",
            ),
        )
    )
    with pytest.raises(PortfolioError):
        hypothetical_snapshot(p, valuation_time=NOW, knowledge_cutoff=NOW, world_id="actual")
    with pytest.raises(PortfolioError):
        Position(
            legs=(p.legs[0], replace(p.legs[0], contract=contract(world="actual"), leg_id="b"))
        )


@pytest.mark.parametrize("bad", [1.0, float("nan"), True, "nan", "01.0", D("Infinity")])
def test_money_rejects_binary_floats_and_noncanonical_text(bad):
    with pytest.raises(PortfolioError):
        replace(fill(1, "2"), premium=bad)


@pytest.mark.parametrize("bad", [True, 0, 1.0, -1.5])
def test_quantity_requires_signed_integer(bad):
    with pytest.raises(PortfolioError):
        replace(fill(1, "2"), quantity=bad)


def test_strict_portfolio_codecs_and_mapping_ownership():
    c = contract(world="actual")
    f = fill(1, "2")
    terms = {c.key: c}
    trade = Trade(label="trade", currency="USD", executions=(f,), contracts=terms)
    terms.clear()
    assert len(trade.contracts) == 1
    for obj in (c, f, trade):
        assert loads(dumps(obj)) == obj
        data = json.loads(dumps(obj))
        data["unknown"] = True
        with pytest.raises(PortfolioError):
            loads(json.dumps(data))
    data = json.loads(dumps(f))
    data["premium"] = 2.0
    with pytest.raises(PortfolioError):
        loads(json.dumps(data))
    incomplete = executed([f], completeness="partial")
    assert incomplete.cumulative_cash is None and incomplete.realized is None
    with pytest.raises(IncompleteBasisError):
        prepare_terminal(incomplete, terminal_assumption=ASSUMPTION)


def test_positions_only_evidence_keeps_basis_unavailable():
    c = contract(world="actual")
    p = Position(
        legs=(
            OptionLeg(
                contract=c, quantity=1, allocated_entry=None, allocated_fees=D(0), leg_id="held"
            ),
        )
    )
    obs = select_as_of(
        [position_record(p, record_id="inventory", stamp=c.stamp)], knowledge_cutoff=NOW
    )
    trade = Trade(
        label="inventory",
        currency="USD",
        executions=(),
        contracts={c.key: c},
        completeness="positions_only",
        inventory_record_id="inventory",
    )
    s = position_snapshot(trade, observations=obs, valuation_time=NOW, policy="fifo")
    assert s.origin == "positions_only" and s.cumulative_cash is None and s.open_basis is None
    with pytest.raises(IncompleteBasisError):
        expiry_profile(s, [100.0], terminal_assumption=ASSUMPTION)


def test_random_cash_and_closed_cycle_against_exact_fraction_oracle():
    from fractions import Fraction
    import random

    rng = random.Random(314159)
    for _ in range(30):
        inventory = 0
        fills = []
        cash = Fraction(0)
        for i in range(8):
            n = rng.choice((-3, -2, -1, 1, 2, 3))
            price = D(rng.randrange(1, 1000)) / 100
            fee = D(rng.randrange(0, 100)) / 100
            fills.append(fill(n, str(price), fee=str(fee), id=f"f{i}", seconds=i))
            cash -= n * 100 * Fraction(price) + Fraction(fee)
            inventory += n
        mark = D("3.125")
        snapshot = executed(fills, valuation_time=NOW + timedelta(seconds=20))
        assert Fraction(snapshot.cumulative_cash) == cash
        assert sum(leg.quantity for leg in snapshot.position.legs) == inventory
        realized, unrealized, total = snapshot.accounting(inventory * 100 * mark)
        assert Fraction(total) == cash + inventory * 100 * Fraction(mark)
        # Liquidating the entire residual position at the mark realizes that total.
        if inventory:
            fills.append(fill(-inventory, str(mark), id="liquidate", seconds=10))
        closed = executed(fills, valuation_time=NOW + timedelta(seconds=20))
        assert not closed.position.legs
        assert closed.realized == closed.cumulative_cash == total


def test_snapshot_tampering_and_delta_revisions_are_refused():
    r = record()
    s = select_as_of([r], knowledge_cutoff=NOW)
    with pytest.raises(ObservationError):
        replace(s, digest="changed")
    with pytest.raises(ObservationError):
        replace(s, selected_records=())
    revision = replace(
        r,
        record_id="next",
        payload_schema="delta/1",
        stamp=replace(r.stamp, version=1, revision_of=r.record_id),
    )
    with pytest.raises(RevisionError):
        select_as_of([r, revision], knowledge_cutoff=NOW)
    snapshot = executed([fill(1, "2")])
    data = json.loads(dumps(snapshot))
    data["realized"] = "99"
    with pytest.raises(PortfolioError):
        loads(json.dumps(data))


def test_financial_identifiers_reject_mutable_values():
    from fast_vollib.portfolio import OpenLot

    with pytest.raises(ObservationError):
        replace(contract(), source_symbol_id=["mutable"])
    with pytest.raises(ObservationError):
        replace(fill(1, "2"), order_id={"mutable": True})
    with pytest.raises(ObservationError):
        OpenLot(
            contract_key=["mutable"],
            execution_id="x",
            quantity=1,
            premium=D(2),
            opening_fees=D(0),
            basis=D(200),
        )
    with pytest.raises(ObservationError):
        replace(executed([fill(1, "2")]), accounting_policy=["mutable"])
