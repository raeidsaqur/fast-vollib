from dataclasses import replace
from datetime import timedelta
from decimal import Decimal as D
import json

import numpy as np
import pytest

from fast_vollib.observations import select_as_of
from fast_vollib.pnl import (
    OptionQuote,
    PnLError,
    QuoteFlags,
    QuotePolicy,
    QuoteSnapshot,
    expiry_profile,
    export_pnl,
    mark_position,
    plot_pnl,
    quote_from_dict,
    quote_record,
    quote_to_dict,
    quotes_as_of,
)
from fast_vollib.pnl.serialization import loads

from ._fixtures import ASSUMPTION, NOW, WORLD, executed, fill, stamp
from .test_terminal import condor, snap


def policy(**kw):
    return QuotePolicy(
        **(
            dict(
                name="current_indications",
                mark_rule="midpoint",
                max_quote_age_s=60.0,
                max_underlier_age_s=60.0,
                permitted_delay_s=0,
                allow_zero_size=False,
            )
            | kw
        )
    )


def book(snapshot=None, **kw):
    snapshot = snapshot or snap(condor("b", "4"))
    mids = [D(".80"), D("1.40"), D("1.20"), D(".70")]
    q = {
        leg.contract.key: OptionQuote(
            contract_key=leg.contract.key,
            bid=m - D(".05"),
            ask=m + D(".05"),
            bid_size=10,
            ask_size=10,
            last=D("999"),
            last_trade_time=NOW - timedelta(days=1),
            flags=QuoteFlags.OK,
            stamp=stamp(world=snapshot.world_id),
        )
        for leg, m in zip(snapshot.position.legs, mids)
    }
    return QuoteSnapshot(
        **(
            dict(
                quotes=q,
                underlying=D(600),
                stamp=stamp(world=snapshot.world_id),
                valuation_time=snapshot.valuation_time,
                knowledge_cutoff=snapshot.knowledge_cutoff,
                delay_s=0,
            )
            | kw
        )
    )


def change_first(q, **kw):
    quotes = dict(q.quotes)
    key = next(iter(quotes))
    quotes[key] = replace(quotes[key], **kw)
    return replace(q, quotes=quotes)


def test_midpoint_46_after_costs_42_and_close_side_semantics():
    s = snap(condor("b", "4"))
    q = book(s)
    midpoint = mark_position(s, q, policy=policy())
    net = mark_position(s, q, policy=policy(), future_costs=D(4))
    close = mark_position(s, q, policy=policy(mark_rule="close_estimate"))
    assert midpoint.total[0, 0] == 46 and net.total[0, 0] == 42
    assert midpoint.cumulative_cash == 156 and midpoint.paid_fees == 4
    assert close.total[0, 0] == 26
    assert "not an executable multi-leg fill" in close.meta.caveat
    assert midpoint.realized == 0 and midpoint.unrealized == 46
    assert midpoint.metrics is None and midpoint.breakevens is None
    np.testing.assert_array_equal(midpoint.leg_values[0, 0], [80.0, -140.0, -120.0, 70.0])
    restored = loads(net.to_json())
    assert restored.total[0, 0] == 42 and restored.quote_details[0].mark == D(".80")
    assert restored.meta.quote_policy == net.meta.quote_policy
    assert restored.meta.basis == "quote_mark"


def test_missing_side_never_uses_last_or_zero_and_one_side_close_is_usable():
    s = snap(condor("b", "4"))
    q = change_first(book(s), ask=None, ask_size=None)
    with pytest.raises(PnLError):
        mark_position(s, q, policy=policy())
    diagnostic = mark_position(s, q, policy=policy(), diagnostic=True)
    assert not diagnostic.valid[0, 0] and np.isnan(diagnostic.total[0, 0])
    assert np.isnan(diagnostic.leg_values[0, 0, 0]) and diagnostic.leg_valid.sum() == 3
    assert diagnostic.quote_details[0].mark is None
    assert diagnostic.quote_details[0].flags & QuoteFlags.ONE_SIDED
    close = mark_position(s, q, policy=policy(mark_rule="close_estimate"))
    assert close.valid.all() and close.quote_details[0].mark == D(".75")
    missing = replace(q, quotes={})
    assert not mark_position(s, missing, policy=policy(), diagnostic=True).valid.any()
    with pytest.raises(PnLError):
        mark_position(s, missing, policy=policy())
    data = json.loads(diagnostic.to_json())
    assert data["profile"]["total"][0][0] == {"status": "missing"}


@pytest.mark.parametrize("flags", [QuoteFlags(n) for n in range(128)])
def test_every_source_flag_combination(flags):
    s = snap(condor("b", "4"))
    q = change_first(book(s), flags=flags)
    invalid = bool(
        flags
        & (
            QuoteFlags.MISSING
            | QuoteFlags.ONE_SIDED
            | QuoteFlags.CROSSED
            | QuoteFlags.STALE
            | QuoteFlags.HALTED
            | QuoteFlags.ZERO_SIZE
            | QuoteFlags.DELAYED
        )
    )
    profile = mark_position(s, q, policy=policy(), diagnostic=True)
    assert bool(profile.valid[0, 0]) == (not invalid)
    assert profile.quote_details[0].flags & flags == flags
    if invalid:
        with pytest.raises(PnLError):
            mark_position(s, q, policy=policy())


def test_sizes_delay_indications_and_unknown_time_are_explicit():
    s = snap(condor("b", "4"))
    q = change_first(book(s), bid_size=0, delay_s=15)
    with pytest.raises(PnLError):
        mark_position(s, q, policy=policy())
    p = mark_position(s, q, policy=policy(permitted_delay_s=15, allow_zero_size=True))
    assert p.valid.all() and p.quote_details[0].flags & QuoteFlags.ZERO_SIZE
    assert p.quote_details[0].flags & QuoteFlags.DELAYED and p.quote_details[0].delay_s == 15
    q = change_first(book(s), bid_size=None)
    with pytest.raises(PnLError):
        mark_position(s, q, policy=policy())
    assert mark_position(s, q, policy=policy(allow_unknown_size=True)).valid.all()
    q = change_first(
        book(s),
        quote_time_known=False,
        stamp=stamp(timing_note="source quote time unknown; event is retrieval"),
    )
    with pytest.raises(PnLError):
        mark_position(s, q, policy=policy())
    p = mark_position(s, q, policy=policy(allow_unknown_quote_time=True))
    assert p.quote_details[0].age_s is None and p.quote_details[0].quote_time is None
    with pytest.raises(PnLError):
        change_first(book(s), quote_time_known=False)
    with pytest.raises(PnLError):
        mark_position(s, replace(book(s), delay_s=None), policy=policy())
    assert mark_position(
        s, replace(book(s), delay_s=None), policy=policy(allow_unknown_delay=True)
    ).valid.all()


@pytest.mark.parametrize(
    "field,value",
    [
        ("bid", 1.2),
        ("ask", D("-1")),
        ("bid_size", True),
        ("ask_size", 1.5),
        ("delay_s", -1),
        ("flags", 128),
    ],
)
def test_quote_input_domains(field, value):
    with pytest.raises(ValueError):
        change_first(book(), **{field: value})


def test_quote_underlier_age_event_and_knowledge_cutoff_are_separate():
    s = snap(condor("b", "4"))
    q = book(s)
    for changed in (
        change_first(q, stamp=stamp(event=NOW - timedelta(seconds=61))),
        change_first(q, stamp=stamp(event=NOW + timedelta(seconds=1))),
        change_first(q, stamp=stamp(event=NOW, available=NOW + timedelta(seconds=1))),
        replace(q, stamp=stamp(event=NOW - timedelta(seconds=61))),
        replace(q, stamp=stamp(event=NOW + timedelta(seconds=1))),
        replace(q, stamp=stamp(event=NOW, available=NOW + timedelta(seconds=1))),
    ):
        with pytest.raises(PnLError):
            mark_position(s, changed, policy=policy())
        assert not mark_position(s, changed, policy=policy(), diagnostic=True).valid.any()
    for changed in (
        replace(q, valuation_time=NOW + timedelta(seconds=1)),
        replace(q, knowledge_cutoff=NOW + timedelta(seconds=1)),
    ):
        with pytest.raises(PnLError):
            mark_position(s, changed, policy=policy(), diagnostic=True)
    old = change_first(q, last_trade_time=NOW - timedelta(days=500))
    assert mark_position(s, old, policy=policy()).valid.all()


def test_quotes_as_of_revisions_cutoffs_and_payload_ownership():
    s = snap(condor("b", "4"))
    q = book(s)
    quote = next(iter(q.quotes.values()))
    wire = quote_to_dict(quote)
    assert quote_from_dict(wire) == quote
    record = quote_record(quote, record_id="original")
    correction = replace(
        quote,
        bid=D(".65"),
        stamp=stamp(available=NOW + timedelta(seconds=10), revision_of="original", version=1),
    )
    revised = quote_record(correction, record_id="corrected")
    early = select_as_of([record, revised], knowledge_cutoff=NOW, world_id=WORLD)
    later = select_as_of(
        [record, revised], knowledge_cutoff=NOW + timedelta(seconds=11), world_id=WORLD
    )
    a = quotes_as_of(early, valuation_time=NOW, underlying=D(600), stamp=q.stamp, delay_s=0)
    b = quotes_as_of(later, valuation_time=NOW, underlying=D(600), stamp=q.stamp, delay_s=0)
    assert a.quotes[quote.contract_key].bid == D(".75")
    assert b.quotes[quote.contract_key].bid == D(".65")
    assert a.digest != b.digest
    source = dict(q.quotes)
    saved = replace(q, quotes=source)
    before = saved.digest
    source.clear()
    assert saved.digest == before and len(saved.quotes) == 4
    with pytest.raises(ValueError):
        quote_from_dict(wire | {"unknown": 1})


def test_realized_and_unrealized_are_not_replaced_by_marks():
    s = executed([fill(2, "2", id="open", seconds=-2), fill(-1, "3", id="close", seconds=-1)])
    q = book(s)
    first = next(iter(q.quotes.values()))
    q = replace(q, quotes={first.contract_key: replace(first, bid=D("2.4"), ask=D("2.6"))})
    p = mark_position(s, q, policy=policy())
    assert (p.realized, p.unrealized, p.total[0, 0]) == (100, 50, 150)
    p2 = mark_position(
        s, change_first(q, bid=D("3.4"), ask=D("3.6")), policy=policy(), future_costs=D(5)
    )
    assert (p2.realized, p2.unrealized, p2.total[0, 0]) == (100, 150, 245)


def test_plot_and_export_retain_policy_status_and_fresh_tables(tmp_path):
    matplotlib = pytest.importorskip("matplotlib", reason="optional viz extra is not installed")

    matplotlib.use("Agg")
    s = snap(condor("b", "4"))
    q = book(s)
    mark = mark_position(s, q, policy=policy(mark_rule="close_estimate"))
    terminal = expiry_profile(s, np.linspace(560.0, 640.0, 81), terminal_assumption=ASSUMPTION)
    fig = plot_pnl([terminal], mark=mark, show_legs=True)
    assert any(
        "not an executable multi-leg fill" in " ".join(text.get_text().split())
        for text in fig.texts
    )
    paths = export_pnl(mark, tmp_path / "indication", figure=fig)
    assert len(paths) == 3 and all(p.exists() for p in paths)
    assert (
        "quote_mark" in paths[1].read_text()
        and "not an executable multi-leg fill" in paths[1].read_text()
    )
    frame = mark.to_frame()
    frame.loc[0, "total"] = 0
    assert mark.total[0, 0] == 26
    with pytest.raises(ValueError):
        mark.total.setflags(write=True)
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_closed_inventory_needs_no_live_quote_and_unknown_basis_stays_unavailable():
    s = executed([fill(1, "2", id="open", seconds=-2), fill(-1, "3", id="close", seconds=-1)])
    q = book(s, underlying=None, delay_s=None)
    p = mark_position(s, q, policy=policy())
    assert p.total[0, 0] == 100 and p.realized == 100 and p.unrealized == 0
    unknown = replace(
        executed([fill(1, "2", id="inventory", seconds=-1)]),
        origin="positions_only",
        completeness="positions_only",
        cumulative_cash=None,
        realized=None,
    )
    p = mark_position(unknown, book(unknown), policy=policy())
    assert p.leg_valid.all() and not p.valid.any() and p.cumulative_cash is None
    assert p.realized is None and p.unrealized is None


def test_executable_quote_example():
    from pathlib import Path
    import runpy

    plt = pytest.importorskip("matplotlib.pyplot", reason="optional viz extra is not installed")

    runpy.run_path(str(Path(__file__).resolve().parents[2] / "examples/quote_marks.py"))
    plt.close("all")


def test_codecs_reject_unknown_fields_and_inconsistent_mark_values():
    p = mark_position(snap(condor("b", "4")), book(), policy=policy())
    wire = json.loads(p.to_json())
    wire["profile"]["quote_details"][0]["mark"] = {"decimal": "999"}
    with pytest.raises(PnLError):
        loads(json.dumps(wire))
    legacy = expiry_profile(snap(condor()), [600.0], terminal_assumption=ASSUMPTION).to_dict()
    assert (
        "quote_details" not in legacy["profile"] and "quote_policy" not in legacy["profile"]["meta"]
    )
    legacy["profile"]["quote_details"] = []
    with pytest.raises(PnLError):
        loads(json.dumps(legacy))
