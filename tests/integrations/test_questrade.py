from dataclasses import replace
from datetime import datetime, timezone
from decimal import Decimal as D
import json

import pytest

from fast_vollib.instruments import InstrumentRef, SettlementType
from fast_vollib.integrations.questrade import (
    ContractTerms,
    NormalizationContext,
    activity_record,
    broker_position_record,
    get_mapping,
    load_saved_json,
    normalize_activities,
    normalize_executions,
    normalize_option_quotes,
    normalize_positions,
    normalize_symbols,
    position_evidence,
    reconcile_executions_activities,
)
from fast_vollib.observations import select_as_of
from fast_vollib.observations.serialization import dumps as record_json, loads as record_loads
from fast_vollib.pnl import quote_from_dict, quote_to_dict
from fast_vollib.portfolio.contracts import require_standard
from fast_vollib.portfolio.serialization import (
    contract_from_dict,
    contract_to_dict,
    execution_from_dict,
    execution_to_dict,
    position_from_dict,
)

NOW = datetime(2026, 1, 1, 13, tzinfo=timezone.utc)
VERSION = "questrade.saved/1:examples-percent"
CONTEXT = NormalizationContext(
    namespace="00000000-0000-4000-8000-000000000001",
    retrieved_at=NOW,
    available_time=NOW,
    availability_note="availability assumed retrieval for this saved response",
    world_id="hypothetical:offline",
)


def payloads():
    return dict(
        symbols={
            "symbols": [
                dict(
                    symbolId=1,
                    symbol="ACME20Nov26C100.00",
                    optionType="Call",
                    optionExerciseType="American",
                    optionExpiryDate="2026-11-20T00:00:00-05:00",
                    optionStrikePrice="100",
                    currency="USD",
                    optionContractDeliverables={
                        "underlyings": [
                            dict(multiplier=100, underlyingSymbol="ACME", underlyingSymbolId=2)
                        ],
                        "cashInLieu": 0,
                    },
                )
            ]
        },
        executions={
            "executions": [
                dict(
                    id=5,
                    symbolId=1,
                    quantity=1,
                    side="BTO",
                    price="2.00",
                    timestamp="2026-01-01T12:00:00Z",
                    orderId=10,
                    totalCost="2.00",
                    commission="1.00",
                    orderPlacementCommission=0,
                    executionFee=0,
                    secFee=0,
                    canadianExecutionFee=0,
                    notes="PRIVATE_SENTINEL",
                )
            ]
        },
        quotes={
            "optionQuotes": [
                dict(
                    symbolId=1,
                    bidPrice="2.4",
                    askPrice="2.6",
                    bidSize=10,
                    askSize=10,
                    lastTradePrice="2.5",
                    lastTradeTime="2026-01-01T11:00:00Z",
                    delay=0,
                    isHalted=False,
                    volatility="25",
                    delta="0.5",
                )
            ]
        },
        positions={
            "positions": [
                dict(
                    symbolId=1,
                    openQuantity=1,
                    currentMarketValue="250",
                    currentPrice="2.5",
                    averageEntryPrice="2",
                    closedPnl=0,
                    openPnl="49",
                    totalCost="201",
                    isRealTime=True,
                    isUnderReorg=False,
                )
            ]
        },
        activities={
            "activities": [
                dict(
                    tradeDate="2026-01-01T00:00:00-05:00",
                    transactionDate="2026-01-01T00:00:00-05:00",
                    settlementDate="2026-01-02T00:00:00-05:00",
                    action="Buy",
                    symbolId=1,
                    currency="USD",
                    quantity=1,
                    price="2.00",
                    grossAmount="-200",
                    commission="-1",
                    netAmount="-201",
                    type="Trades",
                    description="PRIVATE_SENTINEL",
                )
            ]
        },
    )


def contracts(data=None, context=CONTEXT):
    return normalize_symbols(
        (data or payloads())["symbols"],
        terms={
            "1": ContractTerms(
                contract_key="synthetic:ACME:call:100",
                underlier=InstrumentRef(identifier="ACME", currency="USD"),
                premium_multiplier=D(100),
                settlement=SettlementType.PHYSICAL,
                assumption_note="caller declares standard premium multiplier and share settlement",
            )
        },
        context=context,
        mapping_version=VERSION,
    )


def execution(data=None, fee_policy="separate_components"):
    return normalize_executions(
        (data or payloads())["executions"],
        contracts=contracts(),
        context=CONTEXT,
        mapping_version=VERSION,
        fee_policy=fee_policy,
    )


def activities(data=None, commission_policy="negative_deduction"):
    return normalize_activities(
        (data or payloads())["activities"],
        contracts=contracts(),
        context=CONTEXT,
        mapping_version=VERSION,
        activity_ids=["00000000-0000-4000-8000-000000000002"],
        commission_policy=commission_policy,
    )


def test_lossless_json_and_mapping_variants():
    value = load_saved_json(
        '{"amount": 0.10000000000000000000000000001, "id": 18446744073709551615}'
    )
    assert value["amount"] == D("0.10000000000000000000000000001") and value["id"] == 2**64 - 1
    for invalid in ('{"a":1,"a":2}', '{"a":NaN}'):
        with pytest.raises(ValueError):
            load_saved_json(invalid)
    with pytest.raises(ValueError):
        get_mapping("unknown")
    data = payloads()
    data["executions"]["executions"][0]["price"] = 2.0
    with pytest.raises(ValueError):
        execution(data)
    with pytest.raises(ValueError):
        replace(CONTEXT, namespace="12345678")
    with pytest.raises(ValueError):
        replace(CONTEXT, available_time=datetime(2026, 1, 1))


def test_contract_execution_quote_and_observation_roundtrips():
    c = contracts()[0]
    e = execution()[0]
    q = normalize_option_quotes(
        payloads()["quotes"], contracts=(c,), context=CONTEXT, mapping_version=VERSION
    )[0]
    assert (
        c.exercise == "american"
        and c.settlement is SettlementType.PHYSICAL
        and c.expiry_instant is None
    )
    assert c.deliverable == "100 ACME" and e.quantity == 1 and e.premium == D(2)
    assert contract_from_dict(contract_to_dict(c)) == c
    assert execution_from_dict(execution_to_dict(e)) == e
    assert quote_from_dict(quote_to_dict(q)) == q
    assert not q.quote_time_known and q.stamp.event_time == NOW and "unknown" in q.stamp.timing_note
    assert q.reported_estimates["iv_annualized_decimal"] == D(".25")
    assert q.stamp.mapping_version == e.stamp.mapping_version == c.stamp.mapping_version == VERSION
    assert q.stamp.available_time == NOW
    data = payloads()
    data["quotes"] = {"quotes": data["quotes"]["optionQuotes"]}
    with pytest.raises(ValueError):
        normalize_option_quotes(
            data["quotes"], contracts=(c,), context=CONTEXT, mapping_version=VERSION
        )
    q2 = normalize_option_quotes(
        data["quotes"],
        contracts=(c,),
        context=CONTEXT,
        mapping_version="questrade.saved/1:properties-decimal",
    )[0]
    assert q2.reported_estimates["iv_annualized_decimal"] == D(25)
    data = payloads()
    data["symbols"]["symbols"][0]["optionContractDeliverables"]["cashInLieu"] = "1.25"
    adjusted = contracts(data)[0]
    assert "unsupported" in adjusted.deliverable
    with pytest.raises(ValueError):
        require_standard(adjusted)


def test_fee_composition_and_signed_quantities_are_explicit():
    data = payloads()
    row = data["executions"]["executions"][0]
    row["commission"] = "2"
    row["executionFee"] = "0.5"
    separate = execution(data)[0]
    included = execution(data, fee_policy="commission_includes_components")[0]
    assert sum(f.amount for f in separate.fees) == D("2.5")
    assert sum(f.amount for f in included.fees) == D(2)
    with pytest.raises(ValueError):
        execution(data, fee_policy="guess")
    row["executionFee"] = "3"
    with pytest.raises(ValueError):
        execution(data, fee_policy="commission_includes_components")
    for side, sign in (
        ("Buy", 1),
        ("Sell", -1),
        ("Short", -1),
        ("Cov", 1),
        ("BTO", 1),
        ("STC", -1),
        ("STO", -1),
        ("BTC", 1),
    ):
        data = payloads()
        data["executions"]["executions"][0]["side"] = side
        assert execution(data)[0].quantity == sign
    data = payloads()
    data["executions"]["executions"][0]["quantity"] = -1
    with pytest.raises(ValueError):
        execution(data)
    data = payloads()
    data["executions"]["executions"][0]["totalCost"] = "200"
    with pytest.raises(ValueError):
        execution(data)
    data = payloads()
    data["executions"]["executions"].append(dict(data["executions"]["executions"][0]))
    assert len(execution(data)) == 1
    data["executions"]["executions"][1]["commission"] = "7"
    with pytest.raises(ValueError):
        execution(data)


def test_positions_are_inventory_only_and_bad_documented_types_are_refused():
    rows = normalize_positions(
        payloads()["positions"], contracts=contracts(), context=CONTEXT, mapping_version=VERSION
    )
    p = rows[0]
    assert p.reported["entry_history"] == "unknown"
    evidence = position_evidence(rows, contracts=contracts(), record_id="selected:inventory")
    position = position_from_dict(dict(evidence.payload))
    assert position.legs[0].allocated_entry is None
    record = broker_position_record(p, record_id="source:position")
    assert record_loads(record_json(record)).payload_digest == record.payload_digest
    for field, value in (
        ("totalCost", False),
        ("isRealTime", "Individual"),
        ("openQuantity", D(".5")),
        ("isUnderReorg", True),
    ):
        data = payloads()
        data["positions"]["positions"][0][field] = value
        with pytest.raises(ValueError):
            normalize_positions(
                data["positions"], contracts=contracts(), context=CONTEXT, mapping_version=VERSION
            )


def test_activity_execution_reconciliation_never_sums_both_ledgers():
    e = execution()
    a = activities()
    links = {a[0].activity_id: (e[0].execution_id,)}
    report = reconcile_executions_activities(
        e, a, contracts=contracts(), links=links, trade_timezone="America/New_York"
    )
    assert (
        report.status == "consistent"
        and report.matched_execution_cash == report.matched_activity_cash == D(-201)
    )
    assert not hasattr(report, "total_cash")
    pending = reconcile_executions_activities(
        e, a, contracts=contracts(), links={}, trade_timezone="America/New_York"
    )
    assert (
        pending.status == "incomplete"
        and pending.unmatched_executions
        and pending.unmatched_trade_activities
    )
    bad = replace(a[0], net_amount=D(-202))
    with pytest.raises(ValueError):
        reconcile_executions_activities(
            e, (bad,), contracts=contracts(), links=links, trade_timezone="America/New_York"
        )
    record = activity_record(a[0], record_id="synthetic:activity")
    encoded = record_json(record)
    assert "PRIVATE_SENTINEL" not in encoded and "PRIVATE_SENTINEL" not in json.dumps(
        execution_to_dict(e[0])
    )
    assert record_loads(encoded).payload_digest == record.payload_digest
    store = select_as_of((record, record), knowledge_cutoff=NOW, world_id=CONTEXT.world_id)
    assert len(store.selected_records) == 1


def test_unknown_variants_fields_and_missing_fee_evidence_are_refused():
    data = payloads()
    data["executions"]["executions"][0].pop("secFee")
    with pytest.raises(ValueError):
        execution(data)
    data = payloads()
    data["executions"]["executions"][0]["totalCommission"] = "4"
    with pytest.raises(ValueError):
        execution(data)
    data = payloads()
    data["quotes"]["optionQuotes"][0]["timestamp"] = "2026-01-01T00:00:00Z"
    with pytest.raises(ValueError):
        normalize_option_quotes(
            data["quotes"], contracts=contracts(), context=CONTEXT, mapping_version=VERSION
        )
    data = payloads()
    data["activities"]["activities"][0]["commission"] = "1"
    with pytest.raises(ValueError):
        activities(data)
    assert activities(data, commission_policy="positive_charge")[0].paid_fees == 1


def test_nontrade_cash_is_retained_as_audit_evidence_without_creating_fills():
    data = payloads()
    row = data["activities"]["activities"][0]
    row.update(
        type="Interest",
        action="",
        symbolId=0,
        quantity=0,
        price=0,
        grossAmount=0,
        commission=0,
        netAmount="-320.08",
    )
    activity = activities(data)[0]
    assert (
        activity.kind == "Interest"
        and activity.net_amount == D("-320.08")
        and activity.contract_key is None
    )
    report = reconcile_executions_activities(
        execution(), (activity,), contracts=contracts(), links={}, trade_timezone="America/New_York"
    )
    assert report.status == "incomplete" and report.other_activities == (activity.activity_id,)


def test_revision_metadata_is_caller_declared_and_never_inferred_from_refetch():
    original = execution()[0]
    ctx = replace(CONTEXT, revisions={original.execution_id: (1, "prior:execution:record")})
    revised = normalize_executions(
        payloads()["executions"],
        contracts=contracts(),
        context=ctx,
        mapping_version=VERSION,
        fee_policy="separate_components",
    )[0]
    assert revised.stamp.version == 1 and revised.stamp.revision_of == "prior:execution:record"


def test_synthetic_notebook_runs_without_a_client():
    from pathlib import Path

    plt = pytest.importorskip("matplotlib.pyplot", reason="optional viz extra is not installed")

    root = Path(__file__).resolve().parents[2]
    notebook = json.loads((root / "notebooks/offline_questrade.ipynb").read_text())
    namespace = {}
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert not cell["outputs"] and cell["execution_count"] is None
            exec(compile("".join(cell["source"]), "offline_notebook", "exec"), namespace)
    assert namespace["example"]["mark"].total[0, 0] == 42
    plt.close("all")


def test_many_fills_one_activity_and_no_reused_execution_link():
    data = payloads()
    second = dict(data["executions"]["executions"][0])
    second.update(id=6, price="3", totalCost="3")
    data["executions"]["executions"].append(second)
    data["activities"]["activities"][0].update(
        quantity=2, price="2.5", grossAmount="-500", commission="-2", netAmount="-502"
    )
    fills, rows = execution(data), activities(data)
    links = {rows[0].activity_id: tuple(e.execution_id for e in fills)}
    result = reconcile_executions_activities(
        fills, rows, contracts=contracts(), links=links, trade_timezone="America/New_York"
    )
    assert result.matched_execution_cash == D(-502)
    duplicate = replace(rows[0], activity_id=CONTEXT.identity("activity", "other"))
    with pytest.raises(ValueError):
        reconcile_executions_activities(
            fills,
            (*rows, duplicate),
            contracts=contracts(),
            links=links | {duplicate.activity_id: (fills[0].execution_id,)},
            trade_timezone="America/New_York",
        )


def test_cash_comparisons_refuse_currency_and_world_mixing():
    e, a = execution(), activities()
    with pytest.raises(ValueError):
        reconcile_executions_activities(
            e,
            (replace(a[0], currency="CAD"),),
            contracts=contracts(),
            links={},
            trade_timezone="America/New_York",
        )
    with pytest.raises(ValueError):
        reconcile_executions_activities(
            e,
            (replace(a[0], stamp=replace(a[0].stamp, world_id="hypothetical:other")),),
            contracts=contracts(),
            links={},
            trade_timezone="America/New_York",
        )
