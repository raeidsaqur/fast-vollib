"""Synthetic saved option chain and executions, with explicit reconciliation links."""

from datetime import datetime, timezone
from decimal import Decimal as D

from fast_vollib.instruments import InstrumentRef, SettlementType
from fast_vollib.integrations.questrade import (
    ContractTerms,
    NormalizationContext,
    normalize_activities,
    normalize_executions,
    normalize_option_quotes,
    normalize_symbols,
    reconcile_executions_activities,
)
from fast_vollib.observations import ObservationStamp
from fast_vollib.pnl import QuotePolicy, QuoteSnapshot, mark_position
from fast_vollib.portfolio import OptionLeg, Position, hypothetical_snapshot

origin = datetime(2026, 10, 16, 15, tzinfo=timezone.utc)
world = "hypothetical:saved-options-example"
context = NormalizationContext(
    namespace="00000000-0000-4000-8000-000000000010",
    retrieved_at=origin,
    available_time=origin,
    availability_note="synthetic response known at retrieval",
    world_id=world,
)
version = "questrade.saved/1:examples-percent"
ref = InstrumentRef(identifier="SPY", currency="USD")
# Every value below is synthetic. No account identifier or authenticated client is used.
roles = (
    ("Put", "580", "BTO", "1.20", "0.75", "0.85"),
    ("Put", "585", "STO", "2.00", "1.35", "1.45"),
    ("Call", "615", "STO", "1.80", "1.15", "1.25"),
    ("Call", "620", "BTO", "1.00", "0.65", "0.75"),
)
symbol_rows = []
execution_rows = []
quote_rows = []
activity_rows = []
terms = {}
for i, (kind, strike, side, premium, bid, ask) in enumerate(roles, 1):
    key = f"synthetic:SPY:2026-11-20:{kind}:{strike}"
    terms[str(i)] = ContractTerms(
        contract_key=key,
        underlier=ref,
        premium_multiplier=D(100),
        settlement=SettlementType.PHYSICAL,
        assumption_note="synthetic standard 100-share terms",
    )
    symbol_rows.append(
        dict(
            symbolId=i,
            symbol=key,
            optionType=kind,
            optionExerciseType="American",
            optionExpiryDate="2026-11-20T00:00:00-05:00",
            optionStrikePrice=strike,
            currency="USD",
            optionContractDeliverables={
                "underlyings": [
                    dict(multiplier=100, underlyingSymbol="SPY", underlyingSymbolId=10)
                ],
                "cashInLieu": 0,
            },
        )
    )
    execution_rows.append(
        dict(
            id=i,
            symbolId=i,
            quantity=1,
            side=side,
            price=premium,
            timestamp=origin.isoformat(),
            orderId=1,
            totalCost=premium,
            commission="1.00",
            orderPlacementCommission=0,
            executionFee=0,
            secFee=0,
            canadianExecutionFee=0,
        )
    )
    quote_rows.append(
        dict(
            symbolId=i,
            bidPrice=bid,
            askPrice=ask,
            bidSize=10,
            askSize=10,
            lastTradePrice=None,
            lastTradeTime=None,
            delay=0,
            isHalted=False,
            volatility="20",
        )
    )
    signed = 1 if side == "BTO" else -1
    gross = -signed * D(100) * D(premium)
    activity_rows.append(
        dict(
            tradeDate="2026-10-16T00:00:00-04:00",
            transactionDate=origin.isoformat(),
            settlementDate="2026-10-19T00:00:00-04:00",
            action="Buy" if signed > 0 else "Sell",
            symbolId=i,
            currency="USD",
            quantity=signed,
            price=premium,
            grossAmount=str(gross),
            commission="-1.00",
            netAmount=str(gross - 1),
            type="Trades",
        )
    )
contracts = normalize_symbols(
    {"symbols": symbol_rows}, terms=terms, context=context, mapping_version=version
)
executions = normalize_executions(
    {"executions": execution_rows},
    contracts=contracts,
    context=context,
    mapping_version=version,
    fee_policy="separate_components",
    source_sequences={str(i): i for i in range(1, 5)},
)
quotes = normalize_option_quotes(
    {"optionQuotes": quote_rows}, contracts=contracts, context=context, mapping_version=version
)
activities = normalize_activities(
    {"activities": activity_rows},
    contracts=contracts,
    context=context,
    mapping_version=version,
    activity_ids=[f"00000000-0000-4000-8000-{i:012d}" for i in range(1, 5)],
    commission_policy="negative_deduction",
)
links = {a.activity_id: (e.execution_id,) for a, e in zip(activities, executions)}
reconciliation = reconcile_executions_activities(
    executions, activities, contracts=contracts, links=links, trade_timezone="America/New_York"
)
assert reconciliation.status == "consistent"
assert reconciliation.matched_execution_cash == reconciliation.matched_activity_cash == D(156)
# The caller explicitly chooses this grouping and a hypothetical entry interpretation.
by_key = {c.key: c for c in contracts}
position = Position.from_legs(
    OptionLeg(
        contract=by_key[e.contract_key],
        quantity=e.quantity,
        allocated_entry=e.premium,
        allocated_fees=sum((f.amount for f in e.fees), D(0)),
        leg_id=e.contract_key,
    )
    for e in executions
)
snapshot = hypothetical_snapshot(
    position, valuation_time=origin, knowledge_cutoff=origin, world_id=world
)
underlying_stamp = ObservationStamp(
    event_time=origin, available_time=origin, recorded_at=origin, source="synthetic", world_id=world
)
book = QuoteSnapshot(
    quotes={q.contract_key: q for q in quotes},
    underlying=D(600),
    stamp=underlying_stamp,
    delay_s=0,
    valuation_time=origin,
    knowledge_cutoff=origin,
)
# The endpoint provides last-trade time, not source quote time: no freshness claim.
policy = QuotePolicy(
    name="synthetic_indications_unknown_quote_time",
    mark_rule="midpoint",
    max_quote_age_s=60.0,
    max_underlier_age_s=60.0,
    permitted_delay_s=0,
    allow_zero_size=False,
    allow_unknown_quote_time=True,
)
mark = mark_position(snapshot, book, policy=policy, future_costs=D(4))
assert mark.total[0, 0] == 42
assert all(d.age_s is None for d in mark.quote_details)
