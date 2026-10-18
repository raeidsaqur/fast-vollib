"""Pure normalization and explicit reconciliation of saved Questrade responses."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date, datetime, timezone
from decimal import Decimal, localcontext
import json
from typing import Any
from uuid import UUID
from zoneinfo import ZoneInfo

from ...instruments import InstrumentRef, OptionType, SettlementType
from ...observations import ObservationRecord, ObservationStamp
from ...observations.records import freeze, text, utc
from ...pnl.quotes import OptionQuote, QuoteFlags
from ...portfolio import Execution, FeeRecord, ListedContract, OptionLeg, Position
from ...portfolio._validate import LEDGER_CONTEXT, money
from ...portfolio.serialization import position_record
from .mappings import FEE_FIELDS, ROW_FIELDS, SIDE_SIGN, EndpointMapping, get_mapping


class NormalizationError(ValueError):
    """Saved payloads do not establish the requested economic interpretation."""


def _pairs(pairs: Any) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise NormalizationError("duplicate saved JSON key")
        result[key] = value
    return result


def load_saved_json(value: str | bytes) -> Any:
    """Lossless JSON numbers; no token, client, file or network access."""
    return json.loads(
        value,
        parse_float=Decimal,
        parse_int=int,
        object_pairs_hook=_pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(
            NormalizationError("nonfinite saved JSON number")
        ),
    )


def _lossless(value: Any) -> None:
    if isinstance(value, float):
        raise NormalizationError("binary float payloads are ambiguous; use load_saved_json")
    if isinstance(value, Mapping):
        for v in value.values():
            _lossless(v)
    elif isinstance(value, (tuple, list)):
        for v in value:
            _lossless(v)


def _rows(payload: Any, key: str) -> tuple[Mapping[str, Any], ...]:
    _lossless(payload)
    if (
        not isinstance(payload, Mapping)
        or set(payload) != {key}
        or not isinstance(payload[key], (list, tuple))
    ):
        raise NormalizationError("saved endpoint envelope does not match its named mapping")
    if any(not isinstance(row, Mapping) for row in payload[key]):
        raise NormalizationError("saved endpoint rows must be objects")
    allowed = ROW_FIELDS["quotes" if key == "optionQuotes" else key]
    if any(set(row) - allowed for row in payload[key]):
        raise NormalizationError("unknown row fields require a new endpoint mapping")
    return tuple(payload[key])


def _required(row: Mapping[str, Any], keys: Sequence[str]) -> None:
    if not isinstance(row, Mapping) or not set(keys).issubset(row):
        raise NormalizationError("saved payload misses required mapped fields")
    if any(k in row for k in ("accountId", "accountNumber", "access_token", "refresh_token")):
        raise NormalizationError("account identifiers and credentials are not normalization inputs")


def _money(value: Any, *, nonnegative: bool = False) -> Decimal:
    if type(value) is int:
        value = Decimal(value)
    return money(value, nonnegative=nonnegative)


def _int(value: Any, *, nonnegative: bool = False, nonzero: bool = False) -> int:
    if type(value) is int:
        result = value
    elif isinstance(value, Decimal) and value.is_finite() and value == value.to_integral_value():
        result = int(value)
    else:
        raise NormalizationError("mapped integer field must be lossless and integral")
    if abs(result) > 10**12 or (nonnegative and result < 0) or (nonzero and result == 0):
        raise NormalizationError("mapped integer field is out of domain")
    return result


def _id(value: Any) -> str:
    if type(value) is int or isinstance(value, Decimal):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, Decimal))
            or (
                isinstance(value, Decimal)
                and (not value.is_finite() or value != value.to_integral_value())
            )
            or value < 0
            or value > 2**64 - 1
        ):
            raise NormalizationError(
                "numeric source identifier must be an unsigned integral 64-bit value"
            )
        result = str(int(value))
    elif isinstance(value, str) and value.strip():
        result = value
    else:
        raise NormalizationError("source identifier must be nonempty text or integral numeric data")
    return result


def _time(value: Any) -> datetime:
    if not isinstance(value, str):
        raise NormalizationError("source timestamp must be ISO text")
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None or result.utcoffset() is None:
        raise NormalizationError("source timestamp needs an explicit timezone")
    return result.astimezone(timezone.utc)


@dataclass(frozen=True, slots=True, kw_only=True)
class NormalizationContext:
    namespace: str
    retrieved_at: datetime
    available_time: datetime
    availability_note: str
    world_id: str
    revisions: Mapping[str, tuple[int, str]] | None = None

    def __post_init__(self) -> None:
        try:
            namespace = str(UUID(self.namespace))
        except (ValueError, TypeError, AttributeError) as exc:
            raise NormalizationError(
                "namespace must be a stable opaque UUID, never an account number"
            ) from exc
        object.__setattr__(self, "namespace", namespace)
        for name in ("retrieved_at", "available_time"):
            object.__setattr__(self, name, utc(getattr(self, name)))
        text(self.availability_note, "availability_note")
        text(self.world_id, "world_id")
        revisions = {}
        for key, value in (self.revisions or {}).items():
            text(key, "revision source identity")
            pair = tuple(value)
            if len(pair) != 2 or type(pair[0]) is not int or pair[0] < 1:
                raise NormalizationError(
                    "corrections require a positive version and explicit ancestor record ID"
                )
            text(pair[1], "revision ancestor")
            revisions[key] = pair
        object.__setattr__(self, "revisions", freeze(revisions))

    def identity(self, kind: str, source_id: Any) -> str:
        return f"questrade:{self.namespace}:{kind}:{_id(source_id)}"

    def stamp(
        self, *, event: datetime, source_id: str, mapping: EndpointMapping, note: str
    ) -> ObservationStamp:
        version, ancestor = (self.revisions or {}).get(source_id, (0, None))
        return ObservationStamp(
            event_time=event,
            version=version,
            revision_of=ancestor,
            available_time=self.available_time,
            recorded_at=self.retrieved_at,
            source="questrade",
            source_record_id=source_id,
            world_id=self.world_id,
            mapping_version=mapping.version,
            timing_note=f"Availability: {self.availability_note}. {note}",
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class ContractTerms:
    contract_key: str
    underlier: InstrumentRef
    premium_multiplier: Decimal
    settlement: SettlementType
    assumption_note: str

    def __post_init__(self) -> None:
        text(self.contract_key, "contract_key")
        text(self.assumption_note, "assumption_note")
        if not isinstance(self.underlier, InstrumentRef) or not isinstance(
            self.settlement, SettlementType
        ):
            raise NormalizationError("explicit underlier reference and settlement enum required")
        multiplier = _money(self.premium_multiplier, nonnegative=True)
        if multiplier <= 0:
            raise NormalizationError("premium multiplier must be positive")
        object.__setattr__(self, "premium_multiplier", multiplier)


def normalize_symbols(
    payload: Any,
    *,
    terms: Mapping[str, ContractTerms],
    context: NormalizationContext,
    mapping_version: str,
) -> tuple[ListedContract, ...]:
    mapping = get_mapping(mapping_version)
    contracts = []
    for row in _rows(payload, "symbols"):
        _required(
            row,
            (
                "symbolId",
                "symbol",
                "optionType",
                "optionExerciseType",
                "optionExpiryDate",
                "optionStrikePrice",
                "optionContractDeliverables",
                "currency",
            ),
        )
        sid = _id(row["symbolId"])
        if sid not in terms or not isinstance(terms[sid], ContractTerms):
            raise NormalizationError("explicit listed terms required for each source symbol")
        given = terms[sid]
        if row["optionType"] not in ("Call", "Put") or row["optionExerciseType"] not in (
            "American",
            "European",
        ):
            raise NormalizationError("unsupported source option type or exercise designation")
        raw_date = row["optionExpiryDate"]
        _time(raw_date)
        # Preserve the source legal date, not the UTC date of a midnight timestamp.
        expiry = datetime.fromisoformat(raw_date.replace("Z", "+00:00")).date()
        delivery = row["optionContractDeliverables"]
        if (
            not isinstance(delivery, Mapping)
            or set(delivery) != {"underlyings", "cashInLieu"}
            or not isinstance(delivery["underlyings"], (list, tuple))
        ):
            raise NormalizationError("unknown contract deliverable variant")
        pieces = []
        for component in delivery["underlyings"]:
            _required(component, ("multiplier", "underlyingSymbol", "underlyingSymbolId"))
            count = _int(component["multiplier"], nonnegative=True, nonzero=True)
            symbol = text(component["underlyingSymbol"], "deliverable symbol")
            pieces.append(
                {
                    "multiplier": count,
                    "underlyingSymbol": symbol,
                    "underlyingSymbolId": _id(component["underlyingSymbolId"]),
                }
            )
        cash = _money(delivery["cashInLieu"], nonnegative=True)
        if pieces and given.settlement != SettlementType.PHYSICAL:
            raise NormalizationError(
                "source share deliverables cannot be relabelled cash settlement"
            )
        standard = (
            len(pieces) == 1
            and pieces[0]["multiplier"] == 100
            and pieces[0]["underlyingSymbol"] == given.underlier.identifier
            and cash == 0
        )
        descriptor = (
            (
                f"100 {given.underlier.identifier}"
                if given.settlement == SettlementType.PHYSICAL
                else "standard_cash_equivalent:100"
            )
            if standard
            else "unsupported_broker_deliverables:"
            + json.dumps({"underlyings": pieces, "cashInLieu": str(cash)}, sort_keys=True)
        )
        source_id = context.identity("symbol", sid)
        contracts.append(
            ListedContract(
                key=given.contract_key,
                underlier=given.underlier,
                option_type=OptionType.CALL if row["optionType"] == "Call" else OptionType.PUT,
                strike=_money(row["optionStrikePrice"], nonnegative=True),
                expiry_date=expiry,
                exercise=row["optionExerciseType"].lower(),
                settlement=given.settlement,
                multiplier=given.premium_multiplier,
                deliverable=descriptor,
                currency=row["currency"],
                source_symbol_id=sid,
                stamp=context.stamp(
                    event=context.retrieved_at,
                    source_id=source_id,
                    mapping=mapping,
                    note="Symbol event time assumed retrieval; expiry remains a legal date. Listed terms: "
                    + given.assumption_note,
                ),
            )
        )
    if len({c.key for c in contracts}) != len(contracts) or len(
        {c.source_symbol_id for c in contracts}
    ) != len(contracts):
        raise NormalizationError("duplicate or conflicting source contract identities")
    return tuple(contracts)


def _lookup(contracts: Sequence[ListedContract]) -> dict[str, ListedContract]:
    result: dict[str, ListedContract] = {}
    for contract in contracts:
        if (
            not isinstance(contract, ListedContract)
            or contract.source_symbol_id is None
            or contract.source_symbol_id in result
        ):
            raise NormalizationError("contracts require unique source symbol identities")
        result[contract.source_symbol_id] = contract
    return result


def _fees(row: Mapping[str, Any], *, policy: str, currency: str) -> tuple[FeeRecord, ...]:
    _required(row, FEE_FIELDS)
    values = {name: _money(row[name], nonnegative=True) for name in FEE_FIELDS}
    other = sum((values[k] for k in FEE_FIELDS if k != "commission"), Decimal(0))
    if policy == "separate_components":
        commission = values["commission"]
    elif policy == "commission_includes_components":
        commission = values["commission"] - other
        if commission < 0:
            raise NormalizationError("included fee components exceed declared total commission")
    else:
        raise NormalizationError("explicit supported fee composition policy required")
    parts = (
        ("commission", commission),
        ("other", values["orderPlacementCommission"]),
        ("exchange", values["executionFee"] + values["canadianExecutionFee"]),
        ("regulatory", values["secFee"]),
    )
    return tuple(FeeRecord(kind=kind, amount=value, currency=currency) for kind, value in parts)


def normalize_executions(
    payload: Any,
    *,
    contracts: Sequence[ListedContract],
    context: NormalizationContext,
    mapping_version: str,
    fee_policy: str,
    source_sequences: Mapping[str, int] | None = None,
) -> tuple[Execution, ...]:
    mapping = get_mapping(mapping_version)
    lookup = _lookup(contracts)
    executions: dict[str, Execution] = {}
    with localcontext(LEDGER_CONTEXT):
        for row in _rows(payload, "executions"):
            _required(
                row,
                (
                    "id",
                    "symbolId",
                    "quantity",
                    "side",
                    "price",
                    "timestamp",
                    "orderId",
                    "totalCost",
                    *FEE_FIELDS,
                ),
            )
            sid = _id(row["symbolId"])
            if sid not in lookup:
                raise NormalizationError("execution references unknown contract")
            contract = lookup[sid]
            if contract.stamp.world_id != context.world_id:
                raise NormalizationError("execution and contract worlds disagree")
            if row["side"] not in SIDE_SIGN:
                raise NormalizationError("unsupported execution side")
            quantity = _int(row["quantity"], nonnegative=True, nonzero=True)
            premium = _money(row["price"], nonnegative=True)
            if _money(row["totalCost"], nonnegative=True) != premium * quantity:
                raise NormalizationError(
                    "execution totalCost does not match the named positive premium-times-quantity mapping"
                )
            source = _id(row["id"])
            identity = context.identity("execution", source)
            execution = Execution(
                contract_key=contract.key,
                quantity=SIDE_SIGN[row["side"]] * quantity,
                premium=premium,
                execution_id=identity,
                order_id=context.identity("order", row["orderId"]),
                source_sequence=None if source_sequences is None else source_sequences.get(source),
                fees=_fees(row, policy=fee_policy, currency=contract.currency),
                stamp=context.stamp(
                    event=_time(row["timestamp"]),
                    source_id=identity,
                    mapping=mapping,
                    note=f"Fee composition: {fee_policy}; source totalCost is premium times unsigned contracts, without multiplier.",
                ),
            )
            if identity in executions and executions[identity] != execution:
                raise NormalizationError("conflicting repeated execution identity")
            executions[identity] = execution
    return tuple(executions.values())


def normalize_option_quotes(
    payload: Any,
    *,
    contracts: Sequence[ListedContract],
    context: NormalizationContext,
    mapping_version: str,
) -> tuple[OptionQuote, ...]:
    mapping = get_mapping(mapping_version)
    lookup = _lookup(contracts)
    quotes: dict[str, OptionQuote] = {}
    with localcontext(LEDGER_CONTEXT):
        for row in _rows(payload, mapping.quote_envelope):
            _required(
                row,
                (
                    "symbolId",
                    "bidPrice",
                    "askPrice",
                    "bidSize",
                    "askSize",
                    "lastTradePrice",
                    "lastTradeTime",
                    "delay",
                    "isHalted",
                ),
            )
            sid = _id(row["symbolId"])
            if sid not in lookup:
                raise NormalizationError("option quote references unknown contract")
            if lookup[sid].stamp.world_id != context.world_id:
                raise NormalizationError("quote and contract worlds disagree")
            if type(row["isHalted"]) is not bool:
                raise NormalizationError("isHalted must be boolean")
            if "quoteTime" in row or "timestamp" in row:
                raise NormalizationError(
                    "source quote time requires a separately versioned mapping"
                )
            flags = QuoteFlags.HALTED if row["isHalted"] else QuoteFlags.OK
            delay = None if row["delay"] is None else _int(row["delay"], nonnegative=True)
            if delay:
                flags |= QuoteFlags.DELAYED
            estimates: dict[str, Any] = {
                "source": "questrade",
                "iv_unit": mapping.iv_unit,
                "mapping_version": mapping.version,
            }
            for name in ("volatility", "delta", "gamma", "theta", "vega", "rho"):
                if name in row and row[name] is not None:
                    value = _money(row[name], nonnegative=name == "volatility")
                    estimates[name] = value
                    if name == "volatility":
                        estimates["iv_annualized_decimal"] = value / (
                            100 if mapping.iv_unit == "percent" else 1
                        )
            identity = context.identity("quote", sid) + ":" + context.retrieved_at.isoformat()
            quote = OptionQuote(
                contract_key=lookup[sid].key,
                bid=None if row["bidPrice"] is None else _money(row["bidPrice"], nonnegative=True),
                ask=None if row["askPrice"] is None else _money(row["askPrice"], nonnegative=True),
                bid_size=None if row["bidSize"] is None else _int(row["bidSize"], nonnegative=True),
                ask_size=None if row["askSize"] is None else _int(row["askSize"], nonnegative=True),
                last=None
                if row["lastTradePrice"] is None
                else _money(row["lastTradePrice"], nonnegative=True),
                last_trade_time=None
                if row["lastTradeTime"] is None
                else _time(row["lastTradeTime"]),
                flags=flags,
                delay_s=delay,
                quote_time_known=False,
                reported_estimates=estimates,
                stamp=context.stamp(
                    event=context.retrieved_at,
                    source_id=identity,
                    mapping=mapping,
                    note="Source quote time unknown; event time assumed retrieval. Last-trade time is not quote time.",
                ),
            )
            if sid in quotes and quotes[sid] != quote:
                raise NormalizationError("conflicting duplicate option quote")
            quotes[sid] = quote
    return tuple(quotes.values())


@dataclass(frozen=True, slots=True, kw_only=True)
class ObservedPosition:
    contract_key: str
    quantity: int
    reported: Mapping[str, Any]
    stamp: ObservationStamp

    def __post_init__(self) -> None:
        text(self.contract_key, "contract_key")
        object.__setattr__(self, "quantity", _int(self.quantity))
        if not isinstance(self.stamp, ObservationStamp):
            raise NormalizationError("position requires an observation stamp")
        object.__setattr__(self, "reported", freeze(dict(self.reported)))


def normalize_positions(
    payload: Any,
    *,
    contracts: Sequence[ListedContract],
    context: NormalizationContext,
    mapping_version: str,
) -> tuple[ObservedPosition, ...]:
    mapping = get_mapping(mapping_version)
    lookup = _lookup(contracts)
    positions = []
    for row in _rows(payload, "positions"):
        _required(
            row,
            (
                "symbolId",
                "openQuantity",
                "currentMarketValue",
                "currentPrice",
                "averageEntryPrice",
                mapping.closed_pnl_key,
                mapping.open_pnl_key,
                "totalCost",
                "isRealTime",
                "isUnderReorg",
            ),
        )
        sid = _id(row["symbolId"])
        if sid not in lookup:
            raise NormalizationError("position references unknown contract")
        if lookup[sid].stamp.world_id != context.world_id:
            raise NormalizationError("position and contract worlds disagree")
        if type(row["isRealTime"]) is not bool or type(row["isUnderReorg"]) is not bool:
            raise NormalizationError("position real-time/reorganization fields must be booleans")
        if row["isUnderReorg"]:
            raise NormalizationError("reorganization requires an unsupported lifecycle mapping")
        alternate = {"closedPnL", "openPnL", "closedPnl", "openPnl"} - {
            mapping.closed_pnl_key,
            mapping.open_pnl_key,
        }
        if any(k in row for k in alternate):
            raise NormalizationError("mixed position payload variants")
        reported = {
            "source": "questrade",
            "entry_history": "unknown",
            "isRealTime": row["isRealTime"],
        }
        for name in (
            "currentMarketValue",
            "currentPrice",
            "averageEntryPrice",
            mapping.closed_pnl_key,
            mapping.open_pnl_key,
            "totalCost",
        ):
            reported[name] = _money(row[name])
        positions.append(
            ObservedPosition(
                contract_key=lookup[sid].key,
                quantity=_int(row["openQuantity"]),
                reported=reported,
                stamp=context.stamp(
                    event=context.retrieved_at,
                    source_id=context.identity("position", sid)
                    + ":"
                    + context.retrieved_at.isoformat(),
                    mapping=mapping,
                    note="Position event time assumed retrieval; reported average entry and P&L are not a complete execution ledger.",
                ),
            )
        )
    if len({p.contract_key for p in positions}) != len(positions):
        raise NormalizationError("duplicate positions require explicit resolution")
    return tuple(positions)


def position_evidence(
    selected: Sequence[ObservedPosition], *, contracts: Sequence[ListedContract], record_id: str
) -> ObservationRecord:
    """Explicit caller grouping into inventory-only evidence, never invented fills."""
    if not selected:
        raise NormalizationError("explicit selected inventory rows required")
    lookup = {c.key: c for c in contracts}
    legs = []
    # Different source IDs are expected; time/cutoff/world must agree.
    first = selected[0].stamp
    for p in selected:
        if p.contract_key not in lookup or (
            p.stamp.event_time,
            p.stamp.available_time,
            p.stamp.world_id,
        ) != (first.event_time, first.available_time, first.world_id):
            raise NormalizationError("selected inventory evidence is incompatible")
        if p.quantity:
            legs.append(
                OptionLeg(
                    contract=lookup[p.contract_key],
                    quantity=p.quantity,
                    allocated_entry=None,
                    allocated_fees=Decimal(0),
                    leg_id=p.contract_key,
                )
            )
    return position_record(
        Position.from_legs(legs),
        record_id=record_id,
        stamp=replace(first, source_record_id=None, version=0, revision_of=None),
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class CashActivity:
    activity_id: str
    contract_key: str | None
    quantity: int
    price: Decimal
    trade_date: date
    gross_amount: Decimal
    paid_fees: Decimal
    net_amount: Decimal
    currency: str
    kind: str
    stamp: ObservationStamp

    def __post_init__(self) -> None:
        for name in ("activity_id", "currency", "kind"):
            text(getattr(self, name), name)
        object.__setattr__(self, "quantity", _int(self.quantity))
        if len(self.currency) != 3 or not self.currency.isalpha() or not self.currency.isupper():
            raise NormalizationError("activity currency must be an uppercase three-letter code")
        if type(self.trade_date) is not date:
            raise NormalizationError("activity requires the source legal trade date")
        for name in ("price", "gross_amount", "paid_fees", "net_amount"):
            object.__setattr__(
                self, name, _money(getattr(self, name), nonnegative=name in ("price", "paid_fees"))
            )
        if not isinstance(self.stamp, ObservationStamp):
            raise NormalizationError("activity requires a stamp")


def normalize_activities(
    payload: Any,
    *,
    contracts: Sequence[ListedContract],
    context: NormalizationContext,
    mapping_version: str,
    activity_ids: Sequence[str],
    commission_policy: str,
) -> tuple[CashActivity, ...]:
    mapping = get_mapping(mapping_version)
    lookup = _lookup(contracts)
    rows = _rows(payload, "activities")
    if len(activity_ids) != len(rows) or len(set(activity_ids)) != len(activity_ids):
        raise NormalizationError("activities without source IDs require unique stable caller UUIDs")
    if commission_policy not in ("negative_deduction", "positive_charge"):
        raise NormalizationError("explicit activity commission sign policy required")
    activities = []
    with localcontext(LEDGER_CONTEXT):
        for row, raw_id in zip(rows, activity_ids):
            try:
                stable = str(UUID(raw_id))
            except (ValueError, TypeError, AttributeError) as exc:
                raise NormalizationError(
                    "activity row identity must be a stable opaque UUID"
                ) from exc
            _required(
                row,
                (
                    "tradeDate",
                    "transactionDate",
                    "settlementDate",
                    "action",
                    "symbolId",
                    "currency",
                    "quantity",
                    "price",
                    "grossAmount",
                    "commission",
                    "netAmount",
                    "type",
                ),
            )
            kind = text(row["type"], "activity type")
            sid = _id(row["symbolId"])
            quantity = _int(row["quantity"])
            price = _money(row["price"], nonnegative=True)
            gross = _money(row["grossAmount"])
            commission = _money(row["commission"])
            net = _money(row["netAmount"])
            fee = -commission if commission_policy == "negative_deduction" else commission
            if fee < 0 or (kind == "Trades" and net != gross - fee):
                raise NormalizationError(
                    "activity commission sign/net identity disagrees with explicit policy"
                )
            contract = lookup.get(sid)
            if contract is not None and contract.stamp.world_id != context.world_id:
                raise NormalizationError("activity and contract worlds disagree")
            if kind == "Trades":
                if (
                    contract is None
                    or row["action"] not in SIDE_SIGN
                    or not quantity
                    or (quantity > 0) != (SIDE_SIGN[row["action"]] > 0)
                ):
                    raise NormalizationError(
                        "trade activity requires a known contract and consistent signed quantity/action"
                    )
                if (
                    row["currency"] != contract.currency
                    or gross != -quantity * contract.multiplier * price
                ):
                    raise NormalizationError(
                        "trade activity gross/currency disagrees with contract units"
                    )
            identity = context.identity("activity", stable)
            _time(row["transactionDate"])
            _time(row["settlementDate"])
            activities.append(
                CashActivity(
                    activity_id=identity,
                    contract_key=None if contract is None else contract.key,
                    quantity=quantity,
                    price=price,
                    trade_date=datetime.fromisoformat(
                        row["tradeDate"].replace("Z", "+00:00")
                    ).date(),
                    gross_amount=gross,
                    paid_fees=fee,
                    net_amount=net,
                    currency=row["currency"],
                    kind=kind,
                    stamp=context.stamp(
                        event=_time(row["tradeDate"]),
                        source_id=identity,
                        mapping=mapping,
                        note=f"Activity commission policy: {commission_policy}; explicit stable caller row identity. No activity amount is added to execution cash.",
                    ),
                )
            )
    return tuple(activities)


@dataclass(frozen=True, slots=True, kw_only=True)
class ReconciliationReport:
    status: str
    matched: tuple[tuple[str, tuple[str, ...]], ...]
    unmatched_executions: tuple[str, ...]
    unmatched_trade_activities: tuple[str, ...]
    other_activities: tuple[str, ...]
    matched_execution_cash: Decimal
    matched_activity_cash: Decimal
    caveat: str = "Consistency of supplied saved rows and explicit links; not proof of complete broker-account reconciliation."

    def __post_init__(self) -> None:
        for name in (
            "matched",
            "unmatched_executions",
            "unmatched_trade_activities",
            "other_activities",
        ):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        object.__setattr__(self, "matched", tuple((aid, tuple(ids)) for aid, ids in self.matched))
        if self.status not in ("consistent", "incomplete"):
            raise NormalizationError("unknown reconciliation status")
        for name in ("matched_execution_cash", "matched_activity_cash"):
            object.__setattr__(self, name, _money(getattr(self, name)))
        if self.matched_execution_cash != self.matched_activity_cash:
            raise NormalizationError("matched cash comparisons must agree")
        if self.status == "consistent" and (
            not self.matched or self.unmatched_executions or self.unmatched_trade_activities
        ):
            raise NormalizationError("consistency status requires complete explicit trade links")


def _event_time(stamp: ObservationStamp) -> datetime:
    if not isinstance(stamp.event_time, datetime):
        raise NormalizationError("execution reconciliation requires instant events")
    return stamp.event_time


def reconcile_executions_activities(
    executions: Sequence[Execution],
    activities: Sequence[CashActivity],
    *,
    contracts: Sequence[ListedContract],
    links: Mapping[str, Sequence[str]],
    trade_timezone: str,
) -> ReconciliationReport:
    """Compare explicit many-execution/one-activity links without summing both ledgers."""
    zone = ZoneInfo(trade_timezone)
    fills = {e.execution_id: e for e in executions}
    cash = {a.activity_id: a for a in activities}
    terms = {c.key: c for c in contracts}
    if any(e.contract_key not in terms for e in executions):
        raise NormalizationError("reconciliation requires every execution's contract terms")
    currencies = {terms[e.contract_key].currency for e in executions} | {
        a.currency for a in activities if a.kind == "Trades"
    }
    worlds = {e.stamp.world_id for e in executions} | {
        a.stamp.world_id for a in activities if a.kind == "Trades"
    }
    if len(currencies) > 1 or len(worlds) > 1:
        raise NormalizationError("reconcile one currency and one world at a time")
    if len(fills) != len(executions) or len(cash) != len(activities):
        raise NormalizationError("reconcile unique identities, not repeated deliveries")
    used: set[str] = set()
    matched = []
    execution_cash = activity_cash = Decimal(0)
    with localcontext(LEDGER_CONTEXT):
        for aid, eids_input in links.items():
            eids = tuple(eids_input)
            if (
                aid not in cash
                or cash[aid].kind != "Trades"
                or not eids
                or len(set(eids)) != len(eids)
                or any(i not in fills or i in used for i in eids)
            ):
                raise NormalizationError("invalid, repeated or missing reconciliation link")
            activity = cash[aid]
            group = [fills[i] for i in eids]
            if any(
                e.contract_key != activity.contract_key
                or e.stamp.world_id != activity.stamp.world_id
                or _event_time(e.stamp).astimezone(zone).date() != activity.trade_date
                for e in group
            ):
                raise NormalizationError("linked records disagree on contract, world or trade date")
            if activity.contract_key not in terms:
                raise NormalizationError("linked activity has no contract terms")
            contract = terms[activity.contract_key]
            gross = sum((-e.quantity * contract.multiplier * e.premium for e in group), Decimal(0))
            fees = sum((f.amount for e in group for f in e.fees), Decimal(0))
            if (
                sum(e.quantity for e in group) != activity.quantity
                or activity.currency != contract.currency
                or gross != activity.gross_amount
                or fees != activity.paid_fees
                or gross - fees != activity.net_amount
            ):
                raise NormalizationError(
                    "linked activity and execution quantities/gross/fees/net do not reconcile"
                )
            used.update(eids)
            matched.append((aid, eids))
            execution_cash += gross - fees
            activity_cash += activity.net_amount
    unfilled = tuple(sorted(set(fills) - used))
    unmatched = tuple(
        sorted(aid for aid, a in cash.items() if a.kind == "Trades" and aid not in links)
    )
    others = tuple(sorted(aid for aid, a in cash.items() if a.kind != "Trades"))
    return ReconciliationReport(
        status="consistent" if matched and not unfilled and not unmatched else "incomplete",
        matched=tuple(sorted(matched)),
        unmatched_executions=unfilled,
        unmatched_trade_activities=unmatched,
        other_activities=others,
        matched_execution_cash=execution_cash,
        matched_activity_cash=activity_cash,
    )


def activity_record(activity: CashActivity, *, record_id: str) -> ObservationRecord:
    payload = {
        name: getattr(activity, name)
        for name in (
            "activity_id",
            "contract_key",
            "quantity",
            "price",
            "gross_amount",
            "paid_fees",
            "net_amount",
            "currency",
            "kind",
        )
    }
    payload["trade_date"] = activity.trade_date.isoformat()
    return ObservationRecord(
        record_id=record_id,
        kind="cash_activity",
        entities=(activity.contract_key or "currency:" + activity.currency,),
        payload=payload,
        payload_schema="questrade.activity/1",
        stamp=activity.stamp,
    )


def broker_position_record(position: ObservedPosition, *, record_id: str) -> ObservationRecord:
    return ObservationRecord(
        record_id=record_id,
        kind="broker_position",
        entities=(position.contract_key,),
        payload={
            "contract_key": position.contract_key,
            "quantity": position.quantity,
            "reported": position.reported,
        },
        payload_schema="questrade.position/1",
        stamp=position.stamp,
    )
