"""Strict portfolio JSON records. Decimal money is text, never a JSON number."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields
from datetime import date, datetime
import json
from typing import Any, Callable

from ..instruments import InstrumentRef, OptionType, SettlementType
from ..observations import ObservationRecord, ObservationStamp
from ..observations.serialization import load_json
from .contracts import ListedContract
from .errors import PortfolioError
from .position import OptionLeg, Position
from .trades import Execution, FeeRecord, Trade


def _fields(value: Any, names: Any) -> Any:
    if not isinstance(value, Mapping) or set(value) != set(names):
        raise PortfolioError("unknown or missing portfolio codec fields")


def _stamp_to(stamp: Any) -> Any:
    return {
        f.name: (
            [x.isoformat() for x in stamp.event_time]
            if isinstance(stamp.event_time, tuple)
            else stamp.event_time.isoformat()
        )
        if f.name == "event_time"
        else getattr(stamp, f.name).isoformat()
        if f.name in ("available_time", "recorded_at")
        else getattr(stamp, f.name)
        for f in fields(stamp)
    }


def _stamp_from(value: Any) -> Any:
    _fields(value, (f.name for f in fields(ObservationStamp)))
    data = dict(value)
    event = data["event_time"]
    data["event_time"] = (
        datetime.fromisoformat(event)
        if isinstance(event, str)
        else tuple(datetime.fromisoformat(x) for x in event)
    )
    for key in ("available_time", "recorded_at"):
        data[key] = datetime.fromisoformat(data[key])
    return ObservationStamp(**data)


def _money(value: Any) -> Any:
    if not isinstance(value, str):
        raise PortfolioError("wire money must be decimal text")
    from ._validate import money

    return money(value)


def contract_to_dict(contract: ListedContract) -> dict[str, Any]:
    result = {f.name: getattr(contract, f.name) for f in fields(contract)}
    result.update(
        schema="portfolio.contract/1",
        underlier={
            "identifier": contract.underlier.identifier,
            "currency": contract.underlier.currency,
            "asset_class": contract.underlier.asset_class.value
            if contract.underlier.asset_class
            else None,
        },
        option_type=contract.option_type.value,
        settlement=contract.settlement.value,
        strike=str(contract.strike),
        multiplier=str(contract.multiplier),
        expiry_date=contract.expiry_date.isoformat(),
        expiry_instant=contract.expiry_instant.isoformat() if contract.expiry_instant else None,
        stamp=_stamp_to(contract.stamp),
    )
    return result


def contract_from_dict(value: Any) -> ListedContract:
    _fields(value, ["schema", *(f.name for f in fields(ListedContract))])
    if value["schema"] != "portfolio.contract/1":
        raise PortfolioError("unknown contract schema")
    _fields(value["underlier"], ("identifier", "currency", "asset_class"))
    data = dict(value)
    del data["schema"]
    data.update(
        underlier=InstrumentRef(**data["underlier"]),
        option_type=OptionType(data["option_type"]),
        settlement=SettlementType(data["settlement"]),
        strike=_money(data["strike"]),
        multiplier=_money(data["multiplier"]),
        expiry_date=date.fromisoformat(data["expiry_date"]),
        expiry_instant=datetime.fromisoformat(data["expiry_instant"])
        if data["expiry_instant"]
        else None,
        stamp=_stamp_from(data["stamp"]),
    )
    return ListedContract(**data)


def execution_to_dict(fill: Execution) -> dict[str, Any]:
    return {
        "schema": "portfolio.execution/1",
        "contract_key": fill.contract_key,
        "quantity": fill.quantity,
        "premium": str(fill.premium),
        "execution_id": fill.execution_id,
        "source_sequence": fill.source_sequence,
        "order_id": fill.order_id,
        "stamp": _stamp_to(fill.stamp),
        "fees": [
            {"kind": f.kind, "amount": str(f.amount), "currency": f.currency} for f in fill.fees
        ],
    }


def execution_from_dict(value: Any) -> Execution:
    _fields(value, ["schema", *(f.name for f in fields(Execution))])
    if value["schema"] != "portfolio.execution/1":
        raise PortfolioError("unknown execution schema")
    fees = []
    for fee in value["fees"]:
        _fields(fee, ("kind", "amount", "currency"))
        fees.append(
            FeeRecord(kind=fee["kind"], amount=_money(fee["amount"]), currency=fee["currency"])
        )
    data = {k: v for k, v in value.items() if k not in ("schema", "premium", "stamp", "fees")}
    return Execution(
        **data,
        premium=_money(value["premium"]),
        stamp=_stamp_from(value["stamp"]),
        fees=tuple(fees),
    )


def position_to_dict(position: Position) -> dict[str, Any]:
    return {
        "schema": "portfolio.position/1",
        "legs": [
            {
                "contract": contract_to_dict(x.contract),
                "quantity": x.quantity,
                "allocated_entry": str(x.allocated_entry)
                if x.allocated_entry is not None
                else None,
                "allocated_fees": str(x.allocated_fees),
                "leg_id": x.leg_id,
            }
            for x in position.legs
        ],
    }


def position_from_dict(value: Any) -> Position:
    _fields(value, ("schema", "legs"))
    if value["schema"] != "portfolio.position/1":
        raise PortfolioError("unknown position schema")
    legs = []
    for leg in value["legs"]:
        _fields(leg, (f.name for f in fields(OptionLeg)))
        legs.append(
            OptionLeg(
                contract=contract_from_dict(leg["contract"]),
                quantity=leg["quantity"],
                allocated_entry=_money(leg["allocated_entry"])
                if leg["allocated_entry"] is not None
                else None,
                allocated_fees=_money(leg["allocated_fees"]),
                leg_id=leg["leg_id"],
            )
        )
    return Position(legs=tuple(legs))


def trade_to_dict(trade: Trade) -> dict[str, Any]:
    return {
        "schema": "portfolio.trade/1",
        "label": trade.label,
        "currency": trade.currency,
        "completeness": trade.completeness,
        "inventory_record_id": trade.inventory_record_id,
        "contracts": {k: contract_to_dict(c) for k, c in trade.contracts.items()},
        "executions": [execution_to_dict(x) for x in trade.executions],
    }


def trade_from_dict(value: Any) -> Trade:
    _fields(value, ["schema", *(f.name for f in fields(Trade))])
    if value["schema"] != "portfolio.trade/1":
        raise PortfolioError("unknown trade schema")
    return Trade(
        label=value["label"],
        currency=value["currency"],
        completeness=value["completeness"],
        inventory_record_id=value["inventory_record_id"],
        contracts={k: contract_from_dict(v) for k, v in value["contracts"].items()},
        executions=tuple(execution_from_dict(x) for x in value["executions"]),
    )


def snapshot_to_dict(snapshot: Any) -> Any:
    return {
        "schema": "portfolio.snapshot/1",
        **{
            f.name: position_to_dict(snapshot.position)
            if f.name == "position"
            else [
                {
                    "contract_key": x.contract_key,
                    "execution_id": x.execution_id,
                    "quantity": x.quantity,
                    "premium": str(x.premium),
                    "opening_fees": str(x.opening_fees),
                    "basis": str(x.basis),
                }
                for x in snapshot.lots
            ]
            if f.name == "lots"
            else str(getattr(snapshot, f.name))
            if f.name in ("cumulative_cash", "realized", "paid_fees")
            and getattr(snapshot, f.name) is not None
            else getattr(snapshot, f.name).isoformat()
            if f.name in ("valuation_time", "knowledge_cutoff")
            else getattr(snapshot, f.name)
            for f in fields(snapshot)
        },
    }


def snapshot_from_dict(value: Any) -> Any:
    from .lots import OpenLot
    from .position import PositionSnapshot

    _fields(value, ["schema", *(f.name for f in fields(PositionSnapshot))])
    if value["schema"] != "portfolio.snapshot/1":
        raise PortfolioError("unknown snapshot schema")
    data = {k: v for k, v in value.items() if k != "schema"}
    data["position"] = position_from_dict(data["position"])
    lots = []
    for lot in data["lots"]:
        _fields(lot, (f.name for f in fields(OpenLot)))
        lots.append(
            OpenLot(
                **{
                    k: _money(v) if k in ("premium", "opening_fees", "basis") else v
                    for k, v in lot.items()
                }
            )
        )
    data["lots"] = tuple(lots)
    for k in ("valuation_time", "knowledge_cutoff"):
        data[k] = datetime.fromisoformat(data[k])
    for k in ("cumulative_cash", "realized", "paid_fees"):
        if data[k] is not None:
            data[k] = _money(data[k])
    return PositionSnapshot(**data)


_TO: dict[type, Callable[..., dict[str, Any]]] = {
    ListedContract: contract_to_dict,
    Execution: execution_to_dict,
    Position: position_to_dict,
    Trade: trade_to_dict,
}
_FROM: dict[str, Callable[..., Any]] = {
    "portfolio.contract/1": contract_from_dict,
    "portfolio.execution/1": execution_from_dict,
    "portfolio.position/1": position_from_dict,
    "portfolio.trade/1": trade_from_dict,
    "portfolio.snapshot/1": snapshot_from_dict,
}


def dumps(value: Any) -> str:
    from .position import PositionSnapshot

    codec = snapshot_to_dict if isinstance(value, PositionSnapshot) else _TO.get(type(value))
    if codec is None:
        raise PortfolioError("unsupported portfolio codec type")
    return json.dumps(codec(value), sort_keys=True, allow_nan=False)


def loads(value: Any) -> Any:
    data = load_json(value)
    if not isinstance(data, dict) or data.get("schema") not in _FROM:
        raise PortfolioError("unknown portfolio schema")
    return _FROM[data["schema"]](data)


def contract_record(contract: Any, *, record_id: Any) -> Any:
    return ObservationRecord(
        record_id=record_id,
        kind="listed_contract",
        entities=(contract.key,),
        payload=contract_to_dict(contract),
        payload_schema="portfolio.contract/1",
        stamp=contract.stamp,
    )


def execution_record(fill: Any, *, record_id: Any) -> Any:
    return ObservationRecord(
        record_id=record_id,
        kind="execution",
        entities=(fill.execution_id, fill.contract_key),
        payload=execution_to_dict(fill),
        payload_schema="portfolio.execution/1",
        stamp=fill.stamp,
    )


def position_record(position: Any, *, record_id: Any, stamp: Any) -> Any:
    return ObservationRecord(
        record_id=record_id,
        kind="position",
        entities=(record_id,),
        payload=position_to_dict(position),
        payload_schema="portfolio.position/1",
        stamp=stamp,
    )
