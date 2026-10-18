"""Strict terminal report JSON with distinct missing, invalid and unbounded tags."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields
from datetime import date, datetime
from decimal import Decimal
from fractions import Fraction
import json
import math
from typing import Any

import numpy as np

from ..observations.serialization import load_json
from ..portfolio import ListedContract
from ..portfolio.serialization import contract_from_dict, contract_to_dict
from .errors import PnLError
from .evaluate import CoefficientConversion
from .metrics import PayoffMetrics
from .quotes import LegQuoteStatus, QuoteFlags
from .results import PnLProfile, ResultMeta


def _fields(value: Any, cls: Any) -> Any:
    if not isinstance(value, dict) or set(value) != {f.name for f in fields(cls)}:
        raise PnLError("unknown or missing result fields")


def _encode(value: Any) -> Any:
    if value is None:
        return {"status": "missing"}
    if isinstance(value, Decimal):
        if value.is_infinite():
            return {"status": "negative_unbounded" if value < 0 else "positive_unbounded"}
        if not value.is_finite():
            return {"status": "invalid"}
        return {"decimal": str(value)}
    if isinstance(value, Fraction):
        return {"rational": [str(value.numerator), str(value.denominator)]}
    if isinstance(value, datetime):
        return {"utc": value.isoformat()}
    if isinstance(value, date):
        return {"date": value.isoformat()}
    if isinstance(value, np.ndarray):
        return _encode(value.tolist())
    if isinstance(value, (tuple, list)):
        return [_encode(x) for x in value]
    if isinstance(value, ListedContract):
        return contract_to_dict(value)
    if hasattr(value, "__dataclass_fields__"):
        return {f.name: _encode(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Mapping):
        return {k: _encode(v) for k, v in value.items()}
    if isinstance(value, float) and not math.isfinite(value):
        return {
            "status": "invalid"
            if math.isnan(value)
            else "positive_unbounded"
            if value > 0
            else "negative_unbounded"
        }
    if isinstance(value, QuoteFlags):
        return int(value)
    if type(value) in (str, bool, int, float):
        return value
    raise PnLError("unsupported result value")


def _decode(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_decode(x) for x in value)
    if not isinstance(value, dict):
        return value
    if set(value) == {"status"}:
        status = value["status"]
        if status == "missing":
            return None
        if status == "invalid":
            return float("nan")
        if status in ("positive_unbounded", "negative_unbounded"):
            return Decimal("Infinity" if status == "positive_unbounded" else "-Infinity")
        raise PnLError("unknown numeric status")
    if set(value) == {"decimal"}:
        if not isinstance(value["decimal"], str):
            raise PnLError("decimal tag must contain text")
        out = Decimal(value["decimal"])
        if not out.is_finite():
            raise PnLError("nonfinite Decimal needs a status tag")
        return out
    if set(value) == {"rational"}:
        pair = value["rational"]
        if (
            not isinstance(pair, (list, tuple))
            or len(pair) != 2
            or any(not isinstance(x, str) for x in pair)
        ):
            raise PnLError("rational tag requires numerator/denominator text")
        return Fraction(int(pair[0]), int(pair[1]))
    if set(value) == {"utc"}:
        return datetime.fromisoformat(value["utc"])
    if set(value) == {"date"}:
        return date.fromisoformat(value["date"])
    return {k: _decode(v) for k, v in value.items()}


_NEW_META = {"quote_policy"}
_NEW_PROFILE = {"quote_details", "realized", "unrealized"}


def profile_to_dict(profile: PnLProfile) -> dict[str, Any]:
    data = _encode(profile)
    if profile.meta.schema_version == "pnl.result/1":
        # Preserve the original terminal wire contract, including its strict fields.
        for field in _NEW_PROFILE:
            data.pop(field)
        for field in _NEW_META:
            data["meta"].pop(field)
        return {"schema": "pnl.profile/1", "profile": data}
    if profile.meta.basis == "quote_mark":
        data["total"] = [
            [
                _encode(float(value)) if valid else {"status": "missing"}
                for value, valid in zip(row, mask)
            ]
            for row, mask in zip(profile.total, profile.valid)
        ]
        data["leg_values"] = [
            [
                [
                    _encode(float(value)) if valid else {"status": "missing"}
                    for value, valid in zip(row, mask)
                ]
                for row, mask in zip(horizon, hmask)
            ]
            for horizon, hmask in zip(profile.leg_values, profile.leg_valid)
        ]
    if profile.cumulative_cash is None:
        data["total"] = [[{"status": "missing"} for _ in row] for row in profile.total]
    return {
        "schema": "pnl.profile/2",
        "result_type": "european_proxy" if profile.meta.basis == "european_proxy" else "profile",
        "profile": data,
    }


def profile_from_dict(value: Any) -> PnLProfile:
    if not isinstance(value, dict) or value.get("schema") not in ("pnl.profile/1", "pnl.profile/2"):
        raise PnLError("unknown report schema")
    legacy = value["schema"] == "pnl.profile/1"
    if set(value) != ({"schema", "profile"} if legacy else {"schema", "profile", "result_type"}):
        raise PnLError("unknown or missing envelope fields")
    if not legacy and value["result_type"] not in ("profile", "european_proxy"):
        raise PnLError("unknown report result type")
    data = _decode(value["profile"])
    expected = {f.name for f in fields(PnLProfile)} - (_NEW_PROFILE if legacy else set())
    expected_meta = {f.name for f in fields(ResultMeta)} - (_NEW_META if legacy else set())
    if (
        not isinstance(data, dict)
        or set(data) != expected
        or not isinstance(data.get("meta"), dict)
        or set(data["meta"]) != expected_meta
    ):
        raise PnLError("unknown or missing report/metadata fields")
    if data["metrics"] is not None:
        _fields(data["metrics"], PayoffMetrics)
        data["metrics"] = PayoffMetrics(**data["metrics"])
    data["contracts"] = tuple(contract_from_dict(c) for c in data["contracts"])
    data["meta"] = ResultMeta(**data["meta"])
    if legacy != (data["meta"].schema_version == "pnl.result/1"):
        raise PnLError("envelope and result metadata versions disagree")
    errors = []
    for error in data["conversion_errors"]:
        _fields(error, CoefficientConversion)
        errors.append(CoefficientConversion(**error))
    data["conversion_errors"] = tuple(errors)
    if not legacy:
        details = []
        for detail in data["quote_details"]:
            _fields(detail, LegQuoteStatus)
            if type(detail["flags"]) is not int or detail["flags"] & ~127:
                raise PnLError("unknown quote status flags")
            detail["flags"] = QuoteFlags(detail["flags"])
            details.append(LegQuoteStatus(**detail))
        data["quote_details"] = tuple(details)
    if not legacy and value["result_type"] == "european_proxy":
        from .proxy import EuropeanProxyProfile

        return EuropeanProxyProfile(**data)
    return PnLProfile(**data)


def dumps(profile: Any) -> str:
    if not isinstance(profile, PnLProfile):
        from .bands import SurfaceSensitivityBand
        from .greeks_position import PositionGreeks
        from .probability import TerminalProfitProbability

        if isinstance(profile, (SurfaceSensitivityBand, PositionGreeks, TerminalProfitProbability)):
            return profile.to_json()
        raise PnLError("unsupported result type")
    return json.dumps(profile_to_dict(profile), sort_keys=True, allow_nan=False)


def loads(value: Any) -> Any:
    data = load_json(value)
    if isinstance(data, dict):
        if data.get("schema") == "pnl.surface_sensitivity/1":
            from .bands import band_from_dict

            return band_from_dict(data)
        if data.get("schema") == "pnl.position_greeks/1":
            from .greeks_position import greeks_from_dict

            return greeks_from_dict(data)
        if data.get("schema") == "pnl.terminal_probability/1":
            from .probability import probability_from_dict

            return probability_from_dict(data)
    return profile_from_dict(data)
