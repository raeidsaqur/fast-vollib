"""Explicit calendar elapsed-time conventions and caller-assumed expiry resolution."""

from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, time, timezone
from zoneinfo import ZoneInfo

from .contracts import ListedContract
from .errors import PortfolioError


def _aware(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise PortfolioError("datetime must be timezone-aware")
    return value.astimezone(timezone.utc)


def year_fraction(start: date | datetime, end: date | datetime, *, convention: str) -> float:
    denominators = {"act_365": 365, "act_365_25": 365.25, "act_360": 360}
    if convention not in denominators:
        raise PortfolioError("unknown day count")
    if isinstance(start, datetime) and isinstance(end, datetime):
        days = (_aware(end) - _aware(start)).total_seconds() / 86400
    elif type(start) is date and type(end) is date:
        start_date = date(start.year, start.month, start.day)
        end_date = date(end.year, end.month, end.day)
        days = (end_date - start_date).days
    else:
        raise PortfolioError("start/end must both be dates or both aware instants")
    if days < 0:
        raise PortfolioError("negative elapsed time")
    return days / denominators[convention]


def resolve_expiry_instant(contract: ListedContract, *, policy: tuple[time, str]) -> ListedContract:
    """Resolve ``(local_time, IANA_zone)`` as a caller assumption, not a calendar.

    The caller must first check holidays and settlement sessions. Ambiguous and
    nonexistent wall times are refused, including times made invalid by DST.
    """
    if not isinstance(policy, tuple) or len(policy) != 2:
        raise PortfolioError("expiry policy requires (time, IANA timezone)")
    local_time, zone_name = policy
    if (
        not isinstance(local_time, time)
        or local_time.tzinfo is not None
        or not isinstance(zone_name, str)
    ):
        raise PortfolioError("policy needs a naive wall time and an IANA zone name")
    zone = ZoneInfo(zone_name)
    naive = datetime.combine(contract.expiry_date, local_time)
    candidates = {
        naive.replace(tzinfo=zone, fold=f).astimezone(timezone.utc)
        for f in (0, 1)
        if naive.replace(tzinfo=zone, fold=f)
        .astimezone(timezone.utc)
        .astimezone(zone)
        .replace(tzinfo=None)
        == naive
    }
    if len(candidates) != 1:
        raise PortfolioError("expiry wall time is ambiguous or nonexistent")
    return replace(
        contract,
        expiry_instant=candidates.pop(),
        expiry_policy=f"caller_wall_time/1:{local_time.isoformat()}:{zone_name}",
    )
