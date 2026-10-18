"""Strict Decimal money and signed integer quantities."""

from __future__ import annotations

from decimal import Context, Decimal, localcontext
import re
from typing import Any

from .errors import PortfolioError

LEDGER_CONTEXT = Context(prec=50)
DECIMAL_TEXT = re.compile(r"[+-]?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?\Z")


def money(value: Any, *, nonnegative: bool = False) -> Decimal:
    if isinstance(value, str) and DECIMAL_TEXT.fullmatch(value):
        value = Decimal(value)
    if not isinstance(value, Decimal) or not value.is_finite():
        raise PortfolioError("money requires finite Decimal or decimal text, never binary floats")
    # The ledger uses an explicit precision and refuses inputs it cannot retain.
    if len(value.as_tuple().digits) > 50 or abs(value.adjusted()) > 100:
        raise PortfolioError("money exceeds the declared ledger precision/domain")
    if nonnegative and value < 0:
        raise PortfolioError("amount must be nonnegative")
    return value


def quantity(value: Any, *, positive: bool = False) -> int:
    if type(value) is not int or value == 0 or abs(value) > 10**12 or (positive and value < 0):
        raise PortfolioError("quantity must be a nonzero integer (positive for a strategy)")
    return value


def allocate(total: Any, sizes: Any) -> tuple[Decimal, ...]:
    """Pro rata at precision 50; the last allocation owns the exact remainder."""
    total = money(total, nonnegative=True)
    sizes = tuple(sizes)
    if not sizes or any(type(s) is not int or s <= 0 for s in sizes):
        raise PortfolioError("allocation sizes must be positive integers")
    with localcontext(LEDGER_CONTEXT):
        denominator = sum(sizes)
        parts = [total * s / denominator for s in sizes[:-1]]
        return (*parts, total - sum(parts, Decimal(0)))
