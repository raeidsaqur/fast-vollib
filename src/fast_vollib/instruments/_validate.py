"""One validation core, shared by contracts and batches.

Scalar contracts and columnar batches must agree on what a valid instrument
term is -- otherwise a book would validate differently depending on whether it
arrived as objects or as arrays.  Both therefore reduce to the checks in this
module: the scalar helpers here, and their vectorized twins in
:mod:`fast_vollib.instruments.batch`, which reuse the same predicates and the
same message wording with a row index attached.

Not public API.
"""

from __future__ import annotations

from enum import Enum
from typing import TypeVar

import numpy as np

from .enums import OptionType
from .errors import InstrumentValidationError

_E = TypeVar("_E", bound=Enum)

__all__ = [
    "coerce_underlier",
    "ensure_enum",
    "ensure_finite_float",
    "ensure_identifier",
    "ensure_non_negative",
    "ensure_nonzero",
    "ensure_optional_currency",
    "ensure_optional_identifier",
    "ensure_positive",
    "parse_option_type",
]

# Real scalar types accepted for a contract term.  ``bool`` is excluded on
# purpose even though it is an ``int`` subclass: ``strike=True`` is a mistake,
# not a strike of 1.0.  numpy 0-d arrays are excluded too -- invariant: a
# contract holds no arrays at all, not even shapeless ones.
_REAL_SCALARS = (int, float, np.integer, np.floating)


def _valid_values(enum_cls: type[Enum]) -> str:
    return ", ".join(repr(member.value) for member in enum_cls)


def ensure_enum(value: object, enum_cls: type[_E], *, field: str) -> _E:
    """Coerce ``value`` to a member of ``enum_cls``.

    Accepts a member of the enum or its canonical string value.  Any other
    input is an error naming the field and listing the admissible values.
    """
    if isinstance(value, enum_cls):
        return value
    if isinstance(value, str):
        try:
            return enum_cls(value)
        except ValueError:
            pass
    raise InstrumentValidationError(
        f"{field} must be one of {_valid_values(enum_cls)} "
        f"(or the corresponding {enum_cls.__name__} member); got {value!r}."
    )


def parse_option_type(value: object, *, field: str = "option_type") -> OptionType:
    """Coerce ``value`` to an :class:`~fast_vollib.instruments.OptionType`.

    Beyond the canonical ``'call'`` / ``'put'`` spellings this accepts the
    single-character ``'c'`` / ``'p'`` flags the pricing kernels use, in either
    case, matching ``fast_vollib.utils.broadcast.preprocess_flags``.  The
    serialization codec deliberately does *not* accept the short forms: the
    wire format has one spelling per value.

    Examples
    --------
    >>> from fast_vollib.instruments._validate import parse_option_type
    >>> parse_option_type("C")
    <OptionType.CALL: 'call'>
    >>> parse_option_type("put")
    <OptionType.PUT: 'put'>
    """
    if isinstance(value, OptionType):
        return value
    if isinstance(value, str):
        lowered = value.lower()
        if lowered == "c":
            return OptionType.CALL
        if lowered == "p":
            return OptionType.PUT
        try:
            return OptionType(lowered)
        except ValueError:
            pass
    raise InstrumentValidationError(
        f"{field} must be one of 'call', 'put', 'c', 'p' (or an OptionType member); got {value!r}."
    )


def ensure_identifier(value: object, *, field: str) -> str:
    """Return ``value`` as a non-empty trimmed identifier string."""
    if not isinstance(value, str):
        raise InstrumentValidationError(f"{field} must be a string; got {type(value).__name__}.")
    trimmed = value.strip()
    if not trimmed:
        raise InstrumentValidationError(f"{field} must be a non-empty string.")
    return trimmed


def ensure_optional_identifier(value: object, *, field: str) -> str | None:
    """Like :func:`ensure_identifier`, but ``None`` passes through."""
    if value is None:
        return None
    return ensure_identifier(value, field=field)


def ensure_optional_currency(value: object, *, field: str) -> str | None:
    """Return ``value`` as an upper-cased currency code, or ``None``.

    Kept as a plain string rather than an enum because currency lists evolve
    independently of this library. A closed enum would reject new codes until
    the library's vocabulary was updated.
    """
    if value is None:
        return None
    return ensure_identifier(value, field=field).upper()


def ensure_finite_float(value: object, *, field: str) -> float:
    """Return ``value`` as a finite Python ``float``.

    Rejects booleans, complex numbers, arrays, NaN, and the infinities.
    """
    if isinstance(value, (bool, np.bool_)):
        raise InstrumentValidationError(f"{field} must be a real number, not a bool.")
    if not isinstance(value, _REAL_SCALARS):
        raise InstrumentValidationError(
            f"{field} must be a real scalar number; got {type(value).__name__}."
        )
    as_float = float(value)
    if not np.isfinite(as_float):
        raise InstrumentValidationError(f"{field} must be finite; got {as_float!r}.")
    return as_float


def ensure_positive(value: object, *, field: str) -> float:
    """Return ``value`` as a finite ``float`` that is strictly positive."""
    as_float = ensure_finite_float(value, field=field)
    if as_float <= 0.0:
        raise InstrumentValidationError(f"{field} must be strictly positive; got {as_float!r}.")
    return as_float


def ensure_non_negative(value: object, *, field: str) -> float:
    """Return ``value`` as a finite ``float`` that is zero or positive."""
    as_float = ensure_finite_float(value, field=field)
    if as_float < 0.0:
        raise InstrumentValidationError(f"{field} must be non-negative; got {as_float!r}.")
    return as_float


def ensure_nonzero(value: object, *, field: str) -> float:
    """Return ``value`` as a finite, non-zero ``float``.

    Used for ``notional``: zero would make the instrument economically empty
    and would divide by zero when an observed price is converted to unit terms
    before implied-volatility inversion.
    """
    as_float = ensure_finite_float(value, field=field)
    if as_float == 0.0:
        raise InstrumentValidationError(f"{field} must be non-zero.")
    return as_float


def coerce_underlier(value: object, *, field: str = "underlier"):
    """Normalize an underlier argument to an :class:`InstrumentRef`.

    Accepts a reference, an :class:`~fast_vollib.instruments.Asset` (converted
    via :meth:`~fast_vollib.instruments.Asset.ref`), or a bare identifier
    string.  All three collapse to a reference, so a contract never holds an
    edge into an object graph -- which is what keeps equality structural and a
    serialized record self-contained.
    """
    from .base import Asset, InstrumentRef

    if isinstance(value, InstrumentRef):
        return value
    if isinstance(value, Asset):
        return value.ref()
    if isinstance(value, str):
        return InstrumentRef(identifier=value)
    raise InstrumentValidationError(
        f"{field} must be an InstrumentRef, an Asset, or an identifier string; "
        f"got {type(value).__name__}."
    )
