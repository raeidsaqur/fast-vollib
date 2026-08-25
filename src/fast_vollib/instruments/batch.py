"""Columnar batches: the unit of execution.

Scalar contracts provide identity -- a hashable, serializable value
that means one thing.  They are the wrong shape for evaluation.  fast-vollib's
whole point is that pricing a hundred thousand options is one fused kernel
call, and a list of a hundred thousand objects is a Python loop wearing a
costume.

So the two shapes are separate on purpose.  :class:`EuropeanOptionBatch` is a
homogeneous, columnar container over host NumPy arrays that maps directly onto
one kernel call.  Its columns are exactly what the kernels take.

Three constructors, one validation core
---------------------------------------
:meth:`~EuropeanOptionBatch.from_arrays` is the canonical high-throughput path:
columns in, columns out, with no per-row Python object built anywhere -- not an
option, not even an underlier reference.  A test asserts this by counting
constructor calls, because "vectorized" is easy to claim and easy to lose.

:meth:`~EuropeanOptionBatch.from_frame` is a strict mapping front end that
delegates to it.  Every column is named explicitly; nothing is guessed from
column names, and market data is not admitted -- a frame of contracts describes
contracts.

:meth:`~EuropeanOptionBatch.from_instruments` is the bridge from scalar
objects, for the case where you have them already.

All three normalize through the same code, so equivalent data produces equal
columns whichever door it came in by, and all three report an invalid value by
row index rather than as a whole-array complaint.

Equality
--------
Batches deliberately have no ``__eq__``.  For a container of arrays, ``a == b``
has two defensible meanings -- one bool, or an elementwise mask -- and
silently picking either produces wrong control flow in code that expected the
other.  Use :meth:`~EuropeanOptionBatch.equals` for the whole-batch question.
Batches are identity-hashable, which is what caching them wants anyway.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Iterable, Sequence

import numpy as np

from ._validate import ensure_enum
from .base import Asset, InstrumentRef
from .enums import OptionType, PricingModel, SettlementType
from .errors import InstrumentValidationError
from .options import EuropeanOption

if TYPE_CHECKING:
    import pandas as pd

    from ..types import ModelLiteral
    from .market import VanillaMarketInputs

__all__ = [
    "EuropeanOptionBatch",
    "forward_price",
    "log_moneyness",
    "moneyness",
    "time_to_maturity",
]

#: Sentinel for "not specified" inside a string column. Identifiers are
#: required to be non-empty, so the empty string is unambiguous and lets the
#: whole column stay a fixed-width NumPy array instead of a Python object array.
_UNSET = ""

_FLAG_BY_SPELLING = {
    "c": "c",
    "call": "c",
    "p": "p",
    "put": "p",
}


def _readonly(array: np.ndarray) -> np.ndarray:
    array.flags.writeable = False
    return array


def _unwrap_enums(value: object) -> object:
    """Replace enum members with their string values before array conversion.

    ``class X(str, Enum)`` members are strings, so NumPy sizes a ``<U`` array
    from ``len(member)`` but fills it from ``str(member)`` -- which is
    ``"SettlementType.CASH"``, not ``"cash"``. The result is a silently
    truncated column. Unwrapping here means every constructor is safe from it,
    rather than each column normalizer having to remember.
    """
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            return np.array(
                [item.value if isinstance(item, Enum) else item for item in value.tolist()],
                dtype=object,
            )
        return value
    if isinstance(value, (list, tuple)):
        return [item.value if isinstance(item, Enum) else item for item in value]
    return value


def _as_1d(value: object) -> np.ndarray:
    """Coerce any accepted column input to a 1-D NumPy array."""
    from ..utils.broadcast import to_numpy

    array = np.atleast_1d(to_numpy(_unwrap_enums(value)))
    if array.ndim != 1:
        raise InstrumentValidationError(
            f"Batch columns must be one-dimensional; got an array of shape {array.shape}."
        )
    return array


def _broadcast_to(array: np.ndarray, size: int, *, column: str) -> np.ndarray:
    if array.size == size:
        return array
    if array.size == 1:
        return np.repeat(array, size)
    raise InstrumentValidationError(
        f"Batch column {column!r} has length {array.size}, which is neither 1 "
        f"(a broadcast scalar) nor {size} (the batch length)."
    )


def _first_bad_row(mask: np.ndarray) -> int:
    return int(np.argmax(mask))


def _check_numeric(
    array: np.ndarray,
    *,
    column: str,
    positive: bool = False,
    non_negative: bool = False,
    non_zero: bool = False,
) -> np.ndarray:
    if array.dtype == np.bool_:
        raise InstrumentValidationError(f"Batch column {column!r} must be numeric, not boolean.")
    try:
        values = array.astype(np.float64, copy=False)
    except (TypeError, ValueError) as exc:
        raise InstrumentValidationError(
            f"Batch column {column!r} must be numeric; got dtype {array.dtype}."
        ) from exc
    bad = ~np.isfinite(values)
    if bad.any():
        row = _first_bad_row(bad)
        raise InstrumentValidationError(
            f"Batch column {column!r} must be finite; row {row} is {values[row]!r}."
        )
    if positive:
        bad = values <= 0.0
        if bad.any():
            row = _first_bad_row(bad)
            raise InstrumentValidationError(
                f"Batch column {column!r} must be strictly positive; row {row} is {values[row]!r}."
            )
    if non_negative:
        bad = values < 0.0
        if bad.any():
            row = _first_bad_row(bad)
            raise InstrumentValidationError(
                f"Batch column {column!r} must be non-negative; row {row} is {values[row]!r}."
            )
    if non_zero:
        bad = values == 0.0
        if bad.any():
            raise InstrumentValidationError(
                f"Batch column {column!r} must be non-zero; row {_first_bad_row(bad)} is 0.0."
            )
    return _readonly(np.ascontiguousarray(values))


def _as_text(values: np.ndarray, *, column: str, allow_null: bool = False) -> np.ndarray:
    """Validate a column as strings and return it as a fixed-width text array.

    Vectorized rather than looped: the array constructors exist so that a
    100k-row book does not pay Python per row, and a validation loop would give
    that back. Non-string entries are found by round-tripping through ``str``
    and comparing -- ``'1' != 1`` -- which numpy does elementwise in C.
    """
    if values.dtype.kind in {"U", "S"}:
        return values.astype("U", copy=False)
    if values.dtype.kind != "O":
        raise InstrumentValidationError(
            f"Batch column {column!r} must hold strings; got dtype {values.dtype}."
        )

    import pandas as pd

    null_mask = np.asarray(pd.isna(values), dtype=bool)
    if null_mask.any() and not allow_null:
        raise InstrumentValidationError(
            f"Batch column {column!r} must be a non-empty string; "
            f"row {_first_bad_row(null_mask)} is null."
        )
    cleaned = np.where(null_mask, _UNSET, values)
    text = cleaned.astype(str)
    mismatch = (text != cleaned) & ~null_mask
    if mismatch.any():
        row = _first_bad_row(mismatch)
        raise InstrumentValidationError(
            f"Batch column {column!r} must hold strings; row {row} is {values[row]!r}."
        )
    return text


def _normalize_flags(values: np.ndarray) -> np.ndarray:
    """Map an option-type column onto the kernels' ``'c'`` / ``'p'`` flags.

    Accepts exactly what the scalar constructor accepts -- ``'call'``,
    ``'put'``, ``'c'``, ``'p'``, in any case, or ``OptionType`` members -- so
    the two paths cannot disagree about what is valid.
    """
    text = _as_text(values, column="option_type")
    lowered = np.char.lower(text)
    is_call = np.isin(lowered, ("c", "call"))
    is_put = np.isin(lowered, ("p", "put"))
    bad = ~(is_call | is_put)
    if bad.any():
        row = _first_bad_row(bad)
        raise InstrumentValidationError(
            f"Batch column 'option_type' must be one of 'call', 'put', 'c', 'p' "
            f"(or an OptionType member); row {row} is {values[row]!r}."
        )
    return _readonly(np.where(is_call, "c", "p").astype("<U1"))


def _normalize_settlement(values: np.ndarray) -> np.ndarray:
    text = _as_text(values, column="settlement")
    valid_values = tuple(member.value for member in SettlementType)
    bad = ~np.isin(text, valid_values)
    if bad.any():
        row = _first_bad_row(bad)
        raise InstrumentValidationError(
            f"Batch column 'settlement' must be one of "
            f"{', '.join(repr(v) for v in valid_values)}; row {row} is {values[row]!r}."
        )
    return _readonly(text.astype("<U8"))


def _normalize_identifiers(values: np.ndarray, *, column: str, optional: bool) -> np.ndarray:
    text = _as_text(values, column=column, allow_null=optional)
    trimmed = np.char.strip(text)
    if not optional:
        blank = trimmed == _UNSET
        if blank.any():
            raise InstrumentValidationError(
                f"Batch column {column!r} must be a non-empty string; "
                f"row {_first_bad_row(blank)} is blank."
            )
    return _readonly(trimmed)


def _underlier_columns(
    underlier: object, size: int, *, asset_class: object, currency: object
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split an underlier argument into three string columns.

    Storing identifiers, asset classes, and currencies as columns rather than
    as ``InstrumentRef`` objects is what keeps the array path free of per-row
    allocation. References are rebuilt on demand by
    :attr:`EuropeanOptionBatch.underliers`.
    """
    identifiers: object
    classes: object = asset_class
    currencies: object = currency

    if isinstance(underlier, (InstrumentRef, Asset)):
        ref = underlier.ref() if isinstance(underlier, Asset) else underlier
        identifiers = ref.identifier
        classes = ref.asset_class if asset_class is None else asset_class
        currencies = ref.currency if currency is None else currency
    elif isinstance(underlier, str):
        identifiers = underlier
    elif isinstance(underlier, np.ndarray) or isinstance(underlier, Sequence):
        items = list(underlier)
        if items and all(isinstance(item, (InstrumentRef, Asset)) for item in items):
            refs = [item.ref() if isinstance(item, Asset) else item for item in items]
            identifiers = np.array([ref.identifier for ref in refs], dtype=object)
            if asset_class is None:
                classes = np.array(
                    [_UNSET if r.asset_class is None else r.asset_class.value for r in refs],
                    dtype=object,
                )
            if currency is None:
                currencies = np.array(
                    [_UNSET if r.currency is None else r.currency for r in refs], dtype=object
                )
        else:
            identifiers = underlier
    else:
        raise InstrumentValidationError(
            f"Batch column 'underlier' must be an InstrumentRef, an Asset, an identifier "
            f"string, or a sequence of those; got {type(underlier).__name__}."
        )

    identifier_column = _normalize_identifiers(
        _broadcast_to(_as_1d(identifiers), size, column="underlier"),
        column="underlier",
        optional=False,
    )
    class_column = _normalize_enum_column(classes, size, column="asset_class")
    currency_column = _normalize_currency_column(currencies, size)
    return identifier_column, class_column, currency_column


def _normalize_enum_column(value: object, size: int, *, column: str) -> np.ndarray:
    """An optional asset-class column; the empty string means unspecified."""
    from .enums import AssetClass

    if value is None:
        return _readonly(np.full(size, _UNSET, dtype="<U16"))
    raw = _broadcast_to(_as_1d(value), size, column=column)
    text = np.char.strip(_as_text(raw, column=column, allow_null=True))
    valid_values = tuple(member.value for member in AssetClass)
    bad = ~np.isin(text, valid_values) & (text != _UNSET)
    if bad.any():
        row = _first_bad_row(bad)
        raise InstrumentValidationError(
            f"Batch column {column!r} must be one of "
            f"{', '.join(repr(v) for v in valid_values)}; row {row} is {raw[row]!r}."
        )
    return _readonly(text.astype("<U16"))


def _normalize_currency_column(value: object, size: int) -> np.ndarray:
    """An optional currency column, upper-cased; the empty string is unspecified."""
    if value is None:
        return _readonly(np.full(size, _UNSET, dtype="<U8"))
    raw = _broadcast_to(_as_1d(value), size, column="currency")
    text = np.char.upper(np.char.strip(_as_text(raw, column="currency", allow_null=True)))
    return _readonly(text.astype("<U8"))


def _batch_length(*columns: np.ndarray) -> int:
    sizes = {column.size for column in columns if column.size != 1}
    if len(sizes) > 1:
        raise InstrumentValidationError(
            f"Batch columns have inconsistent lengths {sorted(sizes)}; every column must be "
            f"either length 1 (broadcast) or the batch length."
        )
    return sizes.pop() if sizes else 1


@dataclass(frozen=True, slots=True, eq=False)
class EuropeanOptionBatch:
    """A homogeneous, columnar book of European options.

    Attributes
    ----------
    flag : numpy.ndarray
        ``'c'`` / ``'p'``, in the kernels' own convention.
    strike, maturity, notional : numpy.ndarray
        Float64 term columns.
    settlement : numpy.ndarray
        Settlement type per row, as its canonical string.
    underlier_identifier, underlier_asset_class, underlier_currency : numpy.ndarray
        The underlier reference, held as columns.  An empty string means
        unspecified; identifiers are required to be non-empty, so there is no
        ambiguity.
    instrument_id : numpy.ndarray
        Per-row record identity; an empty string means unset.

    Notes
    -----
    All columns are read-only views: a batch is an execution object, but it is
    not a mutable one, and in-place edits would silently bypass validation.

    Build one with :meth:`from_arrays`, :meth:`from_frame`, or
    :meth:`from_instruments`.  Constructing the dataclass directly bypasses
    validation and is not supported.

    Examples
    --------
    >>> import numpy as np
    >>> from fast_vollib.instruments import EuropeanOptionBatch
    >>> batch = EuropeanOptionBatch.from_arrays(
    ...     option_type=["call", "put"], strike=[100.0, 110.0],
    ...     maturity=0.5, underlier="ACME",
    ... )
    >>> len(batch)
    2
    >>> batch.flag.tolist(), batch.maturity.tolist()
    (['c', 'p'], [0.5, 0.5])
    """

    flag: np.ndarray
    strike: np.ndarray
    maturity: np.ndarray
    notional: np.ndarray
    settlement: np.ndarray
    underlier_identifier: np.ndarray
    underlier_asset_class: np.ndarray
    underlier_currency: np.ndarray
    instrument_id: np.ndarray

    # -- construction ---------------------------------------------------------

    @classmethod
    def from_arrays(
        cls,
        *,
        option_type: object,
        strike: object,
        maturity: object,
        underlier: object,
        notional: object = 1.0,
        instrument_id: object = None,
        settlement: object = SettlementType.CASH,
        asset_class: object = None,
        currency: object = None,
    ) -> EuropeanOptionBatch:
        """Build a batch from columns.  The canonical high-throughput path.

        Parameters
        ----------
        option_type : array-like of str
            ``'call'`` / ``'put'`` / ``'c'`` / ``'p'``, any case, or
            ``OptionType`` members.
        strike : array-like of float
            Strictly positive.
        maturity : array-like of float
            Year fractions; non-negative.
        underlier : InstrumentRef, Asset, str, or sequence of those
            A single value is broadcast across the batch.
        notional : array-like of float, default 1.0
            Non-zero; negative denotes a short position.
        instrument_id : array-like of str, optional
            Per-row record identity.
        settlement : SettlementType or str or array-like, default ``"cash"``
        asset_class, currency : optional
            Descriptive metadata for the underlier, when it is given as a bare
            identifier.  Explicit values override what a reference carries.

        Returns
        -------
        EuropeanOptionBatch

        Raises
        ------
        InstrumentValidationError
            With the offending row index, for any column that fails the same
            checks the scalar constructor applies.

        Notes
        -----
        Scalar inputs broadcast; every other column must be either length 1 or
        the batch length.  No per-row Python object is constructed at any point.

        Examples
        --------
        >>> import numpy as np
        >>> from fast_vollib.instruments import EuropeanOptionBatch
        >>> batch = EuropeanOptionBatch.from_arrays(
        ...     option_type=np.array(["c", "p", "c"]),
        ...     strike=np.array([95.0, 100.0, 105.0]),
        ...     maturity=np.array([0.25, 0.5, 1.0]),
        ...     underlier="SPX",
        ...     notional=100.0,
        ... )
        >>> batch.notional.tolist()
        [100.0, 100.0, 100.0]
        """
        option_type_raw = _as_1d(option_type)
        strike_raw = _as_1d(strike)
        maturity_raw = _as_1d(maturity)
        notional_raw = _as_1d(notional)
        settlement_raw = _as_1d(settlement)
        instrument_id_raw = _as_1d(_UNSET if instrument_id is None else instrument_id)

        size = _batch_length(
            option_type_raw,
            strike_raw,
            maturity_raw,
            notional_raw,
            settlement_raw,
            instrument_id_raw,
        )
        underlier_columns = _underlier_columns(
            underlier, size, asset_class=asset_class, currency=currency
        )
        size = max(size, underlier_columns[0].size)

        return cls(
            flag=_normalize_flags(_broadcast_to(option_type_raw, size, column="option_type")),
            strike=_check_numeric(
                _broadcast_to(strike_raw, size, column="strike"),
                column="strike",
                positive=True,
            ),
            maturity=_check_numeric(
                _broadcast_to(maturity_raw, size, column="maturity"),
                column="maturity",
                non_negative=True,
            ),
            notional=_check_numeric(
                _broadcast_to(notional_raw, size, column="notional"),
                column="notional",
                non_zero=True,
            ),
            settlement=_normalize_settlement(
                _broadcast_to(settlement_raw, size, column="settlement")
            ),
            underlier_identifier=_broadcast_to(underlier_columns[0], size, column="underlier"),
            underlier_asset_class=_broadcast_to(underlier_columns[1], size, column="asset_class"),
            underlier_currency=_broadcast_to(underlier_columns[2], size, column="currency"),
            instrument_id=_normalize_identifiers(
                _broadcast_to(instrument_id_raw, size, column="instrument_id"),
                column="instrument_id",
                optional=True,
            ),
        )

    @classmethod
    def from_instruments(cls, options: Iterable[EuropeanOption]) -> EuropeanOptionBatch:
        """Build a batch from scalar contracts, preserving their order.

        Parameters
        ----------
        options : iterable of EuropeanOption
            Must be non-empty and homogeneous.

        Raises
        ------
        InstrumentValidationError
            If the iterable is empty, or holds anything other than European
            options -- a mixed book has no single kernel call, so it is
            rejected rather than silently split.

        Examples
        --------
        >>> from fast_vollib.instruments import EuropeanOption, EuropeanOptionBatch
        >>> options = [
        ...     EuropeanOption(underlier="SPX", option_type="c", strike=k, maturity=0.5)
        ...     for k in (95.0, 105.0)
        ... ]
        >>> EuropeanOptionBatch.from_instruments(options).strike.tolist()
        [95.0, 105.0]
        """
        items = list(options)
        if not items:
            raise InstrumentValidationError("Cannot build a batch from an empty sequence.")
        wrong = [type(item).__name__ for item in items if type(item) is not EuropeanOption]
        if wrong:
            raise InstrumentValidationError(
                f"EuropeanOptionBatch is homogeneous; it cannot hold "
                f"{', '.join(sorted(set(wrong)))}. Group the book by instrument type first."
            )
        return cls.from_arrays(
            option_type=np.array([item.flag for item in items], dtype="<U1"),
            strike=np.array([item.strike for item in items], dtype=np.float64),
            maturity=np.array([item.maturity for item in items], dtype=np.float64),
            notional=np.array([item.notional for item in items], dtype=np.float64),
            settlement=np.array([item.settlement.value for item in items], dtype=object),
            instrument_id=np.array(
                [_UNSET if item.instrument_id is None else item.instrument_id for item in items],
                dtype=object,
            ),
            underlier=[item.underlier for item in items],
        )

    @classmethod
    def from_frame(
        cls,
        frame: pd.DataFrame,
        *,
        option_type_col: str,
        strike_col: str,
        maturity_col: str,
        underlier_col: str,
        notional_col: str | None = None,
        instrument_id_col: str | None = None,
        asset_class: object = None,
        currency: object = None,
    ) -> EuropeanOptionBatch:
        """Build a batch from an explicitly mapped DataFrame.

        Every column is named by the caller.  Nothing is inferred from column
        names, and market columns (spot, rate, volatility, observed price) are
        not accepted: a frame of contracts describes contracts, and the market
        goes to the pricing call as :class:`VanillaMarketInputs`.

        See :func:`fast_vollib.instruments.dataframe.batch_from_frame` for the
        parameter details.  Row order is preserved.
        """
        from .dataframe import batch_from_frame

        return batch_from_frame(
            frame,
            option_type_col=option_type_col,
            strike_col=strike_col,
            maturity_col=maturity_col,
            underlier_col=underlier_col,
            notional_col=notional_col,
            instrument_id_col=instrument_id_col,
            asset_class=asset_class,
            currency=currency,
        )

    # -- reading back ---------------------------------------------------------

    def __len__(self) -> int:
        return int(self.strike.size)

    def __repr__(self) -> str:
        underliers = np.unique(self.underlier_identifier)
        shown = ", ".join(underliers[:3].tolist()) + ("..." if underliers.size > 3 else "")
        return f"EuropeanOptionBatch(n={len(self)}, underliers=[{shown}])"

    @property
    def option_type(self) -> tuple[OptionType, ...]:
        """The option types, as enum members."""
        return tuple(
            OptionType.CALL if flag == "c" else OptionType.PUT for flag in self.flag.tolist()
        )

    @property
    def underliers(self) -> tuple[InstrumentRef, ...]:
        """The underlier references, rebuilt from the columns.

        Materialized on access rather than stored, which is what lets
        :meth:`from_arrays` stay free of per-row allocation.
        """
        return tuple(
            InstrumentRef(
                identifier=identifier,
                asset_class=asset_class or None,
                currency=currency or None,
            )
            for identifier, asset_class, currency in zip(
                self.underlier_identifier.tolist(),
                self.underlier_asset_class.tolist(),
                self.underlier_currency.tolist(),
            )
        )

    def instruments(self) -> tuple[EuropeanOption, ...]:
        """Materialize the batch back into scalar contracts, in order.

        The inverse of :meth:`from_instruments`.  This *does* allocate one
        object per row; it exists for inspection and serialization, not for
        the evaluation path.
        """
        return tuple(
            EuropeanOption(
                instrument_id=instrument_id or None,
                underlier=ref,
                option_type=OptionType.CALL if flag == "c" else OptionType.PUT,
                strike=strike,
                maturity=maturity,
                settlement=SettlementType(settlement),
                notional=notional,
            )
            for flag, strike, maturity, notional, settlement, ref, instrument_id in zip(
                self.flag.tolist(),
                self.strike.tolist(),
                self.maturity.tolist(),
                self.notional.tolist(),
                self.settlement.tolist(),
                self.underliers,
                self.instrument_id.tolist(),
            )
        )

    def to_frame(self) -> pd.DataFrame:
        """The batch as a DataFrame, one row per contract, order preserved.

        Round-trips through :meth:`from_frame` back to equal columns.
        """
        from .dataframe import batch_to_frame

        return batch_to_frame(self)

    def equals(self, other: object) -> bool:
        """Whether two batches hold identical normalized columns.

        Provided instead of ``__eq__`` -- see the module docstring.

        Examples
        --------
        >>> from fast_vollib.instruments import EuropeanOptionBatch
        >>> kwargs = dict(option_type="c", strike=100.0, maturity=1.0, underlier="ACME")
        >>> a = EuropeanOptionBatch.from_arrays(**kwargs)
        >>> a.equals(EuropeanOptionBatch.from_arrays(**kwargs))
        True
        >>> a == EuropeanOptionBatch.from_arrays(**kwargs)
        False
        """
        if not isinstance(other, EuropeanOptionBatch):
            return False
        return all(
            np.array_equal(getattr(self, name), getattr(other, name)) for name in _COLUMN_NAMES
        )


_COLUMN_NAMES: tuple[str, ...] = (
    "flag",
    "strike",
    "maturity",
    "notional",
    "settlement",
    "underlier_identifier",
    "underlier_asset_class",
    "underlier_currency",
    "instrument_id",
)


# --- coordinates --------------------------------------------------------------


def _maturity_of(instrument_or_batch: object) -> Any:
    maturity = getattr(instrument_or_batch, "maturity", None)
    if maturity is None:
        raise InstrumentValidationError(
            f"{type(instrument_or_batch).__name__} has no maturity to read."
        )
    return maturity


def _strike_of(instrument_or_batch: object) -> Any:
    strike = getattr(instrument_or_batch, "strike", None)
    if strike is None:
        raise InstrumentValidationError(
            f"{type(instrument_or_batch).__name__} has no strike; moneyness is "
            f"defined for options only."
        )
    return strike


def time_to_maturity(instrument_or_batch: object) -> Any:
    """Year fraction to maturity, for a contract or a whole batch.

    A coordinate rather than an attribute lookup so that batches and scalars
    read identically at the call site.

    Examples
    --------
    >>> from fast_vollib.instruments import EuropeanOption, time_to_maturity
    >>> time_to_maturity(
    ...     EuropeanOption(underlier="A", option_type="c", strike=1.0, maturity=0.75)
    ... )
    0.75
    """
    return _maturity_of(instrument_or_batch)


def forward_price(
    market: VanillaMarketInputs,
    maturity: Any,
    *,
    model: PricingModel | ModelLiteral,
) -> Any:
    """The forward implied by the market inputs under an explicit model.

    Under Black-76 the market's ``underlying`` *is* the forward.  Under
    Black-Scholes it is a spot and the forward is ``S·e^{rT}``; under
    Black-Scholes-Merton, ``S·e^{(r−q)T}``.  The model is required rather than
    inferred, because those three answers differ and nothing in the market
    object distinguishes them.

    Examples
    --------
    >>> from fast_vollib.instruments import VanillaMarketInputs, forward_price
    >>> market = VanillaMarketInputs(underlying=100.0, rate=0.05)
    >>> round(float(forward_price(market, 1.0, model="black_scholes")), 4)
    105.1271
    >>> forward_price(market, 1.0, model="black")
    100.0
    """
    resolved = ensure_enum(model, PricingModel, field="model")
    underlying = market.require("underlying", operation="compute a forward")
    if resolved is PricingModel.BLACK:
        return underlying
    rate = market.require("rate", operation="compute a forward")
    if resolved is PricingModel.BLACK_SCHOLES:
        return underlying * np.exp(np.asarray(rate) * np.asarray(maturity))
    dividend_yield = market.require(
        "dividend_yield", operation="compute a forward under black_scholes_merton"
    )
    drift = (np.asarray(rate) - np.asarray(dividend_yield)) * np.asarray(maturity)
    return underlying * np.exp(drift)


def moneyness(
    instrument_or_batch: object,
    market: VanillaMarketInputs,
    *,
    model: PricingModel | ModelLiteral,
) -> Any:
    """Forward moneyness ``K / F``.

    Defined against the forward, and in the same orientation as
    :mod:`fast_vollib.surface`, whose axis is ``k = log(K/F)`` -- so
    :func:`log_moneyness` is exactly the log of this, and a batch feeds an
    :class:`~fast_vollib.surface.IVSurface` without a sign flip.

    Examples
    --------
    >>> from fast_vollib.instruments import (
    ...     EuropeanOption, VanillaMarketInputs, moneyness,
    ... )
    >>> option = EuropeanOption(underlier="A", option_type="c", strike=110.0, maturity=1.0)
    >>> market = VanillaMarketInputs(underlying=100.0, rate=0.0)
    >>> float(moneyness(option, market, model="black_scholes"))
    1.1
    """
    forward = forward_price(market, _maturity_of(instrument_or_batch), model=model)
    return _strike_of(instrument_or_batch) / forward


def log_moneyness(
    instrument_or_batch: object,
    market: VanillaMarketInputs,
    *,
    model: PricingModel | ModelLiteral,
) -> Any:
    """Forward log-moneyness ``k = log(K / F)``.

    The coordinate :mod:`fast_vollib.surface` is parametrized in.

    Examples
    --------
    >>> from fast_vollib.instruments import (
    ...     EuropeanOption, VanillaMarketInputs, log_moneyness,
    ... )
    >>> option = EuropeanOption(underlier="A", option_type="c", strike=100.0, maturity=1.0)
    >>> market = VanillaMarketInputs(underlying=100.0, rate=0.0)
    >>> float(log_moneyness(option, market, model="black_scholes"))
    0.0
    """
    return np.log(moneyness(instrument_or_batch, market, model=model))
