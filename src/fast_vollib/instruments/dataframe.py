"""The DataFrame bridge: explicit column mapping, nothing inferred.

Options books usually arrive as tables, and the temptation is to guess: look
for a column called ``strike``, fall back to ``K``, treat ``expiry`` as a
maturity if it looks numeric.  Guessing is how a book gets priced against the
wrong column and nobody finds out until the P&L is wrong, so this bridge does
none of it.  Every column is named by the caller, in the same explicit style
:func:`fast_vollib.price_dataframe` already uses.

Two boundaries are enforced here.

*Contracts only.*  Spot, rate, volatility, and observed prices are market
observations, not contract terms.  They stay out of the contract frame and go
to the pricing call as :class:`~fast_vollib.instruments.VanillaMarketInputs`.

*Homogeneous only.*  One batch is one kernel call, so a frame that mixes
instrument types is rejected rather than silently split; group first.

Validation delegates to :meth:`EuropeanOptionBatch.from_arrays`, so a frame and
the equivalent arrays produce identical columns and identical error messages --
with the row index of the offending row, which for a frame is the thing you
actually need.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .errors import InstrumentValidationError

if TYPE_CHECKING:
    import pandas as pd

    from .batch import EuropeanOptionBatch

__all__ = ["batch_from_frame", "batch_to_frame"]

#: Columns that describe the market rather than the contract. Naming one of
#: these as a contract column is a category error, so it is refused by name.
_MARKET_COLUMNS = frozenset(
    {
        "underlying_price",
        "spot",
        "forward",
        "rate",
        "riskfree_rate",
        "sigma",
        "volatility",
        "iv",
        "implied_volatility",
        "price",
        "dividend",
        "dividend_yield",
        "q",
    }
)


def _column(frame: pd.DataFrame, name: str, *, role: str) -> np.ndarray:
    if name not in frame.columns:
        available = ", ".join(repr(str(column)) for column in frame.columns)
        raise InstrumentValidationError(
            f"Column {name!r} (mapped to {role}) is not in the frame. Available: {available}."
        )
    if str(name).strip().lower() in _MARKET_COLUMNS:
        raise InstrumentValidationError(
            f"Column {name!r} looks like market data, not a contract term. Market "
            f"observations belong in VanillaMarketInputs, not in the contract frame."
        )
    return frame[name].to_numpy()


def batch_from_frame(
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
    """Build a :class:`~fast_vollib.instruments.EuropeanOptionBatch` from a frame.

    Parameters
    ----------
    frame : pandas.DataFrame
        One row per contract.  Row order is preserved; the index is not read.
    option_type_col : str
        Column of ``'call'`` / ``'put'`` / ``'c'`` / ``'p'``, any case.
    strike_col, maturity_col : str
        Numeric columns.  Maturity is a year fraction, never a date.
    underlier_col : str
        Column of underlier identifiers.
    notional_col : str, optional
        Numeric column; defaults to 1.0 for every row when omitted.
    instrument_id_col : str, optional
        Per-row record identity.
    asset_class, currency : optional
        Applied to every underlier reference; scalars, not columns, because
        they describe the book rather than the row.

    Returns
    -------
    EuropeanOptionBatch

    Raises
    ------
    InstrumentValidationError
        If a named column is absent, if a column names market data, or if any
        value fails validation -- reported with its row index.

    Examples
    --------
    >>> import pandas as pd
    >>> from fast_vollib.instruments import EuropeanOptionBatch
    >>> chain = pd.DataFrame(
    ...     {"cp": ["c", "p"], "K": [95.0, 105.0], "T": [0.25, 0.25], "sym": ["SPX", "SPX"]}
    ... )
    >>> batch = EuropeanOptionBatch.from_frame(
    ...     chain, option_type_col="cp", strike_col="K",
    ...     maturity_col="T", underlier_col="sym",
    ... )
    >>> batch.strike.tolist()
    [95.0, 105.0]
    """
    from .batch import EuropeanOptionBatch

    if frame.empty:
        raise InstrumentValidationError("Cannot build a batch from an empty frame.")

    return EuropeanOptionBatch.from_arrays(
        option_type=_column(frame, option_type_col, role="option_type"),
        strike=_column(frame, strike_col, role="strike"),
        maturity=_column(frame, maturity_col, role="maturity"),
        underlier=_column(frame, underlier_col, role="underlier"),
        notional=(1.0 if notional_col is None else _column(frame, notional_col, role="notional")),
        instrument_id=(
            None
            if instrument_id_col is None
            else _column(frame, instrument_id_col, role="instrument_id")
        ),
        asset_class=asset_class,
        currency=currency,
    )


def batch_to_frame(batch: EuropeanOptionBatch) -> pd.DataFrame:
    """Render a batch as a DataFrame, one row per contract, order preserved.

    Column names are the batch's own attribute names, so the frame round-trips
    back through :func:`batch_from_frame` without a second naming convention to
    remember.  Unset optional strings come back as ``None`` rather than as the
    empty-string sentinel the columns use internally.
    """
    import pandas as pd

    def optional(column: np.ndarray) -> list[str | None]:
        return [value or None for value in column.tolist()]

    return pd.DataFrame(
        {
            "instrument_id": optional(batch.instrument_id),
            "underlier": batch.underlier_identifier.tolist(),
            "asset_class": optional(batch.underlier_asset_class),
            "currency": optional(batch.underlier_currency),
            "option_type": ["call" if flag == "c" else "put" for flag in batch.flag.tolist()],
            "strike": batch.strike.tolist(),
            "maturity": batch.maturity.tolist(),
            "settlement": batch.settlement.tolist(),
            "notional": batch.notional.tolist(),
        }
    )
