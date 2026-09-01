"""Scattered implied-volatility observations -- what a surface is fitted to.

A fitted surface is rarely delivered on a regular mesh.  A fitting pipeline
observes a handful of quotes per surface, predicts the implied volatility at
whatever coordinates the evaluation asks for, and hands back one row per point:
a surface label, a coordinate pair, and a value.  :class:`SurfaceObservations`
is that long-format record as a validated, **immutable** container, with the
grouping helpers the scoring functions need (per-surface subsets, per-maturity
smiles) and a round trip to the DataFrame / CSV layout the pipelines exchange.

Coordinates follow the convention of :mod:`fast_vollib.surface`: forward
log-moneyness ``k = log(K / F)`` and year-fraction maturity ``T``.  Optional
bid/ask quotes are *forward-normalised* prices, ``quote / (F * e^{-rT})``, the
scale on which the spread diagnostics compare a prediction to the market.
``NaN`` in ``iv`` (or ``bid`` / ``ask``) means "no quote" and is excluded from
every statistic rather than propagated.

Two properties are load-bearing for reproducible diagnostics:

* **Owned, read-only arrays.**  The constructor copies every input and marks
  the copy non-writable, so a caller mutating its own array afterwards cannot
  change a computed diagnostic.
* **No implicit alignment.**  A prediction is matched to a truth row by a
  stable ``(surface_id, point_id)`` key, or -- opt in -- by exact
  ``(surface_id, T, k)`` equality when those keys are unique on both sides.
  :func:`align_predictions` never guesses from row order, float tolerance, or
  occurrence order.  Duplicate observations at one coordinate are legitimate
  and are kept as separate rows.

This container is the promoted form of what :mod:`fast_vollib.diagnostics`
called ``SurfaceQuotes``.  The class is still exported under that name from the
diagnostics package, which is where it was first published; the definition
lives here because fitting, forecasting, and generative models all consume it,
not only the diagnostics that once owned it.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from ._validate import owned_float_1d, owned_labels, read_only
from .errors import SurfaceTypeError, SurfaceValidationError
from .points import CoordinateConvention, SurfacePoints

if TYPE_CHECKING:  # pragma: no cover - typing only, keeps pandas out of import time
    import pandas as pd

__all__ = ["DEFAULT_COLUMNS", "SurfaceObservations", "align_predictions"]

#: Column names used by :meth:`SurfaceObservations.from_dataframe` /
#: :meth:`SurfaceObservations.to_dataframe` when the caller does not override
#: them -- the long-format layout of IV-surface fitting pipelines: one row per
#: observation, ``id`` labelling the surface a row belongs to.
DEFAULT_COLUMNS: Mapping[str, str] = {
    "surface_id": "id",
    "point_id": "point_id",
    "k": "logmoneyness",
    "T": "maturity",
    "iv": "iv",
    "bid": "bid",
    "ask": "ask",
    "is_call": "is_call",
    "weight": "weight",
    "price": "price",
}

#: Column name accepted on input for ``point_id`` when the canonical one is
#: absent.  Long-format quote files written before the container was promoted
#: out of the diagnostics package spell it this way.
LEGACY_POINT_ID_COLUMN = "quote_id"

_REQUIRED = ("k", "T", "iv")
_OPTIONAL = ("point_id", "bid", "ask", "is_call", "weight", "price")

#: Canonical strings accepted for ``is_call``, matched case-insensitively.
_TRUE_STRINGS = frozenset({"true", "call", "c"})
_FALSE_STRINGS = frozenset({"false", "put", "p"})

_ALIGN_KEYS = ("ids", "coordinates")


def _owned_is_call(value: Any, n: int) -> np.ndarray:
    """A strict boolean array: boolean dtype, numeric ``{0, 1}``, or canonical strings."""
    array = np.asarray(value)
    if array.ndim == 0:
        array = np.repeat(array, n)
    if array.ndim != 1:
        raise SurfaceValidationError(f"is_call must be one-dimensional; got shape {array.shape}.")
    if array.shape != (n,):
        raise SurfaceValidationError(f"is_call must have shape ({n},); got {array.shape}.")
    if array.dtype == np.bool_:
        return read_only(np.array(array, dtype=bool, copy=True))
    if np.issubdtype(array.dtype, np.integer) or np.issubdtype(array.dtype, np.floating):
        numeric = np.array(array, dtype=np.float64, copy=True)
        allowed = (numeric == 0.0) | (numeric == 1.0)
        if not bool(np.all(allowed)):
            bad = numeric[~allowed]
            raise SurfaceValidationError(
                f"Numeric is_call values must be exactly 0 or 1; got {np.unique(bad).tolist()[:5]}."
            )
        return read_only(numeric == 1.0)
    out = np.empty(n, dtype=bool)
    for index, item in enumerate(array.tolist()):
        if isinstance(item, np.generic):
            item = item.item()
        if isinstance(item, (bool, np.bool_)):
            out[index] = bool(item)
            continue
        if not isinstance(item, str):
            raise SurfaceTypeError(
                "is_call entries must be booleans, 0/1, or one of "
                "{true,false,call,put,c,p}; got " + repr(item) + "."
            )
        token = item.strip().lower()
        if token in _TRUE_STRINGS:
            out[index] = True
        elif token in _FALSE_STRINGS:
            out[index] = False
        else:
            raise SurfaceValidationError(
                "is_call strings must be one of {true,false,call,put,c,p} "
                f"(case-insensitive); got {item!r}."
            )
    return read_only(out)


@dataclass(frozen=True, slots=True)
class SurfaceObservations:
    """Scattered observations for one or many surfaces.

    Parameters
    ----------
    k, T, iv:
        Forward log-moneyness, maturity in years, and implied volatility, each
        shape ``(N,)``.  Every ``k`` is finite and every ``T`` is finite and
        strictly positive.  ``iv`` is either ``NaN`` (a missing observation) or
        a finite value ``>= 0``; an infinite ``iv`` is rejected.
    surface_id:
        Shape ``(N,)`` labels -- strings or integers -- grouping points into
        surfaces (a day, an asset-day, a sample id).  A scalar labels every
        row; ``None`` means a single surface labelled ``0``.
    point_id:
        Optional shape ``(N,)`` stable per-point labels, unique within each
        surface.  They are what :func:`align_predictions` keys on by default.
    bid, ask:
        Optional forward-normalised bid/ask, shape ``(N,)``, given together or
        not at all.  Both ``NaN`` on a row marks that row unquoted; exactly one
        ``NaN`` is an error.  Quoted values are finite, non-negative, and obey
        ``bid <= ask``.
    is_call:
        Optional per-point call/put flags.  Accepts boolean dtype, numeric
        ``{0, 1}``, or the canonical strings ``{true,false,call,put,c,p}``
        case-insensitively; arbitrary truthiness is rejected.  ``None`` prices
        every point as a call.
    weight:
        Optional non-negative fitting weights, shape ``(N,)``.  A calibrator
        that supports weighting uses them; one that does not says so in its
        capability metadata rather than ignoring them silently.
    price:
        Optional forward-normalised observed option price, shape ``(N,)``,
        ``NaN`` where absent.  Carried so a price-space error can be computed
        against what was actually traded rather than against a price re-derived
        from the observed implied volatility.
    convention:
        How the coordinates were produced; see
        :class:`~fast_vollib.surface.points.CoordinateConvention`.

    Notes
    -----
    Every stored array is an owned, read-only copy: mutating the arrays passed
    in cannot change a diagnostic computed from this object.  Duplicate rows at
    the same coordinate are legitimate repeated observations and are preserved.

    Examples
    --------
    >>> import numpy as np
    >>> from fast_vollib.surface import SurfaceObservations
    >>> q = SurfaceObservations(k=[-0.1, 0.0, 0.1], T=[0.5, 0.5, 0.5], iv=[0.22, 0.2, 0.21])
    >>> q.n, q.surface_ids()
    (3, [0])
    >>> q.k.flags.writeable
    False
    """

    k: Any
    T: Any
    iv: Any
    surface_id: Any = None
    point_id: Any = None
    bid: Any = None
    ask: Any = None
    is_call: Any = None
    weight: Any = None
    price: Any = None
    convention: CoordinateConvention = CoordinateConvention()
    _points: Any = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        k = owned_float_1d(self.k, "k")
        n = k.size
        T = owned_float_1d(self.T, "T", n)
        iv = owned_float_1d(self.iv, "iv", n)
        if T.size != n or iv.size != n:
            raise SurfaceValidationError(
                f"k, T, and iv must have the same length; got {n}, {T.size}, {iv.size}."
            )
        if not bool(np.all(np.isfinite(k))):
            raise SurfaceValidationError("k must be finite everywhere.")
        if not bool(np.all(np.isfinite(T))):
            raise SurfaceValidationError("T must be finite everywhere.")
        if n and not bool(np.all(T > 0.0)):
            raise SurfaceValidationError("T must be strictly positive everywhere.")
        observed = ~np.isnan(iv)
        if not bool(np.all(np.isfinite(iv[observed]))):
            raise SurfaceValidationError(
                "iv must be NaN (missing) or finite; infinities are rejected."
            )
        if not bool(np.all(iv[observed] >= 0.0)):
            raise SurfaceValidationError("Observed iv must be non-negative.")

        if self.surface_id is None:
            surface_id = read_only(np.zeros(n, dtype=np.int64))
        else:
            surface_id = owned_labels(self.surface_id, n, "surface_id")

        point_id = None
        if self.point_id is not None:
            point_id = owned_labels(self.point_id, n, "point_id")
            if _has_duplicate_pairs(surface_id, point_id):
                raise SurfaceValidationError("point_id must be unique within each surface_id.")

        bid = ask = None
        if (self.bid is None) != (self.ask is None):
            raise SurfaceValidationError("bid and ask must be given together.")
        if self.bid is not None:
            bid = owned_float_1d(self.bid, "bid", n)
            ask = owned_float_1d(self.ask, "ask", n)
            if bid.size != n or ask.size != n:
                raise SurfaceValidationError(f"bid and ask must have length {n}.")
            bid_missing = np.isnan(bid)
            ask_missing = np.isnan(ask)
            if bool(np.any(bid_missing != ask_missing)):
                raise SurfaceValidationError("bid and ask must be missing together on a row.")
            quoted = ~bid_missing
            if not bool(np.all(np.isfinite(bid[quoted]) & np.isfinite(ask[quoted]))):
                raise SurfaceValidationError(
                    "Quoted bid/ask must be finite; infinities are rejected."
                )
            if not bool(np.all((bid[quoted] >= 0.0) & (ask[quoted] >= 0.0))):
                raise SurfaceValidationError("Quoted bid/ask must be non-negative.")
            if bool(np.any(bid[quoted] > ask[quoted])):
                raise SurfaceValidationError("bid must not exceed ask where both are quoted.")

        is_call = None if self.is_call is None else _owned_is_call(self.is_call, n)

        weight = None
        if self.weight is not None:
            weight = owned_float_1d(self.weight, "weight", n)
            if weight.size != n:
                raise SurfaceValidationError(f"weight must have length {n}; got {weight.size}.")
            if not bool(np.all(np.isfinite(weight))):
                raise SurfaceValidationError(
                    "weight must be finite everywhere; a missing weight is 0, not NaN."
                )
            if not bool(np.all(weight >= 0.0)):
                raise SurfaceValidationError("weight must be non-negative.")

        price = None
        if self.price is not None:
            price = owned_float_1d(self.price, "price", n)
            if price.size != n:
                raise SurfaceValidationError(f"price must have length {n}; got {price.size}.")
            quoted_price = ~np.isnan(price)
            if not bool(np.all(np.isfinite(price[quoted_price]))):
                raise SurfaceValidationError(
                    "price must be NaN (missing) or finite; infinities are rejected."
                )
            if not bool(np.all(price[quoted_price] >= 0.0)):
                raise SurfaceValidationError("Observed price must be non-negative.")

        if not isinstance(self.convention, CoordinateConvention):
            raise SurfaceValidationError(
                f"convention must be a CoordinateConvention; got {type(self.convention).__name__}."
            )

        object.__setattr__(self, "k", k)
        object.__setattr__(self, "T", T)
        object.__setattr__(self, "iv", iv)
        object.__setattr__(self, "surface_id", surface_id)
        object.__setattr__(self, "point_id", point_id)
        object.__setattr__(self, "bid", bid)
        object.__setattr__(self, "ask", ask)
        object.__setattr__(self, "is_call", is_call)
        object.__setattr__(self, "weight", weight)
        object.__setattr__(self, "price", price)

    # -- size and structure --------------------------------------------------
    @property
    def n(self) -> int:
        """Number of rows (observed or not)."""
        return int(self.k.size)

    def __len__(self) -> int:
        return self.n

    @property
    def has_spread(self) -> bool:
        """Whether bid/ask quotes are carried."""
        return self.bid is not None

    @property
    def has_point_ids(self) -> bool:
        """Whether stable per-point labels are carried."""
        return self.point_id is not None

    @property
    def has_weights(self) -> bool:
        """Whether fitting weights are carried."""
        return self.weight is not None

    @property
    def has_prices(self) -> bool:
        """Whether observed forward-normalised prices are carried."""
        return self.price is not None

    @property
    def points(self) -> SurfacePoints:
        """The coordinates and identity of these rows, without their values.

        Built once and cached: a calibrator that hands its query domain to a
        definite surface should hand it *this*, so the surface is evaluated at
        exactly the rows it was fitted against.
        """
        cached = self._points
        if cached is None:
            cached = SurfacePoints(
                k=self.k,
                T=self.T,
                surface_id=self.surface_id,
                point_id=self.point_id,
                convention=self.convention,
            )
            object.__setattr__(self, "_points", cached)
        return cached

    def surface_ids(self) -> list[Any]:
        """Distinct surface labels in order of first appearance, as plain scalars."""
        return list(dict.fromkeys(self.surface_id.tolist()))

    def subset(self, index: Any) -> SurfaceObservations:
        """The rows selected by a boolean mask or an index array."""
        index = np.asarray(index)
        return SurfaceObservations(
            k=self.k[index],
            T=self.T[index],
            iv=self.iv[index],
            surface_id=self.surface_id[index],
            point_id=None if self.point_id is None else self.point_id[index],
            bid=None if self.bid is None else self.bid[index],
            ask=None if self.ask is None else self.ask[index],
            is_call=None if self.is_call is None else self.is_call[index],
            weight=None if self.weight is None else self.weight[index],
            price=None if self.price is None else self.price[index],
            convention=self.convention,
        )

    def surfaces(self) -> Iterator[tuple[Any, SurfaceObservations]]:
        """Yield ``(label, observations)`` per surface, in order of first appearance."""
        for label in self.surface_ids():
            yield label, self.subset(self.surface_id == label)

    def smiles(
        self, *, maturity_decimals: int = 6
    ) -> Iterator[tuple[Any, float, np.ndarray, np.ndarray]]:
        """Yield ``(label, T, k_sorted, iv_sorted)`` per ``(surface, maturity)`` smile.

        Maturities are bucketed on ``round(T, maturity_decimals)``; unobserved
        (``NaN``) rows are dropped; ``k`` is sorted ascending with a stable sort
        so equal strikes keep their input order.
        """
        for label, quotes in self.surfaces():
            observed = ~np.isnan(quotes.iv)
            bucket = np.round(quotes.T, maturity_decimals)
            for value in np.unique(bucket[observed]):
                mask = observed & (bucket == value)
                order = np.argsort(quotes.k[mask], kind="stable")
                yield label, float(value), quotes.k[mask][order], quotes.iv[mask][order]

    # -- constructors and conversions ---------------------------------------
    @classmethod
    def from_points(cls, points: SurfacePoints, iv: Any, **fields: Any) -> SurfaceObservations:
        """Attach observed implied volatilities to an existing point set."""
        return cls(
            k=points.k,
            T=points.T,
            iv=iv,
            surface_id=points.surface_id,
            point_id=points.point_id,
            convention=points.convention,
            **fields,
        )

    @classmethod
    def from_dataframe(
        cls, frame: "pd.DataFrame", *, columns: Mapping[str, str] | None = None
    ) -> SurfaceObservations:
        """Build from a long-format DataFrame, preserving row order.

        ``columns`` overrides :data:`DEFAULT_COLUMNS` per field.  The surface
        label is read from the named column, or from the index when the index
        carries that name (``pd.read_csv(..., index_col="id")``); it defaults to
        a single surface when neither exists.  ``point_id`` / ``bid`` / ``ask``
        / ``is_call`` / ``weight`` / ``price`` are used when present.  A file
        that spells the point label ``quote_id`` -- the name this container used
        before it was promoted out of the diagnostics package -- still loads.
        """
        names = {**DEFAULT_COLUMNS, **(columns or {})}
        missing = [names[name] for name in _REQUIRED if names[name] not in frame.columns]
        if missing:
            raise SurfaceValidationError(
                f"DataFrame lacks required column(s) {missing}; has {list(frame.columns)}."
            )
        surface_id: Any = None
        if names["surface_id"] in frame.columns:
            surface_id = frame[names["surface_id"]].to_numpy()
        elif frame.index.name == names["surface_id"]:
            surface_id = frame.index.to_numpy()
        optional = {
            name: frame[names[name]].to_numpy()
            for name in _OPTIONAL
            if names[name] in frame.columns
        }
        if "point_id" not in optional and LEGACY_POINT_ID_COLUMN in frame.columns:
            optional["point_id"] = frame[LEGACY_POINT_ID_COLUMN].to_numpy()
        return cls(
            k=frame[names["k"]].to_numpy(),
            T=frame[names["T"]].to_numpy(),
            iv=frame[names["iv"]].to_numpy(),
            surface_id=surface_id,
            **optional,
        )

    @classmethod
    def from_strikes(
        cls,
        K: Any,
        T: Any,
        iv: Any,
        *,
        forward: Any,
        surface_id: Any = None,
        point_id: Any = None,
        bid: Any = None,
        ask: Any = None,
        is_call: Any = None,
        weight: Any = None,
        price: Any = None,
        market_source: str | None = None,
    ) -> SurfaceObservations:
        """Build from strikes and a forward (scalar or per point): ``k = log(K / F)``."""
        from .points import points_from_strikes

        points = points_from_strikes(
            K,
            T,
            forward=forward,
            surface_id=surface_id,
            point_id=point_id,
            market_source=market_source,
        )
        return cls.from_points(
            points,
            iv,
            bid=bid,
            ask=ask,
            is_call=is_call,
            weight=weight,
            price=price,
        )

    def to_dataframe(self, *, columns: Mapping[str, str] | None = None) -> "pd.DataFrame":
        """The long-format DataFrame (absent optional fields are omitted).

        ``is_call`` is emitted as canonical booleans, missing values stay
        ``NaN``, and row order is preserved, so ``from_dataframe`` round-trips.
        """
        import pandas as pd

        names = {**DEFAULT_COLUMNS, **(columns or {})}
        data: dict[str, Any] = {
            names["surface_id"]: np.array(self.surface_id, copy=True),
            names["k"]: np.array(self.k, copy=True),
            names["T"]: np.array(self.T, copy=True),
            names["iv"]: np.array(self.iv, copy=True),
        }
        for name in _OPTIONAL:
            value = getattr(self, name)
            if value is not None:
                data[names[name]] = np.array(value, copy=True)
        ordered = [names["surface_id"]]
        if self.point_id is not None:
            ordered.append(names["point_id"])
        ordered += [names["k"], names["T"], names["iv"]]
        ordered += [
            names[name]
            for name in ("bid", "ask", "is_call", "weight", "price")
            if names[name] in data
        ]
        return pd.DataFrame(data)[ordered]


def _has_duplicate_pairs(left: np.ndarray, right: np.ndarray) -> bool:
    """Whether any ``(left, right)`` pair repeats."""
    if left.size == 0:
        return False
    pairs = list(zip(left.tolist(), right.tolist()))
    return len(set(pairs)) != len(pairs)


def _id_keys(quotes: SurfaceObservations) -> list[tuple[Any, Any]]:
    if quotes.point_id is None:
        raise SurfaceValidationError(
            "Alignment on stable ids requires point_id on both sides; "
            "pass on='coordinates' only when (surface_id, T, k) is unique on both sides."
        )
    return list(zip(quotes.surface_id.tolist(), quotes.point_id.tolist()))


def _coordinate_keys(quotes: SurfaceObservations, side: str) -> list[tuple[Any, float, float]]:
    keys = list(zip(quotes.surface_id.tolist(), quotes.T.tolist(), quotes.k.tolist()))
    if len(set(keys)) != len(keys):
        raise SurfaceValidationError(
            f"Coordinate alignment requires unique (surface_id, T, k) keys on both sides; "
            f"the {side} side has duplicates."
        )
    return keys


def align_predictions(
    truth: SurfaceObservations,
    predictions: SurfaceObservations,
    *,
    values: Any = None,
    on: str = "ids",
) -> np.ndarray:
    """Reorder a per-row prediction array onto ``truth``'s rows by an explicit key.

    Parameters
    ----------
    truth, predictions:
        The observed rows and the model's rows.  ``predictions`` supplies the
        alignment keys and, by default, the values to reorder.
    values:
        Optional raw per-row predictions parallel to ``predictions``' rows.
        A model can emit negative or non-finite implied volatilities, which a
        :class:`SurfaceObservations` deliberately refuses to hold as
        observations; pass them here (alongside a key-carrying ``predictions``)
        and they are returned **unsanitized**, so the evaluator still classifies
        and counts them instead of having them silently dropped during
        alignment.  Defaults to ``predictions.iv``.
    on:
        ``"ids"`` (default) keys on ``(surface_id, point_id)`` and requires
        ``point_id`` on both sides.  ``"coordinates"`` keys on exact
        ``(surface_id, T, k)`` equality and is permitted only when those keys
        are unique on both sides.

    Returns
    -------
    The prediction values reordered to ``truth``'s row order, shape
    ``(len(truth),)``.

    Raises
    ------
    SurfaceValidationError
        If keys are missing, extra, or duplicated on either side, or if a
        coordinate alignment is requested where the coordinates repeat.  There
        is no float tolerance, no fuzzy join, and no occurrence-order fallback.

    Notes
    -----
    :func:`~fast_vollib.diagnostics.diagnose_fit` never calls this: it consumes
    a prediction array already aligned to the truth rows it is given.
    """
    if on not in _ALIGN_KEYS:
        raise SurfaceValidationError(f"on must be one of {_ALIGN_KEYS}; got {on!r}.")
    truth_keys: list[tuple[Any, ...]]
    pred_keys: list[tuple[Any, ...]]
    if on == "ids":
        truth_keys = _id_keys(truth)
        pred_keys = _id_keys(predictions)
    else:
        truth_keys = _coordinate_keys(truth, "truth")
        pred_keys = _coordinate_keys(predictions, "prediction")

    if len(set(truth_keys)) != len(truth_keys):
        raise SurfaceValidationError("Truth keys must be unique for alignment.")
    if len(set(pred_keys)) != len(pred_keys):
        raise SurfaceValidationError("Prediction keys must be unique for alignment.")

    position = {key: index for index, key in enumerate(pred_keys)}
    missing = [key for key in truth_keys if key not in position]
    if missing:
        raise SurfaceValidationError(
            f"{len(missing)} truth row(s) have no prediction; first missing key {missing[0]!r}."
        )
    extra = len(pred_keys) - len(truth_keys)
    if extra:
        unmatched = set(pred_keys) - set(truth_keys)
        raise SurfaceValidationError(
            f"{len(unmatched)} prediction row(s) match no truth row; "
            f"first extra key {sorted(unmatched, key=repr)[0]!r}."
        )
    source = predictions.iv if values is None else np.asarray(values, dtype=np.float64)
    if source.shape != (predictions.n,):
        raise SurfaceValidationError(
            f"values must have shape ({predictions.n},) to match predictions; got {source.shape}."
        )
    order = np.fromiter((position[key] for key in truth_keys), dtype=np.intp, count=len(truth_keys))
    return np.array(source[order], dtype=np.float64, copy=True)
