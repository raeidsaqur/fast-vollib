"""Scoring a predicted surface: what was measured, on what, and how well it is known.

One evaluation object, produced by one function, carrying every number a
comparison between two surface models needs -- and carrying, alongside each
number, the accounting that says what it was computed over.

*Coverage is part of the error.*  A model that answered a fifth of the points and
fitted those five perfectly has an RMSE of zero, and reporting that number alone
is how a partially-predicted surface comes to look like the best one.  Target
count, valid count, and invalid count are three separate integers here and are
never collapsed into one.

*A finite grid is not a certificate.*  When an arbitrage report is attached, so
is a :class:`VerificationLevel` saying what kind of statement it is: an empirical
pass on the nodes that were checked, a penalty that was applied during training,
a guarantee that follows from the parameterization, or a claim somebody made that
this library did not verify.  Those four are different, and a report that does
not distinguish them will eventually be read as the strongest of them.

*Price-space error needs a market and will not invent one.*  Implied-volatility
error is dimensionless and needs nothing; a price error is a statement about a
specific forward curve.  Asked for one without a
:class:`~fast_vollib.surface.market.SurfaceMarket`, this module raises
:class:`~fast_vollib.surface.errors.MissingMarketStateError` rather than
defaulting the forward to one.

Examples
--------
>>> import numpy as np
>>> from fast_vollib.surface import SurfaceObservations, SurfacePrediction, evaluate_prediction
>>> observations = SurfaceObservations(k=[-0.1, 0.0, 0.1], T=[1.0] * 3, iv=[0.22, 0.20, 0.24])
>>> prediction = SurfacePrediction(points=observations.points, iv=[0.22, 0.20, 0.24])
>>> evaluation = evaluate_prediction(prediction, observations)
>>> evaluation.iv_rmse, evaluation.coverage
(0.0, 1.0)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json
from typing import TYPE_CHECKING, Any

import numpy as np

from .errors import MissingMarketStateError, SurfaceValidationError
from .observations import SurfaceObservations
from .prediction import SurfacePrediction

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .grid import IVSurface
    from .gridspec import SurfaceGridSpec
    from .market import SurfaceMarket
    from .report import ArbitrageReport

__all__ = [
    "SCHEMA_VERSION",
    "MaturityEvaluation",
    "RegionEvaluation",
    "SurfaceEvaluation",
    "VerificationLevel",
    "evaluate_prediction",
    "evaluation_json_schema",
    "render_evaluation_json_schema",
]

#: The wire identifier of the serialized evaluation record.
SCHEMA_VERSION = "fast-vollib-surface-evaluation-v1"

_SCHEMA_ID = (
    "https://raeidsaqur.github.io/fast-vollib/schemas/fast-vollib-surface-evaluation-v1.schema.json"
)

_INV_SQRT_2PI = 0.3989422804014327


class VerificationLevel(str, Enum):
    """What kind of statement an arbitrage claim is.

    Attributes
    ----------
    EMPIRICAL_FINITE_GRID
        The conditions were checked on a stated finite mesh and held there.  It
        says nothing about the continuum between nodes, and nothing about a
        different mesh.
    TRAINING_PENALTY
        A penalty on the violations entered the training objective.  A penalty
        changes what a model is likely to do; it does not decide anything.
    MATHEMATICAL_GUARANTEE
        The parameterization admits no violation, so no grid can find one.
        SSVI's non-decreasing theta is of this kind for the calendar condition.
    EXTERNAL_CLAIM_UNVERIFIED
        A property somebody asserted that this library has not checked.  It is a
        level so that such a claim can be carried without being promoted.

    Notes
    -----
    ``str`` mixin rather than ``StrEnum``: the library floor is Python 3.10.
    """

    EMPIRICAL_FINITE_GRID = "empirical_finite_grid"
    TRAINING_PENALTY = "training_penalty"
    MATHEMATICAL_GUARANTEE = "mathematical_guarantee"
    EXTERNAL_CLAIM_UNVERIFIED = "external_claim_unverified"


@dataclass(frozen=True, slots=True)
class RegionEvaluation:
    """Fit error inside one named region of the surface."""

    name: str
    descriptor: dict[str, Any] | None
    target_count: int
    valid_count: int
    iv_rmse: float | None
    iv_mae: float | None

    def to_dict(self) -> dict[str, Any]:
        """The record as a JSON-safe mapping."""
        return {
            "name": self.name,
            "descriptor": self.descriptor,
            "target_count": int(self.target_count),
            "valid_count": int(self.valid_count),
            "iv_rmse": _number(self.iv_rmse),
            "iv_mae": _number(self.iv_mae),
        }


@dataclass(frozen=True, slots=True)
class MaturityEvaluation:
    """Fit error at one maturity."""

    maturity: float
    target_count: int
    valid_count: int
    iv_rmse: float | None
    iv_mae: float | None

    def to_dict(self) -> dict[str, Any]:
        """The record as a JSON-safe mapping."""
        return {
            "maturity": float(self.maturity),
            "target_count": int(self.target_count),
            "valid_count": int(self.valid_count),
            "iv_rmse": _number(self.iv_rmse),
            "iv_mae": _number(self.iv_mae),
        }


@dataclass(frozen=True, slots=True)
class SurfaceEvaluation:
    """Everything measurable about one predicted surface, with its accounting.

    Attributes
    ----------
    target_count, valid_count, invalid_count : int
        Rows that were observed, rows the model answered usably, and rows it did
        not.  Three integers, never collapsed: an RMSE over the second alone is
        not comparable with one over the first.
    coverage : float | None
        ``valid_count / target_count``, or ``None`` when nothing was targeted.
    iv_rmse, iv_mae, max_absolute_iv_error : float | None
        Implied-volatility error over the valid rows.  ``None`` rather than zero
        when there are none: an unmeasured error is not a small one.
    weighted_iv_rmse : float | None
        Error weighted by the observations' own ``weight`` column, when present.
    price_rmse, price_mae : float | None
        Forward-normalised price error.  Requires a market, and requires the
        observations to carry prices or the market to price the observed implied
        volatilities.
    vega_weighted_iv_rmse : float | None
        Implied-volatility error scaled by Black vega, which is what a
        price-space error is to first order and is defined without a market.
    inside_spread_fraction : float | None
        Fraction of quoted rows whose predicted price lands inside bid/ask.
        ``None`` when the observations carry no spread -- never a zero.
    by_region, by_maturity
        The same accounting, split.
    arbitrage : ArbitrageReport | None
        Hard checks on a materialized mesh, when a grid was supplied.
    verification : VerificationLevel | None
        What kind of statement ``arbitrage`` is.
    native_node_fraction : float | None
        Fraction of evaluated mesh nodes that were the model's own output rather
        than introduced by interpolation.
    grid_shape : tuple[int, int] | None
    market_source : str | None
        Provenance of the market state that entered the price-space numbers.
    """

    target_count: int
    valid_count: int
    invalid_count: int
    iv_rmse: float | None = None
    iv_mae: float | None = None
    max_absolute_iv_error: float | None = None
    weighted_iv_rmse: float | None = None
    price_rmse: float | None = None
    price_mae: float | None = None
    vega_weighted_iv_rmse: float | None = None
    inside_spread_fraction: float | None = None
    by_region: tuple[RegionEvaluation, ...] = ()
    by_maturity: tuple[MaturityEvaluation, ...] = ()
    arbitrage: "ArbitrageReport | None" = None
    verification: VerificationLevel | None = None
    native_node_fraction: float | None = None
    grid_shape: tuple[int, int] | None = None
    market_source: str | None = None
    _coverage: float | None = field(default=None, init=False, repr=False, compare=False)

    @property
    def coverage(self) -> float | None:
        """Fraction of targeted rows the model answered usably."""
        if self.target_count == 0:
            return None
        return self.valid_count / self.target_count

    @property
    def invalid_rate(self) -> float | None:
        """Fraction of targeted rows the model declined or answered unusably."""
        if self.target_count == 0:
            return None
        return self.invalid_count / self.target_count

    def to_dict(self) -> dict[str, Any]:
        """The evaluation as a canonical, JSON-safe mapping.

        Field order is fixed by construction, every number is a plain float or
        ``null``, and no ``NaN`` ever appears: an unavailable measurement is
        ``null``, never a zero standing in for "not measured".
        """
        return {
            "schema": SCHEMA_VERSION,
            "coverage": {
                "target_count": int(self.target_count),
                "valid_count": int(self.valid_count),
                "invalid_count": int(self.invalid_count),
                "coverage": _number(self.coverage),
                "invalid_rate": _number(self.invalid_rate),
            },
            "implied_volatility": {
                "rmse": _number(self.iv_rmse),
                "mae": _number(self.iv_mae),
                "max_absolute_error": _number(self.max_absolute_iv_error),
                "weighted_rmse": _number(self.weighted_iv_rmse),
                "vega_weighted_rmse": _number(self.vega_weighted_iv_rmse),
            },
            "price": {
                "rmse": _number(self.price_rmse),
                "mae": _number(self.price_mae),
                "inside_spread_fraction": _number(self.inside_spread_fraction),
                "market_source": self.market_source,
            },
            "regions": [entry.to_dict() for entry in self.by_region],
            "maturities": [entry.to_dict() for entry in self.by_maturity],
            "arbitrage": None if self.arbitrage is None else self.arbitrage.to_dict(),
            "verification": None if self.verification is None else self.verification.value,
            "grid": None
            if self.grid_shape is None
            else {
                "n_moneyness": int(self.grid_shape[0]),
                "n_maturities": int(self.grid_shape[1]),
                "native_node_fraction": _number(self.native_node_fraction),
            },
        }

    def to_json(self, *, indent: int | None = None) -> str:
        """Canonical, newline-terminated JSON."""
        separators = (",", ":") if indent is None else (",", ": ")
        return (
            json.dumps(
                self.to_dict(),
                ensure_ascii=False,
                allow_nan=False,
                indent=indent,
                separators=separators,
                sort_keys=False,
            )
            + "\n"
        )


def evaluate_prediction(
    prediction: SurfacePrediction,
    observations: SurfaceObservations,
    *,
    market: "SurfaceMarket | None" = None,
    require_prices: bool = False,
    grid: "SurfaceGridSpec | None" = None,
    materialized: "IVSurface | None" = None,
    verification: VerificationLevel | None = None,
    regions: Any = None,
    maturity_decimals: int = 6,
) -> SurfaceEvaluation:
    """Score ``prediction`` against ``observations``.

    Parameters
    ----------
    prediction:
        The model's answer.  Its points must be the observations' own points,
        compared exactly -- there is no reordering and no tolerance, for the same
        reason alignment elsewhere is exact.
    observations:
        The truth.  Rows whose ``iv`` is ``NaN`` are not targets.
    market:
        Market state for the price-space numbers.  Without it those fields are
        ``None``; they are never computed against an assumed forward.
    require_prices:
        Turn the absent market into an error rather than into ``None`` fields.
        A caller who is comparing models on price error wants the failure, not a
        report whose price columns are quietly empty.
    grid:
        Declared mesh for the arbitrage check.  Supplying it also fills
        ``grid_shape`` and ``native_node_fraction``.
    materialized:
        The already-materialized mesh, when the caller has one.  Saves a second
        evaluation of the model; must correspond to ``grid``.
    verification:
        What kind of statement the arbitrage report is.  Defaults to
        :attr:`VerificationLevel.EMPIRICAL_FINITE_GRID` when a report is
        produced, because that is what was measured.
    regions:
        Region descriptors to split by; defaults to the diagnostics package's
        own :data:`~fast_vollib.diagnostics.DEFAULT_REGIONS`.
    maturity_decimals:
        Decimal places at which two maturities are one expiry.

    Raises
    ------
    SurfaceValidationError
        If the prediction does not answer the observations' points.
    MissingMarketStateError
        If ``grid`` was given for a price-space check without a market on either
        the grid or the call.
    """
    if not isinstance(prediction, SurfacePrediction):
        raise SurfaceValidationError(
            f"prediction must be a SurfacePrediction; got {type(prediction).__name__}."
        )
    if not isinstance(observations, SurfaceObservations):
        raise SurfaceValidationError(
            f"observations must be a SurfaceObservations; got {type(observations).__name__}."
        )
    if not prediction.points.matches_domain(observations.points):
        raise SurfaceValidationError(
            "The prediction answers different points than the observations describe. "
            "Evaluate a prediction on the observations' own points, or align it "
            "explicitly with align_predictions first."
        )

    from ..diagnostics.fit import classify_predictions
    from ..diagnostics.regions import DEFAULT_REGIONS

    truth = observations.iv
    predicted = prediction.iv
    classification = classify_predictions(np.where(prediction.valid, predicted, np.nan), truth)
    target = classification.target
    valid = classification.valid
    error = np.where(valid, predicted - truth, 0.0)

    weights = (
        observations.weight
        if observations.weight is not None
        else np.ones(observations.n, dtype=np.float64)
    )
    evaluation_kwargs: dict[str, Any] = {
        "target_count": classification.target_count,
        "valid_count": classification.valid_count,
        "invalid_count": classification.invalid_count,
        "iv_rmse": _rmse(error, valid),
        "iv_mae": _mae(error, valid),
        "max_absolute_iv_error": _max_abs(error, valid),
        "weighted_iv_rmse": _weighted_rmse(error, valid, weights),
        "vega_weighted_iv_rmse": _weighted_rmse(
            error, valid, weights * _black_vega(observations.k, observations.T, truth)
        ),
    }

    state = market if market is not None else (grid.market if grid is not None else None)
    if require_prices and state is None:
        raise MissingMarketStateError(
            "Price-space evaluation needs the market state the quotes were made "
            "against, and none was supplied on the call or on the grid. A price "
            "computed against an invented forward looks exactly like one computed "
            "against the real forward."
        )
    model_normalized, observed_normalized = _normalized_prices(observations, predicted)
    if observations.has_spread:
        evaluation_kwargs["inside_spread_fraction"] = _inside_spread(
            observations, model_normalized, valid
        )
    if state is not None:
        # The forward-normalised call is dimensionless; a *price* is that scaled
        # by the discounted forward, which is where the market actually enters.
        scale = state.discount_at(observations.T) * state.forward_at(observations.T)
        price_error = np.where(valid, scale * (model_normalized - observed_normalized), 0.0)
        evaluation_kwargs["price_rmse"] = _rmse(price_error, valid)
        evaluation_kwargs["price_mae"] = _mae(price_error, valid)
        evaluation_kwargs["market_source"] = state.source

    evaluation_kwargs["by_region"] = _by_region(
        observations, error, valid, DEFAULT_REGIONS if regions is None else regions
    )
    evaluation_kwargs["by_maturity"] = _by_maturity(
        observations, error, target, valid, maturity_decimals
    )

    if grid is not None:
        surface = materialized
        if surface is None:
            raise SurfaceValidationError(
                "A grid was given without a materialized surface. Materialize the model "
                "once with materialize_surface and pass the result, so the arbitrage "
                "report describes the same evaluation the metrics do."
            )
        report = surface.validate()
        evaluation_kwargs["arbitrage"] = report
        evaluation_kwargs["verification"] = (
            VerificationLevel.EMPIRICAL_FINITE_GRID if verification is None else verification
        )
        evaluation_kwargs["grid_shape"] = grid.shape
        evaluation_kwargs["native_node_fraction"] = (
            1.0 if grid.native_mask is None else float(np.mean(grid.native_mask))
        )
    elif verification is not None:
        evaluation_kwargs["verification"] = verification

    return SurfaceEvaluation(**evaluation_kwargs)


# --- helpers ------------------------------------------------------------------


def _number(value: Any) -> float | None:
    """A plain float, or ``None``; ``NaN`` and the infinities become ``None``.

    A metric that could not be computed is absent, not zero and not ``NaN``.
    JSON has no ``NaN``, and a zero standing in for "not measured" is the
    failure this codec exists to prevent.
    """
    if value is None:
        return None
    coerced = float(value)
    if coerced != coerced or coerced in (float("inf"), float("-inf")):
        return None
    return coerced


def _rmse(error: np.ndarray, valid: np.ndarray) -> float | None:
    count = int(np.count_nonzero(valid))
    if count == 0:
        return None
    return float(np.sqrt(float(np.sum(error * error)) / count))


def _mae(error: np.ndarray, valid: np.ndarray) -> float | None:
    count = int(np.count_nonzero(valid))
    if count == 0:
        return None
    return float(np.sum(np.abs(error)) / count)


def _max_abs(error: np.ndarray, valid: np.ndarray) -> float | None:
    if not bool(np.any(valid)):
        return None
    return float(np.max(np.abs(error[valid])))


def _weighted_rmse(error: np.ndarray, valid: np.ndarray, weights: np.ndarray) -> float | None:
    selected = valid & np.isfinite(weights) & (weights > 0.0)
    total = float(np.sum(weights[selected]))
    if total <= 0.0:
        return None
    return float(np.sqrt(float(np.sum(weights[selected] * error[selected] ** 2)) / total))


def _black_vega(k: np.ndarray, T: np.ndarray, iv: np.ndarray) -> np.ndarray:
    """Black vega per unit forward at the observed implied volatility."""
    with np.errstate(all="ignore"):
        w = iv * iv * T
        sqrt_w = np.sqrt(w)
        d1 = (-k + 0.5 * w) / sqrt_w
        vega = _INV_SQRT_2PI * np.exp(-0.5 * d1 * d1) * np.sqrt(T)
    return np.where(np.isfinite(vega), vega, 0.0)


def _normalized_prices(
    observations: SurfaceObservations, predicted: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Forward-normalised model and observed call prices at every row.

    Dimensionless by construction -- the forward divides out of both the price
    and the strike -- so no market state is needed here.  It is needed to turn
    these into prices, which is done by the caller and only when a market exists.
    """
    from .._array_api import numpy_namespace
    from .transforms import undiscounted_call

    xp = numpy_namespace()
    forward = np.ones_like(observations.k)
    with np.errstate(all="ignore"):
        model = undiscounted_call(
            observations.k,
            np.where(np.isfinite(predicted), predicted, 0.0) ** 2 * observations.T,
            forward,
            xp,
        )
        if observations.has_prices:
            observed = np.where(
                np.isnan(observations.price),
                undiscounted_call(observations.k, observations.iv**2 * observations.T, forward, xp),
                observations.price,
            )
        else:
            observed = undiscounted_call(
                observations.k, observations.iv**2 * observations.T, forward, xp
            )
    return np.where(np.isfinite(model), model, np.nan), observed


def _inside_spread(
    observations: SurfaceObservations, model_price: np.ndarray, valid: np.ndarray
) -> float | None:
    """Fraction of quoted, valid rows whose model price lands inside bid/ask."""
    quoted = valid & ~np.isnan(observations.bid) & ~np.isnan(observations.ask)
    count = int(np.count_nonzero(quoted))
    if count == 0:
        return None
    inside = (model_price[quoted] >= observations.bid[quoted]) & (
        model_price[quoted] <= observations.ask[quoted]
    )
    return float(np.count_nonzero(inside)) / count


def _by_region(
    observations: SurfaceObservations, error: np.ndarray, valid: np.ndarray, regions: Any
) -> tuple[RegionEvaluation, ...]:
    out = []
    for region in regions:
        inside = np.asarray(region.mask(observations.k, observations.T), dtype=bool)
        selected = valid & inside
        descriptor = region.describe() if hasattr(region, "describe") else None
        out.append(
            RegionEvaluation(
                name=getattr(region, "name", str(region)),
                descriptor=descriptor,
                target_count=int(np.count_nonzero(inside & ~np.isnan(observations.iv))),
                valid_count=int(np.count_nonzero(selected)),
                iv_rmse=_rmse(error, selected),
                iv_mae=_mae(error, selected),
            )
        )
    return tuple(out)


def _by_maturity(
    observations: SurfaceObservations,
    error: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
    decimals: int,
) -> tuple[MaturityEvaluation, ...]:
    bucket = np.round(observations.T, decimals)
    out = []
    for value in np.unique(bucket):
        selected = bucket == value
        out.append(
            MaturityEvaluation(
                maturity=float(value),
                target_count=int(np.count_nonzero(selected & target)),
                valid_count=int(np.count_nonzero(selected & valid)),
                iv_rmse=_rmse(error, selected & valid),
                iv_mae=_mae(error, selected & valid),
            )
        )
    return tuple(out)


# --- the wire schema ----------------------------------------------------------


def _optional_number() -> dict[str, Any]:
    return {"type": ["number", "null"]}


def _closed(properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": sorted(properties),
        "properties": properties,
    }


def evaluation_json_schema() -> dict[str, Any]:
    """The closed Draft 2020-12 schema for ``fast-vollib-surface-evaluation-v1``.

    Built from the same shape :meth:`SurfaceEvaluation.to_dict` emits, so the two
    cannot disagree: every object declares ``additionalProperties: false`` and
    lists every property as required, and an unavailable number is ``null``
    rather than absent.
    """
    region = _closed(
        {
            "name": {"type": "string"},
            "descriptor": {"type": ["object", "null"]},
            "target_count": {"type": "integer", "minimum": 0},
            "valid_count": {"type": "integer", "minimum": 0},
            "iv_rmse": _optional_number(),
            "iv_mae": _optional_number(),
        }
    )
    maturity = _closed(
        {
            "maturity": {"type": "number", "exclusiveMinimum": 0},
            "target_count": {"type": "integer", "minimum": 0},
            "valid_count": {"type": "integer", "minimum": 0},
            "iv_rmse": _optional_number(),
            "iv_mae": _optional_number(),
        }
    )
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": _SCHEMA_ID,
        "title": "fast-vollib surface-evaluation-v1",
        **_closed(
            {
                "schema": {"const": SCHEMA_VERSION},
                "coverage": _closed(
                    {
                        "target_count": {"type": "integer", "minimum": 0},
                        "valid_count": {"type": "integer", "minimum": 0},
                        "invalid_count": {"type": "integer", "minimum": 0},
                        "coverage": _optional_number(),
                        "invalid_rate": _optional_number(),
                    }
                ),
                "implied_volatility": _closed(
                    {
                        "rmse": _optional_number(),
                        "mae": _optional_number(),
                        "max_absolute_error": _optional_number(),
                        "weighted_rmse": _optional_number(),
                        "vega_weighted_rmse": _optional_number(),
                    }
                ),
                "price": _closed(
                    {
                        "rmse": _optional_number(),
                        "mae": _optional_number(),
                        "inside_spread_fraction": _optional_number(),
                        "market_source": {"type": ["string", "null"]},
                    }
                ),
                "regions": {"type": "array", "items": region},
                "maturities": {"type": "array", "items": maturity},
                "arbitrage": {"type": ["object", "null"]},
                "verification": {
                    "type": ["string", "null"],
                    "enum": [level.value for level in VerificationLevel] + [None],
                },
                "grid": {
                    "type": ["object", "null"],
                    "additionalProperties": False,
                    "required": ["n_moneyness", "n_maturities", "native_node_fraction"],
                    "properties": {
                        "n_moneyness": {"type": "integer", "minimum": 1},
                        "n_maturities": {"type": "integer", "minimum": 1},
                        "native_node_fraction": _optional_number(),
                    },
                },
            }
        ),
    }


def render_evaluation_json_schema() -> str:
    """The schema as the exact text checked in at ``docs/schemas``.

    A test regenerates this and compares it byte for byte with the file, so the
    checked-in artifact can never describe a different format from the code.
    """
    return (
        json.dumps(evaluation_json_schema(), indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
