"""Pooled fit-error accumulators for scattered implied-volatility predictions.

Records store additive **sums and counts** rather than derived means, so a
summary over any grouping divides exactly once: pooled RMSE over a set of
samples equals the RMSE of their concatenation, whatever order they merge in.

Three counts are carried separately and never collapsed, because collapsing
them is how a partially-predicted surface comes to look artificially good:

``target_count``
    Rows whose truth is a finite observation -- the denominator the model was
    asked to cover.
``valid_prediction_count``
    Targets whose prediction is finite and ``>= 0``; only these enter the error
    sums.
``invalid_prediction_count``
    Targets whose prediction is non-finite or negative.  They are excluded from
    the error sums *and* reported, and their presence marks the sample
    ``partial``.

``coverage = valid_prediction_count / target_count`` therefore tells a reader
how much of the surface a headline RMSE actually describes.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .regions import DEFAULT_REGIONS, Region

__all__ = [
    "ErrorSums",
    "FitDiagnostics",
    "PredictionClassification",
    "RegionErrors",
    "classify_predictions",
    "fit_error",
    "fit_error_by_region",
]


@dataclass(frozen=True, slots=True)
class PredictionClassification:
    """Which rows are targets, and which of those carry a usable prediction.

    Computed once per sample and shared by the fit, spread, and sampled
    arbitrage paths, so all three agree on exactly which rows the model
    covered.
    """

    #: ``True`` where truth is a finite observation.
    target: np.ndarray
    #: ``True`` where the row is a target *and* the prediction is finite and ``>= 0``.
    valid: np.ndarray
    #: ``True`` where the row is a target but the prediction is unusable.
    invalid: np.ndarray

    @property
    def target_count(self) -> int:
        return int(np.count_nonzero(self.target))

    @property
    def valid_count(self) -> int:
        return int(np.count_nonzero(self.valid))

    @property
    def invalid_count(self) -> int:
        return int(np.count_nonzero(self.invalid))


def classify_predictions(pred_iv: Any, truth_iv: Any) -> PredictionClassification:
    """Split rows into targets, valid predictions, and invalid predictions.

    A target is a row with finite truth.  A target's prediction is valid iff it
    is finite and non-negative; anything else (``NaN``, ``+/-inf``, a negative
    implied volatility) is invalid.  Rows whose truth is missing are neither.
    """
    pred = np.asarray(pred_iv, dtype=np.float64)
    truth = np.asarray(truth_iv, dtype=np.float64)
    if pred.shape != truth.shape:
        raise ValueError(
            f"pred_iv and truth_iv must have the same shape; got {pred.shape} and {truth.shape}."
        )
    if pred.ndim != 1:
        raise ValueError(f"pred_iv and truth_iv must be one-dimensional; got {pred.shape}.")
    target = np.isfinite(truth)
    with np.errstate(invalid="ignore"):
        usable = np.isfinite(pred) & (pred >= 0.0)
    valid = target & usable
    return PredictionClassification(target=target, valid=valid, invalid=target & ~usable)


@dataclass(frozen=True, slots=True)
class ErrorSums:
    """Additive error sums and counts for one group of ``(prediction, truth)`` pairs."""

    squared_error_sum: float = 0.0
    absolute_error_sum: float = 0.0
    valid_prediction_count: int = 0
    target_count: int = 0
    invalid_prediction_count: int = 0
    max_absolute_error: float | None = None

    def merge(self, other: ErrorSums) -> ErrorSums:
        """Pooled sums of two disjoint groups.  Associative and commutative."""
        if self.max_absolute_error is None:
            worst = other.max_absolute_error
        elif other.max_absolute_error is None:
            worst = self.max_absolute_error
        else:
            worst = max(self.max_absolute_error, other.max_absolute_error)
        return ErrorSums(
            squared_error_sum=self.squared_error_sum + other.squared_error_sum,
            absolute_error_sum=self.absolute_error_sum + other.absolute_error_sum,
            valid_prediction_count=self.valid_prediction_count + other.valid_prediction_count,
            target_count=self.target_count + other.target_count,
            invalid_prediction_count=self.invalid_prediction_count + other.invalid_prediction_count,
            max_absolute_error=worst,
        )

    @property
    def rmse(self) -> float | None:
        """Pooled root-mean-square error, or ``None`` with no valid predictions."""
        if self.valid_prediction_count == 0:
            return None
        return float(np.sqrt(self.squared_error_sum / self.valid_prediction_count))

    @property
    def mae(self) -> float | None:
        """Pooled mean absolute error, or ``None`` with no valid predictions."""
        if self.valid_prediction_count == 0:
            return None
        return float(self.absolute_error_sum / self.valid_prediction_count)

    @property
    def coverage(self) -> float | None:
        """``valid_prediction_count / target_count``, or ``None`` with no targets."""
        if self.target_count == 0:
            return None
        return float(self.valid_prediction_count / self.target_count)

    @property
    def invalid_rate(self) -> float | None:
        """``invalid_prediction_count / target_count``, or ``None`` with no targets."""
        if self.target_count == 0:
            return None
        return float(self.invalid_prediction_count / self.target_count)


def _sums_from_mask(
    pred: np.ndarray,
    truth: np.ndarray,
    classification: PredictionClassification,
    mask: np.ndarray | None = None,
) -> ErrorSums:
    target = classification.target if mask is None else classification.target & mask
    valid = classification.valid if mask is None else classification.valid & mask
    invalid = classification.invalid if mask is None else classification.invalid & mask
    error = pred[valid] - truth[valid]
    absolute = np.abs(error)
    return ErrorSums(
        squared_error_sum=float(np.sum(error * error)),
        absolute_error_sum=float(np.sum(absolute)),
        valid_prediction_count=int(valid.sum()),
        target_count=int(target.sum()),
        invalid_prediction_count=int(invalid.sum()),
        max_absolute_error=float(absolute.max()) if absolute.size else None,
    )


def fit_error(
    pred_iv: Any,
    truth_iv: Any,
    *,
    classification: PredictionClassification | None = None,
) -> ErrorSums:
    """Error sums over every target row.

    Parameters
    ----------
    pred_iv, truth_iv:
        Prediction and truth arrays, already aligned row-for-row.
    classification:
        A classification computed once by the caller; recomputed when omitted.

    Examples
    --------
    >>> from fast_vollib.diagnostics import fit_error
    >>> sums = fit_error([0.21, 0.19], [0.20, 0.20])
    >>> round(sums.rmse, 12), sums.target_count, sums.coverage
    (0.01, 2, 1.0)
    """
    pred = np.asarray(pred_iv, dtype=np.float64)
    truth = np.asarray(truth_iv, dtype=np.float64)
    if classification is None:
        classification = classify_predictions(pred, truth)
    return _sums_from_mask(pred, truth, classification)


@dataclass(frozen=True, slots=True)
class RegionErrors:
    """Error sums inside a named region and, when named, outside it."""

    name: str
    complement_name: str | None
    descriptor: dict[str, Any] | None
    inside: ErrorSums
    outside: ErrorSums | None

    def merge(self, other: RegionErrors) -> RegionErrors:
        if (self.name, self.complement_name, self.descriptor) != (
            other.name,
            other.complement_name,
            other.descriptor,
        ):
            raise ValueError(
                f"Cannot merge region {self.name!r} with a differently defined region "
                f"{other.name!r}; region descriptors must match exactly."
            )
        if (self.outside is None) != (other.outside is None):
            raise ValueError(f"Region {self.name!r} disagrees on whether a complement is scored.")
        outside: ErrorSums | None = None
        if self.outside is not None and other.outside is not None:
            outside = self.outside.merge(other.outside)
        return RegionErrors(
            name=self.name,
            complement_name=self.complement_name,
            descriptor=self.descriptor,
            inside=self.inside.merge(other.inside),
            outside=outside,
        )


@dataclass(frozen=True, slots=True)
class FitDiagnostics:
    """Overall fit error plus the per-region split."""

    overall: ErrorSums
    regions: tuple[RegionErrors, ...] = ()

    def merge(self, other: FitDiagnostics) -> FitDiagnostics:
        """Pooled diagnostics.  Regions must match in order and definition."""
        if len(self.regions) != len(other.regions):
            raise ValueError(
                f"Cannot merge fit diagnostics with {len(self.regions)} and "
                f"{len(other.regions)} regions."
            )
        return FitDiagnostics(
            overall=self.overall.merge(other.overall),
            regions=tuple(
                left.merge(right) for left, right in zip(self.regions, other.regions, strict=True)
            ),
        )

    def region(self, name: str) -> RegionErrors:
        """The region scored under ``name``."""
        for entry in self.regions:
            if entry.name == name:
                return entry
        raise KeyError(f"No region named {name!r}; have {[r.name for r in self.regions]}.")


def fit_error_by_region(
    pred_iv: Any,
    truth_iv: Any,
    k: Any,
    T: Any,
    regions: Sequence[Region] = DEFAULT_REGIONS,
    *,
    classification: PredictionClassification | None = None,
) -> FitDiagnostics:
    """Overall error sums plus one :class:`RegionErrors` per region descriptor.

    Each region partitions the target rows into inside / outside, so the two
    halves' counts add back to the overall counts exactly.  A region whose
    ``describe()`` returns ``None`` (an arbitrary callable predicate) is scored
    but carries no descriptor, and the serialized contract refuses it.
    """
    pred = np.asarray(pred_iv, dtype=np.float64)
    truth = np.asarray(truth_iv, dtype=np.float64)
    if classification is None:
        classification = classify_predictions(pred, truth)
    k_arr = np.asarray(k, dtype=np.float64)
    T_arr = np.asarray(T, dtype=np.float64)
    if k_arr.shape != pred.shape or T_arr.shape != pred.shape:
        raise ValueError(
            f"k and T must have shape {pred.shape}; got {k_arr.shape} and {T_arr.shape}."
        )
    scored: list[RegionErrors] = []
    for region in regions:
        mask = np.asarray(region.mask(k_arr, T_arr), dtype=bool)
        if mask.shape != pred.shape:
            raise ValueError(
                f"Region {region.name!r} returned shape {mask.shape}; expected {pred.shape}."
            )
        complement = region.complement_name
        scored.append(
            RegionErrors(
                name=region.name,
                complement_name=complement,
                descriptor=region.describe(),
                inside=_sums_from_mask(pred, truth, classification, mask),
                outside=(
                    None
                    if complement is None
                    else _sums_from_mask(pred, truth, classification, ~mask)
                ),
            )
        )
    return FitDiagnostics(
        overall=_sums_from_mask(pred, truth, classification), regions=tuple(scored)
    )
