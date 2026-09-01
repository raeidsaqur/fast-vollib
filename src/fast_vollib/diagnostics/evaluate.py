"""One-call evaluators: everything measurable about one predicted sample.

:func:`diagnose_fit` classifies the predictions **once** -- which rows are
targets, which of those the model actually covered -- and hands that single
classification to the fit, spread, and sampled-arbitrage paths, so all three
blocks describe exactly the same rows.  Truth is never mutated and invalid
predictions are never hidden: they are excluded from the error sums *and*
counted, and their presence marks the sample ``partial``.

:func:`diagnose_surface` is the separate rectangular-grid path, re-exported
here for symmetry; it needs a genuine mesh, not a scattered cloud.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .arbitrage import (
    RaggedArbitrageConfig,
    diagnose_surface,
    quotes_to_surface,
    sampled_arbitrage,
)
from .fit import classify_predictions, fit_error_by_region
from .quotes import SurfaceQuotes
from .regions import DEFAULT_REGIONS, Region
from .report import SampleDiagnostics
from .spread import spread_consistency

__all__ = ["diagnose_fit", "diagnose_surface", "quotes_to_surface"]


def diagnose_fit(
    pred_iv: Any,
    truth: SurfaceQuotes,
    *,
    regions: Sequence[Region] = DEFAULT_REGIONS,
    ragged: RaggedArbitrageConfig | None = None,
    spread: bool = True,
    arbitrage: bool = True,
) -> SampleDiagnostics:
    """Fit, spread, and sampled-arbitrage diagnostics for one predicted sample.

    Parameters
    ----------
    pred_iv:
        Raw predicted implied volatilities, already aligned row-for-row with
        ``truth`` (use :func:`~fast_vollib.diagnostics.align_predictions` first
        if they are not).  Negative and non-finite values are permitted here
        and are classified rather than rejected.
    truth:
        The observed quotes.  Rows whose ``iv`` is ``NaN`` are not targets.
    regions:
        Region descriptors to split the fit error by.
    ragged:
        Sampled-arbitrage settings; defaults to
        :class:`~fast_vollib.diagnostics.RaggedArbitrageConfig`.
    spread:
        Compute spread consistency when the quotes carry bid/ask.  The block is
        ``None`` when they do not -- never a zero.
    arbitrage:
        Compute the sampled butterfly/calendar checks.

    Returns
    -------
    :class:`~fast_vollib.diagnostics.SampleDiagnostics`

    Examples
    --------
    >>> import numpy as np
    >>> from fast_vollib.diagnostics import SurfaceQuotes, diagnose_fit
    >>> truth = SurfaceQuotes(k=[-0.1, 0.0, 0.1], T=[0.5, 0.5, 0.5], iv=[0.2, 0.2, 0.2])
    >>> sample = diagnose_fit(np.array([0.21, 0.2, 0.19]), truth)
    >>> sample.fit.overall.target_count, sample.quality_status
    (3, 'complete')
    """
    pred = np.asarray(pred_iv, dtype=np.float64)
    if pred.ndim == 0:
        pred = np.full(truth.n, float(pred))
    if pred.shape != (truth.n,):
        raise ValueError(
            f"pred_iv must have shape ({truth.n},) to align with truth; got {pred.shape}. "
            "Use align_predictions to reorder a differently ordered prediction set."
        )
    classification = classify_predictions(pred, truth.iv)
    return SampleDiagnostics(
        fit=fit_error_by_region(
            pred,
            truth.iv,
            truth.k,
            truth.T,
            regions,
            classification=classification,
        ),
        spread=(spread_consistency(pred, truth, classification=classification) if spread else None),
        ragged=(
            sampled_arbitrage(pred, truth, config=ragged, classification=classification)
            if arbitrage
            else None
        ),
    )
