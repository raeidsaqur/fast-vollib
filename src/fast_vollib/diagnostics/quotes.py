"""The scattered-quote container, under the name it was first published as.

:class:`~fast_vollib.surface.observations.SurfaceObservations` began life here,
as ``SurfaceQuotes``, when scattered quotes were an input to diagnostics and
nothing else.  They are now the input to fitting, forecasting, and generative
sampling as well, so the definition lives in
:mod:`fast_vollib.surface.observations` -- one container, promoted rather than
copied, so a calibrator and a diagnostic cannot drift into disagreeing about
what a quote is.

This module keeps the original names working.  ``SurfaceQuotes`` is the
promoted class itself, not a subclass and not a wrapper, so an object built
through either name is the same object and passes every ``isinstance`` check
against the other.

One field was renamed in the promotion: ``quote_id`` became ``point_id``, to
match :class:`~fast_vollib.surface.points.SurfacePoints`, which every model
family queries through.  A long-format file that spells the column ``quote_id``
still loads -- see
:meth:`~fast_vollib.surface.observations.SurfaceObservations.from_dataframe`.

Examples
--------
>>> from fast_vollib.diagnostics import SurfaceQuotes
>>> from fast_vollib.surface import SurfaceObservations
>>> SurfaceQuotes is SurfaceObservations
True
>>> SurfaceQuotes(k=[0.0], T=[1.0], iv=[0.2]).n
1
"""

from __future__ import annotations

from ..surface.observations import (
    DEFAULT_COLUMNS,
    SurfaceObservations,
    align_predictions,
)

__all__ = ["DEFAULT_COLUMNS", "SurfaceQuotes", "align_predictions"]

#: The scattered-observation container, under its original published name.
SurfaceQuotes = SurfaceObservations
