"""Distributions over surfaces, and the evaluation that scores every draw.

The contracts a generative surface model satisfies live in
:mod:`fast_vollib.surface.protocols`; what lives here is the machinery that makes
them measurable -- an aggregation that materializes every draw, checks it by the
same hard conditions a deterministic model faces, and reports the resulting
distribution of outcomes rather than one number about an average nobody trades.

One concrete distribution ships with it.
:class:`GaussianFieldSurfaceDistribution` multiplies a fitted surface's total
variance by a log-normal random field: real enough that its draws are surfaces
and its violations are genuine, simple enough that a test can turn the violation
rate from nearly zero to nearly one by changing one length scale, and honest
about having no dynamics. Production learned models are later milestones and will
fit these same contracts.

Importing this package pulls in numpy only.
"""

from __future__ import annotations

from .distributions import (
    MAX_JOINT_POINTS,
    GaussianFieldSurfaceDistribution,
    GaussianFieldSurfaceGenerator,
)
from .evaluation import (
    CONDITIONS,
    DEFAULT_SEVERITY_QUANTILES,
    SCHEMA_VERSION,
    GenerativeArbitrageReport,
    evaluate_samples,
    generative_json_schema,
    render_generative_json_schema,
    wilson_interval,
)

__all__ = [
    "CONDITIONS",
    "DEFAULT_SEVERITY_QUANTILES",
    "MAX_JOINT_POINTS",
    "SCHEMA_VERSION",
    "GaussianFieldSurfaceDistribution",
    "GaussianFieldSurfaceGenerator",
    "GenerativeArbitrageReport",
    "evaluate_samples",
    "generative_json_schema",
    "render_generative_json_schema",
    "wilson_interval",
]
