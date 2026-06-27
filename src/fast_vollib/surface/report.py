"""Result containers for the arbitrage-evaluation harness.

These mirror the public return conventions of the rest of ``fast_vollib``:
the dataclasses are the native representation, and :meth:`ArbitrageReport.as`
serializes to the ``ReturnAsLiteral`` shapes (``"dict"`` / ``"json"``) used
elsewhere in the package.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .._typing import ArrayLike  # noqa: F401

# Violation taxonomy — the four no-arbitrage condition families (design §4).
ViolationType = str  # 'butterfly' | 'calendar' | 'vertical' | 'bound'
Severity = str  # 'minor' | 'moderate' | 'severe'
Origin = str  # 'native' | 'interpolation_induced'


@dataclass(frozen=True)
class ArbitrageViolation:
    """A single localized no-arbitrage condition failure.

    Attributes
    ----------
    type:
        Which condition family failed: ``'butterfly'`` (convexity / negative
        density), ``'calendar'`` (total-variance crossing), ``'vertical'``
        (call monotonicity / slope), or ``'bound'`` (price box).
    severity:
        Magnitude bucket relative to ``tolerance``: ``minor`` (<2×),
        ``moderate`` (<5×), ``severe`` (≥5×).
    value:
        Normalized, dimensionless violation magnitude (see design §5).
    tolerance:
        The tolerance the value was compared against.
    location:
        ``(k, T)`` context of the violation (log-moneyness and maturity).
    origin:
        ``'native'`` if every node in the violation's stencil is a generator
        node; ``'interpolation_induced'`` if the stencil touches an
        interpolated node (design §5, artifact-vs-arbitrage separation).
    index:
        ``(i, j)`` grid index for downstream highlighting / UI overlays.
    """

    type: ViolationType
    severity: Severity
    value: float
    tolerance: float
    location: tuple[float, float]
    origin: Origin = "native"
    index: tuple[int, int] | None = None


@dataclass
class ArbitrageReport:
    """Full arbitrage diagnostic for a surface (design §9).

    The report deliberately exposes *both* the scalar composite ``sas`` and
    its named ``metrics`` components — a single number hides which condition
    failed, so the composite is never reported alone (design §5).
    """

    passed: bool
    metrics: dict[str, float]
    sas: float
    violations: list[ArbitrageViolation] = field(default_factory=list)
    by_condition: dict[str, dict[str, Any]] = field(default_factory=dict)
    native: dict[str, float] = field(default_factory=dict)
    interpolation_induced: dict[str, float] = field(default_factory=dict)
    trust_mask: Any = None
    tolerance: float = 0.0
    context: dict[str, Any] = field(default_factory=dict)

    # -- summaries -----------------------------------------------------------
    @property
    def n_violations(self) -> int:
        return len(self.violations)

    def severity_counts(self) -> dict[str, int]:
        counts = {"minor": 0, "moderate": 0, "severe": 0}
        for v in self.violations:
            counts[v.severity] = counts.get(v.severity, 0) + 1
        return counts

    # -- serialization (mirrors ReturnAsLiteral) -----------------------------
    def to_dict(self) -> dict[str, Any]:
        trust = self.trust_mask
        if trust is not None and not isinstance(trust, list):
            trust = np.asarray(trust).tolist()
        return {
            "passed": bool(self.passed),
            "metrics": {k: _to_float(v) for k, v in self.metrics.items()},
            "sas": _to_float(self.sas),
            "violations": [asdict(v) for v in self.violations],
            "by_condition": self.by_condition,
            "native": {k: _to_float(v) for k, v in self.native.items()},
            "interpolation_induced": {
                k: _to_float(v) for k, v in self.interpolation_induced.items()
            },
            "severity_counts": self.severity_counts(),
            "trust_mask": trust,
            "tolerance": _to_float(self.tolerance),
            "context": self.context,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    def render(self, return_as: str = "report") -> Any:
        """Return the report in the requested ``ReturnAsLiteral``-style shape."""
        if return_as in ("report", "object", None):
            return self
        if return_as == "dict":
            return self.to_dict()
        if return_as == "json":
            return self.to_json()
        raise ValueError(
            f"Unsupported return_as={return_as!r}; expected 'report', 'dict', or 'json'."
        )

    def __repr__(self) -> str:  # concise, human-friendly
        counts = self.severity_counts()
        return (
            f"ArbitrageReport(passed={self.passed}, sas={self.sas:.4g}, "
            f"violations={self.n_violations} "
            f"[minor={counts['minor']}, moderate={counts['moderate']}, "
            f"severe={counts['severe']}])"
        )


def _to_float(value: Any) -> float:
    """Coerce numpy/torch/jax scalars to a plain Python float for JSON."""
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(np.asarray(value))
