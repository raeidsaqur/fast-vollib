"""Normalized metrics, localization, and the ``validate_surface`` orchestrator.

This is the **host (numpy) report path**: it computes the arbitrage fields via
the shared kernels (:func:`~fast_vollib.surface.arbitrage.compute_fields`),
materializes them to numpy, and turns them into the calibrated, dimensionless
metric vector and the localized :class:`~fast_vollib.surface.report.ArbitrageReport`
of design §5 / §9.

Every metric is normalized so it compares across generators, meshes, and
underlyings.  The scalar :data:`SAS` composite is reported **only alongside its
components** — a single number hides which condition failed, and the weighting
is an explicit, documented modeling choice (design §5, §17).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .arbitrage import compute_fields
from .report import ArbitrageReport, ArbitrageViolation

# np.trapezoid is the numpy>=2.0 spelling; np.trapz is the <2.0 alias.
_trapz = getattr(np, "trapezoid", None) or np.trapz

if TYPE_CHECKING:
    from .grid import IVSurface

# Default SAS weights — a modeling choice, documented and overridable. The
# principled normalization/weighting is itself an open research question
# (design §17); always inspect the components, not just the scalar.
DEFAULT_SAS_WEIGHTS: dict[str, float] = {
    "bfly_frac": 0.30,
    "ndm": 0.20,
    "cal_frac": 0.20,
    "cal_depth": 0.10,
    "vert_frac": 0.10,
    "bound_frac": 0.10,
}

DEFAULT_TOLERANCE = 1e-6
DEFAULT_TRUST_TOLERANCE = 1e-6
DEFAULT_MAX_VIOLATIONS = 2000

# Severity bands on the *normalized* (dimensionless) violation magnitude.
# Comparable across conditions; not tied to the detection tolerance.
SEVERITY_MINOR_MAX = 0.01
SEVERITY_MODERATE_MAX = 0.10


def validate_surface(
    surf: "IVSurface",
    *,
    tolerance: float = DEFAULT_TOLERANCE,
    trust_tolerance: float = DEFAULT_TRUST_TOLERANCE,
    weights: dict[str, float] | None = None,
    max_violations: int = DEFAULT_MAX_VIOLATIONS,
    compute_trust: bool = True,
    return_as: str = "report",
) -> Any:
    """Evaluate every no-arbitrage condition on ``surf`` and return a report.

    Parameters
    ----------
    surf:
        The :class:`~fast_vollib.surface.grid.IVSurface` to evaluate.
    tolerance:
        Normalized slack tolerance below which a condition is deemed satisfied.
    trust_tolerance:
        Round-trip ``σ→C→σ'`` residual below which a node is trusted (§6).
    weights:
        SAS component weights; defaults to :data:`DEFAULT_SAS_WEIGHTS`.
    max_violations:
        Cap on the number of localized violations returned (most-severe first);
        the total counts are always reported in ``by_condition``.
    compute_trust:
        Whether to compute the LBR round-trip trust mask.
    return_as:
        ``"report"`` (default), ``"dict"``, or ``"json"`` (design return
        conventions).
    """
    xp = surf.namespace()
    k2d, T2d, w, forward2d, discount2d = surf.broadcast(xp)
    fields = compute_fields(k2d, w, forward2d, discount2d, xp, shared_k=surf.shared_k)

    # Materialize to numpy for localization / metrics (detaches torch/jax).
    to_np = xp.to_numpy
    k2d = to_np(k2d)
    T2d = to_np(T2d)
    iv = to_np(surf.iv)
    forward2d = to_np(forward2d)
    g = to_np(fields.g)
    density = to_np(fields.density)
    density_K = to_np(fields.density_strikes)
    cal_depth = to_np(fields.cal_depth)
    cal_rel = to_np(fields.cal_rel)
    vert_mag = to_np(fields.vert_mag)
    bound_mag = to_np(fields.bound_mag)

    valid = np.isfinite(iv)
    native = surf.native_mask if surf.native_mask is not None else np.ones_like(valid, dtype=bool)

    # Per-field validity (all stencil nodes quoted) and native-origin masks.
    v_int = valid[:-2] & valid[1:-1] & valid[2:]  # k-stencil (Nk-2, Nt)
    n_int = native[:-2] & native[1:-1] & native[2:]
    v_cal = valid[:, :-1] & valid[:, 1:]  # (Nk, Nt-1)
    n_cal = native[:, :-1] & native[:, 1:]
    v_vert = valid[:-1] & valid[1:]  # (Nk-1, Nt)
    n_vert = native[:-1] & native[1:]
    v_node = valid  # (Nk, Nt)
    n_node = native

    # ---- normalized metrics (design §5) -----------------------------------
    # bfly_frac is the secondary Durrleman-g view; the localized butterfly
    # violations use the robust, per-slice-normalized price-space convexity
    # magnitude (density < 0), gated on the same dimensionless tolerance as
    # every other condition — never raw `density < 0`, whose O(h²) wing noise
    # would manufacture spurious violations on near-degenerate strikes.
    bfly_mag = to_np(fields.bfly_mag)  # = relu(-density) / slice |f|-scale
    bfly_g_viol = v_int & (g < -tolerance)
    bfly_frac = _frac(bfly_g_viol, v_int)
    bfly_viol = v_int & (bfly_mag > tolerance)
    ndm_slice = _ndm_per_slice(density, density_K, v_int)
    ndm = float(np.nanmax(ndm_slice)) if ndm_slice.size else 0.0

    cal_viol = v_cal & (cal_depth > tolerance)
    cal_frac = _frac(cal_viol, v_cal)
    cal_depth_max = _masked_max(cal_rel, v_cal)

    vert_viol = v_vert & (vert_mag > tolerance)
    vert_frac = _frac(vert_mag > tolerance, v_vert)
    bound_viol = v_node & (bound_mag > tolerance)
    bound_frac = _frac(bound_viol, v_node)

    metrics = {
        "ndm": ndm,
        "ndm_mean": float(np.nanmean(ndm_slice)) if ndm_slice.size else 0.0,
        "bfly_frac": bfly_frac,
        "cal_depth_max": cal_depth_max,
        "cal_frac": cal_frac,
        "vert_frac": vert_frac,
        "bound_frac": bound_frac,
    }
    wts = weights or DEFAULT_SAS_WEIGHTS
    sas = _sas(metrics, wts)

    # ---- localize violations (most-severe first) --------------------------
    violations: list[ArbitrageViolation] = []
    by_condition: dict[str, dict[str, Any]] = {}

    _collect(
        violations,
        by_condition,
        "butterfly",
        magnitude=bfly_mag,
        mask=bfly_viol,
        native_mask=n_int,
        coord_k=k2d[1:-1],
        coord_T=T2d[1:-1],
        index_shift=(1, 0),
        tolerance=tolerance,
    )
    _collect(
        violations,
        by_condition,
        "calendar",
        magnitude=cal_rel,
        mask=cal_viol,
        native_mask=n_cal,
        coord_k=k2d[:, :-1],
        coord_T=T2d[:, :-1],
        index_shift=(0, 0),
        tolerance=tolerance,
    )
    _collect(
        violations,
        by_condition,
        "vertical",
        magnitude=vert_mag,
        mask=vert_viol,
        native_mask=n_vert,
        coord_k=k2d[:-1],
        coord_T=T2d[:-1],
        index_shift=(0, 0),
        tolerance=tolerance,
    )
    _collect(
        violations,
        by_condition,
        "bound",
        magnitude=bound_mag,
        mask=bound_viol,
        native_mask=n_node,
        coord_k=k2d,
        coord_T=T2d,
        index_shift=(0, 0),
        tolerance=tolerance,
    )

    violations.sort(key=lambda v: v.value, reverse=True)
    truncated = len(violations) > max_violations
    if truncated:
        violations = violations[:max_violations]

    # ---- artifact separation (native vs interpolation-induced) -----------
    native_counts = {
        "butterfly": int(np.sum(bfly_viol & n_int)),
        "calendar": int(np.sum(cal_viol & n_cal)),
        "vertical": int(np.sum(vert_viol & n_vert)),
        "bound": int(np.sum(bound_viol & n_node)),
    }
    interp_counts = {
        "butterfly": int(np.sum(bfly_viol & ~n_int)),
        "calendar": int(np.sum(cal_viol & ~n_cal)),
        "vertical": int(np.sum(vert_viol & ~n_vert)),
        "bound": int(np.sum(bound_viol & ~n_node)),
    }

    # ---- round-trip trust mask (design §6) --------------------------------
    trust_mask = (
        _trust_mask(k2d, iv * iv * T2d, forward2d, T2d, iv, valid, trust_tolerance)
        if compute_trust
        else None
    )

    # An empty / all-NaN surface has nothing to certify — don't read as clean.
    coverage = float(np.mean(valid)) if valid.size else 0.0
    passed = (
        coverage > 0.0
        and bfly_frac == 0.0
        and not bool(np.any(bfly_viol))
        and cal_frac == 0.0
        and vert_frac == 0.0
        and bound_frac == 0.0
    )

    report = ArbitrageReport(
        passed=bool(passed),
        metrics=metrics,
        sas=sas,
        violations=violations,
        by_condition={
            **by_condition,
            "_truncated": truncated,
            "_max_violations": max_violations,
        },
        native=_dict_floats(native_counts),
        interpolation_induced=_dict_floats(interp_counts),
        trust_mask=trust_mask,
        tolerance=tolerance,
        context={
            "backend": xp.name,
            "shared_k": surf.shared_k,
            "Nk": surf.Nk,
            "Nt": surf.Nt,
            "coverage": coverage,
            "weights": dict(wts),
            "calendar_form": "total_variance" if surf.shared_k else "undiscounted_call",
        },
    )
    return report.render(return_as)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _frac(viol: np.ndarray, valid: np.ndarray) -> float:
    """Fraction of *valid* entries that violate (NaN-safe denominator)."""
    denom = int(np.sum(valid))
    if denom == 0:
        return 0.0
    return float(np.sum(viol & valid)) / denom


def _masked_max(values: np.ndarray, valid: np.ndarray) -> float:
    sel = values[valid]
    sel = sel[np.isfinite(sel)]
    return float(np.max(sel)) if sel.size else 0.0


def _ndm_per_slice(density: np.ndarray, strikes: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Negative-density mass ratio per maturity slice (design §5 ``ndm``).

    ``ndm = ∫ max(−f, 0) dK / ∫ |f| dK`` over the quoted interior strikes,
    integrated by the trapezoid rule.  Slices with < 2 valid points → 0.
    """
    Nt = density.shape[1]
    out = np.zeros(Nt, dtype=np.float64)
    for j in range(Nt):
        col_valid = valid[:, j] & np.isfinite(density[:, j])
        if int(np.sum(col_valid)) < 2:
            continue
        f = density[col_valid, j]
        K = strikes[col_valid, j]
        order = np.argsort(K)
        f, K = f[order], K[order]
        neg = _trapz(np.maximum(-f, 0.0), K)
        tot = _trapz(np.abs(f), K)
        out[j] = float(neg / tot) if tot > 0 else 0.0
    return out


def _sas(metrics: dict[str, float], weights: dict[str, float]) -> float:
    """Convex combination of bounded, normalized components → ``[0, 1]``."""
    comp = {
        "bfly_frac": _clip01(metrics["bfly_frac"]),
        "ndm": _clip01(metrics["ndm"]),
        "cal_frac": _clip01(metrics["cal_frac"]),
        # depth is unbounded above → squash to [0,1) so it stays comparable.
        "cal_depth": float(np.tanh(metrics["cal_depth_max"])),
        "vert_frac": _clip01(metrics["vert_frac"]),
        "bound_frac": _clip01(metrics["bound_frac"]),
    }
    wsum = sum(weights.values()) or 1.0
    return float(sum(weights.get(k, 0.0) * v for k, v in comp.items()) / wsum)


def _clip01(x: float) -> float:
    return float(min(max(x, 0.0), 1.0))


def _severity(value: float, tolerance: float | None = None) -> str:
    """Bucket a *normalized* violation magnitude into a severity band.

    All ``value``s are dimensionless and comparable across conditions
    (per-slice density scale, relative crossing depth, ∂c̃/∂K slack), so
    severity keys off absolute magnitude bands rather than a ratio to the
    detection tolerance — the latter is fixed at ~1e-6 and would label every
    real violation ``severe``.  ``tolerance`` is accepted but unused.
    """
    del tolerance
    if value < SEVERITY_MINOR_MAX:
        return "minor"
    if value < SEVERITY_MODERATE_MAX:
        return "moderate"
    return "severe"


def _collect(
    violations: list[ArbitrageViolation],
    by_condition: dict[str, dict[str, Any]],
    kind: str,
    *,
    magnitude: np.ndarray,
    mask: np.ndarray,
    native_mask: np.ndarray,
    coord_k: np.ndarray,
    coord_T: np.ndarray,
    index_shift: tuple[int, int],
    tolerance: float,
) -> None:
    """Append localized violations for one condition and record summary counts."""
    idx = np.argwhere(mask)
    by_condition[kind] = {
        "count": int(idx.shape[0]),
        "max_magnitude": float(np.max(magnitude[mask])) if idx.shape[0] else 0.0,
    }
    di, dj = index_shift
    for i, j in idx:
        val = float(magnitude[i, j])
        origin = "native" if bool(native_mask[i, j]) else "interpolation_induced"
        violations.append(
            ArbitrageViolation(
                type=kind,
                severity=_severity(val, tolerance),
                value=val,
                tolerance=tolerance,
                location=(float(coord_k[i, j]), float(coord_T[i, j])),
                origin=origin,
                index=(int(i) + di, int(j) + dj),
            )
        )


def _trust_mask(
    k2d: np.ndarray,
    w: np.ndarray,
    forward2d: np.ndarray,
    T2d: np.ndarray,
    iv: np.ndarray,
    valid: np.ndarray,
    trust_tolerance: float,
) -> np.ndarray:
    """LBR round-trip trust mask: ``σ → C → σ'``, trusted where ``|σ'−σ| < tol``.

    Prices the surface analytically (undiscounted forward call) and re-inverts
    with the Jäckel "Let's Be Rational" solver.  Nodes whose residual exceeds
    ``trust_tolerance`` (deep wings, degenerate quotes) are flagged as
    low-confidence (design §6).
    """
    from ..jackel.jackel_iv import jackel_iv_black
    from ._xp import numpy_namespace
    from .transforms import undiscounted_call

    np_ns = numpy_namespace()
    c_tilde = undiscounted_call(k2d, w, forward2d, np_ns)
    strikes = forward2d * np.exp(k2d)
    with np.errstate(all="ignore"):
        sigma_rt = jackel_iv_black(
            c_tilde.ravel(),
            forward2d.ravel(),
            strikes.ravel(),
            T2d.ravel(),
            is_call=True,
        ).reshape(iv.shape)
        residual = np.abs(sigma_rt - iv)
    trust = valid & np.isfinite(residual) & (residual < trust_tolerance)
    return trust


def _dict_floats(d: dict[str, int]) -> dict[str, float]:
    return {k: float(v) for k, v in d.items()}
