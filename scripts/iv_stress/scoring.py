# Author: Raeid Saqur
# Date: 2026-11-19
"""Vectorised scoring under tests/fixtures/iv_adjudication/protocol.json (fv-iv-protocol/1)."""

from __future__ import annotations

import numpy as np
import pandas as pd

# The library's only on_error message (v0.2.4) is "Option price is below intrinsic value."
SIGNAL_MARKERS = ("below intrinsic value",)


def route_signalled(warnings: list[str]) -> bool:
    return any(m in w for w in warnings for m in SIGNAL_MARKERS)


def _f(col: pd.Series) -> np.ndarray:
    return pd.to_numeric(col, errors="coerce").to_numpy(float)


def score_class(ref: pd.DataFrame) -> np.ndarray:
    """Scoring class per row from the reference CSV columns."""
    cls = ref["cls"].astype(str).to_numpy()
    lo_s = ref["sigma_int_lo_status"].fillna("").astype(str).to_numpy()
    hi_s = ref["sigma_int_hi_status"].fillna("").astype(str).to_numpy()
    empty = ref["interval_empty"].astype(str).str.lower().to_numpy() == "true"
    out = np.full(len(ref), "unscored", dtype=object)
    out[cls == "invalid_input"] = "invalid_input"
    out[cls == "expiry"] = "expiry"
    out[np.isin(cls, ["unresolved", "reference_exception"])] = "excluded"
    scorable = ~np.isin(cls, ["invalid_input", "expiry", "unresolved", "reference_exception"])
    out[scorable & empty] = "inconsistent"
    ne = scorable & ~empty
    lo_open = lo_s == "touches_lower_bound"
    hi_open = hi_s == "touches_upper_bound"
    out[ne & lo_open & ~hi_open] = "open_at_zero"
    out[ne & hi_open & ~lo_open] = "open_at_infinity"
    out[ne & lo_open & hi_open] = "open_both"
    out[ne & ~lo_open & ~hi_open & (cls == "interior")] = "strict"
    out[ne & ~lo_open & ~hi_open & (cls != "interior")] = (
        "excluded"  # bounded set without a point root
    )
    beta = pd.to_numeric(ref["beta"], errors="coerce").to_numpy(float)
    out[(cls == "lower_bound") & (beta == 0)] = "zero_quote"
    return out


def tolerances(ref: pd.DataFrame, solver: str, proto: dict) -> tuple[np.ndarray, np.ndarray]:
    t = proto["tolerance"]["float64"]
    rel = t["rel_tol"][solver]
    r = _f(ref["ref_f64"])
    h = (
        np.nan_to_num(_f(ref["h_quote"]))
        + np.nan_to_num(_f(ref["h_forward"]))
        + np.nan_to_num(_f(ref["h_discount"]))
    )
    tol = np.maximum(t["abs_floor"], rel * np.nan_to_num(r)) + t["M"] * h
    return r, tol


def score(
    ref: pd.DataFrame, cand: np.ndarray, solver: str, proto: dict, signalled: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Return (scoring class, pass flag) per row."""
    t = proto["tolerance"]["float64"]
    rel = t["rel_tol"][solver]
    sc = score_class(ref)
    r, tol = tolerances(ref, solver, proto)
    lo = _f(ref["sigma_int_lo"])
    hi = _f(ref["sigma_int_hi"])
    fin = np.isfinite(cand)
    nan_sig = np.isnan(cand) & signalled
    ok = np.zeros(len(ref), dtype=bool)
    m = sc == "strict"
    ok[m] = fin[m] & (np.abs(cand[m] - r[m]) <= tol[m])
    m = sc == "open_at_zero"
    tol_hi = np.maximum(t["abs_floor"], rel * np.nan_to_num(hi))
    ok[m] = (fin[m] & (cand[m] >= 0) & (cand[m] <= hi[m] + tol_hi[m])) | nan_sig[m]
    m = sc == "open_at_infinity"
    tol_lo = np.maximum(t["abs_floor"], rel * np.nan_to_num(lo))
    ok[m] = (fin[m] & (cand[m] >= lo[m] - tol_lo[m])) | nan_sig[m]
    m = sc == "open_both"
    ok[m] = (fin[m] & (cand[m] >= 0)) | nan_sig[m]
    m = sc == "zero_quote"
    ok[m] = np.isnan(cand[m]) | (fin[m] & (cand[m] >= 0) & (cand[m] <= hi[m] + tol_hi[m]))
    m = np.isin(sc, ["inconsistent", "invalid_input"])
    ok[m] = nan_sig[m]
    m = np.isin(sc, ["expiry", "excluded"])
    ok[m] = True  # recorded, not scored
    return sc, ok
