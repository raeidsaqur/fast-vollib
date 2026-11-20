# Author: Raeid Saqur
# Date: 2026-11-19
"""Implied-volatility validation cases and the frozen scoring protocol.

One case driver for the test suite and ``scripts/validation_matrix.py``: it loads the adjudicated
fixtures, the canonical grid and the held-out sample (``tests/fixtures/iv_adjudication``), runs an
implied-volatility route on them, and scores the result under ``protocol.json``:

* rows are scored by the set of volatilities consistent with the rounded quote (strict, open at
  zero / infinity, zero quote, inconsistent, invalid input, expiry);
* strict rows must lie within ``max(abs_floor, rel_tol * ref) + M * (h_quote + h_forward +
  h_discount)`` of the reference, or, outside a solver's declared domain, be NaN with the signal;
* "with the signal" is decided per row: the row is re-run alone with ``on_error="raise"`` and must
  raise the library's row-level ``ValueError``.
"""

from __future__ import annotations

from dataclasses import dataclass
import gzip
import io
import json
from pathlib import Path
import re
from typing import Any
import warnings

import numpy as np
import pandas as pd

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "iv_adjudication"
DBL_MIN = 2.2250738585072014e-308
#: Library row-level messages (any of them identifies the on_error signal).
SIGNAL_MARKERS = ("Implied volatility is undefined", "below intrinsic value")

SOLVERS = ("halley", "jackel")
HALLEY_BACKENDS = ("numpy", "numba", "torch", "jax")
JACKEL_BACKENDS = ("numpy", "torch", "jax")


def protocol() -> dict:
    return json.loads((FIXTURES / "protocol.json").read_text(encoding="utf-8"))


def _hex(v: Any) -> float:
    if v is None or (isinstance(v, float) and np.isnan(v)) or v in ("", "nan"):
        return float("nan")
    return float.fromhex(v) if isinstance(v, str) else float(v)


def _frame(df: pd.DataFrame, price_col: str = "price_hex") -> pd.DataFrame:
    spot = df.get("S_hex", pd.Series([""] * len(df))).fillna("").astype(str) != ""
    out = pd.DataFrame(index=df.index)
    out["key"] = df["key"]
    out["model"] = df["model"]
    out["flag"] = df["flag"].astype(str)
    out["U"] = [
        _hex(sx) if sp else _hex(fx)
        for sx, fx, sp in zip(df.get("S_hex", df["F_hex"]), df["F_hex"], spot)
    ]
    out["K"] = df["K_hex"].map(_hex)
    out["T"] = df["T_hex"].map(_hex)
    out["r"] = (
        df["r_hex"].map(lambda v: 0.0 if v in ("", None) else _hex(v)) if "r_hex" in df else 0.0
    )
    q = df["q_hex"] if "q_hex" in df else pd.Series([""] * len(df), index=df.index)
    out["q"] = q.map(
        lambda v: 0.0 if v in ("", None) or (isinstance(v, float) and np.isnan(v)) else _hex(v)
    )
    out["price"] = df[price_col].map(_hex)
    out["cls"] = df["cls"].astype(str)
    for col in ("ref_f64", "sigma_int_lo", "sigma_int_hi", "h_quote", "h_forward", "h_discount"):
        out[col] = df[col + "_hex"].map(_hex) if col + "_hex" in df else np.nan
    out["lo_status"] = df["sigma_int_lo_status"].fillna("").astype(str)
    out["hi_status"] = df["sigma_int_hi_status"].fillna("").astype(str)
    out["empty"] = df["interval_empty"].astype(str).str.lower() == "true"
    out["beta"] = pd.to_numeric(df["beta"], errors="coerce")
    out["scoring"] = scoring_class(out)
    return out.reset_index(drop=True)


def scoring_class(df: pd.DataFrame) -> np.ndarray:
    cls = df["cls"].to_numpy()
    lo_open = df["lo_status"].to_numpy() == "touches_lower_bound"
    hi_open = df["hi_status"].to_numpy() == "touches_upper_bound"
    empty = df["empty"].to_numpy()
    out = np.full(len(df), "excluded", dtype=object)
    scorable = ~np.isin(cls, ["invalid_input", "expiry", "unresolved", "reference_exception"])
    out[cls == "invalid_input"] = "invalid_input"
    out[cls == "expiry"] = "expiry"
    out[scorable & empty] = "inconsistent"
    ne = scorable & ~empty
    out[ne & lo_open & ~hi_open] = "open_at_zero"
    out[ne & hi_open & ~lo_open] = "open_at_infinity"
    out[ne & lo_open & hi_open] = "open_both"
    out[ne & ~lo_open & ~hi_open & (cls == "interior")] = "strict"
    out[(cls == "lower_bound") & (df["beta"].to_numpy() == 0)] = "zero_quote"
    return out


def load_grid(stride: int = 1) -> pd.DataFrame:
    """Canonical 81x81 x {C, P} grid; ``stride`` keeps every stride-th point on each axis."""
    with gzip.open(FIXTURES / "sweep81_reference.csv.gz", "rt") as fh:
        df = pd.read_csv(fh, dtype=str, keep_default_na=False)
    i = df["i"].astype(int)
    keep = ((i // 81) % stride == 0) & ((i % 81) % stride == 0)
    df = df[keep].copy()
    df["key"] = "grid:" + df["flag"] + ":" + df["i"]
    df["model"] = "black"
    return _frame(df)


def load_heldout() -> pd.DataFrame:
    with gzip.open(FIXTURES / "heldout_chains.csv.gz", "rt") as fh:
        df = pd.read_csv(fh, dtype=str, keep_default_na=False)
    df["key"] = "heldout:" + df["id"]
    df["model"] = "black"
    return _frame(df)


def load_fixtures() -> pd.DataFrame:
    records = json.loads((FIXTURES / "fixtures.json").read_text(encoding="utf-8"))["fixtures"]
    rows = []
    for f in records:
        ref = f["reference"]
        model = f["model"]
        row = {
            k: f.get(k) or ""
            for k in ("id", "flag", "F_hex", "S_hex", "q_hex", "K_hex", "T_hex", "r_hex")
        }
        row.update(price_hex=f["quote_hex"], key="fixtures:" + f["id"])
        row["model"] = (
            "black"
            if model.startswith("black76")
            else "black_scholes_merton"
            if "merton" in model
            else "black_scholes"
        )
        row.update(
            {
                k: ref.get(k)
                for k in (
                    "cls",
                    "sigma_int_lo_status",
                    "sigma_int_hi_status",
                    "interval_empty",
                    "beta",
                )
            }
        )
        for col in (
            "ref_f64",
            "sigma_int_lo",
            "sigma_int_hi",
            "h_quote",
            "h_forward",
            "h_discount",
        ):
            row[col + "_hex"] = ref.get(col + "_hex") or ""
        rows.append(row)
    return _frame(pd.DataFrame(rows))


# --- running a route --------------------------------------------------------------------------


@dataclass
class RouteResult:
    values: np.ndarray
    signalled: np.ndarray  # per row (NaN rows only), from re-running the row alone
    lib_warnings_per_call: int
    #: per row: NaN, and the aggregated warning of its batch call counts every NaN row of that call
    batch_accounted: np.ndarray | None = None


_COUNT = re.compile(r"Implied volatility is undefined for (\d+) of (\d+) rows")


def _call(solver: str, backend: str, sub: pd.DataFrame, on_error: str) -> np.ndarray:
    from fast_vollib import fast_implied_volatility, fast_implied_volatility_black

    model = sub["model"].iloc[0]
    args = (sub["price"].to_numpy(), sub["U"].to_numpy(), sub["K"].to_numpy())
    kw = dict(solver=solver, backend=backend, on_error=on_error, return_as="numpy")
    if model == "black":
        return np.asarray(
            fast_implied_volatility_black(
                *args, sub["r"].to_numpy(), sub["T"].to_numpy(), sub["flag"].to_numpy(), **kw
            ),
            dtype=float,
        )
    q = sub["q"].to_numpy() if model == "black_scholes_merton" else None
    return np.asarray(
        fast_implied_volatility(
            *args,
            sub["T"].to_numpy(),
            sub["r"].to_numpy(),
            sub["flag"].to_numpy(),
            q,
            model=model,
            **kw,
        ),
        dtype=float,
    )


def run_route(df: pd.DataFrame, solver: str, backend: str) -> RouteResult:
    """One call per model (mixed rows in one batch), then per-row signal checks on NaN rows."""
    values = np.full(len(df), np.nan)
    accounted = np.zeros(len(df), dtype=bool)
    worst = 0
    for _, idx in df.groupby("model").indices.items():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            values[idx] = _call(solver, backend, df.iloc[idx], "warn")
        worst = max(worst, sum(any(m in str(w.message) for m in SIGNAL_MARKERS) for w in caught))
        counts = [m for w in caught if (m := _COUNT.search(str(w.message)))]
        nan_rows = int(np.isnan(values[idx]).sum())
        if nan_rows and any(int(m[1]) == nan_rows and int(m[2]) == len(idx) for m in counts):
            accounted[idx] = np.isnan(values[idx])
    signalled = np.zeros(len(df), dtype=bool)
    for i in np.flatnonzero(np.isnan(values)):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _call(solver, backend, df.iloc[[i]], "raise")
        except ValueError as exc:
            signalled[i] = any(m in str(exc) for m in SIGNAL_MARKERS)
    return RouteResult(values, signalled, worst, accounted)


def score(
    df: pd.DataFrame, result: RouteResult, solver: str, proto: dict | None = None
) -> np.ndarray:
    """Per-row pass/fail under the frozen protocol (declared domain per solver)."""
    proto = proto or protocol()
    t = proto["tolerance"]["float64"]
    rel = t["rel_tol"][solver]
    c = result.values
    sc = df["scoring"].to_numpy()
    ref = df["ref_f64"].to_numpy(float)
    lo = df["sigma_int_lo"].to_numpy(float)
    hi = df["sigma_int_hi"].to_numpy(float)
    h = np.nan_to_num(df[["h_quote", "h_forward", "h_discount"]].to_numpy(float)).sum(axis=1)
    tol = np.maximum(t["abs_floor"], rel * np.nan_to_num(ref)) + t["M"] * h
    fin = np.isfinite(c)
    nan_sig = np.isnan(c) & result.signalled
    domain = (proto["declared_domain"].get("current") or {}).get(solver)
    in_dom = np.ones(len(df), dtype=bool)
    if domain:
        in_dom = ~(df["beta"].to_numpy(float) < DBL_MIN)
    ok = np.zeros(len(df), dtype=bool)
    m = (sc == "strict") & in_dom
    ok[m] = fin[m] & (np.abs(c[m] - ref[m]) <= tol[m])
    m = (sc == "strict") & ~in_dom
    ok[m] = nan_sig[m]
    tol_hi = np.maximum(t["abs_floor"], rel * np.nan_to_num(hi))
    tol_lo = np.maximum(t["abs_floor"], rel * np.nan_to_num(lo))
    m = sc == "open_at_zero"
    ok[m] = (fin[m] & (c[m] >= 0) & (c[m] <= hi[m] + tol_hi[m])) | nan_sig[m]
    m = sc == "open_at_infinity"
    ok[m] = (fin[m] & (c[m] >= lo[m] - tol_lo[m])) | nan_sig[m]
    m = sc == "open_both"
    ok[m] = (fin[m] & (c[m] >= 0)) | nan_sig[m]
    m = sc == "zero_quote"
    ok[m] = np.isnan(c[m]) | (fin[m] & (c[m] >= 0) & (c[m] <= hi[m] + tol_hi[m]))
    m = np.isin(sc, ["inconsistent", "invalid_input"])
    ok[m] = nan_sig[m]
    m = np.isin(sc, ["expiry", "excluded"])
    ok[m] = True
    return ok


def available_routes() -> list[tuple[str, str]]:
    from fast_vollib.backends import available_backends

    have = set(available_backends())
    routes = [("halley", b) for b in HALLEY_BACKENDS if b in have]
    routes += [("jackel", b) for b in JACKEL_BACKENDS if b in have]
    return routes


def read_csv_text(text: str) -> pd.DataFrame:  # pragma: no cover - helper for scripts
    return pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)
