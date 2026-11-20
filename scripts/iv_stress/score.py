# Author: Raeid Saqur
# Date: 2026-11-19
"""Score route outputs (run_route.py) under the frozen protocol with PER-ROW signal attribution.

Same classes, tolerances and pass rules as scoring.py (the frozen protocol); the only change
is that `signalled` is a per-row array (from run_route.py) instead of one flag per batch. Also
reports the library's documented preferred outcomes, which are stricter than the
accepted sets:
  lower_bound (exact quote at the lower bound) -> 0.0
  upper_bound / above_upper_bound              -> NaN + signal
  zero_quote                                   -> NaN + signal
  expiry                                       -> 0.0

    <py> score.py --set grid --routes-dir <dir> [--protocol protocol.json] [--fails fails.csv]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import evalsets  # noqa: E402
import scoring as s0  # noqa: E402

CLASSES = [
    "strict",
    "open_at_zero",
    "zero_quote",
    "open_at_infinity",
    "open_both",
    "inconsistent",
    "invalid_input",
    "expiry",
    "excluded",
]


def solver_of(route: str) -> str:
    return "halley" if route.startswith("halley") else "jackel"


def score(ref: pd.DataFrame, cand: np.ndarray, solver: str, proto: dict, sig: np.ndarray):
    t = proto["tolerance"]["float64"]
    rel = t["rel_tol"][solver]
    sc = s0.score_class(ref)
    r, tol = s0.tolerances(ref, solver, proto)
    lo = ref["sigma_int_lo"].to_numpy(float)
    hi = ref["sigma_int_hi"].to_numpy(float)
    fin = np.isfinite(cand)
    nan_sig = np.isnan(cand) & sig
    ok = np.zeros(len(ref), dtype=bool)
    dom = (proto.get("declared_domain", {}).get("current") or {}).get(solver)
    in_dom = np.ones(len(ref), dtype=bool)
    if dom:  # revision 3: Jaeckel refuses beta < DBL_MIN (reference beta of the exact quote)
        assert dom["rule"].startswith("beta >= DBL_MIN"), dom
        beta = pd.to_numeric(ref["beta"], errors="coerce").to_numpy(float)
        in_dom = ~(beta < 2.2250738585072014e-308)
    m = (sc == "strict") & in_dom
    ok[m] = fin[m] & (np.abs(cand[m] - r[m]) <= tol[m])
    m = (sc == "strict") & ~in_dom
    ok[m] = nan_sig[m]
    tol_hi = np.maximum(t["abs_floor"], rel * np.nan_to_num(hi))
    tol_lo = np.maximum(t["abs_floor"], rel * np.nan_to_num(lo))
    m = sc == "open_at_zero"
    ok[m] = (fin[m] & (cand[m] >= 0) & (cand[m] <= hi[m] + tol_hi[m])) | nan_sig[m]
    m = sc == "open_at_infinity"
    ok[m] = (fin[m] & (cand[m] >= lo[m] - tol_lo[m])) | nan_sig[m]
    m = sc == "open_both"
    ok[m] = (fin[m] & (cand[m] >= 0)) | nan_sig[m]
    m = sc == "zero_quote"
    ok[m] = np.isnan(cand[m]) | (fin[m] & (cand[m] >= 0) & (cand[m] <= hi[m] + tol_hi[m]))
    m = np.isin(sc, ["inconsistent", "invalid_input"])
    ok[m] = nan_sig[m]
    m = np.isin(sc, ["expiry", "excluded"])
    ok[m] = True
    # preferred outcomes (the documented row policy)
    cls = ref["cls"].astype(str).to_numpy()
    pref = np.full(len(ref), -1, dtype=int)  # -1 not applicable, 0 miss, 1 meets
    m = (cls == "lower_bound") & (sc != "zero_quote")
    pref[m] = cand[m] == 0.0
    m = np.isin(cls, ["upper_bound", "above_upper_bound"]) | (sc == "zero_quote")
    pref[m] = nan_sig[m]
    m = cls == "expiry"
    pref[m] = cand[m] == 0.0
    score.in_dom = in_dom  # type: ignore[attr-defined]
    return sc, ok, r, tol, pref


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--routes-dir", required=True)
    ap.add_argument("--protocol", default=os.path.join(evalsets.FIX, "protocol.json"))
    ap.add_argument("--out")
    ap.add_argument("--fails")
    a = ap.parse_args()
    proto = json.load(open(a.protocol))
    ref = evalsets.load(a.set)
    summary: dict = {"set": a.set, "rows": len(ref), "routes": {}}
    fails = []
    print(
        f"{'route':24s}{'dev':5s} "
        + " ".join(f"{c[:11]:>12s}" for c in CLASSES[:7])
        + "   pref(miss)  maxW ignOK  max_rel"
    )
    for path in sorted(glob.glob(os.path.join(a.routes_dir, f"{a.set}__*.csv"))):
        name = os.path.basename(path)[len(a.set) + 2 : -4]
        meta = json.load(open(path + ".meta.json"))
        out = (
            pd.read_csv(path, dtype={"iv_hex": str, "key": str}, keep_default_na=False)
            .set_index("key")
            .loc[ref.key]
        )
        c = out.iv_hex.map(float.fromhex).to_numpy()
        sig = out.signalled.astype(int).to_numpy().astype(bool)
        sc, ok, r, tol, pref = score(ref, c, solver_of(meta["route"]), proto, sig)
        in_dom = score.in_dom  # type: ignore[attr-defined]
        res = {
            "route": meta["route"],
            "device": meta["device"],
            "fired": meta["fired"],
            "status": meta["status"],
            "max_lib_warnings_per_call": meta["max_lib_warnings_per_call"],
            "lib_warnings_under_ignore": meta["lib_warnings_under_ignore"],
            "by_class": {},
        }
        ign = out.ignore_identical.astype(str)
        res["ignore_identical_all"] = bool((ign == "1").all()) if (ign != "").any() else None
        cells = []
        for cl in CLASSES:
            m = sc == cl
            if not m.any():
                cells.append("-")
                continue
            d = {
                "eligible": int(m.sum()),
                "pass": int((ok & m).sum()),
                "nan": int((np.isnan(c) & m).sum()),
                "nan_signalled": int((np.isnan(c) & sig & m).sum()),
            }
            if cl == "strict":
                rr = (np.abs(c - r) / r)[m & np.isfinite(c)]
                d["max_rel_err"] = float(rr.max()) if rr.size else None
                d["outside_domain"] = int((m & ~in_dom).sum())
            res["by_class"][cl] = d
            cells.append(f"{d['pass']}/{d['eligible']}")
        pm = pref >= 0
        res["preferred"] = {"applicable": int(pm.sum()), "met": int((pref == 1).sum())}
        summary["routes"][name] = res
        mx = res["by_class"].get("strict", {}).get("max_rel_err")
        print(
            f"{name:24s}{meta['device'][:4]:5s} "
            + " ".join(f"{x:>12s}" for x in cells[:7])
            + f"  {res['preferred']['met']}/{res['preferred']['applicable']:<6d} {meta['max_lib_warnings_per_call']:>3d}"
            f"  {str(res['ignore_identical_all'])[:4]:5s} {mx if mx is None else format(mx, '.1e')}"
        )
        for k in np.where(~ok | (pref == 0))[0]:
            fails.append(
                {
                    "route": name,
                    "key": ref.key.iloc[k],
                    "cls": ref.cls.iloc[k],
                    "scoring": sc[k],
                    "pass": bool(ok[k]),
                    "pref": int(pref[k]),
                    "ref": r[k],
                    "cand": c[k],
                    "abs_err": abs(c[k] - r[k]),
                    "tol": tol[k],
                    "signalled": bool(sig[k]),
                    "beta": ref.beta.iloc[k],
                    "price": ref.price.iloc[k],
                    "U": ref.U.iloc[k],
                    "K": ref.K.iloc[k],
                    "T": ref["T"].iloc[k],
                    "r": ref.r.iloc[k],
                    "q": ref.q.iloc[k],
                    "model": ref.model.iloc[k],
                    "flag": ref.flag.iloc[k],
                }
            )
    if a.out:
        json.dump(summary, open(a.out, "w"), indent=1)
    if a.fails:
        pd.DataFrame(fails).to_csv(a.fails, index=False)


if __name__ == "__main__":
    main()
