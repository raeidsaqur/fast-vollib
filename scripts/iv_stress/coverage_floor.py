# Author: Raeid Saqur
# Date: 2026-11-20
"""Coverage floor against a baseline release on one evaluation set.

For each route present in both folders, on the strict rows inside the solver's declared domain:
finite coverage of the baseline and the candidate (the protocol's `reporting.coverage_floor`
compares these), rows each passes under the protocol, and rows the baseline solved within the
tolerance that the candidate refuses (NaN). Writes a JSON record with --out.

    <py> coverage_floor.py --set devset --base <v0.2.4 route dir> --cand <route dir> [--out f.json]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evalsets  # noqa: E402
import score as S  # noqa: E402
import scoring as s0  # noqa: E402

DBL_MIN = 2.2250738585072014e-308


def _load(path: str, keys: pd.Series) -> np.ndarray:
    d = pd.read_csv(path, dtype={"iv_hex": str}, keep_default_na=False).set_index("key")
    return d.iv_hex.map(float.fromhex).reindex(keys).to_numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--base", required=True, help="baseline route outputs (run_route.py)")
    ap.add_argument("--cand", required=True, help="candidate route outputs (run_route.py)")
    ap.add_argument("--protocol", default=os.path.join(evalsets.FIX, "protocol.json"))
    ap.add_argument("--out")
    a = ap.parse_args()
    proto = json.load(open(a.protocol))
    ref = evalsets.load(a.set)
    sc = s0.score_class(ref)
    beta = pd.to_numeric(ref.beta, errors="coerce").to_numpy(float)
    dom = proto["declared_domain"]["current"] or {}
    out: dict = {"set": a.set, "routes": {}}
    for p in sorted(glob.glob(os.path.join(a.cand, f"{a.set}__*.csv"))):
        route = os.path.basename(p)[len(a.set) + 2 : -4]
        bp = os.path.join(a.base, os.path.basename(p))
        if not os.path.exists(bp):
            continue
        solver = S.solver_of(route)
        in_dom = ~(beta < DBL_MIN) if dom.get(solver) else np.ones(len(ref), bool)
        m = (sc == "strict") & in_dom
        b, c = _load(bp, ref.key), _load(p, ref.key)
        ok_b = S.score(ref, b, solver, proto, np.zeros(len(ref), bool))[1]
        ok_c = S.score(ref, c, solver, proto, np.isnan(c))[1]
        r = {
            "eligible": int(m.sum()),
            "finite_base": int(np.isfinite(b[m]).sum()),
            "finite_cand": int(np.isfinite(c[m]).sum()),
            "pass_base": int(ok_b[m].sum()),
            "pass_cand": int(ok_c[m].sum()),
            "base_pass_cand_nan": int((m & ok_b & np.isnan(c)).sum()),
            "base_finite_cand_nan": int((m & np.isfinite(b) & np.isnan(c)).sum()),
        }
        r["floor_met"] = r["finite_cand"] >= r["finite_base"]
        out["routes"][route] = r
        print(route, r)
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
