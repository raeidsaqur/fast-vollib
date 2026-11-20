# Author: Raeid Saqur
# Date: 2026-11-19
"""Cross-route consistency: max relative disagreement of each route against a base route, per set.

<py> compare_routes.py --dir <routes_dir> --base halley_numpy_cpu [--sets grid,heldout,...]
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd


def load(path):
    d = pd.read_csv(path, dtype={"iv_hex": str}, keep_default_na=False).set_index("key")
    return d.iv_hex.map(float.fromhex)


ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True)
ap.add_argument("--base", required=True)
ap.add_argument("--sets", default="grid,heldout,fixtures,devset")
a = ap.parse_args()
for s in a.sets.split(","):
    bp = os.path.join(a.dir, f"{s}__{a.base}.csv")
    if not os.path.exists(bp):
        continue
    x = load(bp)
    for p in sorted(glob.glob(os.path.join(a.dir, f"{s}__*.csv"))):
        name = os.path.basename(p)[len(s) + 2 : -4]
        if name == a.base:
            continue
        y = load(p).loc[x.index]
        m = np.isfinite(x) & np.isfinite(y) & (x > 0)
        rel = (abs(x - y) / x)[m]
        print(
            f"{s:9s} {name:26s} nan-mismatch={int((np.isnan(x) != np.isnan(y)).sum()):4d} "
            f"zero-mismatch={int(((x == 0) != (y == 0)).sum()):4d} max_rel={rel.max():.1e} "
            f"n>1e-12={int((rel > 1e-12).sum())} n>1e-9={int((rel > 1e-9).sum())}"
        )
