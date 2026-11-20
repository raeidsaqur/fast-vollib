# Author: Raeid Saqur
# Date: 2026-11-19
"""Development set for boundary and full-model logic (r != 0, q != 0, spot and forward form).

The canonical grid has r = 0 in forward form, so intrinsic and discount are exact there and it
cannot expose boundary misclassification. This set rounds them. Quotes:
  exact    : exact price at sigma, correctly rounded
  lb, lb-1, lb+1, lb+2 : the exact lower bound D*max(F-K,0) correctly rounded, and ulp neighbours
  ub, ub-1, ub+1       : the exact upper bound (D*F call, D*K put) correctly rounded, and neighbours
  zero, neg            : 0.0 and -1e-3 (OTM and ITM)
Writes the reference_inverse CSV input (flag, F_hex | S_hex q_hex, K_hex, T_hex, r_hex, price_hex,
sigma_gen_hex) plus id/model/kind columns.

    <py> make_devset.py --out devset_input.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mpmath  # noqa: E402
import reference_inverse as R  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfgs = []
    for r in (-0.01, 0.03, 0.12):
        cfgs.append(("black", r, 0.0))
        cfgs.append(("black_scholes", r, 0.0))
        cfgs.append(("black_scholes_merton", r, 0.025))
    rows = []
    n = 0
    for model, r, q in cfgs:
        for T in (0.02, 0.5, 3.0):
            for x in (-3.0, -1.0, -0.2, 0.0, 0.2, 1.0, 3.0):
                U = 100.0  # F (forward form) or S (spot form)
                with mpmath.workdps(60):
                    if model == "black":
                        Fm = R.exact(U)
                    else:
                        Fm = R.exact(U) * mpmath.exp((R.exact(r) - R.exact(q)) * R.exact(T))
                    K = R.nearest(Fm * mpmath.exp(-mpmath.mpf(x)))
                for flag in ("c", "p"):
                    kw = dict(F=U, K=K, T=T, is_call=flag == "c", r=r)
                    if model != "black":
                        kw.update(F=float("nan"), spot=U, q=q)
                    c = R.Contract(**kw)
                    quotes = []
                    for sig in (0.005, 0.03, 0.2, 1.0, 3.0):
                        with mpmath.workdps(80):
                            quotes.append(("exact", R.nearest(R.black_price(c, R.exact(sig))), sig))
                    with mpmath.workdps(80):
                        Fx, Kx, Tx, Dx = c.mp_inputs()
                        lb = Dx * max((Fx - Kx) if flag == "c" else (Kx - Fx), mpmath.mpf(0))
                        ub = Dx * (Fx if flag == "c" else Kx)
                    lbq, ubq = R.nearest(lb), R.nearest(ub)
                    quotes += [
                        ("lb", lbq, None),
                        ("lb-1", math.nextafter(lbq, -math.inf), None),
                        ("lb+1", math.nextafter(lbq, math.inf), None),
                        ("lb+2", math.nextafter(math.nextafter(lbq, math.inf), math.inf), None),
                        ("ub", ubq, None),
                        ("ub-1", math.nextafter(ubq, -math.inf), None),
                        ("ub+1", math.nextafter(ubq, math.inf), None),
                        ("zero", 0.0, None),
                        ("neg", -1e-3, None),
                    ]
                    for kind, p, sig in quotes:
                        if kind == "lb-1" and lbq == 0.0:
                            continue  # would be negative: covered by "neg"
                        row = {
                            "id": f"DV-{n:05d}",
                            "model": model,
                            "kind": kind,
                            "flag": flag,
                            "K_hex": K.hex(),
                            "T_hex": float(T).hex(),
                            "r_hex": float(r).hex(),
                            "price_hex": float(p).hex(),
                            "sigma_gen_hex": float(sig).hex() if sig is not None else "",
                        }
                        if model == "black":
                            row.update(F_hex=U.hex(), S_hex="", q_hex="")
                        else:
                            row.update(F_hex="", S_hex=U.hex(), q_hex=float(q).hex())
                        rows.append(row)
                        n += 1
    cols = [
        "id",
        "model",
        "kind",
        "flag",
        "F_hex",
        "S_hex",
        "q_hex",
        "K_hex",
        "T_hex",
        "r_hex",
        "price_hex",
        "sigma_gen_hex",
    ]
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(len(rows), "rows")


if __name__ == "__main__":
    main()
