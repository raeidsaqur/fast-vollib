# Author: Raeid Saqur
# Date: 2026-11-19
"""Fresh-seed confirmation set (built after the solver; run once and reported as-is).

Random contracts off the development lattice, forward and spot form, r != 0 and q != 0:
model uniform over {black, black_scholes, black_scholes_merton}; U (F or S) log-uniform [1, 5000];
x = ln(F/K) uniform [-3, 3]; T log-uniform [0.003, 10]; r uniform [-0.02, 0.15]; q uniform [0, 0.06]
(BSM); sigma log-uniform [0.003, 4]. 70% exact quotes (correctly rounded price), 30% boundary
neighbours: lower bound +- {0, 1, 2, 5} ULPs and upper bound - {1, 2, 5} / + {0, 1} ULPs.

    <py> make_confirmset.py --out confirm_input.csv [--n 3000] [--seed 20261121]
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mpmath  # noqa: E402
import reference_inverse as R  # noqa: E402


def ulps(v: float, n: int) -> float:
    for _ in range(abs(n)):
        v = math.nextafter(v, math.inf if n > 0 else -math.inf)
    return v


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=20261121)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    rows = []
    for i in range(a.n):
        model = rng.choice(["black", "black_scholes", "black_scholes_merton"])
        U = math.exp(rng.uniform(math.log(1.0), math.log(5000.0)))
        x = rng.uniform(-3.0, 3.0)
        T = math.exp(rng.uniform(math.log(0.003), math.log(10.0)))
        r = rng.uniform(-0.02, 0.15)
        q = rng.uniform(0.0, 0.06) if model == "black_scholes_merton" else 0.0
        flag = rng.choice(["c", "p"])
        with mpmath.workdps(60):
            Fm = (
                R.exact(U)
                if model == "black"
                else R.exact(U) * mpmath.exp((R.exact(r) - R.exact(q)) * R.exact(T))
            )
            K = R.nearest(Fm * mpmath.exp(-mpmath.mpf(x)))
        kw = dict(F=U, K=K, T=T, is_call=flag == "c", r=r)
        if model != "black":
            kw.update(F=float("nan"), spot=U, q=q)
        c = R.Contract(**kw)
        sig = None
        if rng.random() < 0.7:
            kind = "exact"
            sig = math.exp(rng.uniform(math.log(0.003), math.log(4.0)))
            with mpmath.workdps(80):
                p = R.nearest(R.black_price(c, R.exact(sig)))
        else:
            with mpmath.workdps(80):
                Fx, Kx, Tx, Dx = c.mp_inputs()
                lb = R.nearest(Dx * max((Fx - Kx) if c.is_call else (Kx - Fx), mpmath.mpf(0)))
                ub = R.nearest(Dx * (Fx if c.is_call else Kx))
            kind = rng.choice(
                ["lb", "lb+1", "lb+2", "lb+5", "lb-1", "lb-2", "ub-1", "ub-2", "ub-5", "ub", "ub+1"]
            )
            base, off = (lb, kind[2:]) if kind.startswith("lb") else (ub, kind[2:])
            p = ulps(base, int(off) if off else 0)
            if p < 0:
                p = 0.0
        row = {
            "id": f"CF-{i:05d}",
            "model": model,
            "kind": kind,
            "flag": flag,
            "K_hex": K.hex(),
            "T_hex": T.hex(),
            "r_hex": r.hex(),
            "price_hex": float(p).hex(),
            "sigma_gen_hex": sig.hex() if sig else "",
        }
        if model == "black":
            row.update(F_hex=U.hex(), S_hex="", q_hex="")
        else:
            row.update(F_hex="", S_hex=U.hex(), q_hex=q.hex())
        rows.append(row)
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
