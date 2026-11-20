# Author: Raeid Saqur
# Date: 2026-11-19
"""Load the v0.2.5 evaluation sets into one frame: inputs + reference, decimal columns for scoring.

Sets: grid (canonical 81x81, forward form, r = 0), heldout (the varied-parameter sample: forward form, r != 0; used during
development), fixtures (adjudication fixtures, incl. spot-form W-*), devset (forward and spot form, r != 0, q != 0,
rounded boundary quotes; development data), confirm (fresh-seed random contracts with
boundary neighbours, built after the solver; run once), confirm2 (second fresh seed 20261123,
generated after the bounds refinement and the Jäckel policy; run once on the final solver).
"""

from __future__ import annotations

import gzip
import json
import math
import os

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIX = os.path.join(REPO, "tests", "fixtures", "iv_adjudication")
#: folder with {devset,confirm,confirm2}_{input,reference}.csv (the archived evidence's
#: `stress_sets/`, or the output folder of make_devset.py / make_confirmset.py)
ART = os.environ.get("FV_IV_SETS", os.path.join(REPO, "artifacts", "v0.2.5", "l1"))

REF_HEX = ["ref_f64", "sigma_int_lo", "sigma_int_hi", "h_quote", "h_forward", "h_discount"]


def _fx(v) -> float:
    if v is None or (isinstance(v, float) and math.isnan(v)) or v == "" or v == "nan":
        return math.nan
    return float.fromhex(v) if isinstance(v, str) else float(v)


def _finish(df: pd.DataFrame, name: str) -> pd.DataFrame:
    for c in REF_HEX:
        df[c] = [_fx(v) for v in df[c + "_hex"]] if c + "_hex" in df else np.nan
    df["beta"] = pd.to_numeric(df["beta"], errors="coerce")
    for c in ("sigma_int_lo_status", "sigma_int_hi_status"):
        df[c] = df[c].fillna("").astype(str)
    df["interval_empty"] = df["interval_empty"].astype(str)
    df["set"] = name
    return df.reset_index(drop=True)


def _inputs_from_hex(df: pd.DataFrame, price_col: str = "price_hex") -> pd.DataFrame:
    spot = (
        df["S_hex"].fillna("").astype(str) != ""
        if "S_hex" in df
        else pd.Series(False, index=df.index)
    )
    df["U"] = [
        _fx(s) if sp else _fx(f)
        for s, f, sp in zip(df.get("S_hex", [None] * len(df)), df["F_hex"], spot)
    ]
    df["K"] = [_fx(v) for v in df["K_hex"]]
    df["T"] = [_fx(v) for v in df["T_hex"]]
    df["r"] = (
        [0.0 if (isinstance(v, float) or v in ("", None)) else _fx(v) for v in df["r_hex"]]
        if "r_hex" in df
        else 0.0
    )
    q = df["q_hex"] if "q_hex" in df else pd.Series([""] * len(df))
    df["q"] = [0.0 if (v is None or (isinstance(v, float)) or v == "") else _fx(v) for v in q]
    df["price"] = [_fx(v) for v in df[price_col]]
    df["spot"] = spot.to_numpy()
    return df


def load(name: str) -> pd.DataFrame:
    if name == "grid":
        df = pd.read_csv(
            os.path.join(FIX, "sweep81_reference.csv.gz"), dtype=str, keep_default_na=False
        )
        df = _inputs_from_hex(df)
        df["model"] = "black"
        df["key"] = "grid:" + df.flag + ":" + df.i
        return _finish(df, name)
    if name == "heldout":
        df = pd.read_csv(
            os.path.join(FIX, "heldout_chains.csv.gz"), dtype=str, keep_default_na=False
        )
        df = _inputs_from_hex(df)
        df["model"] = "black"
        df["key"] = "heldout:" + df["id"]
        return _finish(df, name)
    if name == "fixtures":
        d = json.load(open(os.path.join(FIX, "fixtures.json")))
        recs = []
        for f in d["fixtures"]:
            ref = f["reference"]
            rec = {
                k: f.get(k)
                for k in (
                    "id",
                    "role",
                    "flag",
                    "F_hex",
                    "S_hex",
                    "q_hex",
                    "K_hex",
                    "T_hex",
                    "r_hex",
                )
            }
            rec["price_hex"] = f["quote_hex"]
            rec.update(
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
            rec.update({c + "_hex": ref.get(c + "_hex") for c in REF_HEX})
            m = f["model"]
            rec["model"] = (
                "black"
                if m.startswith("black76")
                else "black_scholes_merton"
                if "merton" in m
                else "black_scholes"
            )
            recs.append(rec)
        df = pd.DataFrame(recs)
        df = _inputs_from_hex(df)
        df["key"] = "fixtures:" + df["id"]
        return _finish(df, name)
    if name in ("devset", "confirm", "confirm2"):
        inp = pd.read_csv(os.path.join(ART, f"{name}_input.csv"), dtype=str, keep_default_na=False)
        ref = pd.read_csv(
            os.path.join(ART, f"{name}_reference.csv"), dtype=str, keep_default_na=False
        )
        ref = ref.drop(columns=[c for c in ref.columns if c in inp.columns and c != "id"])
        df = inp.merge(ref, on="id", how="left", validate="one_to_one")
        df = _inputs_from_hex(df)
        df["key"] = f"{name}:" + df["id"]
        return _finish(df, name)
    raise ValueError(name)
