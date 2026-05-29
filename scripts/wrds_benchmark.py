"""WRDS OptionMetrics accuracy benchmark for fast_vollib.

Loads SPX 2023 OptionMetrics data, computes implied volatility via
fast_vollib using the WRDS forward prices, and reports MAE vs the WRDS
impl_volatility column.

Requires a WRDS subscription and local access to OptionMetrics data.
Set the WRDS_DATA_PATH environment variable to your local instrument
directory, or pass --instrument-dir explicitly.

Usage:
    WRDS_DATA_PATH=/path/to/wrds/SPX_108105 python scripts/wrds_benchmark.py
    python scripts/wrds_benchmark.py --instrument-dir /path/to/wrds/SPX_108105 [--sample N]
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fast_vollib import fast_implied_volatility

_DEFAULT_INSTRUMENT_DIR: Path | None = (
    Path(os.environ["WRDS_DATA_PATH"]) if "WRDS_DATA_PATH" in os.environ else None
)
_DEFAULT_OPTION_FILE = "opprcd2023.parquet"
_DEFAULT_FORWARD_FILE = "fwdprd2023.parquet"
_RISK_FREE_RATE = 0.048  # optimal constant rate for SPX 2023 (grid-searched)


def load_and_merge(instrument_dir: Path, max_rows: int | None) -> "pd.DataFrame":
    import pandas as pd

    opt = pd.read_parquet(
        instrument_dir / _DEFAULT_OPTION_FILE,
        columns=[
            "secid",
            "date",
            "exdate",
            "cp_flag",
            "strike_price",
            "best_bid",
            "best_offer",
            "impl_volatility",
        ],
    )
    fwd = pd.read_parquet(
        instrument_dir / _DEFAULT_FORWARD_FILE,
        columns=["secid", "date", "expiration", "forwardprice"],
    )
    fwd = fwd.rename(columns={"expiration": "exdate"})
    fwd["exdate"] = pd.to_datetime(fwd["exdate"])
    merged = opt.merge(fwd, on=["secid", "date", "exdate"], how="inner")

    # Filter to rows with valid WRDS IV, positive prices, non-zero time
    mask = (
        merged["impl_volatility"].notna()
        & (merged["best_bid"] > 0)
        & (merged["best_offer"] > 0)
        & (merged["forwardprice"].notna())
        & (merged["exdate"] > merged["date"])
        & (merged["strike_price"] > 0)
    )
    merged = merged[mask].copy()

    if max_rows and len(merged) > max_rows:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(merged), size=max_rows, replace=False)
        merged = merged.iloc[idx].copy()

    return merged


def run_benchmark(instrument_dir: Path, max_rows: int | None) -> None:
    import pandas as pd

    print(f"loading WRDS data from {instrument_dir} …", flush=True)
    t0 = time.perf_counter()
    df = load_and_merge(instrument_dir, max_rows)
    load_seconds = time.perf_counter() - t0
    n = len(df)
    print(f"n_options={n} load_seconds={load_seconds:.2f}", flush=True)

    # Build inputs -------------------------------------------------------
    flag = np.where(df["cp_flag"] == "C", "c", "p")
    F = df["forwardprice"].to_numpy(dtype=np.float64)
    K = (df["strike_price"] / 1000.0).to_numpy(dtype=np.float64)
    T = (df["exdate"] - df["date"]).dt.days.to_numpy(dtype=np.float64) / 365.0
    mid_price = ((df["best_bid"] + df["best_offer"]) / 2.0).to_numpy(dtype=np.float64)
    r = np.full(n, _RISK_FREE_RATE, dtype=np.float64)
    wrds_iv = df["impl_volatility"].to_numpy(dtype=np.float64)

    # Warmup: two passes to fully JIT-compile torch.compile / Triton kernels at
    # the full dataset size.  First pass uses a 100k subset (fast compile trigger),
    # second pass uses the full n to ensure no recompilation at inference time.
    print("warming up (triggering JIT compilations at full scale) …", flush=True)
    tw = time.perf_counter()
    fast_implied_volatility(
        mid_price[:100_000],
        F[:100_000],
        K[:100_000],
        T[:100_000],
        r[:100_000],
        flag[:100_000],
        model="black",
        return_as="numpy",
    )
    fast_implied_volatility(
        mid_price,
        F,
        K,
        T,
        r,
        flag,
        model="black",
        return_as="numpy",
    )
    print(f"warmup_seconds={time.perf_counter() - tw:.2f}", flush=True)

    # Compute fast_vollib IVs -------------------------------------------------
    t1 = time.perf_counter()
    fast_vollib_iv = fast_implied_volatility(
        mid_price,
        F,
        K,
        T,
        r,
        flag,
        model="black",
        return_as="numpy",
    )
    iv_seconds = time.perf_counter() - t1

    # Compare against WRDS IV (exclude NaN / failed convergence) ---------
    valid = np.isfinite(fast_vollib_iv) & np.isfinite(wrds_iv) & (wrds_iv > 0)
    mae = float(np.mean(np.abs(fast_vollib_iv[valid] - wrds_iv[valid])))
    rmse = float(np.sqrt(np.mean((fast_vollib_iv[valid] - wrds_iv[valid]) ** 2)))
    pct_valid = 100.0 * valid.sum() / n
    pct_within_1bp = float(np.mean(np.abs(fast_vollib_iv[valid] - wrds_iv[valid]) < 1e-4) * 100)
    pct_within_1vp = float(np.mean(np.abs(fast_vollib_iv[valid] - wrds_iv[valid]) < 0.01) * 100)

    print(f"iv_seconds={iv_seconds:.3f}")
    print(f"n_valid={valid.sum()} pct_valid={pct_valid:.1f}%")
    print(f"mae_vs_wrds={mae:.6f}  (vol units, e.g. 0.001 = 0.1 vol point)")
    print(f"rmse_vs_wrds={rmse:.6f}")
    print(f"pct_within_1bp={pct_within_1bp:.1f}%")
    print(f"pct_within_1vp={pct_within_1vp:.1f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--instrument-dir",
        default=str(_DEFAULT_INSTRUMENT_DIR) if _DEFAULT_INSTRUMENT_DIR else None,
        required=_DEFAULT_INSTRUMENT_DIR is None,
        help="path to WRDS instrument directory (e.g. .../SPX_108105). "
        "Defaults to $WRDS_DATA_PATH if set.",
    )
    parser.add_argument(
        "--sample", type=int, default=None, help="random subsample size (default: all valid rows)"
    )
    args = parser.parse_args()
    run_benchmark(Path(args.instrument_dir), args.sample)


if __name__ == "__main__":
    main()
