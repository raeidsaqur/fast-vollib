# Author: Raeid Saqur
# Date: 2026-11-19
"""Implied-volatility validation matrix: one record per (case, route), plus a manifest.

Runs every requested route on the adjudicated evaluation sets of ``tests/fixtures/iv_adjudication``
(fixtures, canonical grid, varied-parameter sample) through the public functional API, scores each row
under the frozen protocol (``tests/_cases.py``) and writes:

* ``records.csv.gz``: set, key, model, flag, hex inputs (U = F or S, K, T, r, q, quote), point
  class, scoring class, reference, route, backend, device, output (hex), signalled (the row
  re-run alone raises under ``on_error="raise"``), batch_accounted (the batch call's aggregated
  warning counts all its NaN rows), py_lets_be_rational on the same quote (hex), error vs the
  reference, error vs py_lets_be_rational, verdict;
* ``summary.json``: pass / eligible counts per (set, route, scoring class);
* ``manifest.json``: schema version, source SHA and dirty flag, ``uv.lock`` hash, protocol
  revision, versions (Python, NumPy, SciPy, Numba, PyTorch, JAX, Triton, CUDA, py_lets_be_rational,
  mpmath), CPU / GPU model, architecture, thread count, set specification and SHA-256 of every
  output. No host names, user names or paths are recorded.

    python scripts/validation_matrix.py --out <dir> [--sets fixtures,grid,heldout] [--grid-stride 1]
        [--routes halley:numpy,jackel:torch,...]

Device selection follows the environment (``CUDA_VISIBLE_DEVICES``, ``JAX_PLATFORMS``).
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
import _cases  # noqa: E402

SCHEMA = "fv-iv-validation-matrix/1"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _version(module: str) -> str | None:
    try:
        return str(__import__(module).__version__)
    except Exception:
        return None


def _device(backend: str) -> str:
    if backend == "torch":
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    if backend == "jax":
        import jax

        return str(jax.default_backend())
    return "cpu"


def _lbr(df: pd.DataFrame) -> np.ndarray:
    """py_lets_be_rational on the same quote (undiscounted forward form), NaN where it fails."""
    try:
        from py_lets_be_rational import implied_volatility_from_a_transformed_rational_guess as lbr
    except ImportError:
        return np.full(len(df), np.nan)
    out = np.full(len(df), np.nan)
    for i, row in enumerate(df.itertuples()):
        forward = row.U if row.model == "black" else row.U * np.exp((row.r - row.q) * row.T)
        undiscounted = row.price * np.exp(row.r * row.T)
        try:
            out[i] = lbr(undiscounted, forward, row.K, row.T, 1 if row.flag == "c" else -1)
        except Exception:
            pass
    return out


def _cpu_model() -> str | None:
    """CPU model name: /proc/cpuinfo on x86, lscpu on Arm (no "model name" line there)."""
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    try:
        out = subprocess.run(["lscpu"], capture_output=True, text=True, timeout=10).stdout
        for line in out.splitlines():
            if line.strip().startswith("Model name:"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or None


def _manifest(
    sets: dict[str, int], routes: list[tuple[str, str]], files: list[Path]
) -> dict[str, Any]:
    def git(*args: str) -> str:
        run = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True)
        return run.stdout.strip()

    m: dict[str, Any] = {
        "schema": SCHEMA,
        "source_sha": git("rev-parse", "HEAD"),
        "source_dirty": bool(git("status", "--porcelain", "--", "src", "scripts", "tests")),
        "uv_lock_sha256": _sha256(ROOT / "uv.lock") if (ROOT / "uv.lock").exists() else None,
        "protocol": {k: _cases.protocol()[k] for k in ("schema", "revision")},
        "python": platform.python_version(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "cpus_allowed": len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else os.cpu_count(),
        "threads": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "NUMBA_NUM_THREADS")},
        "versions": {
            m: _version(m)
            for m in (
                "numpy",
                "scipy",
                "numba",
                "torch",
                "jax",
                "jaxlib",
                "triton",
                "mpmath",
                "py_lets_be_rational",
            )
        },
        "sets": sets,
        "routes": [f"{s}:{b}:{_device(b)}" for s, b in routes],
        "files": {f.name: _sha256(f) for f in files},
    }
    m["cpu_model"] = _cpu_model()
    try:
        import torch

        m["cuda"] = torch.version.cuda
        m["gpu_model"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        pass
    return m


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--sets", default="fixtures,grid,heldout")
    ap.add_argument("--grid-stride", type=int, default=1)
    ap.add_argument(
        "--routes", default="", help="comma-separated solver:backend; default: all available"
    )
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    routes = (
        [tuple(r.split(":")) for r in a.routes.split(",") if r]
        if a.routes
        else _cases.available_routes()
    )
    loaders = {
        "fixtures": _cases.load_fixtures,
        "grid": lambda: _cases.load_grid(stride=a.grid_stride),
        "heldout": _cases.load_heldout,
    }
    frames, summary, sets = [], {}, {}
    for name in a.sets.split(","):
        df = loaders[name]()
        sets[name] = len(df)
        lbr = _lbr(df)
        for solver, backend in routes:
            result = _cases.run_route(df, solver, backend)
            ok = _cases.score(df, result, solver)
            ref = df["ref_f64"].to_numpy(float)
            with np.errstate(divide="ignore", invalid="ignore"):
                err_ref = np.abs(result.values - ref) / ref
                err_lbr = np.abs(result.values - lbr) / lbr
            rec = pd.DataFrame(
                {
                    "set": name,
                    "key": df["key"],
                    "model": df["model"],
                    "flag": df["flag"],
                    "U_hex": df["U"].map(float.hex),
                    "K_hex": df["K"].map(float.hex),
                    "T_hex": df["T"].map(float.hex),
                    "r_hex": df["r"].map(float.hex),
                    "q_hex": df["q"].map(float.hex),
                    "quote_hex": df["price"].map(float.hex),
                    "cls": df["cls"],
                    "scoring": df["scoring"],
                    "ref_hex": [float(v).hex() for v in ref],
                    "solver": solver,
                    "backend": backend,
                    "device": _device(backend),
                    "iv_hex": [float(v).hex() for v in result.values],
                    "signalled": result.signalled,
                    "batch_accounted": result.batch_accounted,
                    "lbr_hex": [float(v).hex() for v in lbr],
                    "rel_err_ref": err_ref,
                    "rel_err_lbr": err_lbr,
                    "pass": ok,
                }
            )
            frames.append(rec)
            route = f"{solver}:{backend}:{_device(backend)}"
            for cls, grp in rec.groupby("scoring"):
                summary.setdefault(name, {}).setdefault(route, {})[cls] = {
                    "eligible": int(len(grp)),
                    "pass": int(grp["pass"].sum()),
                }
            summary[name][route]["lib_warnings_per_call_max"] = int(result.lib_warnings_per_call)
            print(
                name,
                route,
                {
                    c: f"{v['pass']}/{v['eligible']}"
                    for c, v in summary[name][route].items()
                    if isinstance(v, dict)
                },
                flush=True,
            )
    records = out / "records.csv.gz"
    with gzip.open(records, "wt") as fh:
        pd.concat(frames).to_csv(fh, index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    manifest = _manifest(sets, routes, [records, out / "summary.json"])
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
