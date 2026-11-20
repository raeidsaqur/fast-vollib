# Author: Raeid Saqur
# Date: 2026-11-19
"""Frozen IV timing harness for the v0.2.5 performance budget (v0.2.4 vs candidate, back to back).

One fresh process per (source tree, route). Each process loads the same saved,
seeded inputs, validates the route's identity and output, then records the cold
first call and 7 warm repetitions (each stored). End-to-end host->host timing:
NumPy arrays in, NumPy array out (GPU routes therefore include H2D/D2H and sync).

Usage:
    <python> iv_timing.py make-inputs --out inputs.npz [--n 1000000]
    <python> iv_timing.py run --src <checkout>/src --inputs inputs.npz --routes numpy,numba,torch \
        --out records.jsonl [--label v0.2.4]
The manifest records versions and hardware model only (no host names or paths).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from typing import Any

WARM_REPS = 7
SEED = 20261119


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def make_inputs(out: str, n: int) -> None:
    import numpy as np

    rng = np.random.default_rng(SEED)
    S = np.full(n, 100.0)
    x = rng.uniform(-0.5, 0.5, n)  # log-moneyness ln(S/K)
    K = S * np.exp(-x)
    t = rng.uniform(0.05, 2.0, n)
    r = np.full(n, 0.02)
    sigma = rng.uniform(0.10, 0.60, n)
    is_call = rng.random(n) < 0.5
    # Price with an explicit, closed-form NumPy/SciPy evaluator so inputs do not
    # depend on fast-vollib's backend selection.
    from scipy.special import ndtr

    sq = sigma * np.sqrt(t)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * t) / sq
    d2 = d1 - sq
    df = np.exp(-r * t)
    call = S * ndtr(d1) - K * df * ndtr(d2)
    put = K * df * ndtr(-d2) - S * ndtr(-d1)
    price = np.where(is_call, call, put)
    np.savez(
        out, S=S, K=K, t=t, r=r, sigma=sigma, is_call=is_call, price=price, seed=np.int64(SEED)
    )
    print(json.dumps({"inputs": os.path.basename(out), "n": n, "sha256": _sha256(out)}))


ROUTES = {
    # label: (callable-kind, backend, extra)
    "numpy": ("halley", "numpy"),
    "numba": ("halley", "numba"),
    "torch": ("halley", "torch"),  # CUDA visible + triton installed -> Triton kernel
    "jax": ("halley", "jax"),
    "jackel_numpy": ("jackel", "numpy"),
}


def _route_identity(kind: str, backend: str) -> list[str]:
    """Wrap terminal functions; returns a list that is filled on call."""
    import importlib

    fired: list[str] = []
    targets = []
    if kind == "halley":
        targets = [("fast_vollib.backends.%s_backend" % backend, "implied_volatility")]
        if backend == "torch":
            targets += [
                ("fast_vollib.backends.triton_kernels", "bsm_iv_triton"),  # v0.2.4
                ("fast_vollib.backends.triton_kernels", "iv_otm_triton"),  # v0.2.5
                ("fast_vollib.backends.torch_backend", "_get_compiled_halley"),
            ]
    else:
        targets = [("fast_vollib.jackel.%s_backend" % backend, "implied_volatility")]
    for mod_name, attr in targets:
        try:
            mod = importlib.import_module(mod_name)
        except Exception:
            continue
        if not hasattr(mod, attr):
            continue
        orig = getattr(mod, attr)

        def make(orig=orig, label=f"{mod_name.split('.')[-1]}.{attr}"):
            def w(*a, **k):
                if label not in fired:
                    fired.append(label)
                return orig(*a, **k)

            return w

        setattr(mod, attr, make())
    return fired


def child(route: str, inputs: str) -> None:
    import numpy as np

    kind, backend = ROUTES[route]
    d = np.load(inputs)
    price, S, K, t, r = d["price"], d["S"], d["K"], d["t"], d["r"]
    flag = np.where(d["is_call"], "c", "p")
    fired = _route_identity(kind, backend)

    if kind == "halley":
        from fast_vollib import fast_implied_volatility

        def call() -> Any:
            return fast_implied_volatility(
                price, S, K, t, r, flag, return_as="numpy", backend=backend
            )
    else:
        from fast_vollib.jackel import numpy_backend as jnb
        from fast_vollib.utils.broadcast import preprocess_flags

        fl = preprocess_flags(flag)

        def call() -> Any:
            return jnb.implied_volatility("black_scholes", price, S, K, t, r, fl)

    sync = None
    if backend == "torch":
        import torch

        if torch.cuda.is_available():
            sync = torch.cuda.synchronize
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t0 = time.perf_counter()
        out = call()
        if sync:
            sync()
        cold = time.perf_counter() - t0
        warm = []
        for _ in range(WARM_REPS):
            t0 = time.perf_counter()
            out = call()
            if sync:
                sync()
            warm.append(time.perf_counter() - t0)
    out = np.asarray(out, dtype=float)
    err = np.abs(out - d["sigma"]) / d["sigma"]
    finite = np.isfinite(out)
    rec = {
        "route": route,
        "kind": kind,
        "backend": backend,
        "fired": fired,
        "n": int(price.size),
        "cold_s": cold,
        "warm_s": warm,
        "warm_median_s": float(np.median(warm)),
        "warm_min_s": float(np.min(warm)),
        "warm_max_s": float(np.max(warm)),
        "finite_frac": float(finite.mean()),
        "relerr_vs_generating_median": float(np.nanmedian(err)),
        "relerr_vs_generating_p99": float(np.nanpercentile(err, 99)),
        "relerr_vs_generating_max": float(np.nanmax(err)),
        "out_sha256": hashlib.sha256(np.nan_to_num(out, nan=-1.0).tobytes()).hexdigest(),
    }
    print("@@REC@@" + json.dumps(rec))


def _manifest(src: str) -> dict[str, Any]:
    import numpy as np

    repo = os.path.dirname(os.path.abspath(src))
    sha = subprocess.run(
        ["git", "-C", repo, "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "-C", repo, "status", "--porcelain", "--", "src"],
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    lock = os.path.join(repo, "uv.lock")
    m: dict[str, Any] = {
        "schema": "fv-iv-timing/1",
        "source_sha": sha,
        "source_dirty_src": dirty,
        "uv_lock_sha256": _sha256(lock) if os.path.exists(lock) else None,
        "python": platform.python_version(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "cpu_count": os.cpu_count(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "numba_num_threads_env": os.environ.get("NUMBA_NUM_THREADS"),
    }
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    m["cpu_model"] = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    if "cpu_model" not in m:
        m["cpu_model"] = platform.processor() or None
    for mod in ("numba", "torch", "jax", "triton"):
        try:
            m[mod] = __import__(mod).__version__
        except Exception:
            m[mod] = None
    try:
        import torch

        m["torch_cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            m["gpu_model"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    try:
        q = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,driver_version",
                "--format=csv,noheader",
                "-i",
                os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0] or "0",
            ],
            capture_output=True,
            text=True,
            timeout=20,
        ).stdout.strip()
        m["gpu_util_mem_driver_at_start"] = q
    except Exception:
        pass
    try:
        m["loadavg_at_start"] = os.getloadavg()
    except OSError:
        pass
    return m


def run(src: str, inputs: str, routes: list[str], out: str, label: str) -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.abspath(src)
    man = _manifest(src)
    man.update({"label": label, "inputs_sha256": _sha256(inputs), "warm_reps": WARM_REPS})
    with open(out, "a") as fh:
        for route in routes:
            proc = subprocess.run(
                [sys.executable, os.path.abspath(__file__), "child", route, inputs],
                env=env,
                capture_output=True,
                text=True,
                timeout=3600,
            )
            rec = None
            for line in proc.stdout.splitlines():
                if line.startswith("@@REC@@"):
                    rec = json.loads(line[7:])
            if rec is None:
                rec = {
                    "route": route,
                    "status": "crash",
                    "rc": proc.returncode,
                    "stderr_tail": proc.stderr.splitlines()[-5:],
                }
            rec["manifest"] = man
            fh.write(json.dumps(rec) + "\n")
            print(
                f"{label:8s} {route:13s} fired={rec.get('fired')} cold={rec.get('cold_s', float('nan')):.4f}s "
                f"warm_med={rec.get('warm_median_s', float('nan')):.4f}s "
                f"[{rec.get('warm_min_s', float('nan')):.4f},{rec.get('warm_max_s', float('nan')):.4f}] "
                f"p99err={rec.get('relerr_vs_generating_p99', float('nan')):.2e}",
                flush=True,
            )


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("make-inputs")
    a.add_argument("--out", required=True)
    a.add_argument("--n", type=int, default=1_000_000)
    b = sub.add_parser("run")
    b.add_argument("--src", required=True)
    b.add_argument("--inputs", required=True)
    b.add_argument("--routes", default="numpy,numba,torch")
    b.add_argument("--out", required=True)
    b.add_argument("--label", default="")
    c = sub.add_parser("child")
    c.add_argument("route")
    c.add_argument("inputs")
    ns = ap.parse_args()
    if ns.cmd == "make-inputs":
        make_inputs(ns.out, ns.n)
    elif ns.cmd == "run":
        run(ns.src, ns.inputs, ns.routes.split(","), ns.out, ns.label)
    else:
        child(ns.route, ns.inputs)


if __name__ == "__main__":
    main()
