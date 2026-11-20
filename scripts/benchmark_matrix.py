# Author: Raeid Saqur
# Date: 2026-11-19
"""Same-host benchmark matrix: fast-vollib routes and baselines, repeated timings per record.

Protocol
--------
* **Isolation.** Every (library or route, workload, N) cell runs in a fresh process
  (py_vollib_vectorized and fast-vollib's compatibility helpers patch upstream symbols). Before
  timing, a fast-vollib cell checks on its cold call that the expected terminal backend function
  ran (for torch IV on CUDA, the fused Triton kernel) and validates its output on
  well-conditioned rows; a cell that fails either check is recorded, not timed. Baselines are
  validated only.
* **Workloads.** ``price``, ``iv`` and ``greeks`` separately. IV inverts identical precomputed
  prices (Black-Scholes, all rows valid by construction). ``price+iv`` is a labelled combined
  workload for comparison with earlier reports only.
* **Sizes.** N = 10^2 ... 10^6 by default.
* **Timing.** One cold first call, then ``--reps`` (default 7) warm repetitions, each stored. GPU
  routes are timed end to end host to host with device synchronisation; for torch (Triton) and
  JAX Halley IV a device-resident time (inputs already on the device, CUDA events / blocking
  wait) is recorded separately.
* **Fairness.** All cells of one run are on one host; thread settings are recorded. A cell that
  exceeds ``--timeout`` is recorded as censored with the timeout as a lower bound.
* **Reporting.** Absolute seconds and contracts per second per record; ratios are left to the
  analysis and only between cells with the same N, workload and scope.

The manifest records versions and hardware models only (no host names or paths).

    python scripts/benchmark_matrix.py --out <dir> [--sizes 100,1000,...] [--routes ...]
    python scripts/benchmark_matrix.py --list-routes
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "fv-benchmark-matrix/1"
SEED = 20261119

#: route -> (library, workloads)
ROUTES: dict[str, tuple[str, tuple[str, ...]]] = {
    **{
        f"fast_vollib:halley:{b}": ("fast_vollib", ("price", "iv", "greeks", "price+iv"))
        for b in ("numpy", "numba", "torch", "jax")
    },
    **{f"fast_vollib:jackel:{b}": ("fast_vollib", ("iv",)) for b in ("numpy", "torch", "jax")},
    "py_vollib_vectorized": ("py_vollib_vectorized", ("price", "iv", "greeks")),
    "py_vollib": ("py_vollib", ("price", "iv")),
    "py_lets_be_rational": ("py_lets_be_rational", ("iv",)),
}
#: Scalar Python baselines are only run up to this N by default (they are censored above).
SCALAR_MAX_N = 100_000


def make_inputs(n: int) -> dict[str, np.ndarray]:
    """Seeded Black-Scholes inputs: |ln(S/K)| <= 0.5, T in [0.05, 2], sigma in [0.1, 0.6]."""
    rng = np.random.default_rng(SEED)
    s = np.full(n, 100.0)
    k = s * np.exp(rng.uniform(-0.5, 0.5, n))
    t = rng.uniform(0.05, 2.0, n)
    r = np.full(n, 0.02)
    sigma = rng.uniform(0.1, 0.6, n)
    flag = np.where(rng.random(n) < 0.5, "c", "p")
    from scipy.special import ndtr

    d1 = (np.log(s / k) + (r + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    call = s * ndtr(d1) - k * np.exp(-r * t) * ndtr(d2)
    put = k * np.exp(-r * t) * ndtr(-d2) - s * ndtr(-d1)
    return {
        "flag": flag,
        "s": s,
        "k": k,
        "t": t,
        "r": r,
        "sigma": sigma,
        "price": np.where(flag == "c", call, put),
    }


# --- child: one cell -------------------------------------------------------------------------


def _sync(backend: str) -> Callable[[], None]:
    if backend == "torch":
        import torch

        if torch.cuda.is_available():
            return torch.cuda.synchronize
    return lambda: None


def _fast_vollib_cell(
    route: str, workload: str, x: dict[str, np.ndarray]
) -> tuple[Callable[[], Any], list[str], str]:
    import importlib

    from fast_vollib import fast_black_scholes, fast_implied_volatility, get_all_greeks

    _, solver, backend = route.split(":")
    fired: list[str] = []
    module = (
        importlib.import_module(f"fast_vollib.backends.{backend}_backend")
        if solver == "halley"
        else importlib.import_module(f"fast_vollib.jackel.{backend}_backend")
    )
    for attr in ("implied_volatility", "price_black_scholes", "greeks"):
        if hasattr(module, attr):
            original = getattr(module, attr)

            def spy(*a: Any, _o: Any = original, _n: str = attr, **k: Any) -> Any:
                if f"{module.__name__}.{_n}" not in fired:
                    fired.append(f"{module.__name__}.{_n}")
                return _o(*a, **k)

            setattr(module, attr, spy)
    if solver == "halley" and backend == "torch" and workload in ("iv", "price+iv"):
        import torch

        if torch.cuda.is_available():  # the fused CUDA kernel, not a torch fallback
            from fast_vollib.backends import triton_kernels as tk

            original_kernel = tk.iv_otm_triton

            def kernel_spy(*a: Any, **k: Any) -> Any:
                if "triton_kernels.iv_otm_triton" not in fired:
                    fired.append("triton_kernels.iv_otm_triton")
                return original_kernel(*a, **k)

            tk.iv_otm_triton = kernel_spy  # type: ignore[assignment]
    f, s, k, t, r, sig, p = (x[c] for c in ("flag", "s", "k", "t", "r", "sigma", "price"))
    kw = dict(backend=backend, return_as="numpy")

    def call() -> Any:
        if workload == "price":
            return fast_black_scholes(f, s, k, t, r, sig, **kw)
        if workload == "iv":
            return fast_implied_volatility(p, s, k, t, r, f, solver=solver, on_error="ignore", **kw)
        if workload == "greeks":
            return get_all_greeks(f, s, k, t, r, sig, backend=backend, return_as="dict")
        prices = fast_black_scholes(f, s, k, t, r, sig, **kw)
        return fast_implied_volatility(
            prices, s, k, t, r, f, solver=solver, on_error="ignore", **kw
        )

    return call, fired, backend


def _baseline_cell(
    route: str, workload: str, x: dict[str, np.ndarray]
) -> tuple[Callable[[], Any], list[str], str]:
    f, s, k, t, r, sig, p = (x[c] for c in ("flag", "s", "k", "t", "r", "sigma", "price"))
    if route == "py_vollib_vectorized":
        import py_vollib_vectorized as pvv

        fired = [f"py_vollib_vectorized {pvv.__name__}"]
        if workload == "price":
            return (
                lambda: pvv.vectorized_black_scholes(f, s, k, t, r, sig, return_as="numpy"),
                fired,
                "numpy",
            )
        if workload == "iv":
            return (
                lambda: pvv.vectorized_implied_volatility(
                    p, s, k, t, r, f, return_as="numpy", on_error="ignore"
                ),
                fired,
                "numpy",
            )
        return lambda: pvv.get_all_greeks(f, s, k, t, r, sig, return_as="dict"), fired, "numpy"
    if route == "py_vollib":
        from py_vollib.black_scholes import black_scholes as bs
        from py_vollib.black_scholes.implied_volatility import implied_volatility as bs_iv

        fired = ["py_vollib (scalar loop)"]
        if workload == "price":
            return (
                lambda: np.array(
                    [bs(a, b, c, d, e, g) for a, b, c, d, e, g in zip(f, s, k, t, r, sig)]
                ),
                fired,
                "numpy",
            )

        def iv() -> np.ndarray:
            out = np.empty(len(p))
            for i in range(len(p)):
                try:
                    out[i] = bs_iv(p[i], s[i], k[i], t[i], r[i], f[i])
                except Exception:
                    out[i] = np.nan
            return out

        return iv, fired, "numpy"
    from py_lets_be_rational import implied_volatility_from_a_transformed_rational_guess as lbr

    fired = ["py_lets_be_rational (scalar loop)"]
    fwd = s * np.exp(r * t)
    und = p * np.exp(r * t)

    def iv_lbr() -> np.ndarray:
        out = np.empty(len(p))
        for i in range(len(p)):
            try:
                out[i] = lbr(und[i], fwd[i], k[i], t[i], 1 if f[i] == "c" else -1)
            except Exception:  # e.g. below intrinsic after undiscounting
                out[i] = np.nan
        return out

    return iv_lbr, fired, "numpy"


def _device_resident(route: str, x: dict[str, np.ndarray], reps: int) -> list[float] | None:
    """Halley IV with inputs already on the device (torch -> Triton, JAX), events / blocking."""
    if not route.startswith("fast_vollib:"):
        return None
    _, solver, backend = route.split(":")
    if solver != "halley" or backend not in ("torch", "jax"):
        return None
    if backend == "torch":
        import torch

        if not torch.cuda.is_available():
            return None
        from fast_vollib.backends import triton_kernels as tk

        dev = torch.device("cuda")
        T = lambda v: torch.as_tensor(v, dtype=torch.float64, device=dev)  # noqa: E731
        args = (
            T(x["price"]),
            T(x["s"]),
            T(x["k"]),
            T(x["t"]),
            T(x["r"]),
            torch.zeros_like(T(x["r"])),
        )
        is_call = torch.as_tensor(x["flag"] == "c", device=dev)
        tk.iv_otm_triton("black_scholes", *args, is_call)
        torch.cuda.synchronize()
        out = []
        for _ in range(reps):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            tk.iv_otm_triton("black_scholes", *args, is_call)
            end.record()
            torch.cuda.synchronize()
            out.append(start.elapsed_time(end) / 1e3)
        return out
    import jax
    import jax.numpy as jnp

    if jax.default_backend() == "cpu":
        return None
    from fast_vollib.backends import jax_backend as jb

    jb._ensure_x64()
    args = [jnp.asarray(x[c], dtype=jnp.float64) for c in ("price", "s", "k", "t", "r")]
    args.append(jnp.zeros_like(args[-1]))
    args.append(jnp.asarray(x["flag"] == "c"))
    fn = jb._get_jit_iv("black_scholes")
    jax.block_until_ready(fn(*args))
    out = []
    for _ in range(reps):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        out.append(time.perf_counter() - t0)
    return out


def expected_identity(route: str, workload: str) -> list[str]:
    """Terminal functions the cell must reach (checked on the cold call, before any reference)."""
    if not route.startswith("fast_vollib:"):
        return []
    _, solver, backend = route.split(":")
    if solver == "jackel":
        return [f"fast_vollib.jackel.{backend}_backend.implied_volatility"]
    module = f"fast_vollib.backends.{backend}_backend"
    names = {
        "price": ["price_black_scholes"],
        "iv": ["implied_volatility"],
        "greeks": ["greeks"],
        "price+iv": ["price_black_scholes", "implied_volatility"],
    }[workload]
    expected = [f"{module}.{n}" for n in names]
    if backend == "torch" and workload in ("iv", "price+iv"):
        import torch

        if torch.cuda.is_available():
            expected.append("triton_kernels.iv_otm_triton")
    return expected


def child(route: str, workload: str, n: int, reps: int) -> dict[str, Any]:
    x = make_inputs(n)
    library = ROUTES[route][0]
    if library == "fast_vollib":
        call, fired, backend = _fast_vollib_cell(route, workload, x)
    else:
        call, fired, backend = _baseline_cell(route, workload, x)
    sync = _sync(backend)
    t0 = time.perf_counter()
    out = call()
    sync()
    cold = time.perf_counter() - t0
    fired = list(fired)  # snapshot: the NumPy reference calls below must not count
    missing = [e for e in expected_identity(route, workload) if e not in fired]
    # Validate before timing, on well-conditioned rows (one ulp of the quote moves sigma by less
    # than 1e-12 relative): IV against the generating volatility, prices and delta against the NumPy
    # route. Ill-conditioned rows carry no volatility information for any solver.
    from fast_vollib import fast_black_scholes, get_all_greeks

    greeks0 = get_all_greeks(
        x["flag"], x["s"], x["k"], x["t"], x["r"], x["sigma"], backend="numpy", return_as="dict"
    )
    raw_vega = 100.0 * greeks0["vega"]
    with np.errstate(divide="ignore"):
        cond = np.spacing(x["price"]) / raw_vega / x["sigma"] < 1e-12
    if workload == "iv":
        got, ref, tol = np.asarray(out, float), x["sigma"], 1e-6
    elif workload == "price+iv":
        # Inverts fast-vollib's own prices: compare with the NumPy route on the same prices.
        from fast_vollib import fast_implied_volatility

        prices0 = fast_black_scholes(
            x["flag"],
            x["s"],
            x["k"],
            x["t"],
            x["r"],
            x["sigma"],
            backend="numpy",
            return_as="numpy",
        )
        ref = fast_implied_volatility(
            prices0,
            x["s"],
            x["k"],
            x["t"],
            x["r"],
            x["flag"],
            backend="numpy",
            return_as="numpy",
            on_error="ignore",
        )
        # Each backend inverts its own prices, whose absolute error is ~ eps * S (cancellation
        # in deep out-of-the-money prices); keep rows where that moves sigma by < 1e-8 relative.
        with np.errstate(divide="ignore"):
            cond = cond & (64 * np.finfo(float).eps * x["s"] / raw_vega / x["sigma"] < 1e-8)
        got, tol = np.asarray(out, float), 1e-6
    elif workload == "price":
        ref = fast_black_scholes(
            x["flag"],
            x["s"],
            x["k"],
            x["t"],
            x["r"],
            x["sigma"],
            backend="numpy",
            return_as="numpy",
        )
        got, tol, cond = np.asarray(out, float), 1e-9, np.ones(n, dtype=bool)
    else:
        ref = greeks0["delta"]
        got = np.asarray(out["delta"] if isinstance(out, dict) else out, float).reshape(-1)
        tol, cond = 1e-6, np.ones(n, dtype=bool)
    atol = {"price": 1e-12, "greeks": 1e-6}.get(workload, 0.0)  # delta lies in [-1, 1]
    excess = np.abs(got[cond] - ref[cond]) - (tol * np.abs(ref[cond]) + atol)
    finite = np.isfinite(got[cond])
    valid = bool(finite.all() and (excess.max(initial=-1.0) <= 0.0)) and not missing
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.abs(got[cond] - ref[cond]) / np.abs(ref[cond])
    validation = {
        "rows": int(cond.sum()),
        "nonfinite": int((~finite).sum()),
        "max_rel_err": float(np.nanmax(rel[np.isfinite(rel)], initial=0.0)),
        "rtol": tol,
        "atol": atol,
    }
    warm = []
    if valid:
        for _ in range(reps):
            t0 = time.perf_counter()
            call()
            sync()
            warm.append(time.perf_counter() - t0)
    rec = {
        "route": route,
        "workload": workload,
        "n": n,
        "fired": fired,
        "identity_missing": missing,
        "valid": valid,
        "validation": validation,
        "cold_s": cold,
        "warm_s": warm,
        "warm_median_s": float(np.median(warm)) if warm else None,
        "contracts_per_s": float(n / np.median(warm)) if warm else None,
    }
    if valid and workload == "iv":
        try:
            rec["device_resident_s"] = _device_resident(route, x, reps)
        except Exception as exc:  # keep the end-to-end timings
            rec["device_resident_s"] = None
            rec["device_resident_error"] = f"{type(exc).__name__}: {exc}"[:300]
    return rec


# --- parent ------------------------------------------------------------------------------------


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


def manifest() -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(ROOT), *args], capture_output=True, text=True
        ).stdout.strip()

    m: dict[str, Any] = {
        "schema": SCHEMA,
        "source_sha": git("rev-parse", "HEAD"),
        "source_dirty": bool(git("status", "--porcelain", "--", "src", "scripts")),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "cpus_allowed": len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else os.cpu_count(),
        "threads": {
            k: os.environ.get(k)
            for k in ("OMP_NUM_THREADS", "NUMBA_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "seed": SEED,
    }
    for mod in (
        "numpy",
        "scipy",
        "numba",
        "torch",
        "jax",
        "triton",
        "py_vollib",
        "py_vollib_vectorized",
        "py_lets_be_rational",
    ):
        try:
            m[mod] = str(__import__(mod).__version__)
        except Exception:
            m[mod] = None
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
    ap.add_argument("--out")
    ap.add_argument("--sizes", default="100,1000,10000,100000,1000000")
    ap.add_argument("--routes", default="")
    ap.add_argument("--workloads", default="price,iv,greeks,price+iv")
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--list-routes", action="store_true")
    ap.add_argument("--child", nargs=3, metavar=("ROUTE", "WORKLOAD", "N"), help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if a.list_routes:
        print("\n".join(ROUTES))
        return 0
    if a.child:
        route, workload, n = a.child
        print("@@REC@@" + json.dumps(child(route, workload, int(n), a.reps)))
        return 0
    if not a.out:
        ap.error("--out is required")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(manifest(), indent=1), encoding="utf-8")
    routes = a.routes.split(",") if a.routes else list(ROUTES)
    workloads = a.workloads.split(",")
    with open(out / "records.jsonl", "a", encoding="utf-8") as fh:
        for route in routes:
            for workload in workloads:
                if workload not in ROUTES[route][1]:
                    continue
                for n in (int(v) for v in a.sizes.split(",")):
                    base = {"route": route, "workload": workload, "n": n}
                    if (
                        ROUTES[route][0] in ("py_vollib", "py_lets_be_rational")
                        and n > SCALAR_MAX_N
                    ):
                        rec = {
                            **base,
                            "censored": True,
                            "reason": f"scalar baseline not run above N={SCALAR_MAX_N}",
                        }
                    else:
                        try:
                            proc = subprocess.run(
                                [
                                    sys.executable,
                                    __file__,
                                    "--child",
                                    route,
                                    workload,
                                    str(n),
                                    "--reps",
                                    str(a.reps),
                                ],
                                capture_output=True,
                                text=True,
                                timeout=a.timeout,
                            )
                            line = next(
                                (x for x in proc.stdout.splitlines() if x.startswith("@@REC@@")),
                                None,
                            )
                            rec = (
                                json.loads(line[7:])
                                if line
                                else {**base, "error": proc.stderr[-500:]}
                            )
                        except subprocess.TimeoutExpired:
                            rec = {**base, "censored": True, "timeout_s": a.timeout}
                    fh.write(json.dumps(rec) + "\n")
                    fh.flush()
                    print(
                        route,
                        workload,
                        n,
                        rec.get("warm_median_s"),
                        "valid"
                        if rec.get("valid")
                        else rec.get("reason") or rec.get("error", "")[:80],
                        flush=True,
                    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
