# Author: Raeid Saqur
# Date: 2026-11-19
"""Run one IV route on one evaluation set, with per-row signal attribution.

Batch outputs come from one call per contract group (model, flag, U, T, r, q), the shape of an
option chain, under on_error="warn". Then:
  * per-row signal: every NaN row is re-run alone under on_error="raise"; it is signalled iff the
    library raises with one of its row-level messages (routes without on_error are never signalled);
  * one-signal check: per call, the number of library warnings under "warn" (must be <= 1) and a
    repeat under "ignore" (no library warning, bit-identical output).
Route identity is asserted by wrapping the terminal function(s).

    PYTHONPATH=<checkout>/src <py> run_route.py --set grid --route halley_numpy --out out.csv
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if os.environ.get("FV_BLOCK_NUMBA") == "1":
    sys.modules["numba"] = None  # type: ignore[assignment]

import evalsets  # noqa: E402

# Library row-level messages (v0.2.4 and v0.2.5).
MARKERS = ("below intrinsic value", "Implied volatility is undefined")
SIGNAL_CAPABLE = (
    "halley_numpy",
    "halley_numba",
    "halley_torch",
    "halley_jax",
    "jackel_numpy",
    "jackel_torch",
    "jackel_jax",
)
ROUTES = list(SIGNAL_CAPABLE) + [
    "jackel_host",
    "jackel_torch_ad",
    "jackel_jax_ad",
    "jackel_triton_direct",
    "lbr",
]


def _is_lib(msg: str) -> bool:
    return any(m in msg for m in MARKERS)


def _wrap(mod_name: str, attr: str, fired: list[str], label: str) -> None:
    mod = importlib.import_module(mod_name)
    orig = getattr(mod, attr)
    if getattr(orig, "_l1_wrapped", False):
        return

    def w(*a, **k):
        if label not in fired:
            fired.append(label)
        return orig(*a, **k)

    w._l1_wrapped = True  # type: ignore[attr-defined]
    setattr(mod, attr, w)


def _forward(model, U, T, r, q):
    if model == "black":
        return U
    return U * np.exp((r - q) * T)


def setup(route: str, fired: list[str]) -> str:
    dev = "cpu"
    if route.startswith("halley_"):
        b = route.split("_", 1)[1]
        _wrap(f"fast_vollib.backends.{b}_backend", "implied_volatility", fired, f"halley.{b}")
        if b == "torch":
            import torch

            dev = "cuda" if torch.cuda.is_available() else "cpu"
            try:
                tk = importlib.import_module("fast_vollib.backends.triton_kernels")
                for name in dir(tk):
                    if name.endswith("_triton") and "iv" in name:
                        _wrap("fast_vollib.backends.triton_kernels", name, fired, f"triton.{name}")
            except Exception:
                pass
        if b == "jax":
            import jax

            dev = jax.devices()[0].platform
    elif route in ("jackel_numpy", "jackel_torch", "jackel_jax"):
        b = route.split("_", 1)[1]
        modname = f"fast_vollib.jackel.{b}_backend"
        _wrap(modname, "implied_volatility", fired, f"jackel.{b}")
        if b == "numpy":
            mod = importlib.import_module(modname)
            for attr in ("_jackel_iv_normalized", "_jackel_iv_black"):  # v0.2.5 / v0.2.4 terminal
                if hasattr(mod, attr):
                    _wrap(modname, attr, fired, f"jackel.numpy.host_solver({attr})")
        if b == "torch":
            _wrap(modname, "jackel_iv_black_torch", fired, "jackel.torch.native_kernel")
            import torch

            dev = "cuda" if torch.cuda.is_available() else "cpu"
        if b == "jax":
            mod = importlib.import_module(modname)
            for attr in (
                "_jit_jackel_iv_normalized",
                "_jackel_iv_normalized_j",
                "jackel_iv_black_jax",
            ):
                if getattr(mod, attr, None) is not None:
                    _wrap(modname, attr, fired, f"jackel.jax.native_kernel({attr})")
            import jax

            dev = jax.devices()[0].platform
        if b in ("torch", "jax"):  # tripwires: a host solve on a native route is a fallback
            for attr in ("jackel_iv_black", "jackel_iv_normalized"):
                try:
                    _wrap("fast_vollib.jackel.jackel_iv", attr, fired, f"HOST_FALLBACK({attr})")
                except Exception:
                    pass
    elif route in ("jackel_torch_ad", "jackel_triton_direct"):
        import torch

        dev = "cuda" if torch.cuda.is_available() else "cpu"
    elif route == "jackel_jax_ad":
        import jax

        jax.config.update("jax_enable_x64", True)
        dev = jax.devices()[0].platform
    return dev


def call(route: str, model: str, flag, U, K, T, r, q, p, on_error: str, dev: str, fired: list[str]):
    q_arg = q if model == "black_scholes_merton" else None
    if route.startswith("halley_"):
        from fast_vollib.implied_volatility import fast_implied_volatility

        b = route.split("_", 1)[1]
        out = fast_implied_volatility(
            p, U, K, T, r, flag, q_arg, model=model, backend=b, on_error=on_error, return_as="numpy"
        )
        return np.asarray(out, float)
    if route in ("jackel_numpy", "jackel_torch", "jackel_jax"):
        b = route.split("_", 1)[1]
        mod = importlib.import_module(f"fast_vollib.jackel.{b}_backend")
        return np.asarray(
            mod.implied_volatility(model, p, U, K, T, r, flag, q=q_arg, on_error=on_error), float
        )
    F = _forward(model, U, T, r, q)
    und = np.where(r == 0, p, p * np.exp(r * T))  # raw APIs take the undiscounted price
    if route == "jackel_host":
        from fast_vollib.jackel.jackel_iv import jackel_iv_black

        if "jackel_iv.jackel_iv_black" not in fired:
            fired.append("jackel_iv.jackel_iv_black")
        return np.asarray(jackel_iv_black(und, F, K, T, flag == "c"), float)
    if route == "jackel_torch_ad":
        import torch

        from fast_vollib import jackel

        _wrap(
            "fast_vollib.jackel.differentiable",
            "jackel_iv_black_torch",
            fired,
            "jackel.torch.native_kernel(autograd)",
        )
        tt = lambda v: torch.as_tensor(np.asarray(v, float), dtype=torch.float64, device=dev)  # noqa: E731
        out = jackel.implied_volatility_autograd(
            tt(p),
            tt(U),
            tt(K),
            tt(T),
            tt(r),
            list(flag),
            tt(q) if q_arg is not None else None,
            model=model,
        )
        return out.detach().cpu().numpy()
    if route == "jackel_jax_ad":
        import jax.numpy as jnp

        from fast_vollib import jackel

        if "jackel.jax.autograd" not in fired:
            fired.append("jackel.jax.autograd")
        jj = lambda v: jnp.asarray(np.asarray(v, float), dtype=jnp.float64)  # noqa: E731
        out = jackel.implied_volatility_autograd_jax(
            jj(p),
            jj(U),
            jj(K),
            jj(T),
            jj(r),
            list(flag),
            jj(q) if q_arg is not None else None,
            model=model,
        )
        return np.asarray(out)
    if route == "jackel_triton_direct":
        import torch

        from fast_vollib.jackel.triton_kernels import jackel_iv_triton

        if "jackel.triton_kernels.jackel_iv_triton" not in fired:
            fired.append("jackel.triton_kernels.jackel_iv_triton")
        assert np.all(F == F[0]) and np.all(T == T[0])
        t = lambda v: torch.as_tensor(v, dtype=torch.float64, device="cuda")  # noqa: E731
        out = jackel_iv_triton(
            t(und), float(F[0]), t(K), float(T[0]), torch.as_tensor(flag == "c", device="cuda")
        )
        return out.cpu().numpy()
    if route == "lbr":
        from py_lets_be_rational import implied_volatility_from_a_transformed_rational_guess as lbr

        if "py_lets_be_rational" not in fired:
            fired.append("py_lets_be_rational")
        out = np.empty(p.size)
        for i in range(p.size):
            try:
                out[i] = lbr(und[i], F[i], K[i], T[i], 1 if flag[i] == "c" else -1)
            except Exception:
                out[i] = np.nan
        return out
    raise ValueError(route)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--route", required=True, choices=ROUTES)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    df = evalsets.load(a.set)
    fired: list[str] = []
    dev = setup(a.route, fired)
    n = len(df)
    out = np.full(n, np.nan)
    out_ignore = np.full(n, np.nan)
    signalled = np.zeros(n, dtype=bool)
    raised = np.full(n, "", dtype=object)
    status, err = "ok", ""
    calls = []
    other_warn: set[str] = set()
    cols = ["model", "flag", "U", "T", "r", "q"]
    t0 = time.perf_counter()
    groups = df.groupby(cols, sort=True, dropna=False).indices
    can_signal = a.route in SIGNAL_CAPABLE
    for g, idx in groups.items():
        sub = df.iloc[idx]
        args = (
            sub.model.iloc[0],
            sub.flag.to_numpy(),
            sub.U.to_numpy(),
            sub.K.to_numpy(),
            sub["T"].to_numpy(),
            sub.r.to_numpy(),
            sub.q.to_numpy(),
            sub.price.to_numpy(),
        )
        rec = {"group": [str(x) for x in g], "n": int(len(idx))}
        with warnings.catch_warnings(record=True) as wl:
            warnings.simplefilter("always")
            try:
                out[idx] = call(a.route, *args, on_error="warn", dev=dev, fired=fired)
            except Exception as exc:
                status, err = "error", f"{type(exc).__name__}: {exc}"[:300]
                rec["exception"] = err
        lib = [str(w.message) for w in wl if _is_lib(str(w.message))]
        other_warn |= {
            f"{w.category.__name__}: {str(w.message)[:100]}"
            for w in wl
            if not _is_lib(str(w.message))
        }
        rec["lib_warnings_warn"] = len(lib)
        rec["lib_messages"] = lib[:3]
        if can_signal:
            with warnings.catch_warnings(record=True) as wl2:
                warnings.simplefilter("always")
                try:
                    out_ignore[idx] = call(a.route, *args, on_error="ignore", dev=dev, fired=fired)
                except Exception as exc:
                    rec["exception_ignore"] = f"{type(exc).__name__}: {exc}"[:200]
            rec["lib_warnings_ignore"] = sum(_is_lib(str(w.message)) for w in wl2)
        calls.append(rec)
    if can_signal:
        for i in np.where(np.isnan(out))[0]:
            row = df.iloc[[i]]
            args = (
                row.model.iloc[0],
                row.flag.to_numpy(),
                row.U.to_numpy(),
                row.K.to_numpy(),
                row["T"].to_numpy(),
                row.r.to_numpy(),
                row.q.to_numpy(),
                row.price.to_numpy(),
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    call(a.route, *args, on_error="raise", dev=dev, fired=fired)
                except Exception as exc:
                    raised[i] = f"{type(exc).__name__}: {exc}"[:160]
                    signalled[i] = _is_lib(str(exc))
    secs = time.perf_counter() - t0
    with open(a.out, "w") as fh:
        fh.write("key,iv_hex,signalled,ignore_identical,raised\n")
        for i in range(n):
            same = (np.isnan(out[i]) and np.isnan(out_ignore[i])) or out[i] == out_ignore[i]
            fh.write(
                f"{df.key.iloc[i]},{float(out[i]).hex()},{int(signalled[i])},{int(same) if can_signal else ''},"
                f'"{raised[i].replace(chr(34), chr(39))}"\n'
            )
    meta = {
        "route": a.route,
        "set": a.set,
        "status": status,
        "error": err,
        "fired": fired,
        "device": dev,
        "machine": platform.machine(),
        "n": n,
        "seconds": secs,
        "calls": calls,
        "max_lib_warnings_per_call": max((c["lib_warnings_warn"] for c in calls), default=0),
        "lib_warnings_under_ignore": sum(c.get("lib_warnings_ignore", 0) for c in calls),
        "other_warnings": sorted(other_warn),
    }
    with open(a.out + ".meta.json", "w") as fh:
        json.dump(meta, fh, indent=1)
    print(json.dumps({k: v for k, v in meta.items() if k not in ("calls",)}))


if __name__ == "__main__":
    main()
