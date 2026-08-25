"""Measure the overhead of the instruments layer relative to direct kernel calls.

The benchmark covers two aspects:

1. **Adapter overhead.** Compare ``price_instrument`` with its underlying
   pricing kernel across several batch sizes, including the fixed cost at
   ``N=1``.
2. **Batch construction.** ``from_arrays`` and ``from_frame`` exist so that a
   large book need not materialize one Python object per row. Compare their
   runtime and peak allocation with ``from_instruments``, which consumes scalar
   contract objects.

Run:

    uv run python scripts/benchmark_instruments.py [--backend numpy]

Timings are wall-clock medians over repeated runs after a warmup, with GPU
work synchronized. Nothing here changes numerical behaviour; it only measures.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import statistics
import sys
import time
import tracemalloc
from typing import Any, Callable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fast_vollib.api import get_all_greeks
from fast_vollib.config import get_backend
from fast_vollib.implied_volatility import fast_implied_volatility
from fast_vollib.instruments import (
    EuropeanOption,
    EuropeanOptionBatch,
    VanillaMarketInputs,
    greeks_instrument,
    implied_volatility_instrument,
    price_instrument,
)
from fast_vollib.models import fast_black_scholes

SIZES = (1, 1_000, 100_000)
CONSTRUCTION_SIZES = (1_000, 100_000)


def _sync(backend: str) -> None:
    if backend == "torch":
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except ImportError:
            pass
    elif backend == "jax":
        try:
            import jax

            jax.effects_barrier()
        except (ImportError, AttributeError):
            pass


def timed(fn: Callable[[], Any], *, backend: str, repeats: int, warmup: int = 2) -> float:
    """Median wall-clock milliseconds over ``repeats`` runs."""
    for _ in range(warmup):
        fn()
    _sync(backend)
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        _sync(backend)
        samples.append((time.perf_counter() - start) * 1e3)
    return statistics.median(samples)


def peak_mib(fn: Callable[[], Any]) -> float:
    """Peak Python-heap allocation of one call, in MiB."""
    tracemalloc.start()
    tracemalloc.reset_peak()
    fn()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / (1024 * 1024)


def inputs(n: int) -> dict[str, np.ndarray]:
    return {
        "flag": np.where(np.arange(n) % 2 == 0, "c", "p"),
        "spot": np.full(n, 100.0),
        "strike": np.linspace(80.0, 120.0, n),
        "maturity": np.full(n, 30.0 / 365.0),
        "rate": np.full(n, 0.03),
        "volatility": np.full(n, 0.2),
    }


def row(label: str, *cells: str) -> str:
    return f"| {label:<34} | " + " | ".join(f"{cell:>13}" for cell in cells) + " |"


def header(title: str, columns: tuple[str, ...]) -> None:
    print(f"\n### {title}\n")
    print(row("", *columns))
    print("|" + "-" * 36 + "|" + "|".join(["-" * 15] * len(columns)) + "|")


def benchmark_adapter_overhead(backend: str, repeats: int) -> None:
    header(
        "Adapter overhead vs direct kernel calls (median ms)",
        tuple(f"N={n:,}" for n in SIZES),
    )
    direct_price, adapter_price = [], []
    direct_greeks, adapter_greeks = [], []
    direct_iv, adapter_iv = [], []

    for n in SIZES:
        data = inputs(n)
        batch = EuropeanOptionBatch.from_arrays(
            option_type=data["flag"],
            strike=data["strike"],
            maturity=data["maturity"],
            underlier="ACME",
        )
        market = VanillaMarketInputs(
            underlying=data["spot"],
            rate=data["rate"],
            volatility=data["volatility"],
        )
        quoted = fast_black_scholes(
            data["flag"],
            data["spot"],
            data["strike"],
            data["maturity"],
            data["rate"],
            data["volatility"],
            return_as="numpy",
            backend=backend,
        )
        inversion = VanillaMarketInputs(underlying=data["spot"], rate=data["rate"], price=quoted)

        direct_price.append(
            timed(
                lambda d=data: fast_black_scholes(
                    d["flag"],
                    d["spot"],
                    d["strike"],
                    d["maturity"],
                    d["rate"],
                    d["volatility"],
                    return_as="numpy",
                    backend=backend,
                ),
                backend=backend,
                repeats=repeats,
            )
        )
        adapter_price.append(
            timed(
                lambda b=batch, m=market: price_instrument(
                    b, m, model="black_scholes", backend=backend
                ),
                backend=backend,
                repeats=repeats,
            )
        )
        direct_greeks.append(
            timed(
                lambda d=data: get_all_greeks(
                    d["flag"],
                    d["spot"],
                    d["strike"],
                    d["maturity"],
                    d["rate"],
                    d["volatility"],
                    model="black_scholes",
                    return_as="numpy",
                    backend=backend,
                ),
                backend=backend,
                repeats=repeats,
            )
        )
        adapter_greeks.append(
            timed(
                lambda b=batch, m=market: greeks_instrument(
                    b, m, model="black_scholes", backend=backend
                ),
                backend=backend,
                repeats=repeats,
            )
        )
        direct_iv.append(
            timed(
                lambda d=data, p=quoted: fast_implied_volatility(
                    p,
                    d["spot"],
                    d["strike"],
                    d["maturity"],
                    d["rate"],
                    d["flag"],
                    model="black_scholes",
                    return_as="numpy",
                    backend=backend,
                ),
                backend=backend,
                repeats=repeats,
            )
        )
        adapter_iv.append(
            timed(
                lambda b=batch, m=inversion: implied_volatility_instrument(
                    b, m, model="black_scholes", solver="halley", backend=backend
                ),
                backend=backend,
                repeats=repeats,
            )
        )

    def emit(name: str, direct: list[float], adapter: list[float]) -> None:
        print(row(f"{name}: direct kernel", *[f"{v:.3f}" for v in direct]))
        print(row(f"{name}: instrument adapter", *[f"{v:.3f}" for v in adapter]))
        print(
            row(
                f"{name}: overhead",
                *[f"{a - d:+.3f} ms" for d, a in zip(direct, adapter)],
            )
        )
        print(
            row(
                f"{name}: overhead %",
                *[f"{100 * (a - d) / d:+.1f}%" for d, a in zip(direct, adapter)],
            )
        )

    emit("price", direct_price, adapter_price)
    emit("greeks", direct_greeks, adapter_greeks)
    emit("iv (halley)", direct_iv, adapter_iv)


def benchmark_construction(backend: str, repeats: int) -> None:
    header(
        "Batch construction: time (median ms) and peak heap (MiB)",
        tuple(f"N={n:,}" for n in CONSTRUCTION_SIZES),
    )
    import pandas as pd

    times: dict[str, list[float]] = {"from_arrays": [], "from_frame": [], "from_instruments": []}
    peaks: dict[str, list[float]] = {"from_arrays": [], "from_frame": [], "from_instruments": []}

    for n in CONSTRUCTION_SIZES:
        data = inputs(n)
        frame = pd.DataFrame(
            {
                "cp": data["flag"],
                "K": data["strike"],
                "T": data["maturity"],
                "sym": np.full(n, "ACME"),
            }
        )
        options = [
            EuropeanOption(
                underlier="ACME",
                option_type=str(flag),
                strike=float(strike),
                maturity=float(maturity),
            )
            for flag, strike, maturity in zip(data["flag"], data["strike"], data["maturity"])
        ]

        builders = {
            "from_arrays": lambda d=data: EuropeanOptionBatch.from_arrays(
                option_type=d["flag"],
                strike=d["strike"],
                maturity=d["maturity"],
                underlier="ACME",
            ),
            "from_frame": lambda f=frame: EuropeanOptionBatch.from_frame(
                f,
                option_type_col="cp",
                strike_col="K",
                maturity_col="T",
                underlier_col="sym",
            ),
            "from_instruments": lambda o=options: EuropeanOptionBatch.from_instruments(o),
        }
        for name, builder in builders.items():
            times[name].append(timed(builder, backend=backend, repeats=repeats, warmup=1))
            peaks[name].append(peak_mib(builder))

    for name in times:
        print(row(f"{name}: time", *[f"{v:.3f} ms" for v in times[name]]))
    for name in peaks:
        print(row(f"{name}: peak heap", *[f"{v:.2f} MiB" for v in peaks[name]]))
    print(
        row(
            "from_arrays speedup",
            *[
                f"{times['from_instruments'][i] / times['from_arrays'][i]:.1f}x"
                for i in range(len(CONSTRUCTION_SIZES))
            ],
        )
    )
    print(
        row(
            "from_arrays peak reduction",
            *[
                f"{peaks['from_instruments'][i] / max(peaks['from_arrays'][i], 1e-9):.1f}x"
                for i in range(len(CONSTRUCTION_SIZES))
            ],
        )
    )

    # from_instruments is measured over *pre-built* objects, so the rows above
    # understate the real gap: a caller who starts from arrays would also have
    # to construct those objects first. That cost is quoted separately.
    def build_objects(n: int) -> list[EuropeanOption]:
        return [
            EuropeanOption(underlier="ACME", option_type="c", strike=100.0, maturity=0.5)
            for _ in range(n)
        ]

    def format_object_build_time(n: int) -> str:
        elapsed = timed(
            lambda: build_objects(n),
            backend=backend,
            repeats=3,
            warmup=1,
        )
        return f"{elapsed:.3f} ms"

    print(
        row(
            "(building the N objects: time)",
            *[format_object_build_time(n) for n in CONSTRUCTION_SIZES],
        )
    )
    print(
        row(
            "(building the N objects: peak)",
            *[f"{peak_mib(lambda n=n: build_objects(n)):.2f} MiB" for n in CONSTRUCTION_SIZES],
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default="auto")
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()

    backend = args.backend
    print(f"resolved_backend={get_backend(backend)}  requested={backend}  repeats={args.repeats}")
    benchmark_adapter_overhead(backend, args.repeats)
    benchmark_construction(backend, args.repeats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
