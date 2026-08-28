"""Measure the cost of simulating and pricing, per backend.

Three things are worth knowing before choosing a path budget:

1. **Simulation.** Wall time and peak allocation for the path buffer itself,
   at the sizes a valuation actually uses.
2. **Pricing.** What a terminal payoff and a path payoff add on top of the
   simulation they share.
3. **Antithetic variance reduction.** Whether pairing helps is a property of
   the payoff, not of the estimator, so it is measured per product at an equal
   total path count rather than assumed.

Run:

    uv run python scripts/benchmark_monte_carlo.py [--backend numpy] [--gradients]

Timings are wall-clock medians over repeated runs after a warmup, with device
work synchronized. Peak allocation is Python-heap only, so it measures NumPy
faithfully and undercounts torch and jax, which allocate outside it -- the
figures are comparable within a backend, not across backends. Nothing here
changes numerical behaviour; it only measures.
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

from fast_vollib.instruments import (
    AsianOption,
    BarrierOption,
    EuropeanOption,
    LookbackOption,
    VanillaMarketInputs,
    VarianceSwap,
)
from fast_vollib.processes import GBM
from fast_vollib.simulation import MonteCarloEngine, simulate

#: (paths, steps) pairs spanning an ordinary valuation and a heavy one.
SIZES = ((10_000, 64), (100_000, 64), (100_000, 256))

SPOT, STRIKE, RATE, VOLATILITY, MATURITY = 100.0, 100.0, 0.03, 0.2, 1.0

PRODUCTS: dict[str, Any] = {
    "european": EuropeanOption(
        underlier="ACME", option_type="call", strike=STRIKE, maturity=MATURITY
    ),
    "asian": AsianOption(
        underlier="ACME",
        option_type="call",
        strike=STRIKE,
        averaging_method="arithmetic",
        strike_convention="fixed",
        maturity=MATURITY,
    ),
    "barrier": BarrierOption(
        underlier="ACME",
        option_type="call",
        strike=STRIKE,
        barrier=130.0,
        barrier_type="up_and_out",
        maturity=MATURITY,
    ),
    "lookback": LookbackOption(
        underlier="ACME",
        option_type="call",
        strike=STRIKE,
        strike_convention="fixed",
        maturity=MATURITY,
    ),
    "variance": VarianceSwap(underlier="ACME", strike_variance=VOLATILITY**2, maturity=MATURITY),
}


def _sync(backend: str) -> None:
    if backend == "torch":
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except ImportError:  # pragma: no cover - optional backend
            pass
    elif backend == "jax":
        try:
            import jax

            jax.effects_barrier()
        except (ImportError, AttributeError):  # pragma: no cover - optional backend
            pass


def timed(fn: Callable[[], Any], *, backend: str, repeats: int, warmup: int = 1) -> float:
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


def backend_inputs(backend: str) -> tuple[Any, Any, Any, Any]:
    """``(spot, rate, process, rng)`` native to ``backend``."""
    if backend == "torch":
        import torch

        generator = torch.Generator()
        generator.manual_seed(0)
        return (
            torch.tensor(SPOT, dtype=torch.float64),
            torch.tensor(RATE, dtype=torch.float64),
            GBM(
                torch.tensor(RATE, dtype=torch.float64),
                torch.tensor(VOLATILITY, dtype=torch.float64),
            ),
            generator,
        )
    if backend == "jax":
        import jax
        import jax.numpy as jnp

        jax.config.update("jax_enable_x64", True)
        return (
            jnp.float64(SPOT),
            jnp.float64(RATE),
            GBM(jnp.float64(RATE), jnp.float64(VOLATILITY)),
            jax.random.key(0),
        )
    return SPOT, RATE, GBM.risk_neutral(rate=RATE, volatility=VOLATILITY), 0


def row(label: str, *cells: str) -> str:
    return f"| {label:<28} | " + " | ".join(f"{cell:>13}" for cell in cells) + " |"


def header(title: str, columns: tuple[str, ...]) -> None:
    print(f"\n### {title}\n")
    print(row("", *columns))
    print("|" + "-" * 30 + "|" + "|".join(["-" * 15] * len(columns)) + "|")


def benchmark_simulation(backend: str, repeats: int) -> None:
    columns = tuple(f"{paths // 1000}k x {steps}" for paths, steps in SIZES)
    header("Simulation: median ms", columns)
    spot, _rate, process, rng = backend_inputs(backend)

    times, memory, raw = [], [], []
    for paths, steps in SIZES:
        grid = np.linspace(0.0, MATURITY, steps + 1)

        def run(paths: int = paths, grid: np.ndarray = grid) -> Any:
            return simulate(
                "ACME",
                process,
                initial_state=spot,
                time_grid=grid if backend == "numpy" else _as_native(grid, backend),
                n_paths=paths,
                rng=rng,
            )

        times.append(f"{timed(run, backend=backend, repeats=repeats):.1f}")
        memory.append(f"{peak_mib(run):.1f}")
        raw.append(f"{paths * (steps + 1) * 8 / (1024 * 1024):.1f}")
    print(row("simulate", *times))
    print(row("peak Python heap (MiB)", *memory))
    print(row("raw float64 paths (MiB)", *raw))


def _as_native(grid: np.ndarray, backend: str) -> Any:
    if backend == "torch":
        import torch

        return torch.as_tensor(grid, dtype=torch.float64)
    import jax.numpy as jnp

    return jnp.asarray(grid, dtype=jnp.float64)


def benchmark_pricing(backend: str, repeats: int) -> None:
    columns = tuple(f"{paths // 1000}k x {steps}" for paths, steps in SIZES)
    header("Pricing: median ms, including simulation", columns)
    spot, rate, process, rng = backend_inputs(backend)
    market = VanillaMarketInputs(underlying=spot, rate=rate)
    engine = MonteCarloEngine()

    for name, contract in PRODUCTS.items():
        cells = []
        for paths, steps in SIZES:

            def run(paths: int = paths, steps: int = steps, contract: Any = contract) -> Any:
                return engine.price(
                    contract,
                    market,
                    process=process,
                    n_paths=paths,
                    n_steps=steps,
                    rng=rng,
                    return_native=True,
                )

            cells.append(f"{timed(run, backend=backend, repeats=repeats):.1f}")
        print(row(name, *cells))


def benchmark_antithetic(backend: str, repeats: int) -> None:
    """Variance reduction per product, at an equal total path count.

    Reported as the ratio of ordinary to antithetic standard error. Above 1
    means pairing helped for this product and this seed; it is not a general
    claim, and a value near or below 1 is a real answer rather than a failure.
    """
    del repeats
    header("Antithetic stderr ratio at 100k paths", ("ordinary", "antithetic", "ratio"))
    spot, rate, process, rng = backend_inputs(backend)
    market = VanillaMarketInputs(underlying=spot, rate=rate)

    for name, contract in PRODUCTS.items():
        plain = MonteCarloEngine().price(
            contract, market, process=process, n_paths=100_000, n_steps=64, rng=rng
        )
        paired = MonteCarloEngine(antithetic=True).price(
            contract, market, process=process, n_paths=100_000, n_steps=64, rng=rng
        )
        ratio = plain.stderr / paired.stderr if paired.stderr > 0 else float("inf")
        print(
            row(
                name,
                f"{plain.stderr:.3e}",
                f"{paired.stderr:.3e}",
                f"{ratio:.2f}x",
            )
        )


def benchmark_gradients(backend: str, repeats: int) -> None:
    """The cost of keeping a tape, on the backends that have one."""
    if backend not in {"torch", "jax"}:
        print("\nGradients are measured on the torch and jax backends only.")
        return
    header("Gradient of a price w.r.t. four inputs: median ms", ("100k x 64",))
    paths, steps = 100_000, 64

    if backend == "torch":
        import torch

        def run() -> Any:
            spot = torch.tensor(SPOT, dtype=torch.float64, requires_grad=True)
            rate = torch.tensor(RATE, dtype=torch.float64, requires_grad=True)
            drift = torch.tensor(RATE, dtype=torch.float64, requires_grad=True)
            volatility = torch.tensor(VOLATILITY, dtype=torch.float64, requires_grad=True)
            result = MonteCarloEngine().price(
                PRODUCTS["european"],
                VanillaMarketInputs(underlying=spot, rate=rate),
                process=GBM(drift, volatility),
                n_paths=paths,
                n_steps=steps,
                rng=0,
                return_native=True,
            )
            result.price.backward()
            return spot.grad
    else:
        import jax
        import jax.numpy as jnp

        jax.config.update("jax_enable_x64", True)
        key = jax.random.key(0)

        def priced(spot: Any, rate: Any, drift: Any, volatility: Any) -> Any:
            return (
                MonteCarloEngine()
                .price(
                    PRODUCTS["european"],
                    VanillaMarketInputs(underlying=spot, rate=rate),
                    process=GBM(drift, volatility),
                    n_paths=paths,
                    n_steps=steps,
                    rng=key,
                    return_native=True,
                )
                .price
            )

        gradient = jax.grad(priced, argnums=(0, 1, 2, 3))

        def run() -> Any:
            return gradient(
                jnp.float64(SPOT),
                jnp.float64(RATE),
                jnp.float64(RATE),
                jnp.float64(VOLATILITY),
            )

    print(row("european, four inputs", f"{timed(run, backend=backend, repeats=repeats):.1f}"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default="numpy", choices=["numpy", "torch", "jax"])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--gradients", action="store_true", help="also measure the cost of a backward pass"
    )
    args = parser.parse_args()

    print(f"# Monte Carlo benchmark — backend: {args.backend}")
    benchmark_simulation(args.backend, args.repeats)
    benchmark_pricing(args.backend, args.repeats)
    benchmark_antithetic(args.backend, args.repeats)
    if args.gradients:
        benchmark_gradients(args.backend, args.repeats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
