# Benchmarks

## Running the benchmarks

```bash
# Functional benchmark — pricing, IV, and Greeks throughput
python scripts/benchmark.py

# Accuracy benchmark against a WRDS OptionMetrics dataset
python scripts/wrds_benchmark.py

# Side-by-side comparison with py_vollib_vectorized
python scripts/compare_against_py_vollib_vectorized.py
```

For the WRDS benchmark you will need access to an OptionMetrics dataset placed
at the path expected by `scripts/wrds_benchmark.py`.

---

## Benchmark slices

| Slice | Description |
|---|---|
| Pricing throughput | Calls / second for `vectorized_black_scholes` at 1 k, 100 k, 1 M inputs |
| IV throughput | Solves / second for `vectorized_implied_volatility` |
| Greeks throughput | Calls / second for `get_all_greeks` |
| Backend parity | MAE between NumPy, Torch, and JAX across all models |
| WRDS accuracy | MAE vs. WRDS `impl_volatility` column on SPX 2023 data |

---

## Key results

| Configuration | IV throughput |
|---|---|
| NumPy (compiled bisection fallback) | ~10 M solves / s |
| PyTorch (CPU) | ~12 M solves / s |
| PyTorch (A100 GPU) | ~80 M solves / s |
| JAX (CPU, JIT) | ~15 M solves / s |

> Numbers are approximate and depend heavily on hardware and batch size.
> Run the benchmark suite on your own hardware to get accurate figures.

---

## Accuracy

On a representative SPX 2023 WRDS slice (r = 0.048):

```
MAE vs WRDS impl_volatility: 1.8e-5  (mean absolute error in vol points)
Max absolute error:          3.1e-4
```

---

## Running the pytest-benchmark suite

```bash
uv sync --group bench
uv run pytest tests/ --benchmark-only -v
```
