# Benchmarks

fast-vollib ships benchmark and validation artifacts, but the exact throughput
you observe depends heavily on hardware, backend choice, package versions, and
whether JIT compilation has already been warmed up. This page focuses on what
is reproducible from the repository and what should be cited from upstream
baselines.

---

## Shipped benchmark artifacts

The repository includes four reproducibility entry points:

- `notebooks/fast_vollib_comparison.ipynb`
  Interactive notebook for pricing, IV, and Greeks comparisons.
- `scripts/benchmark.py`
  Quick local timing script for pricing, IV, and all-Greeks runs.
- `scripts/compare_against_py_vollib_vectorized.py`
  Numerical parity check against an installed `py_vollib_vectorized`.
- `scripts/wrds_benchmark.py`
  WRDS OptionMetrics validation script for local institutional datasets.

The notebook is the most complete entry point. The scripts are useful for quick
CI-style or terminal-based checks.

---

## Upstream baseline reference

`py_vollib_vectorized` publishes its own benchmarking page here:

- [py_vollib_vectorized benchmarking](https://py-vollib-vectorized.readthedocs.io/en/latest/benchmarking.html)

Those published upstream timings are a useful historical CPU baseline for the
Python ecosystem. The table below reproduces the values reported on that page.

| Contracts | `pandas.apply` | `for`-loop | `iterrows` | list-comp | `py_vollib_vectorized` |
|----------:|---------------:|-----------:|-----------:|----------:|-----------------------:|
| 10 | 0.037 s | 0.023 s | 0.008 s | 0.023 s | 0.004 s |
| 100 | 0.069 s | 0.226 s | 0.078 s | 0.225 s | 0.002 s |
| 1,000 | 0.652 s | 2.322 s | 0.797 s | 2.291 s | 0.003 s |
| 10,000 | 6.618 s | 23.350 s | 8.186 s | 23.146 s | 0.011 s |
| 100,000 | 60 s cap | 60 s cap | 60 s cap | 60 s cap | 0.095 s |

One important implementation note: the current `py_vollib_vectorized` package
uses `py_lets_be_rational` / Peter Jaeckel's Let’s Be Rational machinery for
implied volatility. It should not be described as a simple
`numpy.vectorize(brentq)` wrapper.

---

## Running the local fast-vollib benchmarks

### Quick timing pass

```bash
python scripts/benchmark.py
```

This script prints timing summaries for:

- Black-Scholes pricing
- implied volatility inversion
- `get_all_greeks`

It also reports NumPy-only timings and a larger-batch timing pass.

### Numerical parity against `py_vollib_vectorized`

```bash
python scripts/compare_against_py_vollib_vectorized.py
```

Expected output is the maximum absolute pricing and IV difference between the
two libraries on a shared synthetic fixture.

### WRDS validation

```bash
python scripts/wrds_benchmark.py --instrument-dir /path/to/wrds/export
```

This script requires local access to WRDS OptionMetrics data and is intended
for institutional environments. It reports aggregate error metrics only.

### Notebook workflow

Open:

```text
notebooks/fast_vollib_comparison.ipynb
```

Run all cells after installing the optional extras for the backend you want to
test.

---

## Jäckel IV solver performance

The `jackel/` module provides a machine-precision IV solver using Jäckel's
*"Let's Be Rational"* algorithm.  The table below shows the optimisation
trajectory on the canonical benchmark grid (N = 100,000 options,
H100 NVL GPU).

### CPU chain (NumPy → Numba, 100k options)

| Stage | Time (ms) | Speedup | Max rel err |
|---|---:|---:|---|
| NumPy baseline (I-1) | 106.5 | 1× | 3.2 × 10⁻¹⁴ |
| + fused vega (I-3) | 88.9 | 1.2× | 3.2 × 10⁻¹⁴ |
| + Numba Householder (I-4) | 58.6 | 1.8× | 3.2 × 10⁻¹⁴ |
| + Numba boundary kernel (I-4b) | 37.0 | 2.9× | 3.2 × 10⁻¹⁴ |
| + Hermite initial guess (I-4c/d) | 15.5 | 6.9× | 1.7 × 10⁻¹⁵ |
| + Numba Hermite kernel (I-4e) | 11.6 | 9.2× | 1.7 × 10⁻¹⁵ |
| + Numba preproc/postproc **(I-4f)** | **8.5** | **12.5×** | **2.2 × 10⁻¹⁵** |

### GPU backends (100k options, CUDA events)

| Backend | Compute (ms) | Wall-clock (ms) | Max rel err |
|---|---:|---:|---|
| torch.compile (I-5) | 2.7 | 4.8 | 2.8 × 10⁻¹⁵ |
| JAX lax.fori_loop (I-6) | 2.4 | 2.4 | 4.9 × 10⁻¹⁵ |
| **Triton single-pass (I-7)** | **0.056** | 2.1 | 9.3 × 10⁻¹⁴ |

The Triton kernel is **11× faster** than the 0.636 ms Halley×8 target and
**1905× faster** than the CPU baseline.

To reproduce:

```bash
uv run python scripts/jackel_triton_bench.py
```

---

## Reporting benchmark results responsibly

When quoting fast-vollib timings, include:

- hardware model
- backend (`numpy`, `torch`, or `jax`)
- Python and package versions
- whether the result is pre- or post-warmup
- batch size and data generation protocol

The repository intentionally does not hard-code a single set of local
fast-vollib throughput tables in the docs, because those numbers drift quickly
with compiler, CUDA, and CPU/GPU changes.
