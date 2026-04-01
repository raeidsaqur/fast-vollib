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
