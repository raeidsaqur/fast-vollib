# Benchmarks

The hardware, backend, and package versions can change the measured throughput.
Throughput can also change if just-in-time (JIT) compilation occurs during the
measurement. This page gives instructions to run the tests from the repository.
For baseline results, refer to the published sources.

---

## Shipped benchmark artifacts

The fast-vollib repository contains the benchmark and validation files below.

- `scripts/benchmark_matrix.py` (also `python scripts/benchmark.py --matrix`)
  Same-host benchmark matrix with per-record timings and a manifest (below).
- `scripts/validation_matrix.py`
  Implied-volatility validation records for every route under the frozen
  protocol of `tests/fixtures/iv_adjudication/`.
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

### Benchmark matrix (same-host protocol)

```bash
python scripts/benchmark_matrix.py --out bench/ --sizes 100,1000,10000,100000,1000000
python scripts/benchmark_matrix.py --list-routes
```

The matrix measures fast-vollib and the baselines on the same host.
The baselines are `py_vollib`, `py_vollib_vectorized` and `py_lets_be_rational`.

- Each route, workload and batch size runs in a new process.
- The script checks the backend and the output before it measures time.
- On CUDA, the PyTorch IV check confirms that the fused Triton kernel runs.
- If a check fails, the script records the failure without a time.
- The script measures pricing, implied volatility and Greeks separately.
- Each IV route uses the same precomputed prices.
- The script records the first call with compilation and seven subsequent calls.
- The GPU times include transfers and synchronization. The GPU Halley routes also report device-resident times.
- The script measures scalar baselines through N = 10⁵. It marks measurements that exceed `--timeout` as censored.

`records.jsonl` has one record for each combination of route, workload and batch size.
`manifest.json` gives the source commit, library versions, CPU and GPU models, and thread settings.

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

## Jäckel IV solver: development history (v0.1.4)

The tables show kernel timings from v0.1.4 development on H100 NVL, for 100,000 options.
Run the benchmark matrix above for current timings.

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
| Triton single-pass (I-7)† | 0.056 | 2.1 | 9.3 × 10⁻¹⁴ |

† **Experimental kernel.** The direct Triton kernel meets the v0.2.5 tolerance on
4,230 of 5,474 strict grid rows. The public functions use a different route.
The error in the table applies only to the near-at-the-money v0.1.4 benchmark grid.

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
