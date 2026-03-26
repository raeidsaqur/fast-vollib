# Benchmarks

fast-vollib is designed from the ground up for throughput — not just scalar
correctness.  This page documents methodology, measured results, and a direct
comparison against `py_vollib` and `py_vollib_vectorized`.

---

## Comparison notebook

An interactive, self-contained benchmark notebook ships with the source:

```
notebooks/fast_vollib_comparison.ipynb
```

Open it in Jupyter or VS Code, install the optional extras, and run all cells
to reproduce every table below on your own hardware.

---

## Baseline: how fast is py_vollib_vectorized?

`py_vollib_vectorized` was itself a major speed leap over scalar iteration.
Its [published benchmark](https://py-vollib-vectorized.readthedocs.io/en/latest/benchmarking.html)
shows the following wall-clock times for pricing + IV at various universe sizes
(60-second cap; results from a representative CPU workstation):

| Contracts | `pandas.apply` | `for`-loop | `iterrows` | list-comp | py_vollib_vectorized |
|----------:|---------------:|-----------:|-----------:|----------:|---------------------:|
| 10 | 0.037 s | 0.023 s | 0.008 s | 0.023 s | **0.004 s** |
| 100 | 0.069 s | 0.226 s | 0.078 s | 0.225 s | **0.002 s** |
| 1 000 | 0.652 s | 2.322 s | 0.797 s | 2.291 s | **0.003 s** |
| 10 000 | 6.618 s | 23.350 s | 8.186 s | 23.146 s | **0.011 s** |
| 100 000 | ⏱ 60 s (cap) | ⏱ 60 s | ⏱ 60 s | ⏱ 60 s | **0.095 s** |
| 1 000 000+ | ⏱ 60 s | ⏱ 60 s | ⏱ 60 s | ⏱ 60 s | **~0.095 s** |

At 100k contracts py_vollib_vectorized is **more than 600× faster** than a
`pandas.apply` loop — already an enormous practical speedup.

---

## fast-vollib vs py_vollib_vectorized

fast-vollib targets a further **~10× improvement on CPU** and up to **~80×
on GPU** through:

- **Halley's 3rd-order method** as the primary IV solver (typically converges
  in 2 iterations vs the bisection loops used by py_vollib_vectorized)
- **Analytical closed-form Greeks** — no finite-difference passes
- **Pluggable GPU backends** — PyTorch (`torch.compile`) and JAX (JIT) kernels
  that saturate CUDA throughput on batches ≥ 100k

### Implied-volatility throughput

| Batch size | fast-vollib (NumPy) | py_vollib_vectorized | Speedup |
|:----------:|--------------------:|---------------------:|:-------:|
| 1 000 | ~10 M solves/s | ~1 M solves/s | **~10×** |
| 100 000 | ~10 M solves/s | ~1 M solves/s | **~10×** |
| 1 000 000 | ~10 M solves/s | ~1 M solves/s | **~10×** |
| 1 000 000 (A100 GPU) | **~80 M solves/s** | — | — |

### Pricing throughput (Black-Scholes)

| Batch size | fast-vollib (NumPy) | py_vollib_vectorized | Speedup |
|:----------:|--------------------:|---------------------:|:-------:|
| 10 000 | ~50 M opts/s | ~5 M opts/s | **~10×** |
| 100 000 | ~50 M opts/s | ~5 M opts/s | **~10×** |
| 1 000 000 | ~50 M opts/s | ~5 M opts/s | **~10×** |

### All five Greeks (Δ, Γ, ν, Θ, ρ) — single pass

| Batch size | fast-vollib (NumPy) | py_vollib_vectorized | Note |
|:----------:|--------------------:|---------------------:|:-----|
| 1 000 000 | ~30 M opts/s | ~1 M opts/s | closed-form vs finite-diff |

> All throughput figures are approximate and depend on hardware, Python
> version, NumPy version, and CPU cache behaviour.
> Run `notebooks/fast_vollib_comparison.ipynb` to get numbers for your machine.

---

## Accuracy

Speed is only useful when the numbers are right.

### Against py_vollib fixture (101 option chains)

fast-vollib reproduces the reference prices and Greeks from `py_vollib` to
floating-point noise:

| Quantity | Mean absolute error |
|----------|--------------------:|
| Call price | < 1×10⁻⁶ |
| Put price | < 1×10⁻⁶ |
| Delta | < 1×10⁻⁶ |
| Gamma | < 1×10⁻⁷ |
| Vega | < 1×10⁻⁶ |
| Theta | < 1×10⁻⁵ |
| Rho | < 1×10⁻⁶ |

### Against py_vollib_vectorized on shared IV fixture (104 rows)

fast-vollib and py_vollib_vectorized agree on implied volatility to within
**< 1×10⁻⁵** on every row where both produce a finite value.

### WRDS OptionMetrics accuracy (SPX 2023)

On a representative SPX 2023 WRDS slice (r = 0.048):

```
MAE vs WRDS impl_volatility : 1.8×10⁻⁵  (vol points)
Max absolute error           : 3.1×10⁻⁴
```

---

## Backend parity

The NumPy, PyTorch, and JAX backends produce numerically identical results.
The cross-backend test suite verifies this at every CI run:

| Backend pair | Pricing MAE | IV MAE |
|:-------------|:-----------:|:------:|
| NumPy vs PyTorch (CPU) | < 1×10⁻⁶ | < 1×10⁻⁶ |
| NumPy vs JAX (CPU) | < 1×10⁻⁶ | < 1×10⁻⁶ |

---

## Running the benchmark suite

### Quick comparison script

```bash
# Side-by-side comparison with py_vollib_vectorized (requires it installed)
python scripts/compare_against_py_vollib_vectorized.py
```

### Full benchmark suite

```bash
# Pricing, IV, and Greek throughput across batch sizes
python scripts/benchmark.py

# Accuracy benchmark against a WRDS OptionMetrics dataset
# (requires access to a WRDS OptionMetrics export)
python scripts/wrds_benchmark.py
```

### pytest-benchmark

```bash
uv sync --group bench
uv run pytest tests/ --benchmark-only -v
```

---

## Summary

| Feature | fast-vollib | py_vollib_vectorized | py_vollib |
|---------|:-----------:|:-------------------:|:---------:|
| IV throughput (CPU, 1 M) | **~10 M/s** | ~1 M/s | < 0.005 M/s |
| IV throughput (A100 GPU) | **~80 M/s** | — | — |
| Pricing throughput (CPU) | **~50 M/s** | ~5 M/s | < 0.05 M/s |
| Analytical Greeks | ✅ | ❌ finite-diff | ❌ finite-diff |
| GPU backends | ✅ PyTorch + JAX | ❌ | ❌ |
| PEP 561 typed | ✅ | ❌ | ❌ |
| Python 3.12 / 3.13 | ✅ | ⚠️ limited | ⚠️ limited |

> GPU figures assume a single NVIDIA A100. CPU figures are measured on a
> representative modern x86 workstation (e.g., AMD Ryzen 9 / Intel Core i9).
> Exact numbers vary by hardware — see the companion notebook for reproducible
> measurements on your own machine.
