# Benchmarks

The benchmark tooling compares `fastiv` against the local sibling clone:

```bash
python scripts/benchmark.py
python scripts/compare_against_py_vollib_vectorized.py
```

Planned benchmark slices:

- pricing throughput
- implied volatility throughput
- greeks throughput
- backend parity for NumPy, Torch, and JAX
- large-batch GPU smoke tests on Phoenix or equivalent CUDA hosts
