# Installation

## Requirements

- Python **3.11** or later
- NumPy ≥ 1.26, SciPy ≥ 1.13 (pulled in automatically)

---

## pip

```bash
pip install fast-vollib
```

## uv

```bash
uv add fast-vollib
```

---

## Optional extras

fast-vollib ships optional extras for GPU and alternate numeric backends.

### PyTorch backend

```bash
pip install "fast-vollib[torch]"
# or
uv add "fast-vollib[torch]"
```

### JAX backend

```bash
pip install "fast-vollib[jax]"
# or
uv add "fast-vollib[jax]"
```

### GPU (Linux only — CUDA 13)

Installs PyTorch from the CUDA 13.0 wheel index plus JAX with CUDA 13 support:

```bash
uv sync --extra cuda
```

> **Note:** The `cuda` extra is Linux-only and requires a CUDA 13.x-capable GPU and driver.

---

## Development install

Clone the repository and install with all optional groups:

```bash
git clone https://github.com/raeidsaqur/fast-vollib.git
cd fast-vollib

# CPU-only (default)
uv sync --all-groups

# GPU (Linux)
uv sync --all-groups --extra cuda
```

### Optional dependency groups

| Group | What it installs |
|---|---|
| `docs` | MkDocs + Material theme for building the documentation site |
| `bench` | pytest-benchmark and RAPIDS packages for benchmarking |

```bash
uv sync --group docs   # docs only
uv sync --group bench  # benchmarks only
```

---

## Verifying the install

```python
import fast_vollib
print(fast_vollib.__version__)   # e.g. "0.1.0"
print(fast_vollib.get_backend()) # "numpy", "torch", or "jax"
```
