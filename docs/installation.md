# Installation

## Requirements

- Python **3.11** or later
- NumPy ≥ 1.26, SciPy ≥ 1.13 (pulled in automatically)

---

## pip

```bash
pip install fastiv
```

## uv

```bash
uv add fastiv
```

---

## Optional extras

fastiv ships optional extras for GPU and alternate numeric backends.

### PyTorch backend

```bash
pip install "fastiv[torch]"
# or
uv add "fastiv[torch]"
```

### JAX backend

```bash
pip install "fastiv[jax]"
# or
uv add "fastiv[jax]"
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
git clone https://github.com/raeid-saqur/fastiv.git
cd fastiv

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
import fastiv
print(fastiv.__version__)   # e.g. "0.1.0"
print(fastiv.get_backend()) # "numpy", "torch", or "jax"
```
