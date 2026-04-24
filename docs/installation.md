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

## TestPyPI development snapshots

Development builds are published to TestPyPI with VCS-derived versions such as `0.1.2.dev3`.

### pip

```bash
pip install --pre \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  fast-vollib
```

### uv

```bash
uv pip install --pre \
  --index https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  fast-vollib
```

Use the normal PyPI channel if you want stable tagged releases only.

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

### Numba backend

```bash
pip install "fast-vollib[numba]"
# or
uv add "fast-vollib[numba]"
```

### Runtime type checking (optional)

Opt-in shape-aware checking of the public API via `jaxtyping` + `beartype`.
Annotations are already present in the library (PEP 563 strings, zero cost
by default); installing this extra only enables the `install_import_hook`
used by `fast_vollib._typing.enable_runtime_checks()`.

```bash
pip install "fast-vollib[typecheck]"
# or
uv add "fast-vollib[typecheck]"
```

See [API Reference → Runtime type checking](api.md#runtime-type-checking)
for usage.

### Multiple backends

```bash
pip install "fast-vollib[torch,jax,numba]"
# or
uv add "fast-vollib[torch,jax,numba]"
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

# Both backends on CPU/MPS (macOS or Linux without CUDA)
uv sync --all-groups --extra torch --extra jax

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
