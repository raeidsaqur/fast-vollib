from __future__ import annotations

from . import jax_backend, numpy_backend, torch_backend


def get_module(name: str):
    if name == "torch":
        return torch_backend
    if name == "jax":
        return jax_backend
    return numpy_backend


def available_backends() -> list[str]:
    backends = ["numpy"]
    if torch_backend.is_available():
        backends.append("torch")
    if jax_backend.is_available():
        backends.append("jax")
    return backends
