"""Backward-compatible imports for the renamed vectorized compatibility module."""

from .py_vollib_vectorized import patch_py_vollib, patch_py_vollib_vectorized

__all__ = ["patch_py_vollib", "patch_py_vollib_vectorized"]
