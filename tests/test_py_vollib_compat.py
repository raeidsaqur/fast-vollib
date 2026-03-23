from __future__ import annotations

import pytest

from fast_vollib import patch_py_vollib


def test_patch_py_vollib_smoke() -> None:
    py_vollib = pytest.importorskip("py_vollib")
    patch_py_vollib()
    import py_vollib.black
    import py_vollib.black_scholes
    import py_vollib.black_scholes_merton

    assert callable(py_vollib.black.black)
    assert callable(py_vollib.black_scholes.black_scholes)
    assert callable(py_vollib.black_scholes_merton.black_scholes_merton)
