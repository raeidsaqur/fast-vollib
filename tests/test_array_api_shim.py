"""The ``surface._xp`` alias must keep resolving to the promoted namespace."""

from __future__ import annotations

import numpy as np

from fast_vollib import _array_api
from fast_vollib.surface import _xp


def test_shim_reexports_are_the_same_objects() -> None:
    assert _xp.ArrayNS is _array_api.ArrayNS
    assert _xp.get_namespace is _array_api.get_namespace
    assert _xp.numpy_namespace is _array_api.numpy_namespace


def test_shim_namespace_is_usable() -> None:
    ns = _xp.get_namespace(np.zeros(3))
    assert ns.name == "numpy"
    assert ns is _array_api.numpy_namespace()
