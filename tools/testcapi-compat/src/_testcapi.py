"""Minimal stand-in for CPython's private ``_testcapi`` module.

Why this exists
---------------
``py_lets_be_rational`` 1.0.x — the version ``py_vollib_vectorized`` jits
against with numba — reads two float limits out of CPython's *internal test*
extension module::

    # py_lets_be_rational/constants.py
    from _testcapi import DBL_MIN, DBL_MAX

``_testcapi`` is a build artefact of CPython's own test suite.  Most
distributions ship it, but some do not: notably the ``install_only``
python-build-standalone builds that ``uv python install`` provides strip the
test modules.  On such an interpreter the import above raises
``ModuleNotFoundError`` and takes ``py_vollib_vectorized`` down with it.  That
is unavoidable wherever a uv-managed interpreter is the only route to a recent
CPython — for instance when the platform ships no system CPython >= 3.12.

Pinning forward is not an option: upstream ``py_lets_be_rational`` 1.1.x drops
the ``_testcapi`` hack but restructures the module such that numba's nopython
type inference fails inside ``py_vollib_vectorized``'s IV solver.  That surfaces
as a runtime ``TypingError`` rather than a clean ``ImportError``, which the
lazy-import guards in :mod:`fast_vollib.compat` cannot catch.  So 1.0.x plus
this shim is the combination that works — see the ``py-lets-be-rational`` pin in
the ``dev`` dependency group.

Safety
------
``lib-dynload`` precedes ``site-packages`` on ``sys.path``, so on an interpreter
that *does* ship the real ``_testcapi`` this module is shadowed and never
imported.  It only takes effect where the genuine article is absent.

The values are exactly what the C module exposes: ``DBL_MIN``/``DBL_MAX`` are
the C ``double`` limits, which CPython also publishes via ``sys.float_info``.

This is a development/benchmarking aid only.  Nothing in ``fast_vollib``'s
runtime imports it; it exists so the reference implementations used for parity
testing can be installed.
"""

from __future__ import annotations

import sys

__all__ = ["DBL_MIN", "DBL_MAX", "DBL_EPSILON"]

DBL_MIN: float = sys.float_info.min
DBL_MAX: float = sys.float_info.max
DBL_EPSILON: float = sys.float_info.epsilon
